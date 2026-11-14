"""Entry point: the JSON-lines bot the arena talks to (rule 5.1).

The loop is deliberately boring and unbreakable:
- read one line, parse, dispatch;
- every handler runs under a hard deadline (`watchdog.guarded`) with a guaranteed reply;
- stdout carries protocol lines only; all logs go to stderr;
- nothing ends the container except `shutdown` or a closed stdin (rule 5.10).

Only the arena's five messages exist: prep / start / turn / end / shutdown.
"""
from __future__ import annotations

import json
import sys

from harness import actions, config, decide, memory, prep, watchdog
from harness.llm import Gateway
from harness.log import kv, log, log_exc
from harness.state import Session


def emit(payload: dict) -> None:
    """The only function allowed to touch stdout."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------- handlers
def handle_prep(msg: dict, session: Session) -> dict:
    """Becoming the specialist: 600 s to turn rules into a spec, a playbook and (optionally) code."""
    session.gateway.configure(msg.get("llm"), cap=config.PREP_BUDGET)
    result = prep.run_prep(msg, session.gateway)
    session.adopt(result)
    return {"type": "ready"}


def handle_start(msg: dict, session: Session) -> dict:
    """A game begins: load the prepared variant, greet, answer `ready` inside the study clock."""
    session.gateway.configure(msg.get("llm"), cap=config.MATCH_BUDGET * config.MATCH_CONTAINER_SHARE)
    session.begin_game(msg)
    study = float(msg.get("study_seconds") or 30)
    game_id = session.game_id

    # Memory first: a cold container inherits what a specialist already built for this game.
    artifacts = prep.load_artifacts(game_id)
    if artifacts is not None:
        session.adopt(artifacts)
    elif session.spec.game_id != game_id:
        session.cold_start(msg)
        if study >= 45 and session.gateway.affordable:
            spec = prep.quick_spec(session.gateway, game_id, session.rules, timeout=min(45.0, study * 0.55))
            if spec is not None:
                session.spec = spec
                session._rebuild()
    elif session.spec.game_id == game_id:
        session._rebuild()

    log(kv(start=msg.get("match_id"), game=game_id, variant=(msg.get("variant") or {}).get("label"),
           seat=msg.get("seat"), opponent=msg.get("opponent"), family=session.spec.family,
           study=study, source=session.spec.source))

    chat = decide.decide_ready(
        gateway=session.gateway,
        spec=session.spec,
        playbook=session.playbook,
        chat_policy=session.chat,
        msg=msg,
        deadline=max(2.0, study - config.SAFETY),
    )
    return {"type": "ready", "chat": chat}


def handle_turn(msg: dict, session: Session) -> dict:
    """One decision: move + chat together, verified, inside the move clock."""
    session.turn += 1
    seconds = float(msg.get("seconds") or 20)
    if session.spec.game_id != session.game_id:
        # Defensive: a `turn` before `start` (or a dropped spec) must not stop us from playing.
        session.cold_start({**session.game, "game_id": session.game_id, "rules": session.rules})

    msg = dict(msg)
    msg["_avoid"] = list(session.avoid)
    decision = decide.decide_turn(
        gateway=session.gateway,
        spec=session.spec,
        playbook=session.playbook,
        engine=session.engine,
        chat_policy=session.chat,
        msg=msg,
        rules=session.rules,
        deadline=max(2.0, seconds - config.SAFETY),
    )
    session.note_action(decision.action)
    log(kv(turn=msg.get("turn"), action=decision.action, source=decision.source,
           chat_chars=len(decision.chat), spend=session.gateway.summary()["spend_usd"]))
    return {"type": "action", "action": decision.action, "chat": decision.chat}


def handle_end(msg: dict, session: Session) -> None:
    session.games_played += 1
    log(kv(end=msg.get("game_index"), result=msg.get("result"), reason=str(msg.get("reason"))[:120],
           match_score=msg.get("match_score"), match_over=msg.get("match_over")))
    decide.observe_end(session.chat, msg, session.game)  # opponent chat profile, for later games
    memory.record_result(
        session.game_id,
        {
            "match": session.game.get("match_id"),
            "opponent": session.game.get("opponent"),
            "game_index": msg.get("game_index"),
            "variant": (session.game.get("variant") or {}).get("label"),
            "seat": session.game.get("seat"),
            "result": msg.get("result"),
            "reason": msg.get("reason"),
            "scores": msg.get("scores"),
            "match_score": msg.get("match_score"),
            "family": session.spec.family,
            "spec_source": session.spec.source,
            "llm": session.gateway.summary(),
        },
    )


# --------------------------------------------------------------------------- fallbacks
def turn_fallback(msg: dict, session: Session) -> dict:
    """Last line of defense: a legal move and a chat line, immediately.

    An empty action is a warning, and the arena re-sends the turn with a fresh clock; a timeout loses the
    game outright (rule 8.2). So we always answer, with the best legal move we can name.
    """
    legal = msg.get("legal_actions")
    action = actions.choose_fallback(legal, session.avoid) or ""
    chat = session.chat.safe_line() if session.chat else "Your move."
    return {"type": "action", "action": action, "chat": chat or "Your move."}


def ready_fallback(session: Session) -> dict:
    chat = session.chat.safe_line() if session.chat else ""
    return {"type": "ready", "chat": chat or "Good luck."}


# --------------------------------------------------------------------------- main loop
def main() -> int:
    # Observations, rules and chat are UTF-8; never let a locale surprise turn them into mojibake.
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    session = Session(gateway=Gateway())
    log(kv(boot=config.VERSION, team=config.TEAM_ID, memory=config.MEMORY_DIR, genpolicy=config.GENPOLICY))

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            return 0
        if not line:
            log("stdin closed: exiting")
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            if not isinstance(msg, dict):
                raise ValueError("not an object")
        except ValueError:
            log(f"ignored non-JSON line: {line[:160]}")
            continue

        kind = msg.get("type")
        try:
            if kind == "prep":
                seconds = float(msg.get("seconds") or 600)
                reply, timed_out = watchdog.guarded(
                    lambda: handle_prep(msg, session),
                    max(5.0, seconds - config.SAFETY * 2),
                    lambda: {"type": "ready"},
                    label="prep",
                )
                if timed_out:
                    log("prep overran: answering ready with whatever we have")
                emit(reply or {"type": "ready"})

            elif kind == "start":
                study = float(msg.get("study_seconds") or 30)
                reply, _ = watchdog.guarded(
                    lambda: handle_start(msg, session),
                    max(2.0, study - config.SAFETY),
                    lambda: ready_fallback(session),
                    label="start",
                )
                emit(reply or ready_fallback(session))

            elif kind == "turn":
                seconds = float(msg.get("seconds") or 20)
                reply, _ = watchdog.guarded(
                    lambda: handle_turn(msg, session),
                    max(1.5, seconds - config.SAFETY),
                    lambda: turn_fallback(msg, session),
                    label="turn",
                )
                emit(reply or turn_fallback(msg, session))

            elif kind == "end":
                handle_end(msg, session)  # no answer; never exits, a specialist plays many matches

            elif kind == "shutdown":
                log(kv(shutdown="bye", **session.summary()))
                break

            else:
                log(f"ignored unknown message type: {kind!r}")
        except Exception as exc:  # noqa: BLE001 - the process must outlive any bug
            log_exc(f"handler {kind} crashed:", exc)
            if kind == "start":
                emit(ready_fallback(session))
            elif kind == "turn":
                emit(turn_fallback(msg, session))

    return 0


if __name__ == "__main__":
    sys.exit(main())
