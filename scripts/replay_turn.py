"""Re-decide one turn of a recorded match with the current harness: "what would we play now?"

    python scripts/replay_turn.py <replay.json> --team adn --game 3 --step 5 [--runs 5] [--env dev.env]

Takes the observation, the legal moves and the chat exactly as our bot had them at that step, runs
`decide_turn` on them with the gateway key from the env file, and prints the move and the chat line next
to what was played in the match. Used for the T1 -> T2 story (what the new chat defense would have done)
and as a regression check on real positions. The engine is the generic one: no prep artifacts are loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def chat_view(doc: dict, team: str, game_index: int, step: int) -> list[dict]:
    """The match chat as `team` saw it when deciding `step` of game `game_index`."""
    view = []
    for line in doc.get("chat", []):
        if line["game"] > game_index or (line["game"] == game_index and line.get("step", 0) >= step):
            continue
        view.append({"from": "you" if line["team"] == team else "opponent", "game": line["game"],
                     "game_name": line.get("game_name"), "phase": line.get("phase", "play"), "turn": line.get("step", 0),
                     "kind": line.get("kind", "chat"), "text": line["text"]})
    return view


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay")
    parser.add_argument("--team", default="adn")
    parser.add_argument("--game", type=int, required=True, help="game index in the match, from 1")
    parser.add_argument("--step", type=int, default=0, help="timeline step whose decision is replayed")
    parser.add_argument("--all", action="store_true", help="replay every decision of the game, in order")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--env", default=str(Path(__file__).resolve().parent.parent / "dev.env"))
    args = parser.parse_args()

    for line in Path(args.env).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.strip().split("=", 1)
            os.environ.setdefault(key, value)

    from harness import decide
    from harness.chatpolicy import ChatPolicy
    from harness.engine import Engine
    from harness.llm import Gateway
    from harness.spec import fallback_spec

    doc = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    game = next(g for g in doc["games"] if g["index"] == args.game)
    seat = game["seats"].index(args.team)
    spec = fallback_spec(game["game_id"], game["rules"])
    if not args.all:
        step = next(s for s in game["timeline"] if s.get("step") == args.step)
        played = (step.get("actions") or {}).get(str(seat))
        if not played:
            print(f"team {args.team} did not move at step {args.step}")
            return 1
        chat = chat_view(doc, args.team, args.game, args.step)
        print(f"{game['game_id']} [{game['variant'].get('label')}] step {args.step}, seat {seat}, family {spec.family}")
        print(f"played then : {played['action']!r}")
        opponent = (step.get("actions") or {}).get(str(1 - seat))
        if opponent:
            print(f"opponent    : {opponent['action']!r}")
    gateway = Gateway()
    if args.all:
        # The whole game through one ChatPolicy, as the specialist container lived it: claim checks run in
        # the background between turns, and each later decision sees what the earlier ones learned.
        policy = ChatPolicy(spec, seed=0)
        policy.opponent = ""
        same = changed = 0
        for item in game["timeline"]:
            mine = (item.get("actions") or {}).get(str(seat))
            if not mine or "observation" not in mine:
                continue
            upto = item.get("step", 0)
            seen = [line for line in chat_view(doc, args.team, args.game, upto)]
            decision = decide.decide_turn(
                gateway=gateway, spec=spec, playbook="", engine=Engine(spec, seed="0"), chat_policy=policy,
                msg={"observation": mine.get("observation", ""), "legal_actions": mine.get("legal_actions"),
                     "chat": seen, "turn": upto, "seat": seat},
                rules=game["rules"], deadline=17.0)
            if policy.cred_thread is not None:
                policy.cred_thread.join(timeout=40)  # the opponent's thinking time, in a real match
            level, true, false = policy.credibility()
            mark = "  " if decision.action == mine["action"] else "->"
            same += decision.action == mine["action"]
            changed += decision.action != mine["action"]
            print(f"step {upto:>2} then {mine['action']!r:8} {mark} now {decision.action!r:8} | trust {level} {true}T/{false}F | {decision.chat[:90]}")
        print(f"same {same}, changed {changed}, spend {gateway.summary()['spend_usd']}")
        return 0
    for run in range(args.runs):
        policy = ChatPolicy(spec, seed=run)
        decision = decide.decide_turn(
            gateway=gateway, spec=spec, playbook="", engine=Engine(spec, seed=str(run)), chat_policy=policy,
            msg={"observation": played.get("observation", ""), "legal_actions": played.get("legal_actions"),
                 "chat": chat, "turn": args.step, "seat": seat},
            rules=game["rules"], deadline=17.0)
        level, true, false = policy.credibility()
        print(f"now #{run + 1}     : {decision.action!r} ({decision.source}) | trust {level} {true}T/{false}F | {decision.chat}")
    print("spend", gateway.summary()["spend_usd"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
