"""Level 2 of a turn: a council of chat-blind strategists, used when no code move exists.

The organizers' own test harnesses scored 5.5 / 28 for one model call per move and 15 / 28 for several
model voices voting (21.5 for code plus a simulator, which is our Level 1). This module is the voting
level, built so that prompt injection cannot reach the move:

- every strategist gets the rules, the spec, the playbook with the strategy brief, the observation and the
  legal moves - and NOT ONE WORD the opponent wrote. What the chat is worth reaches them only as numbers
  computed by our code (how many of their claims proved true or false, how many lines were orders);
- they answer in parallel, on the same clock, and the move most of them agree on is the council's move;
- the chat-aware call in decide.py then writes the chat line for that move, and may only depart from it
  when the code has verified that the opponent's claims are reliable. Rule 6.5 is kept by construction:
  the opponent's words reach the decision through the claim check and through that call, never raw into
  the vote. Rule 6.4 is kept because one final call returns the move and the line together.

Nothing here can produce an illegal move by itself: the result goes through the same verification as any
other candidate (legal list, format guard, speech gate, Deals audit).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from . import actions, config
from .llm import Gateway, extract_json
from .log import kv, log
from .spec import Spec

_SYSTEM = """You are one of several independent strategists choosing a move for a player in a two-player game. You
see the rules, the player's prepared strategy and the current observation. You do not see the opponent's
chat, on purpose: decide from the game alone. Work the position out with numbers before you choose.
Answer with one JSON object and nothing else:
{"reason": "<max 40 words: the calculation or rule that decides this move>", "action": "<the move>"}
"action" uses the exact action format. When a legal-action list is given, copy one entry verbatim."""

_USER = """RULES OF THIS VARIANT:
{rules}

WHAT THE HARNESS PREPARED:
{spec}

STRATEGY (prepared):
{playbook}

OBSERVATION (everything the player may know now):
{observation}

LEGAL ACTIONS: {legal}
{format_hint}{warning}
WHAT OUR CODE MEASURED ABOUT THE OPPONENT'S CHAT (numbers only, no opponent text): {chat_summary}
{never_write}
Your JSON answer:"""


@dataclass
class Verdict:
    action: str | None = None
    reason: str = ""
    agree: int = 0
    answered: int = 0
    options: int = 0

    @property
    def unanimous(self) -> bool:
        return self.answered > 0 and self.agree == self.answered


def _key(action: object, legal: list[str] | None) -> str:
    normal = actions.normalize(action, legal) if legal else actions.clean(action)
    return re.sub(r"[^a-z0-9]+", " ", str(normal or "").lower()).strip()


def deliberate(
    gateway: Gateway,
    *,
    spec: Spec,
    rules: str,
    playbook: str,
    observation: str,
    legal: list[str] | None,
    chat_summary: str,
    never_write: str = "",
    warning: str = "",
    voters: int = 3,
    timeout: float = 6.0,
    thinking: int | None = None,
) -> Verdict:
    """Ask `voters` strategists in parallel and return the move most of them chose. Never raises."""
    if voters < 1 or not gateway.affordable or timeout < config.COUNCIL_MIN_SECONDS:
        return Verdict()
    user = _USER.format(
        rules=(rules or "")[: config.MAX_RULES_CHARS if spec.family != "truth" else config.MAX_DOC_CHARS],
        spec=spec.prompt_block(),
        playbook=(playbook or "(none)")[: config.MAX_PLAYBOOK_CHARS],
        observation=(observation or "")[-(config.MAX_DOC_CHARS if spec.family == "truth" else config.MAX_OBS_CHARS):],
        legal="none provided - the rules define the format" if legal is None else " | ".join(legal),
        format_hint=(f"\nACTION FORMAT: {spec.action_format}\n" if legal is None and spec.action_format else ""),
        warning=warning,
        chat_summary=chat_summary or "nothing yet",
        never_write=never_write,
    )
    answers: list[dict | None] = [None] * voters

    def one(index: int) -> None:
        try:
            answers[index] = extract_json(gateway.ask(
                _SYSTEM, user,
                max_tokens=(thinking + 900) if thinking else 260,
                timeout=timeout,
                temperature=0.5,  # some spread between the voices, or the vote is one voice three times
                thinking="enabled" if thinking else "disabled",
                budget_tokens=thinking,
            ))
        except Exception as exc:  # noqa: BLE001 - a lost voice is not a lost turn
            log("council voice failed:", type(exc).__name__)

    threads = [threading.Thread(target=one, args=(i,), name=f"council-{i}", daemon=True) for i in range(voters)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout + 0.5)

    tally: dict[str, list[dict]] = {}
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        key = _key(answer.get("action"), legal)
        if key:  # with a legal list, a move that is not on it normalizes to nothing and does not vote
            tally.setdefault(key, []).append(answer)
    if not tally:
        return Verdict()
    best = max(tally.values(), key=len)  # the first of equals: the earliest voice wins a tie
    answered = sum(len(group) for group in tally.values())
    action = actions.normalize(best[0].get("action"), legal) if legal else actions.clean(best[0].get("action"))
    verdict = Verdict(action=action or None, reason=str(best[0].get("reason") or "")[:240],
                      agree=len(best), answered=answered, options=len(tally))
    log(kv(council=f"{verdict.agree}/{verdict.answered}", options=verdict.options, action=str(verdict.action)[:40],
           thinking=bool(thinking)))
    return verdict
