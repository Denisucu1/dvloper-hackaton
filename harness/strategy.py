"""A strategy brief for ONE game, written in prep with deep thinking.

Measured before this module existed: prep used 13 to 25 seconds of its 600 and a quarter of a cent of its
75 (pig, connect four, a negotiation and a poker test game). The playbook came from the same single call
that classifies the game, with thinking off. A game nobody has seen deserves more than that: here one
call with a large thinking budget works the game out from its rules - how the winner is really decided,
if-then rules with the numbers computed, what changes in the last turns, the moves that lose outright -
and the result rides in the playbook on every turn.

It is text, not code: it cannot make a move illegal and it cannot break the protocol. The solvers, the
simulator and the search still decide where they exist; the brief is what the model reads where they do not.
"""
from __future__ import annotations

import re

from . import config, memory
from .llm import Gateway
from .log import kv, log
from .spec import Spec

MARK = "=== STRATEGY BRIEF (worked out in prep with thinking) ==="

_SYSTEM = """You are preparing a specialist player for ONE two-player game that will be played many times in a
tournament. You have time to think: work the game out properly from its rules before you write. Compute
the numbers (odds, thresholds, midpoints, break-even points, how many rounds or resources are left) instead
of describing them. Plain text only, no markdown tables, no preamble."""

_USER = """THE GAME ({family} family). RULES OF EVERY VARIANT:
{rules}

Write the strategy brief, at most {limit} characters, in exactly these sections:

WIN CONDITION
How the winner is decided, including ties and what a draw is worth. What the score actually compares.

DECISION RULES
8 to 12 numbered rules of the form "IF <situation in this game's terms> THEN <move>", ordered by how often
they matter, with the numbers worked out from the rules. Cover the opening, the ordinary turn, being ahead,
being behind.

ENDGAME
What changes in the last turns or rounds, and what to do with leftover resources.

BLUNDERS
The 4 moves that lose the most in this game, each in one line.

PER VARIANT
One line for each variant whose best play differs, saying what changes.

Use only facts that are in the rules. If something cannot be known from the rules, do not invent it."""


def build(gateway: Gateway, rules_all: str, spec: Spec, timeout: float) -> str:
    """One call with a large thinking budget. Returns "" on any failure: the playbook stays as it was."""
    if not config.STRATEGY or not gateway.affordable or timeout < config.STRATEGY_MIN_SECONDS:
        return ""
    budget = config.STRATEGY_THINKING
    text = gateway.ask(
        _SYSTEM,
        _USER.format(family=spec.family or "unknown", rules=(rules_all or "")[: config.MAX_DOC_CHARS],
                     limit=config.STRATEGY_CHARS),
        max_tokens=budget + 3000,
        timeout=min(timeout, config.STRATEGY_TIMEOUT),
        temperature=0.2,
        thinking="enabled",
        budget_tokens=budget,
    )
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) < 200 or "DECISION RULES" not in text.upper():
        if text:
            log(f"strategy brief rejected ({len(text)} chars, no DECISION RULES section)")
        return ""
    text = text[: config.STRATEGY_CHARS + 400]
    log(kv(strategy_brief="ready", family=spec.family, chars=len(text)))
    return text


def merge(playbook: str, brief: str) -> str:
    """The brief goes first: it is the part worked out with thinking, and prompts clip from the end."""
    if not brief:
        return playbook
    rest = (playbook or "").split(MARK)[0].strip()
    return f"{MARK}\n{brief}\n\n=== PLAYBOOK NOTES ===\n{rest}".strip()


ARTIFACT = "strategy.txt"


def _path(game_id: object, rules: str):
    return memory.artifact_path(game_id or "unknown", memory.rules_fingerprint(rules or ""), ARTIFACT)


def load(game_id: object, rules: str) -> str:
    return memory.read_text(_path(game_id, rules), "")


def prepare(gateway: Gateway, game_id: object, rules: str, spec: Spec, timeout: float) -> str:
    """A cold container (no prep, 120 s study clock) still gets a brief: built at `start` from this
    variant's rules when the clock allows, cached by rules hash for every later game and container."""
    brief = load(game_id, rules)
    if brief:
        return brief
    brief = build(gateway, rules, spec, timeout)
    if brief:
        try:
            memory.atomic_write_text(_path(game_id, rules), brief)
        except Exception as exc:  # noqa: BLE001 - a cache, never a reason to fail
            log("strategy cache write failed:", type(exc).__name__)
    return brief
