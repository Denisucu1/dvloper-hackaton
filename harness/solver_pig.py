"""Deterministic expert for Pig.

Pig is a one-step push-your-luck decision: the observation gives both banked scores, the
current turn total and the dice variant, so the solver can compare the exact expected value
of one more roll with the guaranteed value of holding. No state is kept between turns.
"""
from __future__ import annotations

import re
from typing import NamedTuple

_TARGET = re.compile(r"\bTarget:\s*(\d+)", re.IGNORECASE)
_TARGET_ALT = re.compile(r"\bfirst\s+to\s+(\d+)", re.IGNORECASE)
_BANKED = re.compile(
    r"^\s*(?:Banked|Score|Total)\s*:\s*you\s+(\d+)\s*,\s*opponent\s+(\d+)",
    re.IGNORECASE | re.MULTILINE,
)
_TURN_TOTAL = re.compile(r"\bTurn\s+total:\s*(\d+)", re.IGNORECASE)
_TWO_DICE = re.compile(r"\b(?:two|2)\s+dice\b", re.IGNORECASE)
_GAME_OVER = re.compile(r"\bgame\s+is\s+over\b|\bgame\s+over\b", re.IGNORECASE)
_OPPONENT_TURN = re.compile(r"\bit\s+is\s+the\s+opponent['\u2019]s\s+turn\b", re.IGNORECASE)


class _PigState(NamedTuple):
    """The numbers the decision needs, already parsed out of the turn text."""

    target: int
    dice: int
    banked: int
    opponent: int
    turn_total: int


def _parse_state(observation: str) -> _PigState | None:
    """Read target, dice variant and both scores from one observation."""
    target_match = _TARGET.search(observation) or _TARGET_ALT.search(observation)
    banked_match = _BANKED.search(observation)
    if target_match is None or banked_match is None:
        return None
    turn_match = _TURN_TOTAL.search(observation)
    dice = 2 if _TWO_DICE.search(observation[:200]) else 1
    return _PigState(
        target=int(target_match.group(1)),
        dice=dice,
        banked=int(banked_match.group(1)),
        opponent=int(banked_match.group(2)),
        turn_total=int(turn_match.group(1)) if turn_match else 0,
    )


def _roll_value(banked: int, turn_total: int, dice: int) -> float:
    """Expected banked score after one more roll followed by a hold.

    A bust keeps the bank but drops the turn total; snake eyes (two dice only) also clears
    the bank. The two-dice branch enumerates all 36 equally likely ordered outcomes.
    """
    if dice >= 2:
        total = 0.0
        for first in range(1, 7):
            for second in range(1, 7):
                if first == 1 and second == 1:
                    continue  # snake eyes: bank and turn total are both lost
                if first == 1 or second == 1:
                    total += banked  # turn lost, bank untouched
                else:
                    total += banked + turn_total + first + second
        return total / 36.0
    total = 0.0
    for face in range(1, 7):
        if face == 1:
            total += banked  # turn lost, bank untouched
        else:
            total += banked + turn_total + face
    return total / 6.0


def _should_roll(state: _PigState) -> bool:
    """Compare holding now with the expected value of one more roll."""
    hold = state.banked + state.turn_total
    if hold >= state.target:
        return False  # holding wins the game immediately
    if _roll_value(state.banked, state.turn_total, state.dice) > hold:
        return True
    # Cheap points are worth risking while behind; this also sends near-ties to the roll.
    return state.turn_total <= state.target // 4 and hold < state.opponent


def _legal_action(action: str, legal: list[str] | None) -> str | None:
    """Return the arena's exact spelling of ``action``, or None when it is absent."""
    if legal is None:
        return action
    for item in legal:
        text = str(item)
        if text.lower() == action:
            return text
    return None


class PigSolver:
    """Expected-value solver for the one-die and two-dice variants of Pig."""

    name = "pig"

    @classmethod
    def matches(cls, game_id: str, spec: object) -> bool:
        """Claim Pig by id, or a spec whose action shape is exactly roll/hold."""
        spec_id = str(getattr(spec, "game_id", "") or "")
        identifier = f"{game_id or ''} {spec_id}".lower()
        if "pig" in identifier:
            return True
        if str(getattr(spec, "family", "") or "").lower() == "pig":
            return True
        examples = " ".join(
            [str(getattr(spec, "action_format", "") or "")]
            + [str(item) for item in (getattr(spec, "action_examples", None) or [])]
            + [str(getattr(spec, "notes", "") or "")]
        ).lower()
        return "roll" in examples and "hold" in examples

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        """Return ``roll`` or ``hold`` for the current turn, or None if not understood."""
        if not observation or _GAME_OVER.search(observation) or _OPPONENT_TURN.search(observation):
            return None
        state = _parse_state(observation)
        if state is None:
            return None
        return _legal_action("roll" if _should_roll(state) else "hold", legal)
