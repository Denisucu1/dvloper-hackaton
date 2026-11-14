"""Deterministic expert for Guess the Number.

The opponent's secret is a fixed number in a known range, and every guess is answered with
``higher`` or ``lower``. That makes the game a pure binary search: the whole observation text
is enough to recompute the remaining interval, so this solver keeps no state between turns.
"""
from __future__ import annotations

import re

_RANGE = re.compile(
    r"GUESS\s+THE\s+NUMBER\s*\(\s*(\d+)\s*(?:[-\u2013\u2014]|to|and)\s*(\d+)\s*\)",
    re.IGNORECASE,
)
_TURN_RANGE = re.compile(r"\bwith\s+n\s+from\s+(\d+)\s+to\s+(\d+)", re.IGNORECASE)
_RANGE_ALT = re.compile(
    r"(?:range|between)\s+(\d+)\s*(?:[-\u2013\u2014]|to|and)\s*(\d+)",
    re.IGNORECASE,
)
_OURS = re.compile(
    r"Your\s+guesses\s+at\s+the\s+opponent['\u2019]s\s+number\s*:",
    re.IGNORECASE,
)
_THEIRS = re.compile(
    r"Opponent['\u2019]s\s+guesses\s+at\s+your\s+number\s*:",
    re.IGNORECASE,
)
_GUESS = re.compile(
    r"you\s+guessed\s+(\d+)\s*(?:->|=>|\u2192)\s*(higher|lower|correct)",
    re.IGNORECASE,
)
_GAME_OVER = re.compile(r"\bgame\s+(?:is\s+)?over\b", re.IGNORECASE)
_OPPONENT_TURN = re.compile(r"\bit\s+is\s+the\s+opponent['\u2019]s\s+turn\b", re.IGNORECASE)


def _parse_range(observation: str) -> tuple[int, int] | None:
    """Return the inclusive hint range announced by the observation, if any."""
    for pattern in (_RANGE, _TURN_RANGE, _RANGE_ALT):
        match = pattern.search(observation)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            if lo <= hi:
                return lo, hi
    return None


def _our_guess_lines(observation: str) -> list[str]:
    """The lines describing guesses *we* made, never the opponent's guesses."""
    ours = _OURS.search(observation)
    start = ours.end() if ours else 0
    theirs = _THEIRS.search(observation, start)
    end = theirs.start() if theirs else len(observation)
    return observation[start:end].splitlines()


def _remaining_interval(observation: str) -> tuple[int, int] | None:
    """Recompute the still-possible interval for the opponent's number from the text alone."""
    bounds = _parse_range(observation)
    if bounds is None:
        return None
    lo, hi = bounds
    for line in _our_guess_lines(observation):
        match = _GUESS.search(line)
        if not match:
            continue
        number = int(match.group(1))
        feedback = match.group(2).lower()
        if feedback == "higher":
            lo = max(lo, number + 1)
        elif feedback == "lower":
            hi = min(hi, number - 1)
        else:  # correct: the secret is pinned down
            lo = hi = number
    return (lo, hi) if lo <= hi else None


def _from_legal(candidate: str, legal: list[str]) -> str | None:
    """Map ``candidate`` onto the arena's exact legal spelling, or give up."""
    items = [str(item) for item in legal]
    for item in items:
        if item == candidate:
            return item
    number = int(candidate.split()[1])
    for item in items:
        digits = re.findall(r"\d+", item)
        if len(digits) == 1 and int(digits[0]) == number:
            return item
    return None


class GuessingSolver:
    """Binary-search solver for number-guessing / higher-lower deduction games."""

    name = "guessing"

    @classmethod
    def matches(cls, game_id: str, spec: object) -> bool:
        """Be conservative: only claim games whose id or action shape says "guess"."""
        spec_id = str(getattr(spec, "game_id", "") or "")
        identifier = f"{game_id or ''} {spec_id}".lower()
        if "guess" in identifier:
            return True
        family = str(getattr(spec, "family", "") or "").lower()
        if family != "guessing":
            return False
        # A bare "guessing" family can describe other hidden-number games; require a deduction cue.
        if any(word in identifier for word in ("number", "deduc", "higher", "lower", "hot-cold", "hot_cold")):
            return True
        action_format = str(getattr(spec, "action_format", "") or "").lower()
        return "guess" in action_format

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        """Guess the midpoint of the remaining interval, never inventing an illegal string."""
        if not observation or _GAME_OVER.search(observation) or _OPPONENT_TURN.search(observation):
            return None
        interval = _remaining_interval(observation)
        if interval is None:
            return None
        lo, hi = interval
        candidate = f"guess {(lo + hi) // 2}"
        if legal is None:
            return candidate
        return _from_legal(candidate, legal)
