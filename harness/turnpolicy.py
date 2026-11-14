"""When a turn is worth spending thinking tokens on.

The gateway's Haiku thinks by default, but thinking tokens count against `max_tokens`, the move clock and
the match budget. We keep it off for cheap, mechanical turns (`has_code_move=True`) and enable bounded
thinking (`THINKING_BUDGET = 1024`) when no deterministic solver or simulator move exists on:
- `bluff` turns without a code move;
- non-numeric `guessing` turns without a code move;
- open-action-space (`legal is None`) turns in `read_opponent`, `deals`, `think_ahead`, or `truth`.
"""
from __future__ import annotations

import re

THINKING_BUDGET = 1024  # the gateway's minimum; the only value that is reliably bounded
# Rulebook 7.3: thinking can overrun its budget, so leave generous headroom for the visible answer.
MAX_TOKENS_WITH_THINKING = THINKING_BUDGET + 1600

_NUMERIC_ACTION = re.compile(r"^\s*(?:guess|pick|choose|say)?\s*[-+]?\d+\s*$", re.IGNORECASE)

_OPEN_SPACE_STRATEGIC_FAMILIES = frozenset(
    {
        "deals",
        "read_opponent",
        "think_ahead",
        "truth",
    }
)


def is_numeric_guessing(legal: list[str] | None) -> bool:
    """True when every legal action is just a number (the GuessingSolver's home turf)."""
    if not legal:
        return False
    return all(_NUMERIC_ACTION.match(str(action)) for action in legal)


def thinking_for(spec, *, has_code_move: bool, legal: list[str] | None) -> tuple[int, str] | None:
    """(budget_tokens, reason) when this turn deserves thinking, else None."""
    if has_code_move:
        return None
    family = str(getattr(spec, "family", "") or "").lower()
    if family == "bluff":
        return THINKING_BUDGET, "bluff without a code move"
    if family == "guessing" and not is_numeric_guessing(legal):
        return THINKING_BUDGET, "non-numeric deduction without a code move"
    if legal is None and family in _OPEN_SPACE_STRATEGIC_FAMILIES:
        return THINKING_BUDGET, f"{family} open-space without a code move"
    return None
