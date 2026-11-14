"""When a turn is worth spending thinking tokens on.

The gateway's Haiku thinks by default, but thinking tokens count against `max_tokens`, the move clock and
the match budget. We keep it off for cheap, mechanical turns and enable it only where the model has to
reason without help from our code:

- turns where a solver or a generated simulator already produced a verified move: no thinking;
- Bluff turns where no code move exists: bounded thinking, because pot odds and reads are exactly the
  kind of reasoning a small model gets wrong when rushed;
- non-numeric Guessing turns (free-text deduction) with no solver move: bounded thinking.

Deliberately NOT enabled for numeric guessing (our GuessingSolver handles it) or Think ahead (the search
handles it) - the clock and the budget are better spent on search.
"""
from __future__ import annotations

import re

THINKING_BUDGET = 1024  # the gateway's minimum; the only value that is reliably bounded
# Rulebook 7.3: thinking can overrun its budget, so leave generous headroom for the visible answer.
MAX_TOKENS_WITH_THINKING = THINKING_BUDGET + 1600

_NUMERIC_ACTION = re.compile(r"^\s*(?:guess|pick|choose|say)?\s*[-+]?\d+\s*$", re.IGNORECASE)


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
    return None
