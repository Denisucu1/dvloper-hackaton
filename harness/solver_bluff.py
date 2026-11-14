"""Deterministic advisor for one-card poker style bluff games.

Why: on the Bluff family the bot was playing from the prompt alone. A tiny, exact pot-odds calculation
plus a read on the opponent's previous hands is cheaper and steadier than a language model guessing.

What it reads from the observation (the arena's text, no hidden state invented):
- the deck size from "(cards 1-N)" and our own card from "Your card: c";
- the current hand's action line from "This hand so far: ..." (check / bet / call / fold);
- the opponent's tendencies from "Earlier hands:" entries: how often they bet, call, fold and go to
  showdown. Entries are attributed by hand number, because the first player alternates every hand.

It never looks at, guesses or uses the opponent's card: only our card and the public action history.
Returns None when the text is not in this shape, so the caller falls back to the model.
"""
from __future__ import annotations

import re
from typing import Iterable

_DECK = re.compile(r"cards?\s*1\s*[-–]\s*(\d+)", re.IGNORECASE)
_CARD = re.compile(r"your card:\s*(\d+)", re.IGNORECASE)
_HAND = re.compile(r"Hand\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
_LINES = re.compile(r"This hand so far:\s*(.*)", re.IGNORECASE)
_EARLIER = re.compile(r"Earlier hands:\s*(.*?)(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)
_ENTRY = re.compile(r"Hand\s+(\d+):\s*(.*?)(?:\s*->\s*(.*))?$", re.IGNORECASE)
_ACTIONS = ("check", "bet", "call", "fold")


class BluffSolver:
    name = "bluff"

    @classmethod
    def matches(cls, game_id: str, spec=None) -> bool:
        gid = str(game_id or "").lower()
        if "bluff" in gid or "poker" in gid:
            return True
        family = str(getattr(spec, "family", "") or "").lower()
        return family == "bluff"

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        text = str(observation or "")
        deck = _DECK.search(text)
        card = _CARD.search(text)
        if not deck or not card:
            return None
        n_cards = int(deck.group(1))
        my_card = int(card.group(1))
        if n_cards < 2 or not 1 <= my_card <= n_cards:
            return None

        line = _LINES.search(text)
        current = self._current_actions(line.group(1) if line else "")
        if current is None:
            return None

        tendencies = self._tendencies(text, seat)
        choice = self._choose(my_card, n_cards, current, tendencies, seat, turn)
        if not choice:
            return None
        return self._legal(choice, legal)

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _current_actions(raw: str) -> list[str] | None:
        """Our current hand's public actions, in order (unknown words are ignored)."""
        raw = raw.strip().rstrip(".")
        if not raw or "no action yet" in raw.lower():
            return []
        acts = [word for word in re.findall(r"[a-z]+", raw.lower()) if word in _ACTIONS]
        if not acts and raw not in ("", "(no action yet)"):
            return None
        return acts

    @staticmethod
    def _tendencies(text: str, seat: int) -> dict:
        """Count OUR opponent's public actions, from the hand summaries.

        Entries are attributed by hand number: the first player alternates every hand, so the actor of
        action `i` in hand `h` is `(h - 1 + i) % 2`.
        """
        empty = {"bet": 0, "call": 0, "fold": 0, "check": 0, "showdowns": 0, "hands": 0}
        block = _EARLIER.search(text)
        if not block:
            return empty
        stats = dict(empty)
        for row in block.group(1).splitlines():
            row = row.strip()
            if not row:
                continue
            match = _ENTRY.search(row)
            if not match:
                continue
            hand_number = int(match.group(1))
            actions = [w for w in re.findall(r"[a-z]+", (match.group(2) or "").lower()) if w in _ACTIONS]
            outcome = (match.group(3) or "").lower()
            stats["hands"] += 1
            first_seat = (hand_number - 1) % 2
            for index, action in enumerate(actions):
                actor = (first_seat + index) % 2
                if actor != int(seat):
                    stats[action] += 1
            if "showdown" in outcome:
                stats["showdowns"] += 1
        return stats

    # ------------------------------------------------------------------ policy
    @staticmethod
    def _choose(my_card: int, n_cards: int, current: list[str], stats: dict, seat: int, turn: int) -> str:
        # Heads-up one-card poker: ante 1 + ante 1, a bet is 1. Pot odds for a call are 1 into 3 => 25%.
        opponents = max(1, n_cards - 1)
        equity = (my_card - 1) / opponents  # P(our card is higher), uniform over the remaining cards
        bet_opportunities = max(1, stats["bet"] + stats["check"] + stats["fold"] + stats["call"])
        fold_rate = stats["fold"] / bet_opportunities
        aggressive = stats["bet"] / max(1, stats["bet"] + stats["check"])

        facing_bet = bool(current) and current[-1] == "bet"
        if facing_bet:
            # Pot odds alone say 25%, but a bettor's range is stronger than average, so the call needs a
            # real edge: fold the bottom third instead of break-even hero calls. Loose bettors loosen it.
            threshold = max(0.25, 0.33 - 0.08 * aggressive)
            if equity >= threshold:
                return "call"
            return "fold"

        # No bet in front of us. Bet for value (thin on small decks), or bluff when they fold a lot.
        value_edge = 0.5 if n_cards <= 4 else 0.62
        if equity >= value_edge or (aggressive > 0.75 and equity >= 0.5):
            return "bet"
        bluff_spot = fold_rate >= 0.45 and equity < 0.4 and (turn + seat) % 3 == 0
        if bluff_spot:
            return "bet"
        return "check"

    @staticmethod
    def _legal(choice: str, legal: Iterable[str] | None) -> str | None:
        if not legal:
            return choice
        wanted = choice.lower()
        for action in legal:
            if str(action).strip().lower() == wanted:
                return str(action)
        return None
