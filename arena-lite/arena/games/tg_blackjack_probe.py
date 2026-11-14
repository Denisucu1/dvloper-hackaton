"""Team adn test game (guess family): Blackjack Duel with a completely public deck.

One shared 52-card deck, hit or stand on your turn, bust over 21. No card is ever hidden: both hands,
both totals and the remaining deck are in every observation. The only randomness is the deck order, so
this is the public-chance probe for `harness/solver_chance.py` (a simulator can expose the pending draw
through `chance_actions()` and expectimax can average over the next card).

    python -m arena practice --games-dir games --game tg_blackjack_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

import random

from ..core import Game

_SUITS = ("S", "H", "D", "C")
_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
_VALUE = {"A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
          "10": 10, "J": 10, "Q": 10, "K": 10}


class TestBlackjack(Game):
    id = "tg_blackjack_probe"
    name = "Blackjack Duel (test)"
    category = "guess"
    blurb = "Hit or stand with every card face up: the deck is public, only its order is unknown."
    howto = ("Both players draw from one shared deck and try to finish closer to 21 without going over. "
             "On your turn hit for another card or stand to lock your total; over 21 busts and cannot win.")
    duplicate_safe = False  # the shuffled deck is hidden future randomness (core.Game.duplicate_safe)

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "52 cards, empty hands", "opening": 0},
            {"label": "52 cards, one opening card each", "opening": 1},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.opening = 1 if int(self.variant.get("opening", 0)) else 0
        self.label = str(self.variant.get("label", ""))
        deck = [rank + suit for suit in _SUITS for rank in _RANKS]
        self.rng.shuffle(deck)
        self.deck = deck                      # index -1 is the next card drawn
        self.hands: list[list[str]] = [[], []]
        self.done = [False, False]            # stood or busted: no more decisions
        self.busted = [False, False]
        self.current = 0
        for _ in range(self.opening):
            for seat in (0, 1):               # every opening card is public
                self.hands[seat].append(self.deck.pop())
        self.history: list[str] = []
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _total(cards: list[str]) -> int:
        """Best blackjack total: count aces as 11, then soften them while the hand is over 21."""
        total = 0
        aces = 0
        for card in cards:
            rank = card[:-1]  # the suit is the last character
            if rank == "A":
                total += 11
                aces += 1
            else:
                total += _VALUE[rank]
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    def _status(self, seat: int) -> str:
        if self.busted[seat]:
            return "busted"
        return "standing" if self.done[seat] else "hitting"

    def _setup_text(self) -> str:
        if self.opening:
            return "Each player is dealt one public opening card, then the first turn begins."
        return "Both players start with no cards (total 0)."

    def _finish(self, trigger: str) -> None:
        self.over = True
        t0, t1 = self._total(self.hands[0]), self._total(self.hands[1])
        b0, b1 = self.busted
        if b0 and b1:
            self.winner = None
            tail = "both players bust"
        elif b0:
            self.winner = 1
            tail = "seat 0 busts, seat 1 wins"
        elif b1:
            self.winner = 0
            tail = "seat 1 busts, seat 0 wins"
        elif t0 > t1:
            self.winner = 0
            tail = "seat 0 has the higher total"
        elif t1 > t0:
            self.winner = 1
            tail = "seat 1 has the higher total"
        else:
            self.winner = None
            tail = "equal totals"
        self.end_reason = f"{trigger}; totals {t0}-{t1} (seat 0 vs seat 1): {tail}"

    def _after_turn(self, seat: int) -> None:
        if not self.deck:
            self._finish("deck empty")
            return
        if self.done[0] and self.done[1]:
            self._finish("both players stood or busted")
            return
        self.current = 1 - seat if not self.done[1 - seat] else seat

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""BLACKJACK DUEL - {self.label}. 2 players, one shared 52-card deck, all cards public.

GOAL
Have the higher hand total that does not exceed 21. A hand over 21 busts and cannot win. Equal totals
are a draw; if both players bust, the game is a draw.

SETUP
One standard 52-card deck is shuffled once, before play, from the match seed. {self._setup_text()} Seat 0
takes the first turn. Every drawn card is public: both hands, both totals and the remaining deck are
visible to both players. No card is ever concealed from either player.

CARD VALUES
A = 11 or 1 (whichever gives the highest total without exceeding 21); 2-10 = face value; J, Q, K = 10.
A hand's total counts every ace as 11 first, then reduces it by 10 per ace while the total is over 21.

YOUR TURN
On your turn choose exactly one of:
  hit    draw the next card of the deck and add it to the drawing player's cards.
  stand  end your participation with your current total.
If your total exceeds 21 after a hit, you bust immediately: you keep the drawn cards, you take no further
turns, and you cannot win. A player who stands or busts is done; the turn then passes to the opponent
while the opponent is still in the hand (a player alone in the hand keeps drawing or standing). The game
ends when both players are done (both stood or busted) or as soon as the deck is empty.

SCORING
At the end compare the totals at or under 21: the higher total wins, equal totals are a draw, and if both
players bust the game is a draw. A player who busts loses to any player who did not bust, even when that
player's total is lower.

ACTION FORMAT
Reply with one word: hit | stand
Examples: "hit", "stand", "HIT", " Stand. " (case, spaces and surrounding punctuation do not matter).
Anything else is illegal (first time: a warning and the same turn again; second time: you lose the game).

WHAT YOU OBSERVE EACH TURN
Both hands with their card totals and status (hitting / standing / busted), whose turn it is, how many
cards remain with their ranks, and your legal actions."""

    def to_move(self) -> list[int]:
        return [] if self.over else [self.current]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self.over or seat != self.current:
            return []
        return self.cap_list(["hit", "stand"])

    def normalize(self, seat: int, action: str) -> str | None:
        if self.over or seat != self.current:
            return None
        wanted = self.clean(action).strip(" .!\"'`")
        return wanted if wanted in ("hit", "stand") else None

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        seat = self.current
        action = self.normalize(seat, actions.get(seat, ""))
        if action is None:
            raise ValueError(f"illegal action for seat {seat}: {actions.get(seat)!r}")
        if action == "stand":
            self.done[seat] = True
            self.history.append(f"seat {seat} stood at {self._total(self.hands[seat])}")
            self._after_turn(seat)
            return
        if not self.deck:  # defensive: a turn is never offered with an empty deck
            self._finish("deck empty")
            return
        card = self.deck.pop()
        self.hands[seat].append(card)
        total = self._total(self.hands[seat])
        if total > 21:
            self.busted[seat] = True
            self.done[seat] = True
            self.history.append(f"seat {seat} hit {card} and busted at {total}")
        else:
            self.history.append(f"seat {seat} hit {card} for {total}")
        self._after_turn(seat)

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        if not self.over:
            return {"winner": None, "scores": [0, 0], "reason": "game in progress"}
        if self.winner is None:
            scores = [0.5, 0.5]
        else:
            scores = [int(self.winner == 0), int(self.winner == 1)]
        return {"winner": self.winner, "scores": scores, "reason": self.end_reason}

    def observation(self, seat: int) -> str:
        opp = 1 - seat
        total = self._total(self.hands[seat])
        opp_total = self._total(self.hands[opp])
        mine = ", ".join(self.hands[seat]) or "(no cards)"
        theirs = ", ".join(self.hands[opp]) or "(no cards)"
        lines = [
            f"BLACKJACK DUEL - {self.label}. You are seat {seat}.",
            f"Your hand: {mine} = {total}" + (" (BUST)" if self.busted[seat] else ""),
            f"Opponent hand: {theirs} = {opp_total}" + (" (BUST)" if self.busted[opp] else ""),
            f"Your status: {self._status(seat)}. Opponent status: {self._status(opp)}.",
            f"Deck: {len(self.deck)} cards left" + self._deck_text() + ".",
        ]
        if self.over:
            lines.append(f"The game is over: {self.end_reason}.")
        else:
            lines.append("It is YOUR turn." if self.current == seat else "It is the opponent's turn.")
            legal = self.legal_actions(seat)
            lines.append("Legal actions: " + (" | ".join(legal) if legal else "(none; not your turn)") + ".")
        return "\n".join(lines)

    def _deck_text(self) -> str:
        counts = {rank: 0 for rank in _RANKS}
        for card in self.deck:
            counts[card[:-1]] += 1
        parts = [f"{rank} x{counts[rank]}" for rank in _RANKS if counts[rank]]
        return (" (remaining ranks: " + ", ".join(parts) + ")") if parts else ""

    def frame(self) -> dict:
        return {
            "label": self.label,
            "opening": self.opening,
            "hands": [list(cards) for cards in self.hands],
            "totals": [self._total(self.hands[0]), self._total(self.hands[1])],
            "status": [self._status(0), self._status(1)],
            "deck": list(self.deck),
            "deck_count": len(self.deck),
            "current": self.current,
            "history": list(self.history),
            "to_move": self.to_move(),
            "over": self.over,
            "result": self.result() if self.over else None,
        }

    def random_action(self, seat: int, rng: random.Random) -> str:
        """Seeded sample-play policy: keep drawing until a reasonable standing total."""
        total = self._total(self.hands[seat])
        if total <= 16:
            return "hit"
        return "stand"
