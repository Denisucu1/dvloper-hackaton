"""Team adn test game (Bluff family): one-card poker over several hands.

Not an official game. A hidden card, a bet, a call or a fold: enough to test bluffing in the chat and
to check that our own card never leaks into a chat line.

    python -m arena practice --games-dir games --game tg_bluff --bot-a ... --bot-b ...
"""
from __future__ import annotations

from ..core import Game


class TestBluff(Game):
    id = "tg_bluff"
    name = "One-Card Poker (test)"
    category = "bluff"
    blurb = "One card each, one bet: is that a bluff or the best card in the deck?"
    howto = "Each hand both players get one hidden card. Bet, call or fold; the higher card wins a showdown. Most chips after the last hand wins."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "3 cards, 6 hands", "cards": 3, "hands": 6},
            {"label": "9 cards, 8 hands", "cards": 9, "hands": 8},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.n_cards = int(self.variant.get("cards", 3))
        self.n_hands = int(self.variant.get("hands", 6))
        self.chips = [0, 0]  # net chips won so far
        self.hand = 0
        self.history: list[str] = []
        self._result: dict | None = None
        self._deal()

    def _deal(self) -> None:
        self.hand += 1
        deck = list(range(1, self.n_cards + 1))
        self.rng.shuffle(deck)
        self.cards = [deck[0], deck[1]]
        self.first = (self.hand - 1) % 2  # who acts first this hand
        self.current = self.first
        self.line: list[str] = []  # actions of this hand, in order

    def rules(self) -> str:
        return f"""ONE-CARD POKER - cards 1 to {self.n_cards}, {self.n_hands} hands.

SETUP
Each hand both players put 1 chip in the pot (the ante) and get one hidden card from a deck with the
cards 1 to {self.n_cards} (each card exists once, so the two cards are different). The higher card is better.
You see only your own card. The player who acts first alternates every hand; Player 0 starts hand 1.

A HAND
The first player chooses "check" or "bet" (a bet adds 1 chip).
- After a check, the second player chooses "check" (showdown) or "bet".
- After a bet, the other player chooses "call" (adds 1 chip, showdown) or "fold" (gives up the pot).
At a showdown the higher card wins the pot. Cards are revealed only at a showdown.

ACTION FORMAT
Exactly one of:  check   bet   call   fold   (the legal ones are listed on every turn).

SCORING
Each hand you win or lose 1 chip (showdown after two checks, or a fold) or 2 chips (showdown after a
bet and a call). After hand {self.n_hands} the player with more chips wins; equal is a draw.
"""

    def to_move(self) -> list[int]:
        return [] if self._result else [self.current]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self._result or seat != self.current:
            return []
        return ["call", "fold"] if self.line and self.line[-1] == "bet" else ["check", "bet"]

    def observation(self, seat: int) -> str:
        lines = [f"ONE-CARD POKER (cards 1-{self.n_cards}). You are Player {seat}.",
                 f"Hand {min(self.hand, self.n_hands)} of {self.n_hands}. Chips: you {self.chips[seat]:+d}, opponent {self.chips[1 - seat]:+d}.",
                 f"Your card: {self.cards[seat]}", "", "Earlier hands:"]
        lines += [f"  {entry}" for entry in self.history[-10:]] or ["  (none yet)"]
        lines.append("")
        lines.append("This hand so far: " + (", ".join(
            f"{'you' if (self.first + i) % 2 == seat else 'opponent'} {act}" for i, act in enumerate(self.line)) or "(no action yet)"))
        if self._result:
            lines.append(f"Game over: {self._result['reason']}.")
        elif seat == self.current:
            lines.append("It is YOUR turn: " + " or ".join(f'"{a}"' for a in self.legal_actions(seat)) + ".")
        else:
            lines.append("It is the opponent's turn.")
        return "\n".join(lines)

    def normalize(self, seat: int, action: str) -> str | None:
        act = self.clean(action)
        return act if act in (self.legal_actions(seat) or []) else None

    def apply(self, actions: dict[int, str]) -> None:
        if self._result:
            return
        seat = self.current
        act = self.normalize(seat, actions.get(seat, ""))
        if act is None:
            raise ValueError(f"illegal action for seat {seat}: {actions.get(seat)!r}")
        self.line.append(act)
        if act == "fold":
            self._settle(1 - seat, 1, f"seat {seat} folded")
        elif act == "call":
            self._showdown(2)
        elif act == "check" and len(self.line) == 2:
            self._showdown(1)
        else:
            self.current = 1 - seat

    def _showdown(self, stake: int) -> None:
        winner = 0 if self.cards[0] > self.cards[1] else 1
        self._settle(winner, stake, f"showdown {self.cards[0]} vs {self.cards[1]} (seat 0 vs seat 1)")

    def _settle(self, winner: int, stake: int, how: str) -> None:
        self.chips[winner] += stake
        self.chips[1 - winner] -= stake
        self.history.append(f"Hand {self.hand}: {', '.join(self.line)} -> {how}; seat {winner} won {stake}")
        if self.hand >= self.n_hands:
            a, b = self.chips
            winner_seat = None if a == b else (0 if a > b else 1)
            scores = [0.5, 0.5] if winner_seat is None else [int(winner_seat == 0), int(winner_seat == 1)]
            self._result = {"winner": winner_seat, "scores": scores, "reason": f"chips {a:+d} vs {b:+d} after {self.n_hands} hands"}
        else:
            self._deal()

    def is_over(self) -> bool:
        return self._result is not None

    def result(self) -> dict:
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def frame(self) -> dict:
        return {"hand": self.hand, "cards": list(self.cards), "chips": list(self.chips), "line": list(self.line),
                "history": list(self.history), "to_move": self.to_move(), "over": self.is_over(), "result": self._result}
