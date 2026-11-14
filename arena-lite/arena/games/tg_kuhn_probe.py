"""Team adn test game (Bluff family): Kuhn poker, the smallest hidden-information betting game.

Three cards (J, Q, K), one each, one betting round, fixed stakes. Small enough that a tabular CFR solution
is exact, which makes it the end-to-end probe for `harness/cfr.py` and the Bluff family. Two variants
differ only in the number of independent hands, so the CFR policy stays valid.

    python -m arena practice --games-dir games --game tg_kuhn_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

from ..core import Game

_RANKS = ("J", "Q", "K")


class TestKuhn(Game):
    id = "tg_kuhn_probe"
    name = "Kuhn Poker (test)"
    category = "bluff"
    blurb = "One card, one bet: is it a bluff or the best card in the deck?"
    howto = "Ante one chip, look at your card, then check or bet. Call or fold to a bet. Highest card wins a showdown."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "J-Q-K, one hand", "hands": 1},
            {"label": "J-Q-K, three hands", "hands": 3},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.hands = max(1, min(3, int(self.variant.get("hands", 1))))
        self.label = str(self.variant.get("label", ""))
        self.chips = [0, 0]
        self.hand = 1
        self.line: list[str] = []
        self.current = 0
        self.cards: list[str] = []
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""
        self._deal()

    def _deal(self) -> None:
        deck = list(_RANKS)
        self.rng.shuffle(deck)
        self.cards = deck[:2]
        self.line = []
        self.current = 0

    # ---------------------------------------------------------------- helpers
    def _legal(self) -> list[str]:
        if self.over:
            return []
        if not self.line:
            return ["check", "bet"]
        if self.line == ["check"]:
            return ["check", "bet"]
        if self.line in (["check", "bet"], ["bet"]):
            return ["call", "fold"]
        return []

    def _finish_hand(self, winner: int | None, stake: int, reason: str) -> None:
        if winner is not None:
            self.chips[winner] += stake
            self.chips[1 - winner] -= stake
        if self.hand >= self.hands:
            self.over = True
            self.winner = 0 if self.chips[0] > self.chips[1] else (1 if self.chips[1] > self.chips[0] else None)
            self.end_reason = (reason + f"; chips {self.chips[0]:+d}/{self.chips[1]:+d}"
                               + (f", seat {self.winner} wins" if self.winner is not None else ", draw"))
            return
        self.hand += 1
        self._deal()

    def _showdown(self, stake: int, how: str) -> None:
        winner = 0 if _RANKS.index(self.cards[0]) > _RANKS.index(self.cards[1]) else 1
        self._finish_hand(winner, stake, f"{how}: {self.cards[0]} vs {self.cards[1]}")

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""KUHN POKER - {self.label}. 2 players, one hidden card each.

GOAL
Win more chips than the opponent over {self.hands} hand(s). The deck is J, Q, K (J lowest, K highest),
each card once; the higher card wins a showdown. You see only your own card.

PLAY
Both players ante 1 chip. Player 0 acts first, then players alternate. One betting round per hand:
- with no bet, choose "check" or "bet" (a bet is 1 chip);
- after a check, the other player chooses "check" (showdown) or "bet";
- facing a bet, choose "call" (1 chip, showdown) or "fold" (give up the pot).
A showdown after two checks wins 1 chip; a showdown after a bet and a call wins 2; a fold gives the pot
to the other player. Cards are revealed only at a showdown. After the last hand the higher chip total wins.

ACTION FORMAT
Exactly one of: check, bet, call, fold (the legal ones are listed every turn).

WHAT YOU OBSERVE EACH TURN
Hand number, your card, the chip total, this hand's actions, and your legal moves."""

    def to_move(self) -> list[int]:
        return [] if self.over else [self.current]

    def legal_actions(self, seat: int) -> list[str] | None:
        if seat != self.current:
            return []
        return self.cap_list(self._legal())

    def normalize(self, seat: int, action: str) -> str | None:
        wanted = self.clean(action).strip(" .!\"'`")
        return wanted if wanted in self._legal() else None

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        action = actions.get(self.current)
        if action not in self._legal():
            return
        actor = self.current
        self.line.append(action)
        if action == "check":
            if self.line == ["check", "check"]:
                self._showdown(1, "showdown after two checks")
                return
            self.current = 1 - actor
            return
        if action == "bet":
            self.current = 1 - actor
            return
        if action == "fold":
            self._finish_hand(1 - actor, 1, f"seat {actor} folded")
            return
        if action == "call":
            self._showdown(2, "bet and call")
            return

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        return {"winner": self.winner, "scores": list(self.chips),
                "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {"label": self.label, "hand": self.hand, "hands": self.hands,
                "cards": list(self.cards), "chips": list(self.chips),
                "line": list(self.line), "to_move": self.to_move(), "over": self.over}

    def observation(self, seat: int) -> str:
        legal = self.legal_actions(seat)
        shown = " or ".join(f'"{a}"' for a in legal) if legal else "(not your turn)"
        return f"""KUHN POKER (J, Q, K). You are Player {seat}.
Hand {self.hand} of {self.hands}. Your card: {self.cards[seat]}
Chips: you {self.chips[seat]:+d}, opponent {self.chips[1 - seat]:+d}.
This hand so far: {", ".join(self.line) or "(no action yet)"}
It is YOUR turn: {shown}."""
