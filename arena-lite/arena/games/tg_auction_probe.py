"""Team adn test game (Deals family): first-price sealed-bid auction with private values.

Both players get a private value for the item and bid at the same time. The higher bid wins the item and
pays its bid, so the profit is value minus bid. It exercises hidden information (private values) plus
simultaneous moves, which is exactly the combination the generic path struggles with.

    python -m arena practice --games-dir games --game tg_auction_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

import re

from ..core import Game

_BID = re.compile(r"^(?:bid|offer)?\s*(\d+)$")


class TestAuction(Game):
    id = "tg_auction_probe"
    name = "Sealed-Bid Auction (test)"
    category = "deals"
    blurb = "One private value, one blind bid, and a profit that depends on both."
    howto = "Both players bid at the same time for an item worth a private value to each. Highest bid wins and pays its bid."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "values 1-20, 3 rounds", "max_value": 20, "rounds": 3},
            {"label": "values 1-100, 1 round", "max_value": 100, "rounds": 1},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.max_value = max(5, min(100, int(self.variant.get("max_value", 20))))
        self.rounds = max(1, min(6, int(self.variant.get("rounds", 3))))
        self.label = str(self.variant.get("label", ""))
        self.profit = [0, 0]
        self.round = 1
        self.values = self._draw()
        self.history: list[str] = []
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""

    def _draw(self) -> list[int]:
        return [self.rng.randint(1, self.max_value), self.rng.randint(1, self.max_value)]

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""SEALED-BID AUCTION - {self.label}. 2 players, simultaneous bids.

GOAL
Earn more profit than your opponent. Each round both players receive a private value for one item,
between 1 and {self.max_value}. Both bid at the same time without seeing the other bid.

PLAY
There are {self.rounds} rounds. The higher bid wins the item and pays its bid; profit = value - bid.
Equal bids mean nobody wins the item (profit 0 for both). Bids may be any integer from 0 to {self.max_value}.
After the last round, the player with more total profit wins; equal profit is a draw.

ACTION FORMAT
"bid <n>" with 0 <= n <= {self.max_value}, for example "bid 7" or "bid 0". Case does not matter. An
illegal bid is refused (first time: warning and the same turn again; second time: you lose the game).

WHAT YOU OBSERVE EACH TURN
Your private value, the profit so far, the previous rounds with both values and bids, and the legal bids
when there are at most 50 of them."""

    def to_move(self) -> list[int]:
        return [] if self.over else [0, 1]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self.over:
            return []
        return self.cap_list([f"bid {n}" for n in range(self.max_value + 1)])

    def normalize(self, seat: int, action: str) -> str | None:
        if self.over:
            return None
        match = _BID.match(self.clean(action).strip(' .!?"\'`'))
        if not match:
            return None
        value = int(match.group(1))
        if not 0 <= value <= self.max_value:
            return None
        return f"bid {value}"

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        first = self.normalize(0, actions.get(0, ""))
        second = self.normalize(1, actions.get(1, ""))
        if first is None or second is None:
            return
        bids = [int(first.split()[-1]), int(second.split()[-1])]
        if bids[0] > bids[1]:
            self.profit[0] += self.values[0] - bids[0]
            result = "seat 0 wins the item"
        elif bids[1] > bids[0]:
            self.profit[1] += self.values[1] - bids[1]
            result = "seat 1 wins the item"
        else:
            result = "equal bids, item unsold"
        self.history.append(
            f"Round {self.round}: values {self.values[0]}/{self.values[1]} "
            f"bids {bids[0]}/{bids[1]} -> {result}"
        )
        if self.round >= self.rounds:
            self.over = True
            if self.profit[0] > self.profit[1]:
                self.winner = 0
            elif self.profit[1] > self.profit[0]:
                self.winner = 1
            self.end_reason = (f"profit {self.profit[0]} to {self.profit[1]}"
                               + (f", seat {self.winner} wins" if self.winner is not None else ", draw"))
            return
        self.round += 1
        self.values = self._draw()

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        return {"winner": self.winner, "scores": list(self.profit),
                "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {"label": self.label, "round": self.round, "profit": list(self.profit),
                "values": list(self.values), "history": list(self.history),
                "to_move": self.to_move(), "over": self.over}

    def observation(self, seat: int) -> str:
        legal = self.legal_actions(seat)
        shown = ", ".join(f'"{a}"' for a in (legal or [])[:10]) or "(more than 50 bids: use bid <n>)"
        return f"""SEALED-BID AUCTION - {self.label}. You are Player {seat}.
Round {self.round} of {self.rounds}. Your private value: {self.values[seat]}.
Profit so far: you {self.profit[seat]}, opponent {self.profit[1 - seat]}.
Previous rounds:
{chr(10).join("  " + line for line in self.history[-5:]) or "  (none yet)"}
It is YOUR turn: bid an integer from 0 to {self.max_value}.
Legal bids (first few): {shown}"""
