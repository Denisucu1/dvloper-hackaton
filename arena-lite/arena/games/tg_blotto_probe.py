"""Team adn test game (Read the opponent family): Colonel Blotto with simultaneous allocation.

Not an official game. It mirrors the shape that cost us a T1 game: both sides split a fixed number of
troops across a few fields at the same time, scoring is per field, so the right move is a mixed strategy,
not a best response to a known opponent move.

    python -m arena practice --games-dir games --game tg_blotto_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

import re

from ..core import Game

_NUMS = re.compile(r"\d+")


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            out.append((first, *rest))
    return out


class TestBlotto(Game):
    id = "tg_blotto_probe"
    name = "Colonel Blotto (test)"
    category = "read"
    blurb = "Split your troops at the same time as your opponent; take more fields than them."
    howto = "Both players secretly split their troops across the fields. The side with more troops on a field wins it. Most fields wins the round."
    duplicate_safe = True  # simultaneous but perfect information, no hidden randomness

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "3 fields, 6 troops, 3 rounds", "fields": 3, "troops": 6, "rounds": 3},
            {"label": "3 fields, 10 troops, 1 round", "fields": 3, "troops": 10, "rounds": 1},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.fields = max(2, min(5, int(self.variant.get("fields", 3))))
        self.troops = max(2, min(20, int(self.variant.get("troops", 6))))
        self.rounds = max(1, min(6, int(self.variant.get("rounds", 3))))
        self.label = str(self.variant.get("label", ""))
        self.points = [0, 0]
        self.round = 1
        self.history: list[str] = []
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""

    # ---------------------------------------------------------------- helpers
    def _allocations(self) -> list[str]:
        return [" ".join(str(n) for n in combo) for combo in _compositions(self.troops, self.fields)]

    def _parse(self, action: str) -> tuple[int, ...] | None:
        numbers = [int(n) for n in _NUMS.findall(str(action))]
        if len(numbers) != self.fields or sum(numbers) != self.troops or any(n < 0 for n in numbers):
            return None
        return tuple(numbers)

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""COLONEL BLOTTO - {self.label}. 2 players, simultaneous moves.

GOAL
Both players split {self.troops} troops across {self.fields} fields. On each field, the side with more
troops wins that field. The side that wins more fields wins the round. Most rounds wins the game.

PLAY
There are {self.rounds} rounds. On every round BOTH players allocate at the same time, without seeing the
other allocation. Both allocations are revealed together. A field with equal troops is a draw and scores
for nobody. Total field wins are compared at the end of the game; equal totals are a draw.

ACTION FORMAT
Exactly {self.fields} non-negative integers that sum to {self.troops}, separated by spaces,
for example {" ".join(["2"] * self.fields) if self.fields else "2 2 2"}.
Any other allocation is illegal (first time: warning and the same turn again; second time: you lose).

WHAT YOU OBSERVE EACH TURN
The round number, the points so far, the previous rounds with both allocations, and your legal moves when
there are at most 50 of them."""

    def to_move(self) -> list[int]:
        return [] if self.over else [0, 1]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self.over or seat not in (0, 1):
            return []
        return self.cap_list(self._allocations())

    def normalize(self, seat: int, action: str) -> str | None:
        if self.over:
            return None
        parsed = self._parse(action)
        return " ".join(str(n) for n in parsed) if parsed else None

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        first = self._parse(actions.get(0, ""))
        second = self._parse(actions.get(1, ""))
        if first is None or second is None:
            return
        won = [0, 0]
        for f in range(self.fields):
            if first[f] > second[f]:
                won[0] += 1
            elif second[f] > first[f]:
                won[1] += 1
        self.points[0] += won[0]
        self.points[1] += won[1]
        self.history.append(
            f"Round {self.round}: you {' '.join(map(str, first))} / opponent "
            f"{' '.join(map(str, second))} -> fields {won[0]}-{won[1]}"
        )
        if self.round >= self.rounds:
            self.over = True
            if self.points[0] > self.points[1]:
                self.winner = 0
            elif self.points[1] > self.points[0]:
                self.winner = 1
            self.end_reason = (f"field points {self.points[0]} to {self.points[1]}"
                               + (f", seat {self.winner} wins" if self.winner is not None else ", draw"))
            return
        self.round += 1

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        return {"winner": self.winner, "scores": list(self.points),
                "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {"label": self.label, "round": self.round, "points": list(self.points),
                "history": list(self.history), "to_move": self.to_move(), "over": self.over}

    def observation(self, seat: int) -> str:
        legal = self.legal_actions(seat)
        shown = ", ".join(f'"{a}"' for a in (legal or [])[:12]) or "(too many to list, use the format)"
        return f"""COLONEL BLOTTO - {self.label}. You are Player {seat}.
Round {self.round} of {self.rounds}. Field points: you {self.points[seat]}, opponent {self.points[1 - seat]}.
Previous rounds:
{chr(10).join("  " + line for line in self.history[-5:]) or "  (none yet)"}
It is YOUR turn: allocate {self.troops} troops across {self.fields} fields as "{' '.join(["2"] * self.fields)}".
Legal allocations (first few): {shown}"""
