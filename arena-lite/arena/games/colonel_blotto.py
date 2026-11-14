"""Colonel Blotto with reserves (Read the opponent family).

RECONSTRUCTION, NOT THE OFFICIAL T1 ENGINE. This is rebuilt from the T1 brief and from our
`tg_blotto_probe` experiments so the pipeline and bench can be run against the T1 game id
`colonel_blotto`. The real T1 game may differ in details we could not observe.

Both players allocate troops to the same fields at the same time. Unlike the older probe, troops are
not reset every round: what a player does not spend stays in reserve for later rounds. Field wins
accumulate across rounds, and the player with more field wins at the end takes the game.

    python -m arena practice --games-dir games --game colonel_blotto --bot-a ... --bot-b ...
"""
from __future__ import annotations

import re
from math import comb

from ..core import MAX_LEGAL_LIST, Game

_TOKEN = re.compile(r"^[0-9]+$")


def _allocations(budget: int, fields: int) -> list[tuple[int, ...]]:
    """Every allocation of AT MOST `budget` troops across `fields` distinct fields."""
    if fields == 1:
        return [(n,) for n in range(budget + 1)]
    out: list[tuple[int, ...]] = []
    for first in range(budget + 1):
        for rest in _allocations(budget - first, fields - 1):
            out.append((first, *rest))
    return out


class ColonelBlotto(Game):
    id = "colonel_blotto"
    name = "Colonel Blotto"
    category = "read"
    blurb = ("Five fields, three rounds, one shrinking reserve: spend too much now and the later "
             "rounds are lost before they start.")
    howto = ("Both players split their remaining troops across the fields at the same time. More troops "
             "on a field wins it; unspent troops stay in reserve for the next round. Most field wins "
             "after the last round wins the game.")
    duplicate_safe = True  # simultaneous, but perfect information: no hidden state or future deal

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            # Too many allocations to list (about 96.5 million): legal_actions() is null.
            {"label": "5 fields, 100 troops, 3 rounds", "fields": 5, "troops": 100, "rounds": 3},
            # C(23, 3) = 1771 allocations at the start: still above the list cap, so null.
            {"label": "3 fields, 20 troops, 3 rounds", "fields": 3, "troops": 20, "rounds": 3},
            # C(7, 3) = 35 allocations at the start (at most 4 across 3 fields), under the
            # MAX_LEGAL_LIST cap, which lets scripts/game_smoke.py play the game generically.
            {"label": "3 fields, 4 troops, 3 rounds", "fields": 3, "troops": 4, "rounds": 3},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.fields = max(2, min(5, int(self.variant.get("fields", 5))))
        self.troops = max(2, min(100, int(self.variant.get("troops", 100))))
        self.rounds = max(1, min(6, int(self.variant.get("rounds", 3))))
        self.label = str(self.variant.get("label", ""))
        self.remaining = [self.troops, self.troops]
        self.points = [0, 0]
        self.round = 1
        self.history: list[dict] = []
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""

    # ---------------------------------------------------------------- helpers
    def _example(self) -> str:
        base, extra = divmod(self.troops, self.fields)
        return " ".join(str(base + (1 if i < extra else 0)) for i in range(self.fields))

    def _parse(self, seat: int, action: str) -> tuple[int, ...] | None:
        parts = self.clean(action).split()
        if len(parts) != self.fields or any(not _TOKEN.match(part) for part in parts):
            return None
        numbers = tuple(int(part) for part in parts)
        if sum(numbers) > self.remaining[seat]:
            return None
        return numbers

    def _history_lines(self, seat: int) -> list[str]:
        lines = []
        for record in self.history[-5:]:
            mine = " ".join(str(n) for n in record["allocation"][seat])
            theirs = " ".join(str(n) for n in record["allocation"][1 - seat])
            wins = record["field_wins"]
            lines.append(
                f"Round {record['round']}: you {mine} / opponent {theirs} "
                f"-> fields {wins[seat]}-{wins[1 - seat]}"
            )
        return lines

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""COLONEL BLOTTO - {self.label}. 2 players, simultaneous moves.

GOAL
Win more fields than your opponent over {self.rounds} rounds. Each player starts with {self.troops}
troops in reserve. Every round, both players allocate any non-negative integers to the {self.fields}
fields at the same time, without seeing the other allocation. On each field the side with more troops
wins that field; equal troops on a field score it for nobody. Unspent troops stay in YOUR reserve and
can be used in later rounds. Field wins accumulate over the whole game. Most field wins after the last
round wins; equal field wins is a draw.

ACTION FORMAT
Exactly {self.fields} non-negative integers separated by spaces, summing to AT MOST your remaining
troops, for example "{self._example()}". You may hold troops back by leaving the sum below your
reserve. Any other allocation is illegal (first time: warning and the same turn again; second time:
you lose the game).

WHAT YOU OBSERVE EACH TURN
The round number, both reserves, both field-win totals, the previous rounds with both allocations and
the fields each side won, and your legal allocations (the full list when there are at most 50 of them,
otherwise a note to write your own allocation)."""

    def to_move(self) -> list[int]:
        return [] if self.over else [0, 1]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self.over or seat not in (0, 1):
            return []
        budget = self.remaining[seat]
        # Count first: materialising ~96.5M strings just to discover they exceed the cap is not viable.
        if comb(budget + self.fields, self.fields) > MAX_LEGAL_LIST:
            return None
        return self.cap_list([" ".join(str(n) for n in allocation)
                              for allocation in _allocations(budget, self.fields)])

    def normalize(self, seat: int, action: str) -> str | None:
        if self.over or seat not in (0, 1):
            return None
        parsed = self._parse(seat, action)
        return " ".join(str(n) for n in parsed) if parsed is not None else None

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        first = self._parse(0, actions.get(0, ""))
        second = self._parse(1, actions.get(1, ""))
        if first is None or second is None:
            return
        wins = [0, 0]
        for field in range(self.fields):
            if first[field] > second[field]:
                wins[0] += 1
            elif second[field] > first[field]:
                wins[1] += 1
        self.points[0] += wins[0]
        self.points[1] += wins[1]
        self.remaining[0] -= sum(first)
        self.remaining[1] -= sum(second)
        self.history.append({
            "round": self.round,
            "allocation": [list(first), list(second)],
            "field_wins": list(wins),
        })
        if self.round >= self.rounds:
            self.over = True
            if self.points[0] > self.points[1]:
                self.winner = 0
            elif self.points[1] > self.points[0]:
                self.winner = 1
            self.end_reason = (
                f"field wins {self.points[0]} to {self.points[1]}"
                + (f", seat {self.winner} wins" if self.winner is not None else ", draw")
            )
            return
        self.round += 1

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        return {"winner": self.winner, "scores": list(self.points),
                "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {
            "label": self.label,
            "round": self.round,
            "rounds": self.rounds,
            "fields": self.fields,
            "troops": self.troops,
            "remaining": list(self.remaining),
            "points": list(self.points),
            "history": [
                {"round": record["round"],
                 "allocation": [list(record["allocation"][0]), list(record["allocation"][1])],
                 "field_wins": list(record["field_wins"])}
                for record in self.history
            ],
            "to_move": self.to_move(),
            "over": self.over,
            "winner": self.winner,
        }

    def observation(self, seat: int) -> str:
        other = 1 - seat
        legal = self.legal_actions(seat)
        if legal is None:
            shown = "(too many to list: write your own allocation)"
        else:
            shown = ", ".join(f'"{a}"' for a in legal)
        history = self._history_lines(seat) or ["(none yet)"]
        ending = (f"Game over: {self.end_reason}." if self.over
                  else (f'It is YOUR turn: allocate AT MOST {self.remaining[seat]} troops across '
                        f'{self.fields} fields as "{self._example()}" '
                        f"(non-negative integers, sum <= {self.remaining[seat]})."))
        return f"""COLONEL BLOTTO - {self.label}. You are Player {seat}.
Round {self.round} of {self.rounds}.
Reserve: you {self.remaining[seat]}, opponent {self.remaining[other]}.
Field wins: you {self.points[seat]}, opponent {self.points[other]}.
Previous rounds:
{chr(10).join("  " + line for line in history)}
{ending}
Legal allocations: {shown}"""
