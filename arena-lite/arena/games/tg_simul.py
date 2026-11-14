"""Team adn test game (Read the opponent family): simultaneous rock-paper-scissors over several rounds.

Not an official game. Both players move at the same time, so the chat is the only channel before a
round: it tests announcements, feints and whether we believe (or obey) what the opponent says it will play.

    python -m arena practice --games-dir games --game tg_simul --bot-a ... --bot-b ...
"""
from __future__ import annotations

from ..core import Game

_BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


class TestSimultaneous(Game):
    id = "tg_simul"
    name = "Duel of Signs (test)"
    category = "read"
    blurb = "Both hands open at once. Everything said before that is a claim."
    howto = "Each round both players pick rock, paper or scissors at the same time. Most points after the last round wins."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "9 rounds, every win 1 point", "rounds": 9, "weights": {"rock": 1, "paper": 1, "scissors": 1}},
            {"label": "9 rounds, rock wins 3", "rounds": 9, "weights": {"rock": 3, "paper": 1, "scissors": 1}},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.rounds = int(self.variant.get("rounds", 9))
        self.weights = dict(self.variant.get("weights") or {"rock": 1, "paper": 1, "scissors": 1})
        self.round = 1
        self.points = [0, 0]
        self.history: list[tuple[str, str]] = []
        self._result: dict | None = None

    def rules(self) -> str:
        w = self.weights
        return f"""DUEL OF SIGNS - {self.rounds} rounds, simultaneous moves.

EACH ROUND
Both players choose at the same time, without seeing the other's choice:
    rock      beats scissors
    paper     beats rock
    scissors  beats paper
The same sign twice is a tie (no points).

ACTION FORMAT
Exactly one of:  rock   paper   scissors

SCORING
A round won with rock scores {w['rock']} point(s), with paper {w['paper']}, with scissors {w['scissors']}.
After round {self.rounds} the player with more points wins; equal is a draw.
You see every earlier round. Chat sent with a move reaches the opponent before the next round.
"""

    def to_move(self) -> list[int]:
        return [] if self._result else [0, 1]

    def observation(self, seat: int) -> str:
        lines = [f"DUEL OF SIGNS. You are Player {seat}.",
                 f"Round {min(self.round, self.rounds)} of {self.rounds}. Points: you {self.points[seat]} - opponent {self.points[1 - seat]}.",
                 "", "Earlier rounds (you / opponent):"]
        if not self.history:
            lines.append("  (none yet)")
        for index, pair in enumerate(self.history, 1):
            mine, theirs = pair[seat], pair[1 - seat]
            outcome = "tie" if mine == theirs else ("you won" if _BEATS[mine] == theirs else "opponent won")
            lines.append(f"  Round {index}: {mine} / {theirs} -> {outcome}")
        lines.append("")
        lines.append(f"Game over: {self._result['reason']}." if self._result
                     else 'Choose now, at the same time as the opponent: "rock", "paper" or "scissors".')
        return "\n".join(lines)

    def legal_actions(self, seat: int) -> list[str] | None:
        return [] if self._result else ["rock", "paper", "scissors"]

    def normalize(self, seat: int, action: str) -> str | None:
        act = self.clean(action)
        return act if not self._result and act in _BEATS else None

    def apply(self, actions: dict[int, str]) -> None:
        if self._result:
            return
        a, b = (self.normalize(seat, actions.get(seat, "")) for seat in (0, 1))
        if a is None or b is None:
            raise ValueError(f"illegal actions: {actions!r}")
        self.history.append((a, b))
        if _BEATS[a] == b:
            self.points[0] += int(self.weights[a])
        elif _BEATS[b] == a:
            self.points[1] += int(self.weights[b])
        if self.round >= self.rounds:
            p, q = self.points
            winner = None if p == q else (0 if p > q else 1)
            scores = [0.5, 0.5] if winner is None else [int(winner == 0), int(winner == 1)]
            self._result = {"winner": winner, "scores": scores, "reason": f"points {p} to {q} after {self.rounds} rounds"}
        else:
            self.round += 1

    def is_over(self) -> bool:
        return self._result is not None

    def result(self) -> dict:
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def frame(self) -> dict:
        return {"round": self.round, "points": list(self.points), "history": [list(pair) for pair in self.history],
                "to_move": self.to_move(), "over": self.is_over(), "result": self._result}
