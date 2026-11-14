"""Probe: the matrix-game solver (``harness/solver_matrix.py``).

Three deterministic matrix fixtures plus one integration fixture:

- Rock-Paper-Scissors payoff matrix: fictitious play must return a mixture near uniform, every
  action above 20% (an exploitable pure strategy would fail);
- a strictly dominated row: its probability must stay below 2%;
- a dominant row: its probability above 98% and a seeded sample must actually pick it;
- a broken model (``legal_actions``/``clone`` raise) must degrade to ``(None, 0.5)``, never raise;
- a 48x48 simultaneous game, the maximum matrix size, must answer a legal move in under 0.5 s.

Score is the fraction of boolean checks that pass. All sampling uses ``random.Random(0)`` so the
probe is reproducible; the production default remains CSPRNG entropy.
"""
from __future__ import annotations

import random
import time

from harness.solver_matrix import MatrixSolver, solve_matrix

from scripts.probes.base import Probe, collect, register, summarize

RPS_PAYOFF = [
    [0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0],
    [0.0, 1.0, 0.5],
]
DOMINANT_PAYOFF = [[1.0, 1.0], [0.0, 0.0]]
# Row 2 is strictly worse than row 0 in every column; a solver that gives it weight is exploitable.
DOMINATED_PAYOFF = [[0.5, 0.5], [0.6, 0.4], [0.1, 0.1]]
MAX_ACTIONS = 48
SAMPLER = random.Random(0)


class RpsGame:
    """One-shot simultaneous rock-paper-scissors."""

    def __init__(self, played: list[tuple[int, str]] | None = None) -> None:
        self.played = list(played or [])

    def update(self, observation: str) -> None:
        return None

    def legal_actions(self, seat: int) -> list[str]:
        return ["rock", "paper", "scissors"]

    def apply(self, seat: int, action: str) -> None:
        self.played.append((int(seat), str(action)))

    def is_terminal(self) -> bool:
        return len(self.played) >= 2

    def to_move(self) -> list[int]:
        return [] if self.is_terminal() else [0, 1]

    def score(self, seat: int) -> float:
        if not self.is_terminal():
            return 0.5
        beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
        mine = next(action for who, action in self.played if who == seat)
        theirs = next(action for who, action in self.played if who != seat)
        if beats[mine] == theirs:
            return 1.0
        if beats[theirs] == mine:
            return 0.0
        return 0.5

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "RpsGame":
        return RpsGame(self.played)


class DominantGame:
    """Seat 0's action ``a`` wins against everything; ``b`` loses against everything."""

    def __init__(self, played: list[tuple[int, str]] | None = None) -> None:
        self.played = list(played or [])

    def legal_actions(self, seat: int) -> list[str]:
        return ["a", "b"]

    def apply(self, seat: int, action: str) -> None:
        self.played.append((int(seat), str(action)))

    def is_terminal(self) -> bool:
        return len(self.played) >= 2

    def to_move(self) -> list[int]:
        return [] if self.is_terminal() else [0, 1]

    def score(self, seat: int) -> float:
        if not self.is_terminal():
            return 0.5
        ours = next(action for who, action in self.played if who == seat)
        value = 1.0 if ours == "a" else 0.0
        return value if seat == 0 else 1.0 - value

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "DominantGame":
        return DominantGame(self.played)


class WideGame:
    """48 actions per seat, one simultaneous ply, neutral payoff: the matrix-size boundary."""

    def __init__(self, played: list[tuple[int, str]] | None = None) -> None:
        self.played = list(played or [])

    def legal_actions(self, seat: int) -> list[str]:
        return [f"a{index}" for index in range(MAX_ACTIONS)]

    def apply(self, seat: int, action: str) -> None:
        self.played.append((int(seat), str(action)))

    def is_terminal(self) -> bool:
        return len(self.played) >= 2

    def to_move(self) -> list[int]:
        return [] if self.is_terminal() else [0, 1]

    def score(self, seat: int) -> float:
        return 0.5

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "WideGame":
        return WideGame(self.played)


class BrokenGame:
    """A generated model that cannot enumerate or clone: the solver must decline, not raise."""

    def legal_actions(self, seat: int) -> list[str]:
        raise RuntimeError("legal_actions broken")

    def to_move(self) -> list[int]:
        return [0, 1]

    def is_terminal(self) -> bool:
        return False

    def score(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "BrokenGame":
        raise RuntimeError("clone broken")


@register
class MatrixProbe(Probe):
    name = "solvers.matrix"
    component = "matrix"

    def run(self) -> dict:
        facts: dict = {}

        def rps_is_uniform() -> bool:
            mixture, value = solve_matrix(RPS_PAYOFF, iterations=3000)
            facts["rps_mixture"] = [round(item, 4) for item in mixture]
            facts["rps_value"] = round(value, 4)
            return (
                len(mixture) == 3
                and abs(sum(mixture) - 1.0) < 1e-9
                and min(mixture) > 0.20
            )

        def dominated_action_is_dropped() -> bool:
            mixture, _value = solve_matrix(DOMINATED_PAYOFF, iterations=3000)
            facts["dominated_mixture"] = [round(item, 5) for item in mixture]
            return len(mixture) == 3 and mixture[2] < 0.02

        def dominant_action_gets_weight() -> bool:
            mixture, _value = solve_matrix(DOMINANT_PAYOFF, iterations=3000)
            facts["dominant_mixture"] = [round(item, 5) for item in mixture]
            return len(mixture) == 2 and mixture[0] > 0.98

        def dominant_action_is_sampled() -> bool:
            action, value = MatrixSolver().propose_joint(
                DominantGame(), 0, seconds=0.5, rng=SAMPLER,
            )
            facts["dominant_sample"] = {"action": action, "value": round(value, 4)}
            return action == "a"

        def broken_model_falls_back() -> bool:
            solver = MatrixSolver()
            broken = solver.propose_joint(BrokenGame(), 0, seconds=0.2)
            missing = solver.propose_joint(None, 0, seconds=0.2)
            facts["broken"] = {"broken": broken, "none": missing}
            return broken == (None, 0.5) and missing == (None, 0.5)

        def max_matrix_is_fast() -> bool:
            started = time.monotonic()
            action, _value = MatrixSolver().propose_joint(
                WideGame(), 0, seconds=0.5, rng=SAMPLER,
            )
            elapsed = time.monotonic() - started
            facts["k48"] = {"action": action, "elapsed_s": round(elapsed, 4)}
            return action in WideGame().legal_actions(0) and elapsed < 0.5

        checks = [
            ("RPS mixture is near uniform (each > 20%)", rps_is_uniform),
            ("dominated action receives < 2%", dominated_action_is_dropped),
            ("dominant action receives > 98%", dominant_action_gets_weight),
            ("dominant action is actually sampled", dominant_action_is_sampled),
            ("broken model returns (None, 0.5)", broken_model_falls_back),
            ("48x48 matrix answers in under 0.5 s", max_matrix_is_fast),
        ]
        run = collect(checks)
        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"RPS {facts.get('rps_mixture')}, dominated "
            f"{facts.get('dominated_mixture', [None])[-1]}, k48 "
            f"{facts.get('k48', {}).get('elapsed_s')} s"
        )
        return summarize(run, notes=notes, details=details)
