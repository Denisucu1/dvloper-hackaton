"""Tests for the simultaneous-game matrix solver."""
from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.solver_matrix import MatrixSolver, solve_matrix
from harness.spec import Spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Row player's payoff; the column player gets 1 - value (zero-sum, and all values stay in [0, 1]).
RPS = [
    [0.0, 0.0, 1.0],  # rock beats scissors
    [1.0, 0.0, 0.0],  # paper beats rock
    [0.0, 1.0, 0.0],  # scissors beats paper
]
DOMINATED = [
    [0.9, 0.8, 0.7],
    [0.8, 0.85, 0.6],
    [0.2, 0.1, 0.3],  # strictly worse than row 0 in every column
]
DOMINANT = [
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [0.0, 0.0, 0.0],
]
WIDE = [[1.0 if row == column else 0.0 for column in range(48)] for row in range(48)]


class MatrixGame:
    """One-shot simultaneous game: score = matrix[our action][their action]."""

    def __init__(
        self,
        matrix: list[list[float]],
        actions: list[str] | None = None,
        picks: dict[int, str] | None = None,
    ) -> None:
        self.matrix = matrix
        self.actions = list(actions) if actions is not None else [str(i) for i in range(len(matrix))]
        self.picks = dict(picks or {})

    def update(self, observation: str) -> None:
        return None

    def legal_actions(self, seat: int) -> list[str]:
        return list(self.actions)

    def apply(self, seat: int, action: str) -> None:
        self.picks[int(seat)] = str(action)

    def is_terminal(self) -> bool:
        return len(self.picks) >= 2

    def to_move(self) -> list[int]:
        return [] if self.is_terminal() else [0, 1]

    def score(self, seat: int) -> float:
        if len(self.picks) < 2:
            return 0.5
        row = self.actions.index(self.picks[0])
        column = self.actions.index(self.picks[1])
        value = self.matrix[row][column]
        return value if int(seat) == 0 else 1.0 - value

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "MatrixGame":
        return type(self)(self.matrix, self.actions, self.picks)


class MatrixSolverTests(unittest.TestCase):
    def test_rps_mixture_is_roughly_uniform(self):
        mixture, value = solve_matrix(RPS, iterations=3000)
        self.assertEqual(len(mixture), 3)
        for probability in mixture:
            self.assertGreater(probability, 0.20)
            self.assertLess(probability, 0.50)
        self.assertAlmostEqual(sum(mixture), 1.0, places=9)
        self.assertTrue(0.0 <= value <= 1.0)

    def test_dominated_action_gets_almost_zero(self):
        mixture, _ = solve_matrix(DOMINATED, iterations=3000)
        self.assertLess(mixture[2], 0.02)
        self.assertGreater(mixture[0] + mixture[1], 0.98)

    def test_dominant_action_is_picked(self):
        mixture, _ = solve_matrix(DOMINANT, iterations=3000)
        self.assertGreater(mixture[0], 0.95)
        solver = MatrixSolver()
        model = MatrixGame(DOMINANT, ["good", "meh", "bad"])
        for seed in range(5):
            action, value = solver.propose_joint(model, 0, rng=random.Random(seed))
            self.assertEqual(action, "good")
            self.assertGreater(value, 0.9)

    def test_propose_joint_samples_a_legal_action(self):
        solver = MatrixSolver()
        model = MatrixGame(RPS, ["rock", "paper", "scissors"])
        picks = {
            solver.propose_joint(model, 0, rng=random.Random(seed))[0]
            for seed in range(20)
        }
        self.assertTrue(picks <= {"rock", "paper", "scissors"})
        self.assertGreaterEqual(len(picks), 2)

    def test_broken_model_never_raises(self):
        class BrokenClone(MatrixGame):
            def clone(self):
                raise RuntimeError("no clone")

        class BrokenActions(MatrixGame):
            def legal_actions(self, seat):
                raise RuntimeError("no actions")

        class BrokenMovers(MatrixGame):
            def to_move(self):
                raise RuntimeError("no movers")

        class Alternating(MatrixGame):
            def to_move(self):
                return [] if self.is_terminal() else [0]

        solver = MatrixSolver()
        for model in (
            None,
            BrokenClone(RPS),
            BrokenActions(RPS),
            BrokenMovers(RPS),
            Alternating(RPS),
            MatrixGame(RPS, picks={0: "rock", 1: "paper"}),
        ):
            action, value = solver.propose_joint(model, 0)
            self.assertEqual((action, value), (None, 0.5))

    def test_broken_apply_cells_fall_back_to_neutral(self):
        class BrokenApply(MatrixGame):
            def apply(self, seat, action):
                raise RuntimeError("cannot apply")

        action, value = MatrixSolver().propose_joint(BrokenApply(RPS), 0)
        self.assertIsNone(action)
        self.assertEqual(value, 0.5)

    def test_k48_is_fast(self):
        solver = MatrixSolver()
        model = MatrixGame(WIDE, [f"a{i}" for i in range(48)])
        started = time.monotonic()
        action, value = solver.propose_joint(model, 0, seconds=1.0, rng=random.Random(0))
        elapsed = time.monotonic() - started
        self.assertIn(action, model.actions)
        self.assertTrue(0.0 <= value <= 1.0)
        self.assertLess(elapsed, 0.5)

    def test_default_sampling_uses_fresh_entropy(self):
        solver = MatrixSolver()
        model = MatrixGame(RPS, ["rock", "paper", "scissors"])
        picks = {solver.propose_joint(model, 0)[0] for _ in range(40)}
        self.assertGreaterEqual(len(picks), 2)

    def test_matches_is_conservative(self):
        solver = MatrixSolver()
        self.assertTrue(
            solver.matches("some_game", Spec(game_id="some_game", family="read_opponent", simultaneous=True))
        )
        self.assertTrue(solver.matches("rps_duel", Spec(game_id="rps_duel", simultaneous=False)))
        self.assertTrue(solver.matches("tg_blotto_probe", Spec(game_id="tg_blotto_probe")))
        self.assertFalse(solver.matches("pig", Spec(game_id="pig", family="guessing", simultaneous=False)))
        self.assertFalse(solver.matches("philosophy", Spec(game_id="philosophy", simultaneous=False)))

    def test_propose_protocol_hook_declines_without_a_model(self):
        self.assertIsNone(
            MatrixSolver().propose(observation="choose", legal=["a", "b"], seat=0)
        )

    def test_module_loads_standalone_for_the_sandbox(self):
        """The sandbox loads this file by path next to search.py, so it must not use package imports."""
        path = os.path.join(ROOT, "harness", "solver_matrix.py")
        module_spec = importlib.util.spec_from_file_location("sandboxed_matrix", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        action, value = module.MatrixSolver().propose_joint(
            MatrixGame(RPS), 0, seconds=0.5, rng=random.Random(0)
        )
        self.assertIn(action, ["0", "1", "2"])
        self.assertTrue(0.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
