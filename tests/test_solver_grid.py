"""Tests for the ported Connect-K fallback solver, including row orientation."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ARENA = os.path.join(ROOT, "arena-lite")
if ARENA not in sys.path:
    sys.path.insert(0, ARENA)

from arena.games.connect_four import ConnectFour  # noqa: E402
from harness.solver_grid import ConnectKSolver, parse_connect_k  # noqa: E402


def make_obs(
    rows: int,
    cols: int,
    k: int,
    board: list[list[str]],
    labels: list[int] | None = None,
) -> str:
    """Build a solver-readable numeric Connect-K observation.

    `board` is top-to-bottom. `labels` can reorder the printed lines so a
    bottom-up board (labels descending) can be tested.
    """
    if labels is None:
        labels = list(range(1, rows + 1))
    row_by_label = {label: board[label - 1] for label in range(1, rows + 1)}
    lines = [f"CONNECT FOUR ({rows}x{cols}, connect {k})"]
    lines.append("Move 7. You are X (opponent is O).")
    for label in labels:
        lines.append(f"  {label} | " + " ".join(row_by_label[label]))
    open_cols = [c for c in range(cols) if board[0][c] == "."]
    lines.append("Open columns: " + ", ".join(str(c) for c in open_cols))
    lines.append('It is your turn. Answer "drop <column>".')
    return "\n".join(lines)


class TestSolverGrid(unittest.TestCase):
    def test_immediate_win(self) -> None:
        # Seven rows, six columns; X has three in a row on the bottom row.
        board = [["."] * 6 for _ in range(7)]
        board[6][1:4] = ["X", "X", "X"]
        solver = ConnectKSolver()
        self.assertIn(solver.propose(observation=make_obs(7, 6, 4, board)), ("drop 0", "drop 4"))

    def test_immediate_block(self) -> None:
        # O has three in a row on the bottom row; X must block at either end.
        board = [["."] * 6 for _ in range(7)]
        board[6][1:4] = ["O", "O", "O"]
        solver = ConnectKSolver()
        self.assertIn(solver.propose(observation=make_obs(7, 6, 4, board)), ("drop 0", "drop 4"))

    def test_wide_9x7_immediate_win(self) -> None:
        # Nine columns x seven rows, the wider Connect-5 shape used by the lab models.
        board = [["."] * 9 for _ in range(7)]
        board[6][1:4] = ["X", "X", "X"]
        solver = ConnectKSolver()
        self.assertIn(solver.propose(observation=make_obs(7, 9, 4, board)), ("drop 0", "drop 4"))

    def test_bottom_up_rows_are_reordered_by_label(self) -> None:
        # The game prints rows 7 -> 1 (bottom-up). Row 7 is the real bottom row and
        # holds X's winning three; a parser that trusts print order puts it on top
        # and misses the immediate win.
        board = [["."] * 6 for _ in range(7)]
        board[6][1:4] = ["X", "X", "X"]
        obs = make_obs(7, 6, 4, board, labels=[7, 6, 5, 4, 3, 2, 1])
        state = parse_connect_k(obs)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state["board"][6][1:4], ["X", "X", "X"])
        self.assertEqual(state["board"][0], ["."] * 6)
        self.assertIn(ConnectKSolver().propose(observation=obs), ("drop 0", "drop 4"))

    def test_real_arena_7x9_letter_columns_win_and_block(self) -> None:
        variant = {"label": "7x9, connect 5", "cols": 9, "rows": 7, "connect": 5}
        solver = ConnectKSolver()

        # Real arena observation shape: dash header (cols x rows), letter columns,
        # "Open columns: drop A, ..." and letter drop actions.
        game = ConnectFour(variant, seed=1)
        obs = game.observation(0)
        state = parse_connect_k(obs)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual((state["cols"], state["rows"], state["k"]), (9, 7, 5))
        self.assertTrue(state["letter_mode"])
        start_move = solver.propose(observation=obs)
        self.assertRegex(start_move, r"^drop [A-I]$")

        # Immediate win: X needs five, so four on the bottom row means drop A or F.
        game = ConnectFour(variant, seed=1)
        game.board[6][1:5] = [0, 0, 0, 0]
        win_move = solver.propose(observation=game.observation(0))
        self.assertIn(win_move, ("drop A", "drop F"))

        # Immediate block: O has the same four on the bottom row.
        game = ConnectFour(variant, seed=2)
        game.board[6][1:5] = [1, 1, 1, 1]
        block_move = solver.propose(observation=game.observation(0))
        self.assertIn(block_move, ("drop A", "drop F"))



if __name__ == "__main__":
    unittest.main()
