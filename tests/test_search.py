import importlib.util
import os
import time
import unittest

from harness import search

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TicTacToe:
    """Minimal perfect-information fixture: just enough to test the search, not a good game."""

    LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6))

    def __init__(self, board=None, seat=0):
        self.board = list(board or ["."] * 9)
        self.seat = seat

    # --- arena surface -----------------------------------------------------
    @staticmethod
    def _cell(action):
        column = "ABC".index(action[0].upper())
        row = int(action[1]) - 1
        return row * 3 + column

    @staticmethod
    def _name(index):
        return "ABC"[index % 3] + str(index // 3 + 1)

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [self._name(i) for i, cell in enumerate(self.board) if cell == "."]

    def apply(self, seat, action):
        index = self._cell(action)
        if self.board[index] != ".":
            raise ValueError("occupied")
        self.board[index] = "X" if seat == 0 else "O"

    def is_terminal(self):
        return self._winner() is not None or "." not in self.board

    def to_move(self):
        if self.is_terminal():
            return []
        x = self.board.count("X")
        o = self.board.count("O")
        return [0 if x == o else 1]

    def score(self, seat):
        winner = self._winner()
        if winner is None:
            return 0.5
        return 1.0 if winner == seat else 0.0

    def evaluate(self, seat):
        other = 1 - seat
        mine = sum(1 for line in self.LINES if all(self.board[i] != ("O" if seat == 0 else "X") for i in line))
        theirs = sum(1 for line in self.LINES if all(self.board[i] != ("X" if seat == 0 else "O") for i in line))
        return 0.5 + 0.05 * (mine - theirs) / 8.0

    def clone(self):
        return TicTacToe(self.board, self.seat)

    def key(self):
        return "".join(self.board)

    def _winner(self):
        for line in self.LINES:
            a, b, c = (self.board[i] for i in line)
            if a != "." and a == b == c:
                return 0 if a == "X" else 1
        return None


def clean(**kwargs):
    return search.best_action(**kwargs)


class SearchTests(unittest.TestCase):
    def test_takes_the_win(self):
        board = ["X", "X", ".", "O", "O", ".", ".", ".", "."]
        action, value = clean(model=TicTacToe(board), seat=0, seconds=0.5)
        self.assertEqual(action, "C1")
        self.assertGreater(value, 0.9)

    def test_blocks_the_loss(self):
        board = ["O", "O", ".", "X", ".", ".", ".", ".", "X"]
        action, _ = clean(model=TicTacToe(board), seat=0, seconds=0.5)
        self.assertEqual(action, "C1")

    def test_returns_a_legal_move_from_empty_board(self):
        model = TicTacToe()
        action, value = clean(model=model, seat=0, seconds=0.3, max_depth=9)
        self.assertIn(action, model.legal_actions(0))
        self.assertTrue(0.0 <= value <= 1.0)

    def test_terminal_state_has_no_move(self):
        board = ["X", "X", "X", "O", "O", ".", ".", ".", "."]
        action, value = clean(model=TicTacToe(board), seat=0, seconds=0.2)
        self.assertIsNone(action)
        self.assertEqual(value, 0.5)

    def test_respects_the_time_budget(self):
        started = time.monotonic()
        clean(model=TicTacToe(), seat=0, seconds=0.2, max_depth=20)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_broken_model_never_raises(self):
        class Broken(TicTacToe):
            def apply(self, seat, action):
                raise RuntimeError("boom")

        action, _ = clean(model=Broken(), seat=0, seconds=0.2)
        self.assertIn(action, ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"])

    def test_none_model(self):
        self.assertEqual(clean(model=None, seat=0), (None, 0.5))

    def test_prefers_center_on_empty_board(self):
        # Not a correctness proof, only a sanity check that evaluation moves the search.
        action, _ = clean(model=TicTacToe(), seat=0, seconds=0.6, max_depth=4)
        self.assertIn(action, ["B2", "A1", "A3", "C1", "C3"])


class StandaloneLoadTests(unittest.TestCase):
    def test_search_module_loads_without_the_package(self):
        """The sandbox loads this file by path, so it must not use relative imports."""
        path = os.path.join(ROOT, "harness", "search.py")
        spec = importlib.util.spec_from_file_location("sandboxed_search", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        action, value = module.best_action(TicTacToe(["X", "X", ".", "O", "O", ".", ".", ".", "."]), 0, 0.3)
        self.assertEqual(action, "C1")
        self.assertGreater(value, 0.9)


if __name__ == "__main__":
    unittest.main()
