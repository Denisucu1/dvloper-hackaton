import textwrap
import os
import tempfile
import unittest

from harness import sandbox

GOOD_MODEL = """
class Model:
    def __init__(self, rules, seat):
        self.seat = seat
        self.board = ["X", "X", ".", "O", "O", ".", ".", ".", "."]
        self.over = False

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [self._name(i) for i, cell in enumerate(self.board) if cell == "."]

    def apply(self, seat, action):
        self.board[self._index(action)] = "X" if seat == 0 else "O"

    def is_terminal(self):
        return self._winner() is not None or "." not in self.board

    def to_move(self):
        return [] if self.is_terminal() else [0]

    def score(self, seat):
        winner = self._winner()
        return 0.5 if winner is None else (1.0 if winner == seat else 0.0)

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        other = Model("", self.seat)
        other.board = list(self.board)
        return other

    def key(self):
        return "".join(self.board)

    @staticmethod
    def _name(index):
        return "ABC"[index % 3] + str(index // 3 + 1)

    @staticmethod
    def _index(action):
        return (int(action[1]) - 1) * 3 + "ABC".index(action[0].upper())

    def _winner(self):
        for a, b, c in ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)):
            if self.board[a] != "." and self.board[a] == self.board[b] == self.board[c]:
                return 0 if self.board[a] == "X" else 1
        return None
"""

BAD_MOVES = """
class Model:
    def __init__(self, rules, seat):
        self.seat = seat
    def update(self, observation):
        return None
    def legal_actions(self, seat):
        return ["Z9"]
    def apply(self, seat, action):
        return None
    def is_terminal(self):
        return False
    def to_move(self):
        return [0]
    def score(self, seat):
        return 0.5
    def evaluate(self, seat):
        return 0.5
    def clone(self):
        return Model("", self.seat)
"""


class ModelSandboxTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="adn-model-")
        self.rules = "TIC-TAC-TOE 3x3: cells A1..C3, three in a row wins."
        self.samples = [{"observation": "board", "legal_actions": ["C1", "C2", "A3", "B3", "C3"], "seat": 0}]

    def write(self, name, source):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent(source))
        return path

    def test_valid_model_is_accepted(self):
        path = self.write("good.py", GOOD_MODEL)
        ok, reason = sandbox.validate_model(path, self.samples, self.rules)
        self.assertTrue(ok, reason)

    def test_wrong_legal_actions_are_rejected(self):
        path = self.write("bad.py", BAD_MOVES)
        ok, reason = sandbox.validate_model(path, self.samples, self.rules)
        self.assertFalse(ok)
        self.assertIn("mismatch", reason)

    def test_network_model_is_rejected(self):
        path = self.write("net.py", "import socket\nclass Model:\n    pass\n")
        ok, reason = sandbox.validate_model(path, self.samples, self.rules)
        self.assertFalse(ok)
        self.assertIn("banned token", reason)

    def test_no_samples_is_rejected(self):
        path = self.write("good.py", GOOD_MODEL)
        ok, _ = sandbox.validate_model(path, [], self.rules)
        self.assertFalse(ok)

    def test_search_plays_through_the_sandbox(self):
        path = self.write("good.py", GOOD_MODEL)
        search_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "harness", "search.py")
        action = sandbox.call_model_best(
            path, search_path, self.rules, "board", ["C1", "C2", "C3"], seat=0, timeout=8.0
        )
        self.assertEqual(action, "C1")  # X on A1+B1 wins on C1

    def test_broken_model_returns_none(self):
        path = self.write("broken.py", "class Model:\n    def __init__(self, rules, seat):\n        raise RuntimeError('no')\n")
        search_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "harness", "search.py")
        self.assertIsNone(sandbox.call_model_best(path, search_path, self.rules, "board", None, 0, 6.0))


if __name__ == "__main__":
    unittest.main()
