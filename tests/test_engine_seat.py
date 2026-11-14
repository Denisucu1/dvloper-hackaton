import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from harness import config, memory
from harness.game_model import ARTIFACT_VERSION
from harness.engine import Engine, seat_from_observation
from harness.spec import Spec

MODEL = '''
class Model:
    """Minimal alternating game where seat 1 must move; no state needed for the test."""

    def __init__(self, rules, seat):
        self.seat = int(seat)

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return ["A1", "A2"]

    def apply(self, seat, action):
        return None

    def is_terminal(self):
        return False

    def to_move(self):
        return [1]

    def score(self, seat):
        return 0.5

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        return Model("", self.seat)
'''


class SeatInferenceTests(unittest.TestCase):
    def test_reads_seat_from_the_observation(self):
        self.assertEqual(seat_from_observation("You are seat 1 and play O.", 0), 1)
        self.assertEqual(seat_from_observation("You are Player 0.", 1), 0)
        self.assertEqual(seat_from_observation("no seat here", 1), 1)

    def test_engine_plays_when_the_caller_lost_the_seat(self):
        """Regression: turn messages do not carry `seat`, so the caller may pass 0 for a seat-1 game."""
        root = tempfile.mkdtemp(prefix="adn-seat-")
        directory = memory.game_dir("seat_test", root)
        (directory / "abc-model.py").write_text(MODEL, encoding="utf-8")
        (directory / "rules.txt").write_text("two moves, seat 1 moves first", encoding="utf-8")
        (directory / "manifest.json").write_text(
            json.dumps({"game_id": "seat_test", "genmodel": "abc-model.py",
                        "artifact_version": ARTIFACT_VERSION}),
            encoding="utf-8",
        )
        spec = Spec(game_id="seat_test", family="think_ahead")
        with mock.patch.object(config, "MEMORY_DIR", root), contextlib.redirect_stderr(io.StringIO()):
            proposal, intent = Engine(spec).propose(
                observation="SOME GAME. You are seat 1 and play O. Your turn.",
                legal=["A1", "A2"],
                seat=0,  # the caller is wrong on purpose
                turn=1,
                timeout=3.0,
            )
        self.assertIn(proposal, ["A1", "A2"])
        self.assertIn("generated model", intent)


if __name__ == "__main__":
    unittest.main()
