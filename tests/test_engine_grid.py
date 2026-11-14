"""The built-in Connect-K solver must run before the generated-model search.

A generated simulator can mis-parse the grid text (it has been measured keeping the `|` separator as a
cell and shifting every column). The hand-written parser is exact, so `Engine.propose` asks the grid
solver first and only falls back to the generated model for everything else.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness import config  # noqa: E402
from harness.engine import Engine  # noqa: E402
from harness.spec import Spec  # noqa: E402

# Real arena shape: `CONNECT FOUR - <cols>x<rows>, connect <k>`, letter column header,
# letter rows and a "drop <letter>" open-column list.
CONNECT_OBS = """CONNECT FOUR - 7x6, connect 4. You are Player 0 and play X; the opponent plays O.
   A B C D E F G
 1 . . . . . . .
 2 . . . . . . .
 3 . . . . . . .
 4 . . . . . . .
 5 . . . . . . .
 6 . X X X . . .
Drops so far: drop B, drop C, drop D
Open columns: drop A, drop B, drop C, drop D, drop E, drop F, drop G
It is YOUR turn. Answer with "drop <column letter>"."""

CONNECT_LEGAL = [f"drop {letter}" for letter in "ABCDEFG"]

OTHER_OBS = """PICK A NUMBER 1-9. You are Player 0.
Numbers left: 1 2 3 4 5 6 7 8 9
It is YOUR turn: pick <n>."""
OTHER_LEGAL = [f"pick {n}" for n in range(1, 10)]


class EngineGridFirstTests(unittest.TestCase):
    def _engine(self, game_id: str) -> Engine:
        # Point memory at an empty directory so no generated model can hide in /memory.
        root = tempfile.mkdtemp(prefix="adn-grid-engine-")
        spec = Spec(game_id=game_id, family="think_ahead")
        with mock.patch.object(config, "MEMORY_DIR", root), \
                contextlib.redirect_stderr(io.StringIO()):
            return Engine(spec)

    def test_connect_k_immediate_win_uses_grid_solver_without_a_model(self) -> None:
        engine = self._engine("connect_four")
        self.assertIsNone(engine.model_path)
        stderr = io.StringIO()
        with mock.patch("harness.engine.sandbox.call_model_search") as model_search, \
                mock.patch("harness.engine.sandbox.call_choose") as model_choose, \
                contextlib.redirect_stderr(stderr):
            action, intent = engine.propose(
                observation=CONNECT_OBS,
                legal=CONNECT_LEGAL,
                seat=0,
                turn=3,
                timeout=3.0,
            )
        self.assertIn(action, ("drop A", "drop E"))  # either end of X X X wins
        self.assertIn("connect_k solver", intent)
        self.assertIn(f"play {action}", intent)
        self.assertIn(f"solver connect_k: {action}", stderr.getvalue())
        model_search.assert_not_called()
        model_choose.assert_not_called()

    def test_non_connect_observation_falls_through_to_the_model_path(self) -> None:
        engine = self._engine("pick_number")
        self.assertIsNone(engine.model_path)
        with mock.patch("harness.engine.sandbox.call_model_search") as model_search, \
                mock.patch("harness.engine.sandbox.call_choose") as model_choose, \
                contextlib.redirect_stderr(io.StringIO()):
            action, intent = engine.propose(
                observation=OTHER_OBS,
                legal=OTHER_LEGAL,
                seat=0,
                turn=1,
                timeout=1.0,
            )
        # No grid move for a non-Connect game: the engine has nothing and asks the model.
        self.assertIsNone(action)
        self.assertNotIn("connect_k solver", intent)
        self.assertNotIn("play drop", intent)
        model_search.assert_not_called()
        model_choose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
