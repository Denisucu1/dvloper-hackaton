"""Tests for the solver smoke script: the real run stays green and the verifier catches bad moves."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

from scripts import solver_smoke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SolverSmokeRunTests(unittest.TestCase):
    def test_solver_smoke_exits_zero_with_a_pass_line(self):
        completed = subprocess.run(
            [sys.executable, "scripts/solver_smoke.py", "--no-color"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        self.assertEqual(completed.returncode, 0, output[-2000:])
        self.assertIn("SOLVER SMOKE PASS", output)
        self.assertIn("proposals", output)


class ValidateActionTests(unittest.TestCase):
    def test_injected_illegal_proposal_is_detected(self):
        ok, detail = solver_smoke.validate_action("drop 9", ["drop 1", "drop 2"])
        self.assertFalse(ok)
        self.assertIn("illegal", detail)

    def test_legal_proposal_normalizes_and_null_list_only_needs_text(self):
        ok, normalized = solver_smoke.validate_action("DROP 2", ["drop 1", "drop 2"])
        self.assertTrue(ok)
        self.assertEqual(normalized, "drop 2")

        ok, text = solver_smoke.validate_action("guess crane", None)
        self.assertTrue(ok)
        self.assertEqual(text, "guess crane")

        self.assertFalse(solver_smoke.validate_action("", ["drop 1"])[0])
        self.assertFalse(solver_smoke.validate_action(None, None)[0])


if __name__ == "__main__":
    unittest.main()
