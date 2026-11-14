"""Tests for the SimultaneousSolver and cold simultaneous family detection."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.solver_simul import SimultaneousSolver, _parse_rps_weights, _rps_distribution
from harness.spec import Spec, fallback_spec


class SimultaneousTests(unittest.TestCase):
    SIMUL_RULES = (
        "DUEL OF SIGNS - 9 rounds, simultaneous moves.\n"
        "Both players choose at the same time, without seeing the other's choice:\n"
        "rock beats scissors, paper beats rock, scissors beats paper.\n"
        "A round won with rock scores 3 point(s), with paper 1, with scissors 1.\n"
        "After round 9 the player with more points wins."
    )

    def test_cold_fallback_spec_classifies_simultaneous_over_guessing(self):
        cold = fallback_spec("tg_simul", self.SIMUL_RULES, [], "fp")
        self.assertEqual(cold.family, "read_opponent")
        self.assertTrue(cold.simultaneous)

    def test_parses_asymmetric_rps_weights_and_counters_rock_spam(self):
        w = _parse_rps_weights(self.SIMUL_RULES)
        self.assertEqual(w, {"rock": 3.0, "paper": 1.0, "scissors": 1.0})
        heavy_rock_obs = "\n".join(f"  Round {i}: rock / rock -> tie" for i in range(1, 7))
        probs = _rps_distribution(self.SIMUL_RULES, heavy_rock_obs)
        # probs order is [rock, paper, scissors]: paper beats rock, so paper dominates.
        self.assertGreater(probs[1], probs[0])
        self.assertGreater(probs[1], probs[2])

    def test_rps_solver_mixes_instead_of_repeating_one_move(self):
        solver = SimultaneousSolver()
        spec = Spec(game_id="tg_simul", family="read_opponent", simultaneous=True)
        self.assertTrue(solver.matches("tg_simul", spec))
        solver.configure(spec, self.SIMUL_RULES)
        picks = {
            solver.propose(
                observation="Round 2 of 9. Choose now.",
                legal=["rock", "paper", "scissors"],
                seat=0,
                turn=2,
            )
            for _ in range(40)
        }
        self.assertEqual(picks, {"rock", "paper", "scissors"})

    def test_does_not_randomize_deals_or_numeric_offers(self):
        solver = SimultaneousSolver()
        deals_spec = Spec(game_id="auction", family="deals", simultaneous=True)
        self.assertFalse(solver.matches("auction", deals_spec))
        sim_spec = Spec(game_id="sim", family="read_opponent", simultaneous=True)
        for sample_actions in (["offer 10", "offer 20", "offer 30"], ["pick 1", "pick 5", "pick 9"]):
            self.assertIsNone(
                solver.propose(observation="Round 1", legal=sample_actions, seat=0)
            )
        solver.configure(sim_spec, "", has_model=True)
        self.assertIsNone(
            solver.propose(observation="Round 1", legal=["left", "right"], seat=0)
        )
        self.assertIn(
            solver.propose(observation="Round 1", legal=["rock", "paper", "scissors"], seat=0),
            {"rock", "paper", "scissors"},
        )


if __name__ == "__main__":
    unittest.main()
