"""Tests for tabular CFR+ and the Kuhn-poker probe."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.cfr import (
    CfrSolver,
    CfrTrainer,
    Policy,
    best_response_value,
    build_kuhn,
    chance,
    decision,
    evaluate_policy,
    kuhn_exploitability,
    kuhn_infosets,
    parse_kuhn_observation,
    regret_matching,
    solve_kuhn,
    terminal,
    validate_tree,
)
from harness.spec import Spec


class KuhnTests(unittest.TestCase):
    def test_tree_has_the_standard_twelve_infosets(self):
        infosets = kuhn_infosets(build_kuhn())
        self.assertEqual(len(infosets), 12)
        self.assertEqual(sum(1 for key in infosets if key.startswith("0|")), 6)
        self.assertEqual(sum(1 for key in infosets if key.startswith("1|")), 6)

    def test_solve_kuhn_exploitability_below_two_percent(self):
        policy = solve_kuhn(iterations=1000)
        exploitability = kuhn_exploitability(policy)
        self.assertLess(exploitability, 0.02)
        values = evaluate_policy(build_kuhn(), policy)
        self.assertAlmostEqual(values[0], -1.0 / 18.0, delta=0.01)

    def test_best_response_values_are_exact_and_bounded(self):
        policy = solve_kuhn(iterations=500)
        best0 = best_response_value(build_kuhn(), policy, 0)
        best1 = best_response_value(build_kuhn(), policy, 1)
        self.assertGreaterEqual(best0, -1.0)
        self.assertLessEqual(best1, 1.0)
        self.assertGreaterEqual((best0 + best1) / 2.0, 0.0)
        self.assertLess((best0 + best1) / 2.0, 0.02)

    def test_solve_is_deterministic(self):
        first = solve_kuhn(iterations=300)
        second = solve_kuhn(iterations=300)
        self.assertEqual(first.strategies, second.strategies)
        self.assertEqual(first.iterations, 300)

    def test_sampling_is_reproducible_with_a_seed(self):
        policy = solve_kuhn(iterations=300, seed=7)
        infoset = "0|K|kb"
        first = [policy.sample(infoset, ["call", "fold"], rng=random.Random(7)) for _ in range(12)]
        second = [policy.sample(infoset, ["call", "fold"], rng=random.Random(7)) for _ in range(12)]
        self.assertEqual(first, second)
        self.assertTrue(set(first) <= {"call", "fold"})


class RegretTests(unittest.TestCase):
    def test_regret_matching_normalises_and_ignores_negative_regret(self):
        self.assertEqual(regret_matching([0.0, 0.0, 0.0]), [1.0 / 3.0] * 3)
        self.assertEqual(regret_matching([2.0, -1.0, 0.0]), [1.0, 0.0, 0.0])
        self.assertEqual(regret_matching([-1.0, -2.0]), [0.5, 0.5])
        self.assertAlmostEqual(sum(regret_matching([0.3, 0.7])), 1.0)

    def test_cfr_plus_single_update_matches_hand_computation(self):
        # One player-0 decision: utility 1 for "a" and -1 for "b". With the initial uniform strategy the
        # node value is 0.0, so CFR+ regrets are [1.0, -1.0] floored to [1.0, 0.0].
        root = decision(0, "I", ("a", "b"), (terminal(1, -1), terminal(-1, 1)))
        trainer = CfrTrainer(root).run(1)
        self.assertEqual(trainer.regrets["I"], [1.0, 0.0])
        self.assertEqual(trainer.strategy_sum["I"], [0.5, 0.5])
        self.assertEqual(trainer.average_policy().strategies["I"], {"a": 0.5, "b": 0.5})

    def test_regret_tables_are_nonnegative_and_shaped_like_actions(self):
        trainer = CfrTrainer(build_kuhn()).run(100)
        self.assertEqual(trainer.iterations_done, 100)
        self.assertEqual(set(trainer.regrets), set(trainer.actions))
        self.assertEqual(set(trainer.strategy_sum), set(trainer.actions))
        for infoset, actions in trainer.actions.items():
            regrets = trainer.regrets[infoset]
            self.assertEqual(len(regrets), len(actions))
            self.assertTrue(all(value >= 0.0 for value in regrets))
            self.assertEqual(len(trainer.strategy_sum[infoset]), len(actions))
        policy = trainer.average_policy()
        for infoset, probabilities in policy.strategies.items():
            self.assertEqual(set(probabilities), set(trainer.actions[infoset]))
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=9)

    def test_tree_validation_rejects_bad_chance_and_inconsistent_infosets(self):
        with self.assertRaises(ValueError):
            validate_tree(chance((0.5, terminal(1, -1))))

        children = (terminal(1, -1), terminal(-1, 1))
        with self.assertRaises(ValueError):
            validate_tree(
                chance(
                    (0.5, decision(0, "same", ("a", "b"), children)),
                    (0.5, decision(0, "same", ("a", "b", "c"), children + (terminal(0, 0),))),
                )
            )
        with self.assertRaises(ValueError):
            validate_tree(
                chance(
                    (0.5, terminal(1, -1)),
                    (0.5, terminal(1, -1, 0)),
                )
            )


class PolicyTests(unittest.TestCase):
    def test_policy_restricts_actions_and_falls_back_to_uniform(self):
        policy = Policy(strategies={"I": {"a": 0.8, "b": 0.2}})
        self.assertAlmostEqual(policy.probs("I", ["a", "b"])["a"], 0.8)
        restricted = policy.probs("I", ["b"])
        self.assertEqual(restricted, {"b": 1.0})
        missing = policy.probs("missing", ["x", "y", "z"])
        self.assertEqual(missing, {"x": 1.0 / 3.0, "y": 1.0 / 3.0, "z": 1.0 / 3.0})
        self.assertEqual(policy.best("I"), "a")
        self.assertIsNone(policy.best("missing"))


class KuhnParserAndSolverTests(unittest.TestCase):
    OBSERVATION = (
        "KUHN POKER. You are Player 0. Your card: K.\n"
        "Player 0 checks\n"
        "Player 1 bets"
    )

    def test_parser_reads_card_and_attributed_history(self):
        turn = parse_kuhn_observation(self.OBSERVATION, ["call", "fold"], seat=1)
        self.assertIsNotNone(turn)
        self.assertEqual(turn.seat, 0)  # "You are Player 0" wins over the caller's fallback
        self.assertEqual(turn.card, "K")
        self.assertEqual(turn.history, "kb")
        self.assertEqual(turn.actions, ("call", "fold"))
        self.assertEqual(turn.infoset, "0|K|kb")

    def test_parser_reads_history_section_without_actors(self):
        turn = parse_kuhn_observation(
            "You are seat 0. You hold the Queen.\nHistory: check, bet", ["call", "fold"], seat=1
        )
        self.assertIsNotNone(turn)
        self.assertEqual((turn.seat, turn.card, turn.history), (0, "Q", "kb"))

    def test_parser_declines_uncertain_or_illegal_text(self):
        self.assertIsNone(parse_kuhn_observation("No card mentioned here.", ["call"], 0))
        self.assertIsNone(parse_kuhn_observation("Your card: Q.\nPlayer 1 checks", ["check", "bet"], 0))
        self.assertIsNone(parse_kuhn_observation("Your card: Q.\nGame over", ["check"], 0))
        # The history is valid but the offered legal actions do not belong to it.
        self.assertIsNone(parse_kuhn_observation(self.OBSERVATION, ["check", "bet"], 0))

    def test_cfr_solver_matches_only_kuhn_and_proposes_legal_actions(self):
        solver = CfrSolver()
        spec = Spec(game_id="kuhn_poker", family="bluff")
        self.assertTrue(solver.matches("kuhn_poker", spec))
        self.assertFalse(solver.matches("pig", Spec(game_id="pig", family="guessing")))
        solver.configure(spec, "Kuhn poker: one card each, check/bet/call/fold.")
        action = solver.propose(observation=self.OBSERVATION, legal=["call", "fold"], seat=0)
        self.assertIn(action, {"call", "fold"})
        self.assertIsNone(
            solver.propose(observation=self.OBSERVATION, legal=[], seat=0)
        )
        self.assertIsNone(
            solver.propose(observation="A unrelated game with no card.", legal=["a", "b"], seat=0)
        )


if __name__ == "__main__":
    unittest.main()
