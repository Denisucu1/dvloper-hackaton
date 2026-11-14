"""Tests for the Leduc expert and the Leduc arena probe."""
from __future__ import annotations

import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ARENA = os.path.join(ROOT, "arena-lite")
if ARENA not in sys.path:
    sys.path.insert(0, ARENA)

from arena.games.tg_leduc_probe import TestLeduc  # noqa: E402
from harness.solver_leduc import LeducSolver, _equity, parse_leduc_observation  # noqa: E402
from harness.spec import Spec  # noqa: E402


def round_one(*, card="Q", seat=0, pot=2, line="(no action yet)", legal='"check" or "bet"'):
    return f"""LEDUC POKER (J, Q, K twice, 2 betting rounds). You are Player {seat}.
Hand 1 of 1. Round 1 of 2.
Your private card: {card}
Community card: (not dealt yet)
Pot: {pot}. Chips: you -1, opponent -1.
This round so far: {line}
It is YOUR turn: {legal}."""


def round_two(*, card="Q", community="K", seat=0, pot=4, line="(no action yet)",
              legal='"check" or "bet"'):
    return f"""LEDUC POKER (J, Q, K twice, 2 betting rounds). You are Player {seat}.
Hand 1 of 1. Round 2 of 2.
Your private card: {card}
Community card: {community}
Earlier rounds: check, check
Pot: {pot}. Chips: you -2, opponent -2.
This round so far: {line}
It is YOUR turn: {legal}."""


class LeducParserTests(unittest.TestCase):
    def test_parses_the_first_round(self):
        turn = parse_leduc_observation(round_one(card="K", pot=2), ["check", "bet"], 0)
        self.assertIsNotNone(turn)
        self.assertEqual((turn.seat, turn.private, turn.community), (0, "K", None))
        self.assertEqual((turn.round, turn.pot, turn.line), (1, 2, ()))
        self.assertFalse(turn.facing_bet)

    def test_parses_the_second_round_with_a_community_card(self):
        turn = parse_leduc_observation(
            round_two(card="Q", community="Q", pot=3, line="check, bet"),
            ["call", "fold"], 1,
        )
        self.assertIsNotNone(turn)
        self.assertEqual((turn.private, turn.community), ("Q", "Q"))
        self.assertEqual((turn.round, turn.pot), (2, 3))
        self.assertEqual(turn.line, ("check", "bet"))
        self.assertTrue(turn.facing_bet)

    def test_round_two_without_a_community_card_is_declined(self):
        text = round_two().replace("Community card: K", "Community card: (not dealt yet)")
        self.assertIsNone(parse_leduc_observation(text, ["check", "bet"], 0))

    def test_broken_action_line_is_declined(self):
        self.assertIsNone(
            parse_leduc_observation(round_one(line="check, bet, call"), ["check", "bet"], 0)
        )


class LeducPolicyTests(unittest.TestCase):
    def setUp(self):
        self.solver = LeducSolver()

    def test_matches_leduc_names_but_not_kuhn(self):
        self.assertTrue(self.solver.matches("mini_poker", Spec(family="bluff")))
        self.assertTrue(self.solver.matches("tg_leduc_probe", Spec(family="bluff")))
        self.assertTrue(self.solver.matches("unknown", Spec(notes="Leduc with two rounds")))
        self.assertFalse(self.solver.matches("tg_kuhn_probe", Spec(family="bluff")))
        self.assertFalse(self.solver.matches("tic_tac_toe", Spec(family="think_ahead")))

    def test_pair_bets_in_the_second_round(self):
        action = self.solver.propose(
            observation=round_two(card="J", community="J", pot=4),
            legal=["check", "bet"], seat=0,
        )
        self.assertEqual(action, "bet")

    def test_weak_private_card_calls_a_bet_once_the_draw_counts(self):
        # J vs a bet with the community card still to come is 0.30, above the 0.25 pot odds:
        # folding this was the measured mini_poker loss (0-1).
        action = self.solver.propose(
            observation=round_one(card="J", pot=3, line="bet"),
            legal=["call", "fold"], seat=1,
        )
        self.assertEqual(action, "call")

    def test_dominated_hand_still_folds_to_a_bet(self):
        # Round two, J with a Q on the board: no draw left, only a J tie (~0.125).
        action = self.solver.propose(
            observation=round_two(card="J", community="Q", pot=3, line="bet"),
            legal=["call", "fold"], seat=1,
        )
        self.assertEqual(action, "fold")

    def test_round_one_equity_enumerates_the_community_card(self):
        jack = _equity("J", None)
        queen = _equity("Q", None)
        king = _equity("K", None)
        self.assertGreaterEqual(jack, 0.30)
        self.assertLessEqual(jack, 0.35)
        self.assertAlmostEqual(queen, 0.50)
        self.assertAlmostEqual(king, 0.70)

    def test_round_two_uses_the_kicker(self):
        # Same high card, different kicker: J+board K loses to Q+board K instead of tying.
        self.assertAlmostEqual(_equity("J", "K"), 0.125)
        self.assertAlmostEqual(_equity("Q", "K"), 0.625)
        self.assertAlmostEqual(_equity("K", "Q"), 0.625)

    def test_smaller_kicker_folds_but_round_one_jack_still_calls(self):
        # Round two, J with a K on the board: 0.125 equity cannot pay 0.25 pot odds (pot 3, call 1).
        self.assertEqual(
            self.solver.propose(
                observation=round_two(card="J", community="K", pot=3, line="check, bet"),
                legal=["call", "fold"], seat=1,
            ),
            "fold",
        )
        # Round one the community draw keeps J at 0.30, so the same 0.25 pot odds are worth a call.
        self.assertEqual(
            self.solver.propose(
                observation=round_one(card="J", pot=3, line="bet"),
                legal=["call", "fold"], seat=1,
            ),
            "call",
        )

    def test_strong_private_card_calls_a_bet(self):
        action = self.solver.propose(
            observation=round_one(card="K", pot=3, line="bet"),
            legal=["call", "fold"], seat=1,
        )
        self.assertEqual(action, "call")

    def test_best_card_bets_when_first_to_act(self):
        action = self.solver.propose(
            observation=round_one(card="K"), legal=["check", "bet"], seat=0,
        )
        self.assertEqual(action, "bet")

    def test_keeps_the_exact_legal_spelling(self):
        action = self.solver.propose(
            observation=round_one(card="K"), legal=["Check", "Bet"], seat=0,
        )
        self.assertEqual(action, "Bet")

    def test_deterministic_given_the_observation(self):
        text = round_one(card="Q")
        first = self.solver.propose(observation=text, legal=["check", "bet"], seat=0, turn=2)
        second = self.solver.propose(observation=text, legal=["check", "bet"], seat=0, turn=2)
        self.assertEqual(first, second)
        self.assertIn(first, ["check", "bet"])

    def test_unreadable_observation_returns_none(self):
        self.assertIsNone(self.solver.propose(observation="no cards here", legal=["check"], seat=0))

    def test_opponent_turn_returns_none(self):
        text = round_one(card="K").replace(
            'It is YOUR turn: "check" or "bet".', "It is the opponent's turn."
        )
        self.assertIsNone(self.solver.propose(observation=text, legal=None, seat=0))

    def test_check_only_stays_within_the_legal_list(self):
        # A strong card would bet, but the arena only offers "check": take the legal fallback.
        action = self.solver.propose(
            observation=round_one(card="K"), legal=["check"], seat=0
        )
        self.assertEqual(action, "check")


class LeducProbeTests(unittest.TestCase):
    def _playout(self, variant, seed, policy=None):
        game = TestLeduc(variant, seed)
        rng = random.Random(seed)
        for _ in range(80):
            if game.is_over():
                break
            actions = {}
            for mover in game.to_move():
                options = game.legal_actions(mover)
                self.assertTrue(options, f"seat {mover} has no legal action")
                if policy is None:
                    choice = rng.choice(list(options))
                else:
                    choice = policy.propose(
                        observation=game.observation(mover), legal=options, seat=mover, turn=0
                    )
                    self.assertIsNotNone(choice, game.observation(mover))
                canonical = game.normalize(mover, str(choice))
                self.assertIn(canonical, options, f"illegal choice {choice!r}")
                actions[mover] = canonical
            game.apply(actions)
        self.assertTrue(game.is_over(), variant.get("label"))
        self.assertEqual(len(game.result()["scores"]), 2)
        return game

    def test_both_variants_accept_a_random_playout(self):
        for index, variant in enumerate(TestLeduc.variants()):
            self._playout(variant, 7 + index)

    def test_solver_self_play_is_legal_and_finishes(self):
        game = self._playout(TestLeduc.variants()[0], 11, policy=LeducSolver())
        self.assertTrue(game.is_over())
        self.assertTrue(game.result()["reason"])

    def test_second_round_observation_reveals_the_community_card(self):
        game = TestLeduc(TestLeduc.variants()[0], 3)
        while game.round == 1 and not game.is_over():
            for mover in game.to_move():
                game.apply({mover: game.normalize(mover, "check")})
        if not game.is_over():
            self.assertNotIn("(not dealt yet)", game.observation(0))
            self.assertIn(game.community, game.observation(0))


if __name__ == "__main__":
    unittest.main()
