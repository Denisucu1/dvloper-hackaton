"""Tests for the ported deterministic `deals` solver.

Covers the split-the-pot bargaining cases that were previously only tested in
`test_t2_upgrades.py`, plus regressions for pot decay parsing, legal-action
snapping, one-shot Prisoner's Dilemma and hyphenated auction detection.
"""

from __future__ import annotations

import unittest

from harness.solver_deals import DealsSolver
from harness.spec import Spec


SPLIT_RULES = (
    "SPLIT THE POT\n"
    "You and your opponent are splitting a pot of 100 coins, shrinking by 5 a turn.\n"
    "Each player has a private walk-away value (10 to 45)."
)


def make_solver(rules: str) -> DealsSolver:
    solver = DealsSolver()
    solver.configure(Spec(game_id="tg_deals", family="deals", hidden_information=True), rules)
    return solver


class TestDealsPort(unittest.TestCase):
    def test_split_the_pot_offer_56(self) -> None:
        solver = make_solver(SPLIT_RULES)
        obs = (
            "SPLIT THE POT. You are Player 0.\n"
            "Your private walk-away value: 23\n"
            "Turn 1 of 8. Current pot: 100 coins.\n"
            "Offers so far: none.\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 100, or "walk".'
        )
        # Turn 1, pot=100, decay=5 -> next_pot=95, low_w=10 -> k_win=56.
        self.assertEqual(solver.propose(observation=obs, legal=None, seat=0, turn=1), "offer 56")

    def test_non_shrinking_pot_offer_61(self) -> None:
        rules = (
            "SPLIT THE POT\n"
            "You and your opponent are splitting a pot of 100 coins.\n"
            "Each player has a private walk-away value (10 to 45)."
        )
        solver = make_solver(rules)
        obs = (
            "SPLIT THE POT. You are Player 0.\n"
            "Your private walk-away value: 28\n"
            "Turn 1 of 8. Current pot: 100 coins.\n"
            "Offers so far: none.\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 100, or "walk".'
        )
        # No "shrink"/"decay" wording: decay must be 0, not an assumed 5.
        self.assertEqual(solver.propose(observation=obs, legal=None, seat=0, turn=1), "offer 61")

    def test_final_turn_unfavourable_offer_walks(self) -> None:
        solver = make_solver(SPLIT_RULES)
        obs = (
            "SPLIT THE POT. You are Player 1.\n"
            "Your private walk-away value: 30\n"
            "Turn 8 of 8. Current pot: 65 coins.\n"
            "Offers so far:\n"
            "  1. opponent offered to keep 55\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 65, or "accept", or "walk".'
        )
        self.assertEqual(solver.propose(observation=obs, legal=None, seat=1, turn=8), "walk")

    def test_one_shot_prisoners_dilemma_defects(self) -> None:
        solver = make_solver("Prisoner's Dilemma. Choose cooperate or defect.")
        self.assertEqual(
            solver.propose(
                observation="Choose cooperate or defect.",
                legal=["cooperate", "defect"],
                seat=0,
                turn=1,
            ),
            "defect",
        )

        # With an explicit iterated marker, round 1 still cooperates and the last round defects.
        solver = make_solver("Iterated Prisoner's Dilemma. Rounds 1-5.")
        self.assertEqual(
            solver.propose(
                observation="Round 1 of 5. Choose cooperate or defect.",
                legal=["cooperate", "defect"],
                seat=0,
                turn=1,
            ),
            "cooperate",
        )
        self.assertEqual(
            solver.propose(
                observation="Round 5 of 5. Choose cooperate or defect.",
                legal=["cooperate", "defect"],
                seat=0,
                turn=5,
            ),
            "defect",
        )

    def test_non_100_initial_pot_does_not_produce_absurd_offer(self) -> None:
        rules = (
            "SPLIT THE POT\n"
            "You and your opponent are splitting a pot of 60 coins.\n"
            "Each player has a private walk-away value (10 to 45)."
        )
        solver = make_solver(rules)
        obs = (
            "SPLIT THE POT. You are Player 0.\n"
            "Your private walk-away value: 20\n"
            "Turn 3 of 8. Current pot: 50 coins.\n"
            "Offers so far: none.\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 50, or "walk".'
        )
        # The pot is only described ("pot of 60"), not said to shrink -> decay 0.
        self.assertEqual(solver.propose(observation=obs, legal=None, seat=0, turn=3), "offer 32")

    def test_decay_is_inferred_from_parsed_initial_pot_not_100(self) -> None:
        rules = (
            "SPLIT THE POT\n"
            "You and your opponent are splitting a pot of 60 coins that shrinks each turn.\n"
            "Each player has a private walk-away value (10 to 45)."
        )
        solver = make_solver(rules)
        obs = (
            "SPLIT THE POT. You are Player 0.\n"
            "Your private walk-away value: 20\n"
            "Turn 3 of 8. Current pot: 50 coins.\n"
            "Offers so far: none.\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 50, or "walk".'
        )
        # Decay = (60 - 50) / (3 - 1) = 5; assuming 100 would imply 25 and a much lower offer.
        self.assertEqual(solver.propose(observation=obs, legal=None, seat=0, turn=3), "offer 29")

    def test_legal_offer_list_snaps_to_closest_numeric_offer(self) -> None:
        solver = make_solver(SPLIT_RULES)
        obs = (
            "SPLIT THE POT. You are Player 0.\n"
            "Your private walk-away value: 23\n"
            "Turn 1 of 8. Current pot: 100 coins.\n"
            "Offers so far: none.\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 100, or "walk".'
        )
        # Target is offer 56; only 50 and 60 are legal, so snap to the nearer 60.
        act = solver.propose(observation=obs, legal=["offer 50", "offer 60"], seat=0, turn=1)
        self.assertIn(act, ("offer 50", "offer 60"))
        self.assertEqual(act, "offer 60")

    def test_second_highest_auction_is_recognised(self) -> None:
        rules = (
            "SEALED-BID AUCTION\n"
            "The highest bidder wins but pays the second-highest bid.\n"
            "Bid up to your private value."
        )
        solver = make_solver(rules)
        obs = "SEALED-BID AUCTION. Your private value: 70. Submit one bid."
        act = solver.propose(
            observation=obs,
            legal=["bid 60", "bid 70", "bid 80"],
            seat=0,
            turn=1,
        )
        self.assertEqual(act, "bid 70")

    def test_auction_bid_snaps_to_nearest_legal_bid(self) -> None:
        rules = (
            "SEALED-BID AUCTION\n"
            "The highest bidder wins but pays the second-highest bid.\n"
            "Bid up to your private value."
        )
        solver = make_solver(rules)
        obs = "SEALED-BID AUCTION. Your private value: 55. Submit one bid."
        act = solver.propose(
            observation=obs,
            legal=["bid 40", "bid 60"],
            seat=0,
            turn=1,
        )
        self.assertEqual(act, "bid 60")

    def test_matches_uses_family_or_clear_game_id_hints(self) -> None:
        solver = DealsSolver()
        self.assertTrue(solver.matches("g", Spec(game_id="g", family="deals")))
        self.assertTrue(solver.matches("tg_auction_1", Spec(game_id="tg_auction_1", family="unknown")))
        self.assertFalse(solver.matches("g", Spec(game_id="g", family="unknown")))
        self.assertFalse(solver.matches("", Spec(game_id="", family="unknown")))


if __name__ == "__main__":
    unittest.main()
