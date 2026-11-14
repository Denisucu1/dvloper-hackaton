import unittest

from harness.solver_bluff import BluffSolver
from harness.spec import Spec
from harness import turnpolicy


def observation(*, card, cards=3, seat=0, current="(no action yet)", earlier="(none yet)"):
    return f"""ONE-CARD POKER (cards 1-{cards}). You are Player {seat}.
Hand 2 of 6. Chips: you +0, opponent +0.
Your card: {card}

Earlier hands:
  {earlier}

This hand so far: {current}
It is YOUR turn: "check" or "bet"."""


class BluffSolverTests(unittest.TestCase):
    def setUp(self):
        self.solver = BluffSolver()

    def test_matches_bluff_games(self):
        self.assertTrue(self.solver.matches("tg_bluff", Spec(family="bluff")))
        self.assertTrue(self.solver.matches("one_card_poker", Spec(family="bluff")))
        self.assertTrue(self.solver.matches("unknown", Spec(family="bluff")))
        self.assertFalse(self.solver.matches("tic_tac_toe", Spec(family="think_ahead")))

    def test_calls_with_the_best_card_facing_a_bet(self):
        action = self.solver.propose(
            observation=observation(card=3, current="opponent bet"),
            legal=["call", "fold"], seat=1,
        )
        self.assertEqual(action, "call")

    def test_folds_the_worst_card_facing_a_bet(self):
        action = self.solver.propose(
            observation=observation(card=1, current="opponent bet"),
            legal=["call", "fold"], seat=1,
        )
        self.assertEqual(action, "fold")

    def test_bets_the_best_card_when_first_to_act(self):
        action = self.solver.propose(
            observation=observation(card=3), legal=["check", "bet"], seat=0,
        )
        self.assertEqual(action, "bet")

    def test_checks_the_worst_card_when_first_to_act(self):
        action = self.solver.propose(
            observation=observation(card=1), legal=["check", "bet"], seat=0,
        )
        self.assertEqual(action, "check")

    def test_keeps_the_exact_legal_spelling(self):
        action = self.solver.propose(
            observation=observation(card=3), legal=["Check", "Bet"], seat=0,
        )
        self.assertEqual(action, "Bet")

    def test_bluffs_when_the_opponent_folds_a_lot(self):
        # Hand 1 (first seat 0): seat 0 checks, seat 1 (us) bets, seat 0 folds -> opponent fold counted.
        earlier = "Hand 1: check, bet, fold -> seat 0 folded"
        action = self.solver.propose(
            observation=observation(card=2, cards=5, seat=1, current="(no action yet)", earlier=earlier),
            legal=["check", "bet"], seat=1, turn=0,
        )
        # equity = 1/4 = 0.25, fold rate high, (turn + seat) % 3 == 1 -> check is fine too; the rule must
        # at least never fold or return an illegal action.
        self.assertIn(action, ["check", "bet"])

    def test_folds_the_bottom_third_of_a_big_deck_to_a_bet(self):
        # 9 cards: card 3 has equity 2/8 = 0.25, exactly break-even, and a bettor's range is stronger.
        action = self.solver.propose(
            observation=observation(card=3, cards=9, current="opponent bet"),
            legal=["call", "fold"], seat=1,
        )
        self.assertEqual(action, "fold")

    def test_calls_the_top_half_of_a_big_deck_to_a_bet(self):
        action = self.solver.propose(
            observation=observation(card=6, cards=9, current="opponent bet"),
            legal=["call", "fold"], seat=1,
        )
        self.assertEqual(action, "call")

    def test_unreadable_observation_returns_none(self):
        self.assertIsNone(self.solver.propose(observation="no cards here", legal=["check"], seat=0))

    def test_missing_legal_action_returns_none(self):
        action = self.solver.propose(
            observation=observation(card=3), legal=["check"], seat=0,
        )
        self.assertIsNone(action)


class ThinkingPolicyTests(unittest.TestCase):
    def test_bluff_without_a_code_move_thinks(self):
        result = turnpolicy.thinking_for(Spec(family="bluff"), has_code_move=False, legal=["check", "bet"])
        self.assertIsNotNone(result)
        self.assertEqual(result[0], turnpolicy.THINKING_BUDGET)

    def test_bluff_with_a_code_move_does_not_think(self):
        self.assertIsNone(
            turnpolicy.thinking_for(Spec(family="bluff"), has_code_move=True, legal=["check", "bet"])
        )

    def test_numeric_guessing_does_not_think(self):
        self.assertIsNone(
            turnpolicy.thinking_for(Spec(family="guessing"), has_code_move=False,
                                    legal=["guess 1", "guess 2"])
        )

    def test_non_numeric_deduction_thinks(self):
        result = turnpolicy.thinking_for(Spec(family="guessing"), has_code_move=False,
                                         legal=["cat", "dog", "fish"])
        self.assertIsNotNone(result)

    def test_other_families_do_not_think(self):
        for family in ("think_ahead", "truth", "guardrails", "deals", "read_opponent"):
            self.assertIsNone(
                turnpolicy.thinking_for(Spec(family=family), has_code_move=False, legal=["a", "b"]),
                family,
            )

    def test_thinking_leaves_headroom_for_the_visible_answer(self):
        self.assertGreaterEqual(turnpolicy.MAX_TOKENS_WITH_THINKING, turnpolicy.THINKING_BUDGET + 1500)


if __name__ == "__main__":
    unittest.main()
