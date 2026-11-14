"""Tests for the deterministic Guess the Number solver."""
import unittest
from types import SimpleNamespace

from harness.solver_guess import GuessingSolver

OPEN_100 = """GUESS THE NUMBER (1-100). You are Player 0.
Your secret number (the opponent is looking for it): 51
Round 1 of 12.

Your guesses at the opponent's number:
  (none yet)
Opponent's guesses at your number:
  (none yet)

It is YOUR turn: "guess <n>" with n from 1 to 100."""

AFTER_HIGHER_100 = """GUESS THE NUMBER (1-100). You are Player 0.
Your secret number (the opponent is looking for it): 51
Round 2 of 12.

Your guesses at the opponent's number:
  Round 1: you guessed 50 -> higher (their number is above 50)
Opponent's guesses at your number:
  (none yet)

It is YOUR turn: "guess <n>" with n from 1 to 100."""

MIXED_100 = """GUESS THE NUMBER (1-100). You are Player 0.
Your secret number (the opponent is looking for it): 51
Round 3 of 12.

Your guesses at the opponent's number:
  Round 1: you guessed 25 -> higher (their number is above 25)
  Round 2: you guessed 47 -> lower (their number is below 47)
Opponent's guesses at your number:
  Round 1: opponent guessed 95 -> lower
  Round 2: opponent guessed 66 -> lower

It is YOUR turn: "guess <n>" with n from 1 to 100."""

SAMPLE_1000 = """GUESS THE NUMBER (1-1000). You are Player 0.
Your secret number (the opponent is looking for it): 76
Round 5 of 15.

Your guesses at the opponent's number:
  Round 1: you guessed 451 -> higher (their number is above 451)
  Round 2: you guessed 832 -> lower (their number is below 832)
  Round 3: you guessed 794 -> lower (their number is below 794)
  Round 4: you guessed 614 -> lower (their number is below 614)
Opponent's guesses at your number:
  Round 1: opponent guessed 262 -> lower
  Round 2: opponent guessed 162 -> lower
  Round 3: opponent guessed 100 -> lower
  Round 4: opponent guessed 73 -> higher

It is YOUR turn: "guess <n>" with n from 1 to 1000."""

SPACED_200 = """GUESS THE NUMBER  ( 1 - 200 ). You are Player 1.

Your guesses at the opponent's number:
  Round 1: you guessed 100 -> higher (their number is above 100)

It is YOUR turn: "guess <n>" with n from 1 to 200."""


def propose(observation: str, legal: list[str] | None = None) -> str | None:
    """Call the solver the same way the harness does."""
    return GuessingSolver().propose(observation=observation, legal=legal, seat=0, turn=0)


class MatchingTests(unittest.TestCase):
    def test_guess_in_id_matches(self):
        self.assertTrue(GuessingSolver.matches("guess_number", None))

    def test_guessing_family_with_number_id_matches(self):
        spec = SimpleNamespace(family="guessing", game_id="", action_format="guess <n>")
        self.assertTrue(GuessingSolver.matches("number_duel", spec))

    def test_guessing_family_alone_is_not_enough_for_other_games(self):
        self.assertFalse(GuessingSolver.matches("pig", SimpleNamespace(family="guessing")))
        self.assertFalse(GuessingSolver.matches("bluff", SimpleNamespace(family="bluff")))


class OpenAndBinarySearchTests(unittest.TestCase):
    def test_first_move_is_the_midpoint(self):
        self.assertEqual(propose(OPEN_100), "guess 50")

    def test_first_move_of_a_1000_range_is_500(self):
        observation = OPEN_100.replace("(1-100)", "(1-1000)").replace("1 to 100", "1 to 1000")
        self.assertEqual(propose(observation), "guess 500")

    def test_after_higher_picks_a_value_above_the_guess(self):
        action = propose(AFTER_HIGHER_100)
        self.assertEqual(action, "guess 75")
        self.assertGreater(int(action.split()[1]), 50)

    def test_mixed_feedback_stays_inside_the_remaining_interval(self):
        action = propose(MIXED_100)
        number = int(action.split()[1])
        self.assertEqual(action, "guess 36")
        self.assertTrue(26 <= number <= 46)

    def test_larger_range_uses_real_sample_feedback(self):
        action = propose(SAMPLE_1000)
        self.assertEqual(action, "guess 532")

    def test_spacing_and_missing_round_line_are_tolerated(self):
        self.assertEqual(propose(SPACED_200), "guess 150")

    def test_opponent_turn_has_no_proposal(self):
        observation = OPEN_100.replace("It is YOUR turn", "It is the opponent's turn")
        self.assertIsNone(propose(observation))


class LegalActionTests(unittest.TestCase):
    def test_exact_guess_string_is_returned(self):
        legal = ["guess 49", "guess 50", "guess 51"]
        action = propose(OPEN_100, legal)
        self.assertEqual(action, "guess 50")
        self.assertIn(action, legal)

    def test_bare_digit_legal_string_is_returned(self):
        legal = ["49", "50", "51"]
        action = propose(OPEN_100, legal)
        self.assertEqual(action, "50")
        self.assertIn(action, legal)

    def test_digits_match_even_when_spelling_differs(self):
        legal = ["GUESS 50", "GUESS 51"]
        self.assertEqual(propose(OPEN_100, legal), "GUESS 50")

    def test_missing_candidate_returns_none_instead_of_inventing_an_action(self):
        self.assertIsNone(propose(OPEN_100, ["guess 10", "guess 20"]))
        self.assertIsNone(propose(OPEN_100, []))


if __name__ == "__main__":
    unittest.main()
