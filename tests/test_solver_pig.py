"""Tests for the deterministic Pig solver."""
import unittest
from types import SimpleNamespace

from harness.solver_pig import PigSolver

ONE_DIE_SAMPLE = """PIG - One die, first to 50. You are seat 0. Target: 50.
Banked: you 17, opponent 7.
Turn 5: it is YOUR turn. Turn total: 5 (rolls this turn: 5).
If you hold now you will have 22. Choose: roll | hold
Previous turns (most recent last):
  Turn 1 (you): 3 -> held for +3. Banked after: you 3, opponent 0.
  Turn 2 (opponent): 4 -> held for +4. Banked after: you 3, opponent 4.
  Turn 3 (you): 6, 3, 3, 2 -> held for +14. Banked after: you 17, opponent 4.
  Turn 4 (opponent): 3 -> held for +3. Banked after: you 17, opponent 7."""

TWO_DICE_SAMPLE = """PIG - Two dice, first to 60 (snake eyes reset). You are seat 0. Target: 60.
Banked: you 22, opponent 8.
Turn 7: it is YOUR turn. Turn total: 0 (rolls this turn: none yet).
If you hold now you will have 22. Choose: roll | hold
Previous turns (most recent last):
  Turn 1 (you): 4+4 -> held for +8. Banked after: you 8, opponent 0.
  Turn 2 (opponent): 5+3 -> held for +8. Banked after: you 8, opponent 8.
  Turn 3 (you): 1+5 -> rolled a 1, turn lost. Banked after: you 8, opponent 8.
  Turn 4 (opponent): 4+1 -> rolled a 1, turn lost. Banked after: you 8, opponent 8.
  Turn 5 (you): 3+2, 5+4 -> held for +14. Banked after: you 22, opponent 8.
  Turn 6 (opponent): 4+1 -> rolled a 1, turn lost. Banked after: you 22, opponent 8."""


def one_die(banked: int, opponent: int, turn_total: int, target: int = 50) -> str:
    """Build the real one-die observation shape with the numbers we want to test."""
    return (
        f"PIG - One die, first to {target}. You are seat 0. Target: {target}.\n"
        f"Banked: you {banked}, opponent {opponent}.\n"
        f"Turn 4: it is YOUR turn. Turn total: {turn_total} (rolls this turn: none yet).\n"
        f"If you hold now you will have {banked + turn_total}. Choose: roll | hold"
    )


def two_dice(banked: int, opponent: int, turn_total: int, target: int = 60) -> str:
    """Build the real two-dice observation shape with the numbers we want to test."""
    return (
        f"PIG - Two dice, first to {target} (snake eyes reset). You are seat 0. Target: {target}.\n"
        f"Banked: you {banked}, opponent {opponent}.\n"
        f"Turn 9: it is YOUR turn. Turn total: {turn_total} (rolls this turn: none yet).\n"
        f"If you hold now you will have {banked + turn_total}. Choose: roll | hold"
    )


def propose(observation: str, legal: list[str] | None = None) -> str | None:
    """Call the solver the same way the harness does."""
    return PigSolver().propose(observation=observation, legal=legal, seat=0, turn=0)


class MatchingTests(unittest.TestCase):
    def test_pig_id_matches(self):
        self.assertTrue(PigSolver.matches("pig", None))
        self.assertTrue(PigSolver.matches("pig_two_dice", SimpleNamespace(family="unknown")))

    def test_roll_and_hold_action_shape_matches(self):
        spec = SimpleNamespace(action_examples=["roll", "hold"], action_format="roll | hold")
        self.assertTrue(PigSolver.matches("dice_race", spec))

    def test_other_games_do_not_match(self):
        self.assertFalse(PigSolver.matches("guess_number", None))
        self.assertFalse(PigSolver.matches("dice_race", SimpleNamespace(action_examples=["roll"])))


class OneDieTests(unittest.TestCase):
    def test_real_sample_rolls_a_small_turn_total(self):
        self.assertEqual(propose(ONE_DIE_SAMPLE), "roll")

    def test_hold_when_holding_wins_the_game(self):
        self.assertEqual(propose(one_die(banked=44, opponent=10, turn_total=6)), "hold")

    def test_high_turn_total_near_the_target_holds(self):
        self.assertEqual(propose(one_die(banked=10, opponent=30, turn_total=25)), "hold")

    def test_expected_value_boundary_is_twenty(self):
        self.assertEqual(propose(one_die(banked=0, opponent=0, turn_total=19)), "roll")
        self.assertEqual(propose(one_die(banked=0, opponent=0, turn_total=20)), "hold")


class TwoDiceTests(unittest.TestCase):
    def test_real_sample_rolls_with_zero_turn_total(self):
        self.assertEqual(propose(TWO_DICE_SAMPLE), "roll")

    def test_hold_when_holding_wins_the_game(self):
        self.assertEqual(propose(two_dice(banked=52, opponent=10, turn_total=8)), "hold")

    def test_high_turn_total_near_the_target_holds(self):
        self.assertEqual(propose(two_dice(banked=10, opponent=30, turn_total=30)), "hold")

    def test_turn_total_zero_rolls_even_when_ahead(self):
        self.assertEqual(propose(two_dice(banked=50, opponent=0, turn_total=0)), "roll")


class LegalActionTests(unittest.TestCase):
    def test_exact_spelling_is_returned(self):
        legal = ["roll", "hold"]
        action = propose(ONE_DIE_SAMPLE, legal)
        self.assertEqual(action, "roll")
        self.assertIn(action, legal)

    def test_case_insensitive_spelling_is_preserved(self):
        self.assertEqual(propose(TWO_DICE_SAMPLE, ["ROLL", "HOLD"]), "ROLL")

    def test_missing_action_returns_none(self):
        self.assertIsNone(propose(one_die(banked=44, opponent=10, turn_total=6), ["roll"]))


class TurnGuardTests(unittest.TestCase):
    def test_opponent_turn_has_no_proposal(self):
        observation = one_die(banked=10, opponent=20, turn_total=5).replace(
            "it is YOUR turn", "it is the opponent's turn"
        )
        self.assertIsNone(propose(observation))

    def test_finished_game_has_no_proposal(self):
        observation = one_die(banked=50, opponent=20, turn_total=0).replace(
            "Turn 4: it is YOUR turn.",
            "The game is over: seat 0 reached 50 (target 50).",
        )
        self.assertIsNone(propose(observation))


if __name__ == "__main__":
    unittest.main()
