"""Offline tests for the generic push-your-luck solver.

The fixtures use the same observation shape the arena sends for the Pig practice game, plus label
variants an unseen dice game is likely to use. Expected decisions are computed here from an
independent 6- and 36-outcome enumeration, so the tests verify the exact expected value rather than
hardcoding the familiar one-die threshold of 20.
"""
import unittest
from types import SimpleNamespace

from harness.solver_luck import LuckSolver

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
    """Real one-die observation shape with the numbers we want to test."""
    return (
        f"PIG - One die, first to {target}. You are seat 0. Target: {target}.\n"
        f"Banked: you {banked}, opponent {opponent}.\n"
        f"Turn 6: it is YOUR turn. Turn total: {turn_total} (rolls this turn: none yet).\n"
        f"If you hold now you will have {banked + turn_total}. Choose: roll | hold"
    )


def two_dice(
    banked: int,
    opponent: int,
    turn_total: int,
    target: int = 60,
    reset_bank: bool = True,
) -> str:
    """Real two-dice observation shape; the header states whether snake eyes resets the bank."""
    rule = "snake eyes reset the bank" if reset_bank else "snake eyes only lose the turn"
    return (
        f"PIG - Two dice, first to {target} ({rule}). You are seat 0. Target: {target}.\n"
        f"Banked: you {banked}, opponent {opponent}.\n"
        f"Turn 9: it is YOUR turn. Turn total: {turn_total} (rolls this turn: none yet).\n"
        f"If you hold now you will have {banked + turn_total}. Choose: roll | hold"
    )


def propose(observation: str, legal: list[str] | None = None, seat: int = 0) -> str | None:
    """Call the solver the same way the harness does."""
    return LuckSolver().propose(observation=observation, legal=legal, seat=seat)


def one_die_roll_ev(banked: int, turn_total: int, bank_reset: bool = False) -> float:
    """Expected banked score after one roll, enumerated from the six faces."""
    bust = 0 if bank_reset else banked
    return (bust + sum(banked + turn_total + face for face in range(2, 7))) / 6.0


def two_dice_roll_ev(banked: int, turn_total: int, reset_bank: bool) -> float:
    """Expected banked score after one roll, enumerated from all 36 ordered outcomes."""
    total = 0
    for first in range(1, 7):
        for second in range(1, 7):
            if first == 1 and second == 1:
                total += 0 if reset_bank else banked
            elif first == 1 or second == 1:
                total += banked
            else:
                total += banked + turn_total + first + second
    return total / 36.0


def expected_action(
    banked: int,
    turn_total: int,
    target: int,
    dice: int,
    reset_bank: bool = False,
) -> str:
    """Reference decision: hold on an immediate win or when rolling is not strictly better."""
    if banked + turn_total >= target:
        return "hold"
    if dice >= 2:
        roll = two_dice_roll_ev(banked, turn_total, reset_bank)
    else:
        roll = one_die_roll_ev(banked, turn_total, reset_bank)
    return "hold" if not roll > banked + turn_total else "roll"


class RealFormatTests(unittest.TestCase):
    def test_one_die_sample_rolls_a_small_turn_total(self):
        self.assertEqual(propose(ONE_DIE_SAMPLE), "roll")

    def test_two_dice_sample_rolls_with_zero_turn_total(self):
        self.assertEqual(propose(TWO_DICE_SAMPLE), "roll")

    def test_hold_when_holding_wins_immediately(self):
        self.assertEqual(propose(one_die(44, 10, 6)), "hold")
        self.assertEqual(propose(two_dice(52, 10, 8)), "hold")

    def test_rolling_at_zero_turn_total(self):
        self.assertEqual(propose(one_die(10, 20, 0)), "roll")
        self.assertEqual(propose(two_dice(50, 0, 0)), "roll")

    def test_high_total_near_the_target_holds(self):
        self.assertEqual(propose(one_die(10, 30, 25)), "hold")
        self.assertEqual(propose(two_dice(10, 30, 40)), "hold")


class ExpectedValueTests(unittest.TestCase):
    def test_one_die_choices_match_exact_expected_value(self):
        target = 300
        for banked in (0, 17, 40):
            for turn_total in range(0, 80):
                observation = one_die(banked, 0, turn_total, target=target)
                expected = expected_action(banked, turn_total, target, dice=1)
                self.assertEqual(
                    propose(observation),
                    expected,
                    msg=f"banked={banked} turn_total={turn_total}",
                )

    def test_two_dice_choices_match_exact_expected_value(self):
        target = 400
        for reset_bank in (False, True):
            for banked in (0, 22, 90):
                for turn_total in range(0, 70, 2):
                    observation = two_dice(
                        banked, 0, turn_total, target=target, reset_bank=reset_bank
                    )
                    expected = expected_action(
                        banked, turn_total, target, dice=2, reset_bank=reset_bank
                    )
                    self.assertEqual(
                        propose(observation),
                        expected,
                        msg=f"reset={reset_bank} banked={banked} turn_total={turn_total}",
                    )

    def test_measured_one_die_threshold_is_about_twenty(self):
        target = 300
        first_hold = next(
            (
                turn_total
                for turn_total in range(0, 60)
                if propose(one_die(0, 0, turn_total, target=target)) == "hold"
            ),
            None,
        )
        self.assertIsNotNone(first_hold)
        self.assertGreaterEqual(first_hold, 18)
        self.assertLessEqual(first_hold, 22)
        self.assertEqual(propose(one_die(0, 0, first_hold - 1, target=target)), "roll")
        self.assertEqual(propose(one_die(0, 0, first_hold, target=target)), "hold")

    def test_snake_eyes_bank_reset_changes_the_boundary(self):
        target = 400
        banked = 90
        reset_decision = [
            propose(two_dice(banked, 0, turn_total, target=target, reset_bank=True))
            for turn_total in range(0, 30)
        ]
        safe_decision = [
            propose(two_dice(banked, 0, turn_total, target=target, reset_bank=False))
            for turn_total in range(0, 30)
        ]
        # Losing the bank on snake eyes makes rolling less attractive, never more.
        self.assertGreaterEqual(safe_decision.count("roll"), reset_decision.count("roll"))
        self.assertNotEqual(safe_decision, reset_decision)


class LabelToleranceTests(unittest.TestCase):
    def test_goal_score_this_turn_labels(self):
        observation = (
            "LUCKY DASH - you are seat 0. Goal: 100.\n"
            "Score: you 40, opponent 30.\n"
            "It is your turn. This turn: 5.\n"
        )
        self.assertEqual(propose(observation), "roll")

    def test_banked_total_and_turn_total_labels(self):
        observation = (
            "LUCKY DASH - you are seat 0. First to 100.\n"
            "Banked total: you 80, opponent 10.\n"
            "It is your turn. Turn total: 25.\n"
        )
        self.assertEqual(propose(observation), "hold")

    def test_your_score_and_opponent_score_lines(self):
        observation = (
            "DICE DASH - first to 100. You are seat 0.\n"
            "Your score: 90.\n"
            "Opponent score: 20.\n"
            "It is YOUR turn. Unbanked: 12.\n"
        )
        self.assertEqual(propose(observation), "hold")

    def test_seat_one_to_move_is_recognised_as_ours(self):
        observation = (
            "DICE DASH - first to 100. You are seat 1. Target: 100.\n"
            "Banked: you 20, opponent 10.\n"
            "Seat 1's turn. Turn total: 4.\n"
        )
        self.assertEqual(propose(observation, seat=1), "roll")


class LegalActionTests(unittest.TestCase):
    def test_exact_spelling_is_returned(self):
        action = propose(ONE_DIE_SAMPLE, ["roll", "hold"])
        self.assertEqual(action, "roll")
        self.assertIn(action, ["roll", "hold"])

    def test_uppercase_spelling_is_preserved(self):
        self.assertEqual(propose(TWO_DICE_SAMPLE, ["ROLL", "HOLD"]), "ROLL")

    def test_mixed_case_hold_spelling_is_preserved(self):
        self.assertEqual(propose(one_die(44, 10, 6), ["Roll", "Hold"]), "Hold")

    def test_bank_synonym_returns_the_arena_spelling(self):
        self.assertEqual(propose(one_die(44, 10, 6), ["Roll", "Bank"]), "Bank")

    def test_missing_action_returns_none(self):
        self.assertIsNone(propose(one_die(44, 10, 6), ["roll"]))
        self.assertIsNone(propose(one_die(10, 10, 4), ["hold"]))

    def test_empty_or_unrelated_legal_list_returns_none(self):
        self.assertIsNone(propose(ONE_DIE_SAMPLE, []))
        self.assertIsNone(propose(ONE_DIE_SAMPLE, ["pass", "draw"]))


class TurnGuardTests(unittest.TestCase):
    def test_opponent_turn_has_no_proposal(self):
        observation = one_die(10, 20, 5).replace(
            "it is YOUR turn", "it is the opponent's turn"
        )
        self.assertIsNone(propose(observation))

    def test_opponent_turn_by_seat_has_no_proposal(self):
        observation = (
            "DICE DASH - first to 100. You are seat 0. Target: 100.\n"
            "Banked: you 20, opponent 10.\n"
            "Seat 1 to move. Turn total: 4.\n"
        )
        self.assertIsNone(propose(observation))

    def test_finished_game_has_no_proposal(self):
        observation = one_die(50, 20, 0).replace(
            "Turn 6: it is YOUR turn.",
            "The game is over: seat 0 reached 50 (target 50).",
        )
        self.assertIsNone(propose(observation))
        self.assertIsNone(propose(one_die(50, 20, 0) + "\nYou win!"))

    def test_unreadable_text_has_no_proposal(self):
        self.assertIsNone(propose("no idea what is happening here"))
        self.assertIsNone(propose("Turn total: 12, score 4, target 50"))
        self.assertIsNone(propose(""))
        self.assertIsNone(propose("   \n  "))


class MatchingTests(unittest.TestCase):
    def test_pig_id_matches_without_a_spec(self):
        self.assertTrue(LuckSolver.matches("pig", None))
        self.assertTrue(LuckSolver.matches("pig_two_dice", SimpleNamespace(family="unknown")))

    def test_luck_words_in_spec_match(self):
        spec = SimpleNamespace(
            game_id="mystery",
            family="unknown",
            notes="Roll a die and hold to bank; a 1 loses the turn.",
            winning_idea="Reach the target first.",
            action_format="roll | hold",
        )
        self.assertTrue(LuckSolver.matches("mystery", spec))

    def test_action_shape_matches_without_luck_words(self):
        spec = SimpleNamespace(action_examples=["roll", "hold"], action_format="roll | hold")
        self.assertTrue(LuckSolver.matches("mystery", spec))

    def test_pig_id_overrides_a_mislabelled_family(self):
        spec = SimpleNamespace(family="think_ahead", notes="roll and hold")
        self.assertTrue(LuckSolver.matches("pig", spec))

    def test_dict_spec_is_supported(self):
        self.assertTrue(
            LuckSolver.matches(
                "dice_dash",
                {"family": "unknown", "notes": "roll a die and hold to bank"},
            )
        )

    def test_auctions_poker_and_boards_do_not_match(self):
        for game_id in (
            "auction_house",
            "poker_night",
            "chess",
            "connect_four",
            "tic_tac_toe",
        ):
            self.assertFalse(LuckSolver.matches(game_id, SimpleNamespace(family="unknown")))

    def test_rejected_families_do_not_match(self):
        self.assertFalse(
            LuckSolver.matches(
                "dice_duel",
                SimpleNamespace(family="bluff", notes="dice, roll, hold, raise"),
            )
        )
        self.assertFalse(
            LuckSolver.matches(
                "board_race",
                SimpleNamespace(family="think_ahead", notes="roll and hold"),
            )
        )

    def test_unknown_game_does_not_match(self):
        self.assertFalse(LuckSolver.matches("guess_number", None))
        self.assertFalse(
            LuckSolver.matches(
                "",
                SimpleNamespace(family="unknown", notes="answer the question about a document"),
            )
        )


if __name__ == "__main__":
    unittest.main()
