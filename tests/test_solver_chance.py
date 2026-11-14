"""Offline tests for the chance/expectimax solver.

The fixtures are intentionally self-contained: they implement the full GameModel surface plus the
optional chance methods, and their `evaluate()` is the exact win probability of a one-die Pig race
computed here by value iteration. That makes the solver's expected value directly comparable with a
DP reference at the same states.
"""
import importlib.util
import os
import time
import unittest
from types import SimpleNamespace

from harness.game_model import apply_chance, chance_of, check_model, has_chance
from harness.solver_chance import ChanceSolver, hold_roll_policy

PIG_TARGET = 30
FACES = (2, 3, 4, 5, 6)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- Pig reference DP
def _dp_solve(target: int = PIG_TARGET, tolerance: float = 1e-9) -> dict:
    """Exact win probability W(p, o, t) for the player to move (value iteration, zero-sum Pig).

    p = our bank, o = opponent bank, t = turn total. A state with p + t >= target is worth 1.0:
    the player to move can hold and win immediately. The only cycles come from repeated busts, so
    Gauss-Seidel value iteration converges (and is simple enough to trust as the reference).
    """
    table = {(p, o, t): 0.5 for p in range(target) for o in range(target) for t in range(target - p)}
    for _ in range(5000):
        delta = 0.0
        for p in range(target):
            for o in range(target):
                for t in range(target - p):
                    hold = 1.0 - table[(o, p + t, 0)]
                    roll = (1.0 - table[(o, p, 0)]) / 6.0
                    for face in FACES:
                        roll += (1.0 if p + t + face >= target else table[(p, o, t + face)]) / 6.0
                    value = hold if hold > roll else roll
                    difference = value - table[(p, o, t)]
                    if difference < 0.0:
                        difference = -difference
                    if difference > delta:
                        delta = difference
                    table[(p, o, t)] = value
        if delta < tolerance:
            return table
    return table


def _dp_values(table: dict, p: int, o: int, t: int, target: int = PIG_TARGET) -> tuple[float, float]:
    """(hold, roll) continuation values for the player to move."""
    if p + t >= target:
        return 1.0, 0.0
    hold = 1.0 - table[(o, p + t, 0)]
    roll = (1.0 - table[(o, p, 0)]) / 6.0
    for face in FACES:
        roll += (1.0 if p + t + face >= target else table[(p, o, t + face)]) / 6.0
    return hold, roll


def _dp_optimal(table: dict, p: int, o: int, t: int, target: int = PIG_TARGET) -> tuple[set[str], float]:
    """Set of optimal actions (ties accepted) and the exact state value."""
    if p + t >= target:
        return {"hold"}, 1.0
    hold, roll = _dp_values(table, p, o, t, target)
    if abs(hold - roll) <= 1e-9:
        return {"hold", "roll"}, max(hold, roll)
    return ({"hold"} if hold > roll else {"roll"}), max(hold, roll)


_PIG_TABLE: dict | None = None


def pig_table() -> dict:
    global _PIG_TABLE
    if _PIG_TABLE is None:
        _PIG_TABLE = _dp_solve(PIG_TARGET)
    return _PIG_TABLE


def setUpModule() -> None:  # warm the reference DP outside the per-move timing test
    pig_table()


# --------------------------------------------------------------------------- fixtures
class PigGame:
    """One-die Pig, first to 30: roll 1..6, a 1 wipes the turn total, a hold banks it."""

    target = PIG_TARGET

    def __init__(self, banked=(0, 0), turn=0, turn_total=0, phase="decide"):
        self.banked = list(banked)
        self.turn = int(turn)
        self.turn_total = int(turn_total)
        self.phase = phase  # "decide" or "chance"

    # --- mandatory GameModel surface -------------------------------------
    def update(self, observation):
        return None

    def legal_actions(self, seat):
        if self.is_terminal() or self.phase != "decide" or int(seat) != self.turn:
            return []
        return ["roll", "hold"]

    def apply(self, seat, action):
        if self.is_terminal() or self.phase != "decide" or int(seat) != self.turn:
            raise ValueError("not this seat's decision")
        action = str(action).strip().lower()
        if action == "roll":
            self.phase = "chance"
        elif action == "hold":
            self.banked[self.turn] += self.turn_total
            self.turn_total = 0
            if not self.is_terminal():
                self.turn = 1 - self.turn
        else:
            raise ValueError("unknown action")

    def is_terminal(self):
        return any(score >= self.target for score in self.banked)

    def to_move(self):
        if self.is_terminal() or self.phase == "chance":
            return []
        return [self.turn]

    def score(self, seat):
        ours = self.banked[seat] >= self.target
        theirs = self.banked[1 - seat] >= self.target
        if ours and theirs:
            return 0.5
        return 1.0 if ours else 0.0 if theirs else 0.5

    def evaluate(self, seat):
        if self.is_terminal():
            return self.score(seat)
        table = pig_table()
        p, o = self.banked[self.turn], self.banked[1 - self.turn]
        if p + self.turn_total >= self.target:
            to_move_value = 1.0
        else:
            to_move_value = table.get((p, o, self.turn_total), 0.5)
        return to_move_value if int(seat) == self.turn else 1.0 - to_move_value

    def clone(self):
        return type(self)(self.banked, self.turn, self.turn_total, self.phase)

    # --- optional chance surface -----------------------------------------
    def chance_actions(self):
        if self.phase != "chance" or self.is_terminal():
            return None
        return [(str(face), 1.0 / 6.0) for face in range(1, 7)]

    def apply_chance(self, outcome):
        if self.phase != "chance" or self.is_terminal():
            raise ValueError("no chance node")
        face = int(outcome)
        if face == 1:
            self.turn_total = 0
            self.turn = 1 - self.turn
        else:
            self.turn_total += face
        self.phase = "decide"

    def sample_state(self, rng):
        child = self.clone()
        child.apply_chance(str(rng.choice(range(1, 7))))
        return child

    def info_state(self):
        return self.key()

    def key(self):
        return f"{tuple(self.banked)}:{self.turn}:{self.turn_total}:{self.phase}"


class PigTwoDice(PigGame):
    """Smoke fixture: two dice, any 1 busts the turn, snake eyes also resets the bank."""

    target = 30
    # 36 ordered outcomes: 10 single-1 busts, 1 snake-eyes, and the 25 no-1 sums.
    OUTCOMES = (
        ("bust", 10.0), ("snake", 1.0),
        ("4", 1.0), ("5", 2.0), ("6", 3.0), ("7", 4.0), ("8", 5.0),
        ("9", 4.0), ("10", 3.0), ("11", 2.0), ("12", 1.0),
    )

    def chance_actions(self):
        if self.phase != "chance" or self.is_terminal():
            return None
        return list(self.OUTCOMES)

    def apply_chance(self, outcome):
        if self.phase != "chance" or self.is_terminal():
            raise ValueError("no chance node")
        outcome = str(outcome)
        if outcome == "bust":
            self.turn_total = 0
            self.turn = 1 - self.turn
        elif outcome == "snake":
            self.turn_total = 0
            self.banked[self.turn] = 0
            self.turn = 1 - self.turn
        else:
            self.turn_total += int(outcome)
        self.phase = "decide"

    def evaluate(self, seat):
        if self.is_terminal():
            return self.score(seat)
        diff = self.banked[int(seat)] - self.banked[1 - int(seat)]
        bonus = 0.10 if int(seat) == self.turn else -0.10
        return min(1.0, max(0.0, 0.5 + 0.45 * diff / self.target + bonus))

    def key(self):
        return f"2d:{tuple(self.banked)}:{self.turn}:{self.turn_total}:{self.phase}"


class DeterministicGame:
    """A valid GameModel with no chance methods at all; it must keep validating unchanged."""

    def __init__(self, done=False):
        self.done = bool(done)

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [] if self.done else ["a", "b"]

    def apply(self, seat, action):
        self.done = True

    def is_terminal(self):
        return self.done

    def to_move(self):
        return [] if self.done else [0]

    def score(self, seat):
        return 0.5

    def clone(self):
        return DeterministicGame(self.done)


class RaisingChance(PigGame):
    """A model that advertises chance support but explodes when it is queried."""

    def chance_actions(self):
        raise RuntimeError("chance_actions is broken")


class WeightedRoll:
    """One decision, then a two-outcome chance node with weights 2:1 (no need for a DP)."""

    def __init__(self, rolled=False, outcome=None):
        self.rolled = bool(rolled)
        self.outcome = outcome

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [] if self.rolled or int(seat) != 0 else ["roll"]

    def apply(self, seat, action):
        self.rolled = True

    def is_terminal(self):
        return self.outcome is not None

    def to_move(self):
        return [] if self.rolled else [0]

    def score(self, seat):
        if self.outcome is None:
            return 0.5
        winner = 1.0 if self.outcome == "win" else 0.0
        return winner if int(seat) == 0 else 1.0 - winner

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        return WeightedRoll(self.rolled, self.outcome)

    def chance_actions(self):
        return [("win", 2.0), ("lose", 1.0)]

    def apply_chance(self, outcome):
        self.outcome = str(outcome)


class Adversary:
    """Seat 0 chooses a branch, then seat 1 picks the worst reply: a pure min node."""

    def __init__(self, first=None, second=None):
        self.first = first
        self.second = second

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        if self.first is None and int(seat) == 0:
            return ["a", "b"]
        if self.first is not None and self.second is None and int(seat) == 1:
            return ["x", "y"]
        return []

    def apply(self, seat, action):
        if int(seat) == 0:
            self.first = str(action)
        else:
            self.second = str(action)

    def is_terminal(self):
        return self.second is not None

    def to_move(self):
        if self.first is None:
            return [0]
        if self.second is None:
            return [1]
        return []

    def score(self, seat):
        if self.first == "b":
            return 0.5
        return 0.0

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        return Adversary(self.first, self.second)

    def chance_actions(self):
        return None  # no random event in this game, but the optional surface exists


# --------------------------------------------------------------------------- tests
class ChanceContractTests(unittest.TestCase):
    def test_deterministic_model_still_validates(self):
        check_model(DeterministicGame())  # must not raise: optional methods are not required
        self.assertFalse(has_chance(DeterministicGame()))

    def test_chance_model_validates_and_is_detected(self):
        check_model(PigGame())
        self.assertTrue(has_chance(PigGame()))
        self.assertEqual(len(chance_of(PigGame(phase="chance"))), 6)

    def test_wrappers_are_safe(self):
        self.assertEqual(chance_of(DeterministicGame()), [])
        self.assertEqual(chance_of(RaisingChance()), [])
        self.assertFalse(apply_chance(DeterministicGame(), "6"))

        game = PigGame(phase="chance")
        self.assertTrue(apply_chance(game, "4"))
        self.assertEqual(game.turn_total, 4)
        self.assertEqual(game.phase, "decide")


class DeterministicFallbackTests(unittest.TestCase):
    def test_missing_chance_surface_returns_none(self):
        action, value = ChanceSolver().propose(DeterministicGame(), 0)
        self.assertIsNone(action)
        self.assertEqual(value, 0.5)

    def test_none_model_is_safe(self):
        self.assertEqual(ChanceSolver().propose(None, 0), (None, 0.5))

    def test_wrong_turn_has_no_move(self):
        self.assertEqual(ChanceSolver().propose(PigGame(turn=1), 0), (None, 0.5))


class RaisingChanceTests(unittest.TestCase):
    def test_broken_chance_method_never_raises(self):
        action, value = ChanceSolver().propose(RaisingChance(), 0, seconds=0.4)
        self.assertTrue(action is None or isinstance(action, str))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class ExpectimaxStructureTests(unittest.TestCase):
    def test_chance_outcomes_are_probability_weighted(self):
        action, value = ChanceSolver().propose(WeightedRoll(), 0, seconds=0.3)
        self.assertEqual(action, "roll")
        self.assertAlmostEqual(value, 2.0 / 3.0, places=9)

    def test_opponent_nodes_minimize(self):
        action, value = ChanceSolver().propose(Adversary(), 0, seconds=0.3)
        self.assertEqual(action, "b")
        self.assertAlmostEqual(value, 0.5, places=9)


class HelperTests(unittest.TestCase):
    def test_matches_is_conservative(self):
        solver = ChanceSolver()
        self.assertTrue(solver.matches("dice_race", SimpleNamespace(family="unknown")))
        self.assertTrue(solver.matches("anything", SimpleNamespace(family="dice")))
        self.assertFalse(solver.matches("tic_tac_toe", SimpleNamespace(family="think_ahead")))
        self.assertFalse(solver.matches("dice_bargain", SimpleNamespace(family="deals")))
        self.assertFalse(solver.matches("pig", None))  # the family solver owns plain Pig

    def test_classic_one_die_threshold_is_twenty(self):
        self.assertEqual(hold_roll_policy(PigGame(turn_total=19)), "roll")
        self.assertEqual(hold_roll_policy(PigGame(turn_total=20)), "hold")
        self.assertEqual(hold_roll_policy(PigGame(turn_total=24)), "hold")

    def test_helper_holds_when_holding_wins(self):
        # 28 banked + 2 turn total reaches the target 30, so rolling would be reckless.
        self.assertEqual(hold_roll_policy(PigGame(banked=(28, 0), turn_total=2)), "hold")

    def test_helper_preserves_legal_spelling_and_returns_none_when_unknown(self):
        self.assertEqual(hold_roll_policy(PigGame(turn_total=1), legal=["ROLL", "HOLD"]), "ROLL")
        self.assertEqual(hold_roll_policy(PigGame(turn_total=25), legal=["ROLL", "HOLD"]), "HOLD")
        self.assertIsNone(hold_roll_policy(SimpleNamespace(), legal=["roll", "hold"]))


class PigOptimalityTests(unittest.TestCase):
    """The solver must play the exact DP-optimal action around the hold/roll boundary."""

    @classmethod
    def setUpClass(cls):
        cls.table = pig_table()
        cls.states = []
        for p in (0, 5, 10, 15, 20, 25, 28):
            for o in (0, 10, 20):
                for t in (16, 18, 19, 20, 21, 22, 24):
                    cls.states.append((p, o, t))

    def test_decisions_match_dp_around_the_boundary(self):
        solver = ChanceSolver()
        matched = 0
        decisions = {"hold": 0, "roll": 0}
        for p, o, t in self.states:
            game = PigGame(banked=(p, o), turn=0, turn_total=t)
            action, _ = solver.propose(game, 0, seconds=0.5, max_depth=4)
            optimal, _ = _dp_optimal(self.table, p, o, t)
            if action in optimal:
                matched += 1
            decisions[action] = decisions.get(action, 0) + 1
        self.assertGreaterEqual(matched, int(0.9 * len(self.states)),
                                f"{matched}/{len(self.states)} matched DP")
        # Both sides of the boundary must actually occur, otherwise the test is vacuous.
        self.assertGreater(decisions.get("hold", 0), 0)
        self.assertGreater(decisions.get("roll", 0), 0)

    def test_value_matches_dp_within_two_percent(self):
        solver = ChanceSolver()
        worst = 0.0
        for p, o, t in self.states:
            game = PigGame(banked=(p, o), turn=0, turn_total=t)
            _, value = solver.propose(game, 0, seconds=0.5, max_depth=4)
            _, expected = _dp_optimal(self.table, p, o, t)
            worst = max(worst, abs(value - expected))
        self.assertLessEqual(worst, 0.02, f"worst value error {worst:.4f}")

    def test_one_propose_is_fast(self):
        game = PigGame(banked=(8, 10), turn=0, turn_total=12)
        started = time.monotonic()
        action, value = ChanceSolver().propose(game, 0, seconds=1.0, max_depth=4)
        elapsed = time.monotonic() - started
        self.assertIn(action, ("roll", "hold"))
        self.assertTrue(0.0 <= value <= 1.0)
        self.assertLess(elapsed, 0.5, f"propose took {elapsed:.3f}s")


class TwoDiceSmokeTests(unittest.TestCase):
    def test_two_dice_proposes_a_legal_action(self):
        game = PigTwoDice()
        action, value = ChanceSolver().propose(game, 0, seconds=0.5, max_depth=2)
        self.assertIn(action, game.legal_actions(0))
        self.assertTrue(0.0 <= value <= 1.0)

    def test_two_dice_holds_when_holding_wins(self):
        game = PigTwoDice(banked=(28, 0), turn_total=2)
        action, value = ChanceSolver().propose(game, 0, seconds=0.5, max_depth=2)
        self.assertEqual(action, "hold")
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_two_dice_snake_eyes_resets_the_bank(self):
        game = PigTwoDice(banked=(10, 5), turn=0, turn_total=6, phase="chance")
        self.assertTrue(apply_chance(game, "snake"))
        self.assertEqual(game.banked, [0, 5])
        self.assertEqual(game.turn, 1)


class StandaloneLoadTests(unittest.TestCase):
    def test_solver_loads_without_the_package(self):
        """The sandbox loads solver_chance.py by path; it must not need harness/game_model.py."""
        path = os.path.join(ROOT, "harness", "solver_chance.py")
        spec = importlib.util.spec_from_file_location("sandboxed_chance", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        action, value = module.ChanceSolver().propose(
            PigGame(banked=(10, 0), turn_total=19), 0, seconds=0.4, max_depth=4
        )
        self.assertIn(action, ("roll", "hold"))
        self.assertTrue(0.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
