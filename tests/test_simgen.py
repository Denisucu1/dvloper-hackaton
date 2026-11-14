"""Tests for the prep step that makes the model write a simulator, not a policy.

Everything here is offline: a FakeGateway scripts the model's answers, and the real sandbox validates
the generated module against real sample-turn shapes (the same path prep uses in the arena).
"""
import tempfile
import unittest
from pathlib import Path

from harness import memory, sandbox, simgen
from harness.game_model import MODEL_SUFFIX
from harness.spec import Spec

RULES = """PICK A NUMBER 1-9. Two players alternate turns. On your turn pick one unused number
from 1 to 9. The player who picks 9 wins immediately; if every number is picked, it is a draw.
Action format: pick <n> with n from 1 to 9."""

OBSERVATION = (
    "PICK A NUMBER 1-9. You are Player 0.\n"
    "Numbers left: 1 2 3 4 5 6 7 8 9\n"
    "It is YOUR turn: pick <n>.\n"
)
LEGAL = [f"pick {n}" for n in range(1, 10)]
SAMPLES = [{"observation": OBSERVATION, "legal_actions": LEGAL, "seat": 0, "turn": 1}]

SPEC = Spec(
    game_id="pick_number",
    family="think_ahead",
    action_format="pick <n>",
    action_examples=["pick 1", "pick 9"],
    winning_idea="pick 9 before the opponent does",
    fingerprint="test",
)

GOOD_MODEL = '''class Model:
    def __init__(self, rules: str, seat: int):
        self.rules = rules
        self.seat = seat
        self.left = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.current = 0
        self.winner = None
        self.finished = False

    def update(self, observation: str) -> None:
        self.left = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.current = self.seat
        self.winner = None
        self.finished = False
        for line in observation.splitlines():
            if "numbers left:" in line.lower():
                part = line.split(":", 1)[1]
                self.left = [int(token) for token in part.split() if token.isdigit()]

    def legal_actions(self, seat: int):
        if self.finished or seat != self.current:
            return []
        return ["pick %d" % n for n in self.left]

    def apply(self, seat: int, action: str) -> None:
        number = int(action.split()[-1])
        self.left = [n for n in self.left if n != number]
        if number == 9:
            self.finished = True
            self.winner = seat
        elif not self.left:
            self.finished = True
            self.winner = None
        else:
            self.current = 1 - seat

    def is_terminal(self) -> bool:
        return self.finished

    def to_move(self):
        return [] if self.finished else [self.current]

    def score(self, seat: int) -> float:
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == seat else 0.0

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self):
        other = Model(self.rules, self.seat)
        other.left = list(self.left)
        other.current = self.current
        other.winner = self.winner
        other.finished = self.finished
        return other

    def key(self) -> str:
        return ",".join(str(n) for n in self.left) + "|" + str(self.current)
'''

# Fails validation honestly: the arena's sample lists "pick 1".."pick 9", this one offers something else.
BAD_MODEL = '''class Model:
    def __init__(self, rules: str, seat: int):
        self.seat = seat

    def update(self, observation: str) -> None:
        return None

    def legal_actions(self, seat: int):
        return ["pick 10"]

    def apply(self, seat: int, action: str) -> None:
        return None

    def is_terminal(self) -> bool:
        return False

    def to_move(self):
        return [self.seat]

    def score(self, seat: int) -> float:
        return 0.5

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self):
        return Model("", self.seat)
'''


class FakeGateway:
    """Scripted stand-in for harness.llm.Gateway; honours every keyword the caller passes."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.kwargs = []

    def ask(self, system, user, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        if not self.replies:
            raise AssertionError("gateway called more often than scripted")
        return self.replies.pop(0)


class SimgenTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="adn-simgen-")
        self.fp = memory.fingerprint(RULES)
        self.notes = []
        self.path = Path(self.dir) / f"{self.fp}{MODEL_SUFFIX}"

    def build(self, gateway, samples=None):
        return simgen.build_model(
            gateway=gateway,
            directory=Path(self.dir),
            fingerprint=self.fp,
            game_id="pick_number",
            rules=RULES,
            samples=SAMPLES if samples is None else samples,
            spec=SPEC,
            notes=self.notes,
            timeout=30.0,
        )

    def test_happy_path_returns_a_valid_simulator(self):
        gateway = FakeGateway([GOOD_MODEL])
        result = self.build(gateway)

        self.assertEqual(result, str(self.path))
        self.assertTrue(Path(result).exists())
        source = Path(result).read_text(encoding="utf-8")
        self.assertIn("class Model", source)
        ok, reason = sandbox.validate_model(str(result), SAMPLES, RULES)
        self.assertTrue(ok, reason)
        self.assertEqual(gateway.calls, 1)

    def test_invalid_first_attempt_is_retried_with_the_error(self):
        gateway = FakeGateway([BAD_MODEL, GOOD_MODEL])
        result = self.build(gateway)

        self.assertEqual(result, str(self.path))
        self.assertEqual(gateway.calls, 2)
        self.assertIn("mismatch", self.notes[0])
        self.assertTrue(Path(result).exists())
        ok, reason = sandbox.validate_model(str(result), SAMPLES, RULES)
        self.assertTrue(ok, reason)

    def test_two_invalid_attempts_leave_no_file(self):
        gateway = FakeGateway([BAD_MODEL, BAD_MODEL])
        result = self.build(gateway)

        self.assertIsNone(result)
        self.assertEqual(gateway.calls, 2)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(Path(self.dir).glob(f"*{MODEL_SUFFIX}")), [])

    def test_no_samples_skips_the_gateway(self):
        gateway = FakeGateway([])
        result = self.build(gateway, samples=[])

        self.assertIsNone(result)
        self.assertEqual(gateway.calls, 0)
        self.assertFalse(self.path.exists())

    def test_prose_before_class_model_is_stripped(self):
        gateway = FakeGateway(["Here is your module:\n" + GOOD_MODEL])
        result = self.build(gateway)

        self.assertEqual(result, str(self.path))
        source = Path(result).read_text(encoding="utf-8")
        self.assertTrue(source.startswith("class Model:"))


class MultiVariantPromptTests(unittest.TestCase):
    """The prompt must force variant-agnostic parsing: one game_id, several board sizes/limits."""

    def test_system_prompt_explains_variants(self):
        prompt = simgen._SYSTEM.lower()
        for phrase in ("several variants", "never hardcode", "dynamically", "every variant"):
            self.assertIn(phrase, prompt)

    def test_sample_blob_shows_variant_and_settings(self):
        samples = [
            {"observation": "grid 3x3", "legal_actions": ["A1"], "seat": 0,
             "variant": "3x3 classic", "settings": {"size": 3}},
            {"observation": "grid 5x5", "legal_actions": ["A1", "B2"], "seat": 0,
             "variant": "5x5, four in a row", "settings": {"size": 5, "connect": 4}},
        ]
        blob = simgen._sample_blob(samples)
        self.assertIn("VARIANT: 3x3 classic", blob)
        self.assertIn("VARIANT: 5x5, four in a row", blob)
        self.assertIn("size", blob)


if __name__ == "__main__":
    unittest.main()
