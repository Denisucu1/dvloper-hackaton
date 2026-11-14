"""Contract tests for chance-aware simulators: sandbox validation and prep wiring.

The fixtures are one-die roll/hold games. `chance_actions()` is only non-empty at a pending chance
node, exactly like the interface in docs/T2-INTERFACES.md, so these tests also prove the validator can
reach a chance node behind a decision (the sample turn the arena actually sends).
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import config, memory, prep, sandbox
from harness.game_model import ARTIFACT_VERSION

RULES = """LUCKY DIE. Roll one six-sided die and hold. First to 20 points wins.
On your turn choose "roll" to add the die face to your turn total (a 1 busts the turn total) or
"hold" to bank the turn total. Action format: roll or hold."""

OBSERVATION = """LUCKY DIE. You are Player 0. Banked: You 0, Opponent 0. Turn total: 0.
It is YOUR turn. Actions: roll, hold."""

LEGAL = ["roll", "hold"]

SAMPLES = [{
    "observation": OBSERVATION,
    "legal_actions": LEGAL,
    "seat": 0,
    "turn": 1,
    "variant": "one die",
}]

SPEC = {
    "family": "think_ahead",
    "action_format": "roll or hold",
    "action_examples": ["roll", "hold"],
    "winning_idea": "bank a turn total that reaches 20 before the opponent does",
    "hidden_information": False,
    "simultaneous": False,
    "scoring": "first to 20 points",
    "speech_rules": "",
    "forbidden_subjects": [],
    "traps": ["rolling a 1 loses the turn total"],
    "notes": "one six-sided die; roll enters a pending chance node",
    "playbook": ["Hold once the turn total is close to 20."],
}

POLICY = '''def choose(observation, legal_actions, seat):
    return "hold" if legal_actions and "hold" in legal_actions else "roll"
'''

CHANCE_MODEL = '''
class Model:
    """One-die roll/hold race: roll enters a chance node, apply_chance resolves the face."""

    def __init__(self, rules, seat):
        self.rules = rules
        self.seat = int(seat)
        self.banked = [0, 0]
        self.turn = int(seat)
        self.turn_total = 0
        self.phase = "decide"
        self.target = 20

    def update(self, observation):
        self.banked = [0, 0]
        self.turn = self.seat
        self.turn_total = 0
        self.phase = "decide"
        for line in str(observation or "").splitlines():
            low = line.lower()
            if "banked:" in low:
                numbers = [int(part) for part in line.replace(",", " ").split() if part.isdigit()]
                if len(numbers) >= 2:
                    self.banked = [numbers[0], numbers[1]]
            if "turn total:" in low:
                numbers = [int(part) for part in line.split(":")[-1].split() if part.isdigit()]
                if numbers:
                    self.turn_total = numbers[0]

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
            self._next_turn()
        else:
            raise ValueError("unknown action")

    def is_terminal(self):
        return any(points >= self.target for points in self.banked)

    def to_move(self):
        if self.is_terminal() or self.phase == "chance":
            return []
        return [self.turn]

    def score(self, seat):
        ours = self.banked[int(seat)] >= self.target
        theirs = self.banked[1 - int(seat)] >= self.target
        if ours and theirs:
            return 0.5
        if ours:
            return 1.0
        if theirs:
            return 0.0
        return 0.5

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        other = Model(self.rules, self.seat)
        other.banked = list(self.banked)
        other.turn = self.turn
        other.turn_total = self.turn_total
        other.phase = self.phase
        other.target = self.target
        return other

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
            self._next_turn()
        else:
            self.turn_total += face
        self.phase = "decide"

    def _next_turn(self):
        if not self.is_terminal():
            self.turn = 1 - self.turn
'''.strip()

BAD_SUM_MODEL = CHANCE_MODEL.replace("1.0 / 6.0", "0.1")
RAISING_MODEL = CHANCE_MODEL.replace(
    '        if self.phase != "chance" or self.is_terminal():\n'
    '            raise ValueError("no chance node")',
    '        raise RuntimeError("boom")',
)
NO_APPLY_MODEL = CHANCE_MODEL.replace("    def apply_chance(self, outcome):",
                                      "    def _unused_apply_chance(self, outcome):")
NO_ACTIONS_MODEL = CHANCE_MODEL.replace("    def chance_actions(self):",
                                        "    def _unused_chance_actions(self):")

DETERMINISTIC_MODEL = '''
class Model:
    def __init__(self, rules, seat):
        self.seat = int(seat)
        self.done = False

    def update(self, observation):
        self.done = False

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

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        other = Model("", self.seat)
        other.done = self.done
        return other
'''.strip()

DETERMINISTIC_SAMPLES = [{"observation": "PICK A OR B.", "legal_actions": ["a", "b"], "seat": 0}]


class ChanceSandboxContractTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="adn-chance-contract-")

    def write(self, name, source):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        return path

    def validate(self, source, samples=None, name="model.py"):
        return sandbox.validate_model(self.write(name, source),
                                      SAMPLES if samples is None else samples, RULES)

    def test_one_die_roll_hold_model_validates(self):
        ok, reason = self.validate(CHANCE_MODEL)
        self.assertTrue(ok, reason)
        self.assertIn("ok on 1 sample(s)", reason)
        self.assertIn("chance node", reason)

    def test_probabilities_not_summing_to_one_are_rejected(self):
        ok, reason = self.validate(BAD_SUM_MODEL)
        self.assertFalse(ok)
        self.assertIn("chance_actions sums to", reason)
        self.assertIn("0.6", reason)

    def test_apply_chance_raising_is_rejected_and_names_the_outcome(self):
        ok, reason = self.validate(RAISING_MODEL)
        self.assertFalse(ok)
        self.assertIn("apply_chance('1')", reason)
        self.assertIn("RuntimeError", reason)
        self.assertIn("boom", reason)

    def test_chance_actions_without_apply_chance_is_rejected(self):
        ok, reason = self.validate(NO_APPLY_MODEL)
        self.assertFalse(ok)
        self.assertIn("chance_actions without apply_chance", reason)

    def test_apply_chance_without_chance_actions_is_rejected(self):
        ok, reason = self.validate(NO_ACTIONS_MODEL)
        self.assertFalse(ok)
        self.assertIn("apply_chance without chance_actions", reason)

    def test_deterministic_model_without_chance_methods_stays_valid(self):
        ok, reason = self.validate(DETERMINISTIC_MODEL, samples=DETERMINISTIC_SAMPLES)
        self.assertTrue(ok, reason)
        self.assertNotIn("chance", reason.lower())


class ChancePrepEndToEndTests(unittest.TestCase):
    """A scripted prep must accept a chance simulator and record it in a v7 manifest."""

    class FakeGateway:
        def __init__(self):
            self.calls = 0
            self.cap = 0.75

        @property
        def available(self):
            return True

        @property
        def affordable(self):
            return True

        def configure(self, info=None, cap=None):
            if cap is not None:
                self.cap = cap

        def ask_json(self, system, user, **kwargs):
            self.calls += 1
            return dict(SPEC)

        def ask(self, system, user, **kwargs):
            self.calls += 1
            return CHANCE_MODEL if "SIMULATES" in system else POLICY

        def summary(self):
            return {"calls": self.calls, "spend_usd": 0.0, "cap_usd": self.cap}

    def message(self):
        return {
            "type": "prep", "game_id": "lucky_die_probe", "game_name": "Lucky Die",
            "variants": [{"label": "one die", "rules": RULES,
                          "sample_turn": dict(SAMPLES[0]), "target": 20}],
            "seconds": 600, "you": "adn", "study_seconds": 30, "move_seconds": 20,
            "llm": {"base_url": "scripted://", "api_key": "sk-test", "model": "haiku"},
        }

    def test_prep_validates_a_chance_model_and_writes_manifest_v7(self):
        root = tempfile.mkdtemp(prefix="adn-chance-prep-")
        gateway = self.FakeGateway()
        with mock.patch.object(config, "MEMORY_DIR", root), \
                contextlib.redirect_stderr(io.StringIO()):
            result = prep.run_prep(self.message(), gateway)
            self.assertIsNotNone(result.genmodel)
            self.assertTrue(Path(result.genmodel).exists())

            ok, reason = sandbox.validate_model(result.genmodel, SAMPLES, RULES)
            self.assertTrue(ok, reason)

            manifest = json.loads(
                (memory.game_dir("lucky_die_probe") / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["artifact_version"], 7)
        self.assertEqual(ARTIFACT_VERSION, 7)
        self.assertTrue(manifest["genmodel"].endswith("-model.py"))
        notes = " | ".join(manifest["notes"])
        self.assertIn("ok on 1 sample(s)", notes)


if __name__ == "__main__":
    unittest.main()
