import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from harness import config, memory, prep, search
from harness.engine import Engine
from harness.spec import Spec

RULES = """SIMULTANEOUS PICK. Both players choose a number at the same time; the higher number wins.
A move is "pick <n>" with n from 1 to N. Equal picks are a draw."""

OBSERVATION = """SIMULTANEOUS PICK. You are seat 0. Both players move at the same time.
Legal picks: 1, 2, 3, 4, 5, 6, 7, 8, 9
It is YOUR turn: "pick <n>" with n one of the legal picks."""

SAMPLE = {
    "type": "turn", "game_index": 1, "turn": 1,
    "observation": OBSERVATION,
    "legal_actions": [f"pick {i}" for i in range(1, 10)],
    "seconds": 20, "chat": [], "seat": 0,
}

SPEC = {
    "family": "read_opponent",
    "action_format": "pick <n>",
    "action_examples": ["pick 1", "pick 9"],
    "winning_idea": "pick higher than the opponent",
    "hidden_information": False,
    "simultaneous": True,
    "scoring": "higher pick wins",
    "speech_rules": "",
    "forbidden_subjects": [],
    "traps": [],
    "notes": "both seats move at once",
    "playbook": ["Pick the highest number: it can never lose."],
}

POLICY = '''def choose(observation, legal_actions, seat):
    return legal_actions[-1] if legal_actions else "pick 1"
'''

MODEL = r'''import re

LEGAL = re.compile(r"Legal picks:\s*(.*)", re.I)


class Model:
    """Simultaneous game: both seats pick at once; the higher number wins."""

    def __init__(self, rules, seat):
        self.seat = int(seat)
        self.open = []
        self.picks = []

    def update(self, observation):
        match = LEGAL.search(observation or "")
        self.open = [int(n) for n in re.findall(r"\d+", match.group(1))] if match else []
        self.picks = []

    def legal_actions(self, seat):
        return [f"pick {n}" for n in self.open]

    def apply(self, seat, action):
        number = int(str(action).split()[-1])
        if number not in self.open:
            raise ValueError("illegal")
        self.open.remove(number)
        self.picks.append((int(seat), number))

    def is_terminal(self):
        return len(self.picks) >= 2

    def to_move(self):
        return [] if self.is_terminal() else [0, 1]

    def score(self, seat):
        chosen = dict(self.picks)
        if len(chosen) < 2:
            return 0.5
        mine, theirs = chosen[seat], chosen[1 - seat]
        return 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        other = Model("", self.seat)
        other.open = list(self.open)
        other.picks = list(self.picks)
        return other
'''


class ScriptedGateway:
    def __init__(self):
        self.calls = 0
        self.spend = 0.0
        self.cap = 0.75

    def configure(self, info=None, cap=None):
        if cap is not None:
            self.cap = cap

    @property
    def available(self):
        return True

    @property
    def affordable(self):
        return True

    def ask_json(self, system, user, **kwargs):
        self.calls += 1
        return dict(SPEC)

    def ask(self, system, user, **kwargs):
        self.calls += 1
        return MODEL if "SIMULATES" in system else POLICY

    def summary(self):
        return {"calls": self.calls, "spend_usd": 0.0, "cap_usd": self.cap}


def message() -> dict:
    return {
        "type": "prep", "game_id": "sim_pick", "game_name": "Simultaneous Pick",
        "variants": [{"label": "1-9", "rules": RULES, "sample_turn": SAMPLE, "size": 9}],
        "seconds": 600, "you": "adn", "study_seconds": 30, "move_seconds": 20,
        "llm": {"base_url": "scripted://", "api_key": "sk-test", "model": "haiku"},
    }


class SimultaneousEndToEndTests(unittest.TestCase):
    """Prep writes a simultaneous simulator, the sandbox validates it, and joint MCTS plays it."""

    def test_prep_to_move_on_a_simultaneous_game(self):
        root = tempfile.mkdtemp(prefix="adn-sim-prep-")
        with mock.patch.object(config, "MEMORY_DIR", root), contextlib.redirect_stderr(io.StringIO()):
            result = prep.run_prep(message(), ScriptedGateway())
            self.assertIsNotNone(result.genmodel)

            manifest = json.loads((memory.game_dir("sim_pick") / "manifest.json").read_text())
            self.assertTrue(manifest["genmodel"])
            notes = " | ".join(manifest["notes"])
            self.assertIn("ok on 1 sample(s)", notes)

            proposal, intent = Engine(result.spec).propose(
                observation=OBSERVATION,
                legal=SAMPLE["legal_actions"],
                seat=0,
                turn=1,
                timeout=2.0,
            )
        self.assertIn(proposal, SAMPLE["legal_actions"])
        self.assertGreaterEqual(int(proposal.split()[-1]), 7)
        self.assertIn("generated model", intent)

    def test_route_for_a_simultaneous_model_is_joint_mcts(self):
        # The action itself is covered above; this pins the router decision that makes it possible.
        class BothToMove:
            def __init__(self):
                self.open = list(range(1, 10))

            def legal_actions(self, seat):
                return [f"pick {n}" for n in self.open]

            def to_move(self):
                return [0, 1]

            def is_terminal(self):
                return len(self.open) < 9

            def score(self, seat):
                return 0.5

            def evaluate(self, seat):
                return 0.5

            def clone(self):
                other = BothToMove()
                other.open = list(self.open)
                return other

            def apply(self, seat, action):
                self.open.remove(int(str(action).split()[-1]))

        meta = search.best_action_meta(BothToMove(), 0, seconds=0.3)
        self.assertEqual(meta["route"], "mcts-joint")
        self.assertIn(meta["action"], [f"pick {i}" for i in range(1, 10)])


class PrepPolicyTests(unittest.TestCase):
    """Cheap code generation is skipped where it cannot help, so prep does not burn its clock/tokens."""

    def test_truth_and_guardrails_have_no_simulator(self):
        for family in ("truth", "guardrails"):
            worth, why = prep.simgen_worthwhile(Spec(family=family))
            self.assertFalse(worth, family)
            self.assertTrue(why)

    def test_hidden_information_is_not_simulated(self):
        worth, why = prep.simgen_worthwhile(Spec(family="think_ahead", hidden_information=True))
        self.assertFalse(worth)
        self.assertIn("hidden", why)

    def test_bluff_and_deals_are_chat_driven(self):
        for family in ("bluff", "deals"):
            self.assertFalse(prep.simgen_worthwhile(Spec(family=family))[0], family)

    def test_open_think_ahead_game_is_worth_simulating(self):
        worth, why = prep.simgen_worthwhile(Spec(family="think_ahead"))
        self.assertTrue(worth, why)

    def test_qa_and_guardrails_skip_the_cheap_policy_too(self):
        for family in ("truth", "guardrails"):
            worth, why = prep.genpolicy_worthwhile(Spec(family=family))
            self.assertFalse(worth, family)
            self.assertTrue(why)
        for family in ("think_ahead", "bluff"):
            self.assertTrue(prep.genpolicy_worthwhile(Spec(family=family))[0], family)


if __name__ == "__main__":
    unittest.main()
