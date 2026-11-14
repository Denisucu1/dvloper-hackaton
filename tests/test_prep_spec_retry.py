"""Prep must survive one unparseable spec answer.

On an unseen game, a JSON parse failure costs the whole structured spec (family, playbook, speech rules),
which changes chat and turn policy for the entire match. One retry is cheap insurance.
"""
import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock

from harness import config, memory, prep

RULES = """BOARD RUSH - 5x5. Two players alternate placing a mark on an empty cell.
Get four in a row to win. A move is a cell name like C3."""

SAMPLE = {
    "observation": "BOARD RUSH. You are seat 0. Empty board.\nIt is your turn.",
    "legal_actions": ["C3"],
    "seat": 0,
    "turn": 1,
}

SPEC = {
    "family": "think_ahead",
    "action_format": "cell like C3",
    "action_examples": ["C3"],
    "winning_idea": "four in a row",
    "hidden_information": False,
    "simultaneous": False,
    "scoring": "line of four wins",
    "speech_rules": "",
    "forbidden_subjects": [],
    "traps": [],
    "notes": "5x5",
    "playbook": ["Take the centre."],
}


class FakeGateway:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.spend = 0.0
        self.cap = 0.75

    def configure(self, info=None, cap=None):
        return None

    @property
    def available(self):
        return True

    @property
    def affordable(self):
        return True

    def ask_json(self, system, user, **kwargs):
        self.calls += 1
        if not self.replies:
            return None
        return self.replies.pop(0)

    def ask(self, system, user, **kwargs):
        raise AssertionError("codegen should be disabled in these tests")

    def summary(self):
        return {"calls": self.calls, "spend_usd": 0.0, "cap_usd": self.cap}


def message() -> dict:
    return {
        "type": "prep", "game_id": "spec_probe", "game_name": "Board Rush",
        "variants": [{"label": "5x5", "rules": RULES, "sample_turn": dict(SAMPLE)}],
        "seconds": 600, "you": "adn", "study_seconds": 30, "move_seconds": 20,
        "llm": {"base_url": "scripted://", "api_key": "sk-test", "model": "haiku"},
    }


class SpecRetryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="adn-spec-retry-")
        self.patch = mock.patch.object(config, "MEMORY_DIR", self.root)
        self.patch.start()
        self.policy = mock.patch.object(config, "GENPOLICY", False)
        self.policy.start()
        self.model = mock.patch.object(prep, "GENMODEL", False)
        self.model.start()

    def tearDown(self):
        self.patch.stop()
        self.policy.stop()
        self.model.stop()

    def run_prep(self, replies):
        gateway = FakeGateway(replies)
        with contextlib.redirect_stderr(io.StringIO()):
            result = prep.run_prep(message(), gateway)
        manifest = json.loads((memory.game_dir("spec_probe") / "manifest.json").read_text())
        return gateway, result, manifest

    def test_failed_parse_is_retried_once(self):
        gateway, result, manifest = self.run_prep([None, dict(SPEC)])
        self.assertEqual(gateway.calls, 2)
        self.assertEqual(result.spec.source, "prep")
        self.assertEqual(result.spec.family, "think_ahead")
        notes = " | ".join(manifest["notes"])
        self.assertIn("spec retry", notes)
        self.assertIn("spec from model (retry)", notes)

    def test_two_failures_fall_back_to_the_regex_spec(self):
        gateway, result, manifest = self.run_prep([None, None])
        self.assertEqual(gateway.calls, 2)
        notes = " | ".join(manifest["notes"])
        self.assertIn("regex fallback spec", notes)
        self.assertIn(result.spec.family, ("think_ahead", "unknown"))

    def test_a_good_first_answer_is_not_retried(self):
        gateway, result, _ = self.run_prep([dict(SPEC)])
        self.assertEqual(gateway.calls, 1)
        self.assertEqual(result.spec.family, "think_ahead")


if __name__ == "__main__":
    unittest.main()
