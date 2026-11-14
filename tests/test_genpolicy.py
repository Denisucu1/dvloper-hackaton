"""Prep must also produce usable code for games that are not boards.

Non-board games are the T2 risk: free-text answers, numeric bids, simultaneous choices, hidden words.
These tests pin two things: the policy prompt covers those regimes, and a failed policy gets a real
repair prompt (the previous module + the concrete error), not a blind resample.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import config, memory, prep, sandbox
from harness.spec import Spec

RULES = """WORD DUEL. Two players guess a hidden five-letter word. A guess is any five-letter word.
Feedback marks exact letters, present letters and absent letters. The first to solve it wins."""

SAMPLE = {
    "observation": "WORD DUEL. You are Player 0. Feedback so far: none.\nIt is YOUR turn: guess a five-letter word.",
    "legal_actions": None,
    "seat": 0,
    "turn": 1,
}

SPEC_DICT = {
    "family": "guessing",
    "action_format": "a five-letter word",
    "action_examples": ["crane", "slate"],
    "winning_idea": "guess words that split the remaining word list",
    "hidden_information": True,
    "simultaneous": False,
    "scoring": "first to solve wins",
    "speech_rules": "",
    "forbidden_subjects": [],
    "traps": [],
    "notes": "free text five-letter guesses",
    "playbook": ["Open with a word with many distinct letters."],
}

BAD_POLICY = '''def choose(observation, legal_actions, seat):
    return "too long and definitely not a five letter word"
'''

GOOD_POLICY = '''def choose(observation, legal_actions, seat):
    return "crane"
'''

GOOD_LIST_POLICY = '''def choose(observation, legal_actions, seat):
    if legal_actions:
        return legal_actions[0]
    return "crane"
'''

BANNED_POLICY = '''def choose(observation, legal_actions, seat):
    __import__("os")
    return "crane"
'''


class FakeGateway:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
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
        return dict(SPEC_DICT)

    def ask(self, system, user, **kwargs):
        self.calls += 1
        self.prompts.append(user)
        if not self.replies:
            raise AssertionError("gateway called more often than scripted")
        return self.replies.pop(0)

    def summary(self):
        return {"calls": self.calls, "spend_usd": 0.0, "cap_usd": self.cap}


class FreeTextValidationTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="adn-genpolicy-")

    def write(self, source):
        path = Path(self.dir) / "policy.py"
        path.write_text(source, encoding="utf-8")
        return str(path)

    def test_free_text_action_is_accepted(self):
        ok, reason = sandbox.validate_module(self.write(GOOD_POLICY), [SAMPLE])
        self.assertTrue(ok, reason)

    def test_empty_action_is_rejected(self):
        path = self.write('def choose(o, l, s):\n    return ""\n')
        ok, reason = sandbox.validate_module(path, [SAMPLE])
        self.assertFalse(ok)
        self.assertIn("no action", reason)

    def test_prose_action_is_rejected(self):
        path = self.write('def choose(o, l, s):\n    return "I think the answer is " + "x" * 140\n')
        ok, reason = sandbox.validate_module(path, [SAMPLE])
        self.assertFalse(ok)
        self.assertIn("implausible free-text", reason)


class PolicyRepairTests(unittest.TestCase):
    def test_retry_shows_the_previous_policy_and_the_error(self):
        directory = Path(tempfile.mkdtemp(prefix="adn-genpolicy-"))
        samples = [{
            "observation": "PICK A NUMBER. Numbers left: 1 2 3",
            "legal_actions": ["pick 1", "pick 2", "pick 3"],
            "seat": 0,
            "turn": 1,
        }]
        bad = 'def choose(o, l, s):\n    return "pick 10"\n'
        gateway = FakeGateway([bad, GOOD_LIST_POLICY])
        notes = []
        with contextlib.redirect_stderr(io.StringIO()):
            result = prep._build_genpolicy(
                gateway, directory, "fptest", RULES, samples,
                Spec(family="think_ahead", action_format="pick <n>"), notes, timeout=30.0,
            )
        self.assertIsNotNone(result)
        self.assertEqual(gateway.calls, 2)
        retry = gateway.prompts[1]
        self.assertIn("PREVIOUS MODULE", retry)
        self.assertIn("pick 10", retry)
        self.assertIn("illegal", retry)
        ok, reason = sandbox.validate_module(result, samples)
        self.assertTrue(ok, reason)

    def test_banned_construct_gets_a_fresh_retry_without_the_banned_source(self):
        directory = Path(tempfile.mkdtemp(prefix="adn-genpolicy-"))
        samples = [{
            "observation": "WORD DUEL. Guess a five-letter word.",
            "legal_actions": None,
            "seat": 0,
            "turn": 1,
        }]
        gateway = FakeGateway([BANNED_POLICY, GOOD_POLICY])
        notes = []
        with contextlib.redirect_stderr(io.StringIO()):
            result = prep._build_genpolicy(
                gateway, directory, "fpbanned", RULES, samples, Spec(family="guessing"), notes, timeout=30.0,
            )
        self.assertIsNotNone(result)
        self.assertEqual(gateway.calls, 2)
        retry = gateway.prompts[1]
        self.assertIn("banned construct", retry)
        self.assertIn("Allowed imports", retry)
        self.assertNotIn("__import__(", retry)
        ok, reason = sandbox.validate_module(result, samples)
        self.assertTrue(ok, reason)

    def test_prompt_covers_non_board_regimes(self):
        prompt = prep._GEN_SYSTEM.lower()
        for phrase in ("free text", "simultaneous", "hidden information", "verbatim", "never invent"):
            self.assertIn(phrase, prompt)


class NonBoardPrepEndToEndTests(unittest.TestCase):
    def test_prep_builds_a_valid_policy_for_a_free_text_game(self):
        root = tempfile.mkdtemp(prefix="adn-nonboard-")
        msg = {
            "type": "prep", "game_id": "word_duel_probe", "game_name": "Word Duel",
            "variants": [{"label": "5 letters", "rules": RULES,
                          "sample_turn": dict(SAMPLE), "size": 5}],
            "seconds": 600, "you": "adn", "study_seconds": 30, "move_seconds": 20,
            "llm": {"base_url": "scripted://", "api_key": "sk-test", "model": "haiku"},
        }
        gateway = FakeGateway([GOOD_POLICY])
        with mock.patch.object(config, "MEMORY_DIR", root), contextlib.redirect_stderr(io.StringIO()):
            result = prep.run_prep(msg, gateway)
            self.assertIsNotNone(result.genpolicy, "a free-text game should still get a policy")
            manifest = json.loads((memory.game_dir("word_duel_probe") / "manifest.json").read_text())
            self.assertTrue(manifest["genpolicy"])
            self.assertFalse(manifest["genmodel"], "hidden word: no deterministic simulator")
            ok, reason = sandbox.validate_module(result.genpolicy, [SAMPLE])
            self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
