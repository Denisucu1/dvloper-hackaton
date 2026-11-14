import os
import tempfile
import textwrap
import unittest

from harness import sandbox

SAMPLES = [
    {
        "observation": "board ... it is your turn",
        "legal_actions": ["A1", "B2", "C3"],
        "seat": 0,
        "turn": 4,
    }
]

GOOD = """
def choose(observation, legal_actions, seat):
    if legal_actions:
        return legal_actions[0]
    return "A1"
"""

ILLEGAL = """
def choose(observation, legal_actions, seat):
    return "Z9"
"""

BANNED = """
import socket
def choose(observation, legal_actions, seat):
    return (legal_actions or ["A1"])[0]
"""

HANG = """
def choose(observation, legal_actions, seat):
    while True:
        pass
"""


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="adn-policy-")

    def write(self, name: str, source: str) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent(source))
        return path

    def test_good_module_validates(self):
        path = self.write("good.py", GOOD)
        ok, reason = sandbox.validate_module(path, SAMPLES)
        self.assertTrue(ok, reason)

    def test_illegal_module_rejected(self):
        path = self.write("bad.py", ILLEGAL)
        ok, reason = sandbox.validate_module(path, SAMPLES)
        self.assertFalse(ok)
        self.assertIn("illegal", reason)

    def test_network_module_rejected(self):
        path = self.write("net.py", BANNED)
        ok, reason = sandbox.validate_module(path, SAMPLES)
        self.assertFalse(ok)
        self.assertIn("banned token", reason)

    def test_no_samples_rejected(self):
        path = self.write("good.py", GOOD)
        ok, _ = sandbox.validate_module(path, [])
        self.assertFalse(ok)

    def test_call_choose_returns_legal(self):
        path = self.write("good.py", GOOD)
        self.assertEqual(sandbox.call_choose(path, "obs", ["B2"], 0), "B2")

    def test_call_choose_normalizes(self):
        path = self.write("case.py", 'def choose(o, l, s):\n    return "b2"\n')
        self.assertEqual(sandbox.call_choose(path, "obs", ["B2"], 0), "B2")

    def test_hanging_module_hits_timeout(self):
        path = self.write("hang.py", HANG)
        self.assertIsNone(sandbox.call_choose(path, "obs", ["A1"], 0, timeout=0.6))


if __name__ == "__main__":
    unittest.main()
