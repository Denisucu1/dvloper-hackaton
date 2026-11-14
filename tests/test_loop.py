import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProtocolLoopTests(unittest.TestCase):
    def test_smoke_script(self):
        """End-to-end: prep -> start -> turns -> end -> second game -> shutdown, with no model."""
        completed = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "smoke.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
