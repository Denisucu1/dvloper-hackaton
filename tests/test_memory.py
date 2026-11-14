import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import config, memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="adn-mem-test-")
        self.patch = mock.patch.object(config, "MEMORY_DIR", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_game_dir_is_filesystem_safe(self):
        path = memory.game_dir("Some Game/../id!")
        self.assertEqual(path.parent, Path(self.root))
        self.assertEqual(path.name, "some_game_id")

    def test_atomic_json_roundtrip(self):
        path = memory.game_dir("g") / "a.json"
        memory.atomic_write_json(path, {"x": 1, "y": [1, 2]})
        self.assertEqual(memory.read_json(path), {"x": 1, "y": [1, 2]})
        leftovers = [name for name in os.listdir(path.parent) if name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_fingerprint_stable_and_sensitive(self):
        a = memory.fingerprint("rules", "v1")
        b = memory.fingerprint("rules", "v1")
        c = memory.fingerprint("rules", "v2")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 16)

    def test_results_append_across_containers(self):
        memory.record_result("g", {"result": "win"})
        memory.record_result("g", {"result": "loss"})
        directory = memory.game_dir("g")
        files = [name for name in os.listdir(directory) if name.startswith("results-")]
        self.assertEqual(len(files), 1)
        lines = (directory / files[0]).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual([json.loads(line)["result"] for line in lines], ["win", "loss"])

    def test_read_missing_returns_default(self):
        self.assertEqual(memory.read_json(memory.game_dir("g") / "nope.json", default={}), {})


if __name__ == "__main__":
    unittest.main()
