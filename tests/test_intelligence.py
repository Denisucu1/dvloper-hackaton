"""Offline, deterministic tests for the replay intelligence stage."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA = os.path.join(ROOT, "arena-lite")
for path in (ROOT, ARENA):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts import intelligence, pipeline  # noqa: E402  (after sys.path)
from arena.registry import get  # noqa: E402


def _policy(game_id: str, game, seat: int) -> str:
    if game_id == "tg_deals":
        return "offer 60" if seat == 0 else "accept"
    if game_id == "pig":
        if seat != 0:
            return "hold"
        return "roll" if game.banked[0] + game.turn_total < game.target else "hold"
    return "hold"


def _drive(game_id: str, variant: dict, seed: int, seats: list[str], ms: int = 500) -> dict:
    """Play a real arena game with a scripted policy and return the replay's game document."""
    game = get(game_id)(dict(variant), seed)
    timeline = []
    for step in range(80):
        if game.is_over():
            break
        movers = list(game.to_move())
        if not movers:
            break
        actions = {}
        normalized = {}
        for seat in movers:
            observation = str(game.observation(seat))
            action = _policy(game_id, game, seat)
            normalized[seat] = game.normalize(seat, action) or action
            actions[str(seat)] = {"action": action, "ms": ms, "observation": observation}
        timeline.append({"step": step, "actions": actions})
        game.apply({seat: normalized[seat] for seat in movers})
    result = game.result()
    return {
        "index": 0,
        "game_id": game_id,
        "category": game.category,
        "variant": dict(variant),
        "seed": seed,
        "seats": list(seats),
        "status": "done",
        "result": result,
        "forfeit": None,
        "timeline": timeline,
    }


def _write_replay(directory: str, name: str, games: list[dict], winner: str = "adn") -> str:
    doc = {
        "match_id": name,
        "a": "adn",
        "b": "rival",
        "winner": winner,
        "score": {"adn": 1, "rival": 0},
        "games": games,
        "chat": [
            {"n": 1, "game": 1, "seat": 0, "team": "adn", "phase": "study",
             "text": "Good luck, let's keep this fair and quick."},
            {"n": 2, "game": 1, "seat": 1, "team": "rival", "phase": "study",
             "text": "Good luck."},
            {"n": 3, "game": 1, "seat": 0, "team": "adn", "phase": "play",
             "text": "Opening at 60 for me."},
        ],
    }
    path = os.path.join(directory, name + ".json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=1)
    return path


def _fixture(directory: str) -> list[dict]:
    deals = get("tg_deals")
    pig = get("pig")
    seats = ["adn", "rival"]
    first = _drive("tg_deals", deals.variants()[0], 3, seats)
    first["index"] = 1
    second = _drive("pig", {"label": "test to 5", "dice": 1, "target": 5}, 7, seats)
    second["index"] = 2
    _write_replay(directory, "round-1", [first])
    _write_replay(directory, "round-2", [second])
    return [first, second]


class IntelligenceAnalysisTests(unittest.TestCase):
    def test_real_games_are_walked_and_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            _fixture(directory)
            result = intelligence.analyse(directory, report={})
            self.assertIsNotNone(result)
            self.assertEqual(len(result["rounds"]), 2)
            self.assertGreaterEqual(result["overall"], 0.0)
            self.assertLessEqual(result["overall"], 100.0)
            self.assertIn(result["confidence"], ("high", "medium", "low"))

            for row in result["rounds"]:
                self.assertIn("legal_pct", row)
                self.assertIn("score", row)
                self.assertIn("judged", row)
                self.assertEqual(row["decisions"], row["legal"])
                self.assertEqual(row["legal_pct"], 100.0)

            pig_row = next(row for row in result["rounds"] if row["game"] == "pig")
            self.assertGreater(pig_row["judged"], 0)
            deals_row = next(row for row in result["rounds"] if row["game"] == "tg_deals")
            self.assertEqual(deals_row["agree"], deals_row["judged"])
            self.assertGreaterEqual(result["inputs"]["judged"], 1)
            self.assertIn("deals", result["inputs"]["judged_by_family"])
            self.assertEqual(result["components"]["legality"]["points"], 30.0)

    def test_corrupted_illegal_action_is_counted_and_lowers_legality(self):
        with tempfile.TemporaryDirectory() as directory:
            _fixture(directory)
            # Corrupt the first pig decision in the replay JSON without touching the real game.
            with open(os.path.join(directory, "round-2.json"), encoding="utf-8") as handle:
                pig_doc = json.load(handle)
            for step in pig_doc["games"][0]["timeline"]:
                entry = (step.get("actions") or {}).get("0")
                if entry:
                    entry["action"] = "banana"
                    break
            with open(os.path.join(directory, "round-2.json"), "w", encoding="utf-8") as handle:
                json.dump(pig_doc, handle, indent=1)

            result = intelligence.analyse(directory, report={})
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result["counters"]["illegal"], 1)
            self.assertEqual(result["components"]["legality"]["points"], 0.0)
            self.assertLess(result["components"]["legality"]["pct"], 100.0)

    def test_bad_game_id_is_an_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            _fixture(directory)
            with open(os.path.join(directory, "round-1.json"), encoding="utf-8") as handle:
                doc = json.load(handle)
            doc["games"][0]["game_id"] = "no_such_game"
            with open(os.path.join(directory, "round-1.json"), "w", encoding="utf-8") as handle:
                json.dump(doc, handle, indent=1)

            result = intelligence.analyse(directory, report={})
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result["counters"]["errors"], 1)
            self.assertEqual(result["components"]["legality"]["points"], 0.0)
            self.assertTrue(any(row["error"] for row in result["rounds"]))

    def test_no_replays_returns_none_and_cli_skips_with_exit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(intelligence.analyse(directory))
            completed = subprocess.run(
                [sys.executable, "scripts/intelligence.py", "--run", directory, "--no-color"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-1000:])
            self.assertIn("INTELLIGENCE SKIPPED", completed.stdout)


class TacticMatchTests(unittest.TestCase):
    def test_numeric_offer_quality_tolerance(self):
        self.assertEqual(intelligence._tactic_match("deals", "offer 60", "offer 56"), (True, True))
        self.assertEqual(intelligence._tactic_match("deals", "offer 51", "offer 56"), (True, True))
        self.assertEqual(intelligence._tactic_match("deals", "offer 40", "offer 56"), (True, False))
        self.assertEqual(intelligence._tactic_match("deals", "accept", "offer 56"), (True, False))

    def test_bids_are_unjudged_unless_exact(self):
        self.assertEqual(intelligence._tactic_match("bluff", "bid 4", "bid 5"), (False, False))
        self.assertEqual(intelligence._tactic_match("bluff", "BID 4", "bid 4"), (True, True))

    def test_discrete_actions_stay_exact(self):
        self.assertEqual(intelligence._tactic_match("think_ahead", "C1", "c1"), (True, True))
        self.assertEqual(intelligence._tactic_match("think_ahead", "C1", "C2"), (True, False))


class PipelineStageTests(unittest.TestCase):
    def _args(self, no_intelligence: bool = False, bench: bool = False) -> SimpleNamespace:
        return SimpleNamespace(fast=True, no_solvers=False, no_prep_trace=True, verbose=False,
                               bench=bench, label="test", games="tic_tac_toe", seeds=1, best_of=1,
                               no_intelligence=no_intelligence)

    def test_intelligence_follows_performance_without_bench(self):
        names = [name for name, _cmd, _timeout in pipeline.stages(self._args())]
        self.assertIn("intelligence", names)
        self.assertLess(names.index("performance"), names.index("intelligence"))
        self.assertNotIn("bench", names)

    def test_intelligence_follows_bench_when_bench_is_requested(self):
        names = [name for name, _cmd, _timeout in pipeline.stages(self._args(bench=True))]
        self.assertIn("bench", names)
        self.assertIn("intelligence", names)
        self.assertGreater(names.index("intelligence"), names.index("bench"))
        self.assertGreater(names.index("intelligence"), names.index("performance"))

    def test_no_intelligence_removes_it_in_both_orders(self):
        for bench in (False, True):
            names = [name for name, _cmd, _timeout in pipeline.stages(
                self._args(no_intelligence=True, bench=bench))]
            self.assertNotIn("intelligence", names)
            self.assertIn("performance", names)
            if bench:
                self.assertIn("bench", names)


if __name__ == "__main__":
    unittest.main()
