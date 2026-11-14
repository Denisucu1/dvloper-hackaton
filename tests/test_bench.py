import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location("bench", os.path.join(ROOT, "scripts", "bench.py"))
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def replay(scores=(1, 0), reason="seat 0 won", warnings=0, with_error=None, chat_lines=None):
    timeline = [
        {"step": 1, "actions": {"0": {"action": "A1", "ms": 1000}, "1": {"action": "B2", "ms": 50}}},
        {"step": 2, "actions": {"0": {"action": "C3", "ms": 3000}}},
    ]
    if with_error:
        timeline.append({"step": 3, "actions": {"0": {"ms": 2000, "error": with_error}}})
    return {
        "status": "done",
        "winner": "adn" if scores[0] > scores[1] else None,
        "games": [
            {
                "index": 1,
                "seats": ["adn", "rival"],
                "warnings": {"0": warnings, "1": 0},
                "study": {"0": {"ms": 1500}, "1": {"ms": 0}},
                "result": {"scores": list(scores), "reason": reason},
                "timeline": timeline,
            }
        ],
        "chat": chat_lines if chat_lines is not None else [
            {"team": "adn", "phase": "study", "text": "hello"},
            {"team": "adn", "phase": "play", "text": "my move"},
            {"team": "adn", "phase": "play", "text": "again"},
            {"team": "rival", "phase": "play", "text": "not ours"},
        ],
    }


def analyse(doc):
    return bench.analyse({"game": "g", "seed": 1, "match_id": "m", "duration_s": 12.0, "doc": doc})


class AnalyseTests(unittest.TestCase):
    def test_score_points_and_move_stats(self):
        row = analyse(replay())
        self.assertEqual(row["points"], 1.0)
        self.assertEqual((row["wins"], row["draws"], row["losses"]), (1, 0, 0))
        self.assertEqual(row["moves"], 2)
        self.assertEqual(row["move_ms_max"], 3000)
        self.assertEqual(row["study_ms_mean"], 1500)
        self.assertEqual(row["chat_lines"], 3)  # the opponent's line is not ours
        self.assertEqual(row["chat_reply_rate"], 1.0)  # 2 moves + 1 ready = 3 expected lines

    def test_draw_and_missing_chat(self):
        row = analyse(replay(scores=(0, 0), chat_lines=[]))
        self.assertEqual(row["points"], 0.5)
        self.assertEqual(row["draws"], 1)
        self.assertEqual(row["chat_reply_rate"], 0.0)

    def test_warnings_counted(self):
        self.assertEqual(analyse(replay(warnings=1))["warnings"], 1)

    def test_timeline_timeout(self):
        row = analyse(replay(scores=(0, 1), with_error="timeout"))
        self.assertEqual(row["timeouts"], 1)
        self.assertEqual(row["points"], 0.0)

    def test_study_phase_failure_is_classified(self):
        doc = replay()
        doc["games"][0]["result"] = {"scores": [0, 1], "reason": "adn crashed (process exited) during study"}
        doc["games"][0]["forfeit"] = True
        row = analyse(doc)
        self.assertEqual(row["study_failures"], 1)
        self.assertEqual(row["crashes"], 1)
        self.assertEqual(row["forfeits"], 1)

    def test_no_double_count_of_timeline_error(self):
        row = analyse(replay(scores=(0, 1), reason="adn made an illegal move", with_error="illegal"))
        self.assertEqual(row["illegal"], 1)


def fake_args(**overrides):
    values = {
        "label": "test",
        "bot": "b",
        "games": ["g"],
        "opponent": ["o"],
        "opponents": ["o"],
        "seeds": 1,
        "best_of": 3,
        "prep": True,
        "prep_seconds": 600,
    }
    values.update(overrides)
    return type("A", (), values)()


class ReportTests(unittest.TestCase):
    def test_delta_direction(self):
        self.assertIn("better", bench.delta_text("warnings", 0, 2))
        self.assertIn("WORSE", bench.delta_text("warnings", 2, 0))
        self.assertIn("better", bench.delta_text("win_pct", 60, 40))
        self.assertEqual(bench.delta_text("wins", 3, 3), "=")

    def test_gate_fails_on_bad_run(self):
        args = fake_args()
        report = bench.aggregate([analyse(replay(warnings=1, chat_lines=[]))], None, None, args)
        problems = bench.gate(report, None)
        self.assertTrue(any("warning" in p for p in problems))
        self.assertTrue(any("chat reply" in p for p in problems))

    def test_gate_passes_clean_run(self):
        args = fake_args()
        report = bench.aggregate([analyse(replay())], None, None, args)
        self.assertEqual(bench.gate(report, None), [])

    def test_gate_catches_score_regression(self):
        args = fake_args()
        good = bench.aggregate([analyse(replay())], None, None, args)
        bad = bench.aggregate([analyse(replay(scores=(0, 1)))], None, None, args)
        problems = bench.gate(bad, good)
        self.assertTrue(any("score dropped" in p for p in problems))

    def test_same_config_filters_baseline(self):
        args = fake_args()
        report = bench.aggregate([analyse(replay())], None, None, args)
        self.assertTrue(bench.same_config(report, args))
        self.assertFalse(bench.same_config(report, fake_args(bot="other")))


if __name__ == "__main__":
    unittest.main()
