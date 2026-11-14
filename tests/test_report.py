"""Tests for the probe report: aggregation, deltas, rendering, failure marking and history."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from scripts import report
from scripts.probes import base


def result(name: str, component: str, score: float, n: int = 1, p50: float = 0.0,
           p95: float = 0.0, notes: str = "", status: str | None = None) -> dict:
    """A report-ready result dict without executing a probe."""
    payload = {
        "name": name,
        "component": component,
        "score": score,
        "n": n,
        "p50_ms": p50,
        "p95_ms": p95,
        "notes": notes or ("passed" if score >= 1.0 else "failed"),
        "details": {"synthetic": True},
        "runtime_ms": 0.1,
    }
    payload["status"] = status or ("pass" if score >= 1.0 else "fail")
    return payload


class _FixedProbe(base.Probe):
    name = "fake.fixed"
    component = "fake"

    def __init__(self, score: float) -> None:
        self._score = score

    def run(self) -> dict:
        return base.make_result(
            score=self._score,
            latencies=[0.5, 1.5],
            n=2,
            notes="synthetic fixed probe",
            details={"fixed": True},
        )


class _CrashProbe(base.Probe):
    name = "fake.crash"
    component = "fake"

    def run(self) -> dict:
        raise RuntimeError("boom")


class AggregationTests(unittest.TestCase):
    def test_component_rows_are_weighted_and_totals_match(self):
        results = [
            result("alpha.one", "alpha", 1.0, n=2, p50=1.0, p95=2.0),
            result("alpha.two", "alpha", 0.5, n=2, p50=3.0, p95=4.0),
            result("beta.one", "beta", 0.0, n=1, p50=5.0, p95=6.0),
        ]
        rows = report.aggregate(results)

        self.assertEqual([row["component"] for row in rows], ["alpha", "beta"])
        alpha = rows[0]
        self.assertAlmostEqual(alpha["score"], 0.75)  # (1.0*2 + 0.5*2) / 4
        self.assertEqual(alpha["n"], 4)
        self.assertEqual(alpha["p50_ms"], 3.0)  # worst case inside the component
        self.assertEqual(alpha["p95_ms"], 4.0)
        self.assertEqual([probe["name"] for probe in alpha["failures"]], ["alpha.two"])

        total = report.totals(results)
        self.assertAlmostEqual(total["score"], 0.6)  # 3.0 weighted points / 5 observations
        self.assertEqual(total["n"], 5)
        self.assertEqual(total["p50_ms"], 5.0)
        self.assertEqual(total["p95_ms"], 6.0)
        self.assertEqual(len(total["failures"]), 2)

    def test_zero_n_still_weights_equally(self):
        rows = report.aggregate([
            result("x.a", "x", 1.0, n=0),
            result("x.b", "x", 0.0, n=0),
        ])
        self.assertAlmostEqual(rows[0]["score"], 0.5)


class DeltaTests(unittest.TestCase):
    def test_better_worse_neutral_and_new(self):
        self.assertEqual(report.classify_delta(0.9, 0.8), "better")
        self.assertEqual(report.classify_delta(0.7, 0.8), "worse")
        self.assertEqual(report.classify_delta(0.8, 0.8), "neutral")
        self.assertEqual(report.classify_delta(0.8, 0.8 + 1e-12), "neutral")
        self.assertEqual(report.classify_delta(0.8, None), "new")

    def test_lower_is_better_direction(self):
        self.assertEqual(report.classify_delta(100, 200, higher_is_better=False), "better")
        self.assertEqual(report.classify_delta(200, 100, higher_is_better=False), "worse")

    def test_row_delta_uses_previous_component_aggregate(self):
        previous = {
            "probes": [
                result("alpha.one", "alpha", 0.5, n=1),
                result("alpha.two", "alpha", 0.5, n=1),
                result("beta.one", "beta", 1.0, n=1),
            ]
        }
        row = {"component": "alpha", "score": 0.75}
        delta = report.delta_for_row(row, previous)
        self.assertEqual(delta["state"], "better")
        self.assertAlmostEqual(delta["diff"], 0.25)
        self.assertIn("better", report.format_delta(delta))
        self.assertIn("+25.0", report.format_delta(delta))

        missing = report.delta_for_row({"component": "gamma", "score": 1.0}, previous)
        self.assertEqual(missing["state"], "new")
        self.assertEqual(report.format_delta(missing), "new")

    def test_neutral_and_worse_labels(self):
        self.assertEqual(report.format_delta({"state": "neutral", "diff": 0.0}), "=")
        worse = report.format_delta({"state": "worse", "diff": -0.05})
        self.assertTrue(worse.startswith("WORSE"))
        self.assertIn("-5.0", worse)


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            result("alpha.one", "alpha", 1.0, n=3, p50=1.0, p95=2.0, notes="alpha is healthy"),
            result("beta.one", "beta", 0.5, n=2, p50=4.0, p95=9.0, notes="beta missed a case"),
        ]

    def test_render_without_color_and_without_tty_does_not_crash(self):
        text = report.render(
            self.results,
            previous=None,
            use_color=False,
            history_path="/tmp/probe-history.jsonl",
            fast=False,
            elapsed_ms=12.0,
        )
        self.assertNotIn("\x1b[", text)
        self.assertIn("T2 PROBE REPORT", text)
        self.assertIn("COMPONENT", text)
        self.assertIn("TOTAL", text)
        self.assertIn("alpha", text)
        self.assertIn("beta", text)
        self.assertIn("100.0", text)
        self.assertIn("50.0", text)
        self.assertIn("FAILURES (1)", text)
        self.assertIn("beta.one", text)
        self.assertIn("probe runtime 0.01 s", text)

    def test_render_with_color_emits_ansi(self):
        text = report.render(self.results, use_color=True, history_path="h.jsonl")
        self.assertIn("\x1b[", text)
        self.assertIn("\x1b[0m", text)

    def test_all_pass_render_has_no_failure_section(self):
        text = report.render(
            [result("alpha.one", "alpha", 1.0, n=1)],
            use_color=False,
            history_path="h.jsonl",
        )
        self.assertIn("ALL PROBES PASSED", text)
        self.assertNotIn("FAILURES", text)

    def test_render_with_previous_shows_delta(self):
        previous = {"probes": [result("alpha.one", "alpha", 0.5, n=3)], "totals": {"score": 0.5}}
        text = report.render(self.results, previous=previous, use_color=False, history_path="h.jsonl")
        self.assertIn("better", text)
        self.assertIn("+50.0", text)  # alpha went 0.5 -> 1.0, i.e. +50 points


class FailureMarkingTests(unittest.TestCase):
    def test_failing_and_crashing_probes_are_marked(self):
        fixed = report.execute_probe(_FixedProbe(0.5))
        crashed = report.execute_probe(_CrashProbe())

        self.assertEqual(fixed["status"], "fail")
        self.assertAlmostEqual(fixed["score"], 0.5)
        self.assertEqual(crashed["status"], "error")
        self.assertEqual(crashed["score"], 0.0)
        self.assertIn("crashed", crashed["notes"])

        rows = report.aggregate([fixed, crashed])
        self.assertEqual(rows[0]["status"], "fail")
        self.assertEqual(len(rows[0]["failures"]), 2)

        text = report.render([fixed, crashed], use_color=False, history_path="h.jsonl")
        self.assertIn("FAILURES (2)", text)
        self.assertIn("fake.fixed", text)
        self.assertIn("fake.crash", text)
        self.assertIn("probe crashed", text)


class CliTests(unittest.TestCase):
    def test_main_without_a_tty_stays_plain_and_writes_history(self):
        root = tempfile.mkdtemp(prefix="adn-report-cli-")
        path = os.path.join(root, "history.jsonl")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = report.main(["--component", "actions", "--fast", "--json", path])
        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertNotIn("\x1b[", output)  # StringIO is not a tty: colour must be off
        self.assertIn("actions", output)
        self.assertIn("TOTAL", output)
        self.assertEqual(len(report.load_history(path)), 1)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="adn-report-test-")
        self.path = os.path.join(self.root, "nested", "report-history.jsonl")
        self.results = [
            result("alpha.one", "alpha", 1.0, n=2, p50=1.0, p95=2.0, notes="ok"),
            result("beta.one", "beta", 0.5, n=1, p50=3.0, p95=4.0, notes="failed"),
        ]

    def test_history_write_read_round_trip(self):
        first = report.make_record(self.results, fast=False, elapsed_ms=12.3,
                                   when="2026-01-01T00:00:00")
        second = report.make_record(self.results, fast=True, elapsed_ms=4.2,
                                    when="2026-01-01T00:00:10")
        report.append_history(self.path, first)
        report.append_history(self.path, second)

        loaded = report.load_history(self.path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["when"], "2026-01-01T00:00:00")
        self.assertAlmostEqual(loaded[0]["totals"]["score"], report.totals(self.results)["score"])
        self.assertEqual(loaded[0]["probes"][0]["name"], "alpha.one")
        self.assertEqual(loaded[0]["probes"][0]["details"], {"synthetic": True})
        self.assertEqual(loaded[1]["fast"], True)
        self.assertEqual(loaded[0]["totals"]["failures"], ["beta.one"])

        self.assertIs(report.previous_record(loaded, fast=False), loaded[0])
        self.assertIs(report.previous_record(loaded, fast=True), loaded[1])
        self.assertIs(report.previous_record(loaded), loaded[1])

    def test_corrupt_lines_are_skipped(self):
        report.append_history(self.path, report.make_record(self.results, fast=False, when="2026-01-01T00:00:00"))
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("{not json}\n\n")
        report.append_history(self.path, report.make_record(self.results, fast=False, when="2026-01-01T00:00:05"))
        loaded = report.load_history(self.path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[-1]["when"], "2026-01-01T00:00:05")

    def test_missing_history_is_empty(self):
        self.assertEqual(report.load_history(os.path.join(self.root, "missing.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
