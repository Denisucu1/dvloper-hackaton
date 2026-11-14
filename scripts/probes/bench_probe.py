"""Probe: read the newest bench report, or a synthetic one when no bench has run yet.

The bench report is the ground truth of a real match set. This probe does not run arena-lite (that
would cost minutes and is the integrator's gate); it parses ``bench/results/**/report.json`` if one
exists and otherwise uses a fixed synthetic report with the same shape. Either way it computes the
figures the acceptance gate cares about: win %, warnings, p95 move time and gateway cost.

Score is the fraction of four gate-style conditions that hold: zero warnings, zero hard failures
(timeouts/crashes/illegal/study failures/forfeits/engine errors), p95 below 7 s and chat reply rate
at least 0.98. ``n`` is the number of games in the report (or matches when the report has no games).
"""
from __future__ import annotations

import json
import os
import time

from scripts.probes.base import ROOT, Probe, make_result, register

P95_LIMIT_MS = 7000.0
CHAT_FLOOR = 0.98

# Fixed shape of a bench report; values are plausible and deliberately healthy so the probe exercises
# the "all gates pass" path even before the first real bench run.
SYNTHETIC = {
    "label": "synthetic",
    "when": "synthetic",
    "config": {"games": ["tic_tac_toe", "pig"], "bot": "synthetic"},
    "totals": {
        "matches": 6, "games": 18, "wins": 13, "draws": 2, "losses": 3, "points": 14.0,
        "warnings": 0, "timeouts": 0, "crashes": 0, "illegal": 0, "study_failures": 0,
        "forfeits": 0, "engine_errors": 0, "setup_errors": 0,
        "chat_reply_rate": 1.0, "chat_chars_mean": 64.0,
        "move_ms_mean": 420.0, "move_ms_p95": 900.0, "move_ms_max": 1800.0,
        "usd": 0.0123, "usd_per_match": 0.00205,
    },
    "matches": [],
}


def _latest_report() -> tuple[str | None, dict]:
    """Newest ``report.json`` under bench/results, or (None, synthetic)."""
    base = os.path.join(ROOT, "bench", "results")
    candidates: list[tuple[str, str, dict]] = []
    if os.path.isdir(base):
        for dirpath, _dirnames, filenames in os.walk(base):
            if "report.json" not in filenames:
                continue
            path = os.path.join(dirpath, "report.json")
            try:
                with open(path, encoding="utf-8") as handle:
                    document = json.load(handle)
            except (OSError, ValueError):
                continue
            if isinstance(document, dict):
                candidates.append((str(document.get("when") or ""), path, document))
    if not candidates:
        return None, dict(SYNTHETIC)
    candidates.sort(key=lambda item: (item[0], item[1]))
    _when, path, document = candidates[-1]
    return path, document


def _summarize(report: dict) -> dict:
    totals = report.get("totals") or {}
    games = int(totals.get("games") or 0) or int(totals.get("matches") or 0)
    wins = int(totals.get("wins") or 0)
    hard_failures = sum(
        int(totals.get(key) or 0)
        for key in ("timeouts", "crashes", "illegal", "study_failures", "forfeits", "engine_errors")
    )
    return {
        "games": games,
        "wins": wins,
        "win_pct": round(100.0 * wins / games, 1) if games else 0.0,
        "warnings": int(totals.get("warnings") or 0),
        "hard_failures": hard_failures,
        "p95_ms": float(totals.get("move_ms_p95") or 0.0),
        "usd": totals.get("usd"),
        "usd_per_match": totals.get("usd_per_match"),
        "chat_reply_rate": float(totals.get("chat_reply_rate") or 0.0),
        "setup_errors": int(totals.get("setup_errors") or 0),
    }


@register
class BenchProbe(Probe):
    name = "bench.replay"
    component = "bench"

    def run(self) -> dict:
        started = time.perf_counter()
        path, report = _latest_report()
        load_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        metrics = _summarize(report)
        extract_ms = (time.perf_counter() - started) * 1000.0

        conditions = [
            ("zero warnings", metrics["warnings"] == 0),
            ("zero hard failures", metrics["hard_failures"] == 0),
            (f"p95 under {P95_LIMIT_MS:.0f} ms", metrics["p95_ms"] < P95_LIMIT_MS),
            (f"chat reply rate at least {CHAT_FLOOR:.2f}", metrics["chat_reply_rate"] >= CHAT_FLOOR),
        ]
        passed = sum(1 for _label, ok in conditions if ok)
        source = path or "synthetic (no bench/results/report.json found)"
        details = {
            "source": source,
            "label": report.get("label"),
            "when": report.get("when"),
            "win_pct": metrics["win_pct"],
            "wins": metrics["wins"],
            "games": metrics["games"],
            "warnings": metrics["warnings"],
            "hard_failures": metrics["hard_failures"],
            "p95_ms": metrics["p95_ms"],
            "usd": metrics["usd"],
            "usd_per_match": metrics["usd_per_match"],
            "chat_reply_rate": metrics["chat_reply_rate"],
            "setup_errors": metrics["setup_errors"],
            "conditions": [label for label, _ok in conditions],
            "failures": [label for label, ok in conditions if not ok],
        }
        notes = (
            f"{'synthetic' if path is None else os.path.basename(os.path.dirname(path))}: "
            f"win {metrics['win_pct']:.1f}%, warnings {metrics['warnings']}, "
            f"p95 {metrics['p95_ms']:.0f}ms, cost "
            + (f"${float(metrics['usd']):.4f}" if metrics["usd"] is not None else "n/a")
        )
        return make_result(
            score=passed / len(conditions),
            latencies=[load_ms, extract_ms],
            n=metrics["games"],
            notes=notes,
            details=details,
        )
