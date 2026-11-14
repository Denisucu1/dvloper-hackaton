#!/usr/bin/env python3
"""Per-component probe scorecard with an ANSI console UI and a JSONL history.

    python3 scripts/report.py                    # run every probe, compare with the previous run
    python3 scripts/report.py --fast             # skip slow probes (subprocess/smoke)
    python3 scripts/report.py --component search # one component (or probe name / prefix)
    python3 scripts/report.py --watch 5          # repeat every 5 s, clear the screen
    python3 scripts/report.py --no-color         # plain text even on a tty
    python3 scripts/report.py --json /tmp/h.jsonl

Scores are printed on a 0-100 scale (probes return 0.0-1.0, see ``scripts/probes/base.py``). DELTA
is the score change in points versus the previous record in the JSONL history; a missing previous
record shows ``new``. Colours are emitted only when stdout is a tty (and ``NO_COLOR`` is unset).

Exit code: 0 when every probe passed, 1 when at least one failed or crashed, 2 when the selection
matched no probe. History is append-only: one JSON object per run, newest last.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.probes import base  # noqa: E402
from scripts.probes import registry, selected  # noqa: E402

PASS_EPSILON = 1e-9
COMPONENT_WIDTH = 18
NOTE_WIDTH = 72
LINE_WIDTH = 136
HISTORY_SCRIPT = os.path.join(ROOT, "scripts", ".report-history.jsonl")
HISTORY_BENCH = os.path.join(ROOT, "bench", "report-history.jsonl")

_STYLES = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
}


def paint(text: str, style: str, use_color: bool) -> str:
    """Wrap ``text`` in one ANSI style, or return it unchanged when colour is off."""
    if not use_color or not text:
        return text
    return f"{_STYLES.get(style, '')}{text}{_STYLES['reset']}"


# --------------------------------------------------------------------------- history
def default_history_path() -> str:
    """``bench/report-history.jsonl`` when a bench directory exists, else the script-side dotfile."""
    if os.path.isdir(os.path.join(ROOT, "bench")):
        return HISTORY_BENCH
    return HISTORY_SCRIPT


def load_history(path: str) -> list[dict]:
    """Read a JSONL history, skipping blank/corrupt lines. Missing file means an empty history."""
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return []
    return records


def append_history(path: str, record: dict) -> None:
    """Append one run to the JSONL history (created with its parent directories if needed)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def previous_record(history: list[dict], fast: bool | None = None) -> dict | None:
    """Newest record, preferring the same ``--fast`` mode so like is compared with like."""
    if not history:
        return None
    if fast is not None:
        for record in reversed(history):
            if bool(record.get("fast")) == bool(fast):
                return record
    return history[-1]


# --------------------------------------------------------------------------- execution
def is_failure(result: dict) -> bool:
    """A probe failed when it crashed or scored below 1.0 (explicit status wins when present)."""
    status = str(result.get("status") or "")
    if status in ("fail", "error"):
        return True
    if status == "pass":
        return False
    return float(result.get("score") or 0.0) < 1.0 - PASS_EPSILON


def _coerce_result(probe: base.Probe, raw: dict, elapsed_ms: float) -> dict:
    """Fill missing keys, clamp the score and attach the probe identity and status."""
    result = {
        "score": base.clamp_score(raw.get("score")),
        "n": int(raw.get("n") or 0),
        "p50_ms": float(raw.get("p50_ms") or 0.0),
        "p95_ms": float(raw.get("p95_ms") or 0.0),
        "notes": str(raw.get("notes") or "").strip(),
        "details": raw.get("details") if isinstance(raw.get("details"), dict) else {},
        "name": probe.name,
        "component": probe.component,
        "slow": bool(probe.slow),
        "runtime_ms": round(elapsed_ms, 3),
    }
    result["status"] = "pass" if result["score"] >= 1.0 - PASS_EPSILON else "fail"
    return result


def execute_probe(probe: base.Probe) -> dict:
    """Run one probe and always return a reportable result; a crash is an ``error`` row, not a raise."""
    started = time.perf_counter()
    try:
        raw = probe.run()
        if not isinstance(raw, dict):
            raise TypeError(f"run() returned {type(raw).__name__}, expected dict")
        return _coerce_result(probe, raw, (time.perf_counter() - started) * 1000.0)
    except Exception as exc:  # noqa: BLE001 - the report must survive a broken probe
        elapsed = (time.perf_counter() - started) * 1000.0
        result = base.make_result(
            score=0.0,
            n=0,
            latencies=[elapsed],
            notes=f"probe crashed: {type(exc).__name__}: {exc}",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )
        result.update({
            "name": probe.name,
            "component": probe.component,
            "slow": bool(probe.slow),
            "runtime_ms": round(elapsed, 3),
            "status": "error",
        })
        return result


def run_probes(*, fast: bool = False, component: str | None = None) -> tuple[list[dict], float]:
    """Execute the selected probes in registry order. Returns (results, elapsed_ms)."""
    probes = selected(component)
    if fast:
        probes = [probe for probe in probes if not probe.slow]
    started = time.perf_counter()
    results = [execute_probe(probe) for probe in probes]
    return results, (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- aggregation
def _weighted_score(probes: list[dict]) -> float:
    total = 0.0
    weight = 0.0
    for probe in probes:
        w = float(probe.get("n") or 0) or 1.0
        total += base.clamp_score(probe.get("score")) * w
        weight += w
    return (total / weight) if weight else 0.0


def _max_metric(probes: list[dict], key: str) -> float:
    return max((float(probe.get(key) or 0.0) for probe in probes), default=0.0)


def aggregate(results: list[dict]) -> list[dict]:
    """Group probes by component: n-weighted score, summed n, worst-case p50/p95, failure list."""
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    for result in results:
        component = str(result.get("component") or result.get("name") or "unknown")
        if component not in groups:
            groups[component] = []
            order.append(component)
        groups[component].append(result)

    rows = []
    for component in order:
        probes = groups[component]
        failures = [probe for probe in probes if is_failure(probe)]
        if failures:
            first = failures[0]
            notes = f"FAIL: {first.get('notes') or 'failed'}"
        else:
            notes = next((str(probe.get("notes")) for probe in probes if probe.get("notes")), "")
            if not notes:
                notes = f"{len(probes)} probe(s) passed"
        rows.append({
            "component": component,
            "score": _weighted_score(probes),
            "n": sum(int(probe.get("n") or 0) for probe in probes),
            "p50_ms": _max_metric(probes, "p50_ms"),
            "p95_ms": _max_metric(probes, "p95_ms"),
            "probes": probes,
            "failures": failures,
            "notes": notes,
            "status": "fail" if failures else "pass",
        })
    return rows


def totals(results: list[dict]) -> dict:
    """One TOTAL row over every probe (not an average of component rows, so n still weights it)."""
    failures = [probe for probe in results if is_failure(probe)]
    return {
        "component": "TOTAL",
        "score": _weighted_score(results),
        "n": sum(int(probe.get("n") or 0) for probe in results),
        "p50_ms": _max_metric(results, "p50_ms"),
        "p95_ms": _max_metric(results, "p95_ms"),
        "probes": list(results),
        "failures": failures,
        "notes": f"{len(failures)} failure(s)" if failures else "all probes passed",
        "status": "fail" if failures else "pass",
    }


# --------------------------------------------------------------------------- deltas
def classify_delta(current, previous, *, higher_is_better: bool = True, tolerance: float = PASS_EPSILON) -> str:
    """``better`` / ``worse`` / ``neutral`` against a previous value, or ``new`` when absent."""
    if current is None or previous is None:
        return "new"
    diff = float(current) - float(previous)
    if abs(diff) <= tolerance:
        return "neutral"
    improved = diff > 0 if higher_is_better else diff < 0
    return "better" if improved else "worse"


def delta_for_row(row: dict, previous: dict | None) -> dict:
    """Compare one component row with the same component in the previous history record."""
    if not previous:
        return {"state": "new", "diff": None}
    previous_probes = [
        probe for probe in (previous.get("probes") or [])
        if str(probe.get("component")) == str(row.get("component"))
    ]
    if not previous_probes:
        return {"state": "new", "diff": None}
    before = _weighted_score(previous_probes)
    state = classify_delta(row.get("score"), before)
    return {"state": state, "diff": float(row.get("score") or 0.0) - before}


def total_delta_for_results(results: list[dict], previous: dict | None) -> dict:
    """TOTAL delta over probes present in both runs, so a filtered view never compares to a full run."""
    if not previous:
        return {"state": "new", "diff": None}
    previous_by_name = {
        str(probe.get("name")): probe for probe in (previous.get("probes") or [])
        if probe.get("name")
    }
    matched_current = [probe for probe in results if str(probe.get("name")) in previous_by_name]
    if not matched_current:
        return {"state": "new", "diff": None}
    before = _weighted_score([previous_by_name[str(probe.get("name"))] for probe in matched_current])
    now = _weighted_score(matched_current)
    return {"state": classify_delta(now, before), "diff": now - before}


def format_delta(delta: dict) -> str:
    """Human label for the DELTA column (points on the 0-100 scale)."""
    state = (delta or {}).get("state", "new")
    if state == "new":
        return "new"
    if state == "neutral":
        return "="
    diff = float((delta or {}).get("diff") or 0.0) * 100.0
    if state == "better":
        return f"better {diff:+.1f}"
    return f"WORSE {diff:+.1f}"


def _delta_style(delta: dict) -> str:
    state = (delta or {}).get("state")
    if state == "better":
        return "green"
    if state == "worse":
        return "red"
    if state == "new":
        return "cyan"
    return "dim"


# --------------------------------------------------------------------------- record
def make_record(results: list[dict], *, fast: bool = False, elapsed_ms: float = 0.0,
                when: str | None = None) -> dict:
    """The JSONL record for one run: probe metrics plus the aggregate totals."""
    total = totals(results)
    return {
        "version": 1,
        "when": when or datetime.now().isoformat(timespec="seconds"),
        "fast": bool(fast),
        "elapsed_ms": round(float(elapsed_ms), 1),
        "probes": [
            {
                "name": result.get("name"),
                "component": result.get("component"),
                "score": result.get("score"),
                "n": result.get("n"),
                "p50_ms": result.get("p50_ms"),
                "p95_ms": result.get("p95_ms"),
                "notes": result.get("notes"),
                "status": result.get("status"),
                "slow": result.get("slow", False),
                "runtime_ms": result.get("runtime_ms", 0.0),
                "details": result.get("details") or {},
            }
            for result in results
        ],
        "totals": {
            "score": total["score"],
            "n": total["n"],
            "p50_ms": total["p50_ms"],
            "p95_ms": total["p95_ms"],
            "probes": len(results),
            "failures": [probe.get("name") for probe in total["failures"]],
        },
    }


# --------------------------------------------------------------------------- rendering
def _row_line(row: dict, delta: dict, use_color: bool) -> str:
    failed = row.get("status") == "fail"
    component = str(row.get("component"))
    score_text = f"{float(row.get('score') or 0.0) * 100.0:5.1f}"
    n_text = f"{int(row.get('n') or 0):>5}"
    p50_text = f"{float(row.get('p50_ms') or 0.0):>7.1f}"
    p95_text = f"{float(row.get('p95_ms') or 0.0):>7.1f}"
    delta_text = f"{format_delta(delta):<14}"
    note = str(row.get("notes") or "")[:NOTE_WIDTH]

    component_cell = paint(f"{component:<{COMPONENT_WIDTH}}", "red" if failed else "green", use_color)
    score_cell = paint(f"{score_text:>6}", "red" if failed else "green", use_color)
    delta_cell = paint(delta_text, _delta_style(delta), use_color)
    note_cell = paint(note, "red" if failed else "dim", use_color)
    return f"{component_cell} {score_cell} {n_text} {p50_text} {p95_text}  {delta_cell} {note_cell}"


def render(results: list[dict], *, previous: dict | None = None, use_color: bool = False,
           history_path: str = "", fast: bool = False, elapsed_ms: float = 0.0) -> str:
    """The whole report as a string. Pure: the caller decides about tty, colours and printing."""
    rows = aggregate(results)
    total = totals(results)
    lines: list[str] = []

    stamp = datetime.now().isoformat(timespec="seconds")
    mode = "fast" if fast else "full"
    lines.append(paint("=" * LINE_WIDTH, "dim", use_color))
    lines.append(paint(" T2 PROBE REPORT", "bold", use_color) + paint(f"   {stamp}", "dim", use_color))
    lines.append(
        paint(
            f" {len(results)} probe(s) · {len(rows)} component(s) · mode {mode} · "
            f"history {history_path or '(not stored)'}",
            "dim",
            use_color,
        )
    )
    lines.append(paint("=" * LINE_WIDTH, "dim", use_color))
    lines.append(
        paint(
            f"{'COMPONENT':<{COMPONENT_WIDTH}} {'SCORE':>6} {'N':>5} {'P50ms':>7} {'P95ms':>7}  "
            f"{'DELTA':<14} NOTE",
            "bold",
            use_color,
        )
    )
    lines.append(paint("-" * LINE_WIDTH, "dim", use_color))

    for row in rows:
        lines.append(_row_line(row, delta_for_row(row, previous), use_color))

    lines.append(paint("-" * LINE_WIDTH, "dim", use_color))
    total_delta = total_delta_for_results(results, previous)
    total_failed = bool(total["failures"])
    total_label = paint(f"{'TOTAL':<{COMPONENT_WIDTH}}", "bold", use_color)
    total_score = paint(
        f"{total['score'] * 100.0:6.1f}",
        "bold red" if total_failed else "bold green",
        use_color,
    )
    total_delta_cell = paint(f"{format_delta(total_delta):<14}", _delta_style(total_delta), use_color)
    lines.append(
        f"{total_label} {total_score} {int(total['n']):>5} "
        f"{float(total['p50_ms']):>7.1f} {float(total['p95_ms']):>7.1f}  "
        f"{total_delta_cell} {len(total['failures'])} failed / {len(results)} probe(s)"
    )

    lines.append("")
    failures = total["failures"]
    if failures:
        lines.append(paint(f"FAILURES ({len(failures)})", "bold red", use_color))
        for probe in failures:
            label = paint(f"  {probe.get('name', '?')}", "red", use_color)
            score = paint(f"{float(probe.get('score') or 0.0) * 100.0:5.1f}", "red", use_color)
            note = str(probe.get("notes") or "failed")[:NOTE_WIDTH * 2]
            lines.append(f"{label} {score}  {note}")
    else:
        lines.append(paint("ALL PROBES PASSED", "bold green", use_color))

    failed_count = len(failures)
    lines.append("")
    lines.append(
        paint(
            f"probe runtime {elapsed_ms / 1000.0:.2f} s · "
            f"{int(total['n'])} observation(s) · {failed_count} failed · "
            f"score {total['score'] * 100.0:.1f}/100",
            "dim",
            use_color,
        )
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the T2 component probes and print a per-component scorecard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="JSONL history file (default: bench/report-history.jsonl when bench/ exists, "
                             "else scripts/.report-history.jsonl)")
    parser.add_argument("--component", default=None, metavar="NAME",
                        help="only probes whose component, name, or name prefix matches NAME")
    parser.add_argument("--fast", action="store_true", help="skip slow probes (smoke subprocess)")
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                        help="repeat every SECONDS and clear the screen")
    parser.add_argument("--no-color", action="store_true", help="never emit ANSI colours")
    return parser


def _run_once(args, history_path: str, use_color: bool) -> int:
    results, elapsed_ms = run_probes(fast=args.fast, component=args.component)
    if not results:
        reason = "the selected probe is slow and --fast skipped it" if args.fast else "no probe matched"
        print(f"report: {reason}", file=sys.stderr)
        return 0 if args.fast else 2

    history = load_history(history_path)
    previous = previous_record(history, fast=args.fast)
    output = render(
        results,
        previous=previous,
        use_color=use_color,
        history_path=history_path,
        fast=args.fast,
        elapsed_ms=elapsed_ms,
    )
    print(output)
    try:
        append_history(history_path, make_record(results, fast=args.fast, elapsed_ms=elapsed_ms))
    except OSError as exc:
        print(f"report: could not write history: {exc}", file=sys.stderr)
    return 1 if any(is_failure(result) for result in results) else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    history_path = args.json or default_history_path()
    use_color = (not args.no_color) and sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    if args.watch is not None:
        try:
            while True:
                if sys.stdout.isatty():
                    sys.stdout.write("\x1b[2J\x1b[H")
                code = _run_once(args, history_path, use_color)
                sys.stdout.flush()
                if code == 2:
                    return code
                time.sleep(max(0.5, float(args.watch)))
        except KeyboardInterrupt:
            print()
            return 0
    return _run_once(args, history_path, use_color)


if __name__ == "__main__":
    sys.exit(main())
