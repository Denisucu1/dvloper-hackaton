#!/usr/bin/env python3
"""One-command testing pipeline: compile, unit tests, probes, solver smoke, protocol, game API, optional bench.

Run from the repo root:

    python3 scripts/pipeline.py                 # fast, offline, ~15 s
    python3 scripts/pipeline.py --bench         # + real arena matches and the gate
    python3 scripts/pipeline.py --no-solvers    # skip the solver smoke stage
    python3 scripts/pipeline.py --json out.json # machine-readable result

The `probes` stage runs the component scorecard; `solvers` then plays one seeded random playout per
arena variant and feeds every observation to the family solvers. Every stage runs even if an earlier
one failed; the summary at the end says what to fix. Exit code is 0 only when every stage passed. No
network is used unless --bench is given (arena matches call the gateway).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(name: str, cmd: list[str], timeout: float) -> dict:
    started = time.time()
    try:
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"name": name, "ok": False, "seconds": round(time.time() - started, 1),
                "note": f"timeout after {int(timeout)} s", "tail": ""}
    elapsed = round(time.time() - started, 1)
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    lines = [line for line in output.strip().splitlines() if line.strip()]
    tail = " | ".join(lines[-3:]) if lines else ""
    note = tail[:180]
    if completed.returncode != 0:
        errors = [line for line in lines if "FAIL" in line or "Error" in line or line.startswith("E ")]
        note = " | ".join((errors or lines)[:2])[:180]
    return {"name": name, "ok": completed.returncode == 0, "seconds": elapsed, "note": note,
            "tail": output[-3000:]}


def stages(args) -> list[tuple[str, list[str], float]]:
    plan = [
        ("compile", [PY, "-m", "compileall", "-q", "main.py", "harness", "scripts", "tests"], 120),
        ("unit tests", [PY, "-m", "unittest", "discover", "-s", "tests"], 300),
        ("probes", [PY, "scripts/report.py", "--no-color"] + (["--fast"] if args.fast else []), 180),
    ]
    if not args.no_solvers:
        plan.append(("solvers", [PY, "scripts/solver_smoke.py"], 120))
    plan += [
        ("protocol", [PY, "scripts/smoke.py"], 120),
        ("games", [PY, "scripts/game_smoke.py", "--no-color"], 120),
        ("performance", [PY, "scripts/lab.py", "--quick"] + (["--verbose"] if args.verbose else []), 240),
    ]
    if not args.no_prep_trace:
        plan.append(("prep trace", [PY, "scripts/prep_trace.py"], 180))
    if args.bench:
        plan.append(("bench", [PY, "scripts/bench.py", "--label", args.label,
                               "--games", args.games, "--seeds", str(args.seeds),
                               "--best-of", str(args.best_of)], 1800))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full local testing pipeline.")
    parser.add_argument("--json", default="", help="write the stage results here")
    parser.add_argument("--fast", action="store_true", help="skip the slow probes")
    parser.add_argument("--bench", action="store_true", help="also run real arena matches (uses the gateway)")
    parser.add_argument("--games", default="tic_tac_toe,tg_connect4,tg_blotto_probe", help="bench games")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--best-of", type=int, default=1)
    parser.add_argument("--label", default="pipeline")
    parser.add_argument("--no-prep-trace", action="store_true")
    parser.add_argument("--no-solvers", action="store_true", help="skip the solver smoke stage")
    parser.add_argument("--verbose", action="store_true", help="print each stage's full output")
    args = parser.parse_args()

    results = []
    print("=" * 78)
    print(" TESTING PIPELINE")
    print("=" * 78)
    for name, cmd, timeout in stages(args):
        print(f"  {name:<12} ... ", end="", flush=True)
        result = run(name, cmd, timeout)
        results.append(result)
        print(("PASS" if result["ok"] else "FAIL") + f"  {result['seconds']}s  {result['note']}")
        if args.verbose and result.get("tail"):
            for line in result["tail"].strip().splitlines():
                print(f"      {line}")

    failed = [result for result in results if not result["ok"]]
    total = round(sum(result["seconds"] for result in results), 1)
    slowest = max(results, key=lambda result: result["seconds"], default=None)
    print("-" * 78)
    print(f"  {len(results) - len(failed)}/{len(results)} stages passed in {total}s")
    if slowest and slowest["seconds"] > 0:
        print(f"  slowest stage: {slowest['name']} ({slowest['seconds']}s)")
    if failed:
        print("  FAILED:")
        for result in failed:
            print(f"    - {result['name']}: {result['note']}")
    else:
        print("  ALL STAGES PASSED")
    print("=" * 78)

    if args.json:
        payload = {
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stages": results,
            "total_seconds": total,
        }
        if slowest:
            payload["slowest"] = {"name": slowest["name"], "seconds": slowest["seconds"]}
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
