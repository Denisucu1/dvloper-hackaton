#!/usr/bin/env python3
"""Prep lab: measure the decision strategies and show the battle-royale improvement per iteration.

This is the performance half of the pipeline. It uses three deterministic fixture simulators (wide
perfect-information, simultaneous allocation, chance/dice) so it runs offline and fast, and it repeats the
prep tournament as a progressive battle royale:

    iteration 1: the incumbent (greedy)
    iteration 2: + alpha-beta
    iteration 3: + the family specialist (matrix for simultaneous, expectimax for chance)

Each iteration reports the winner, its score against a random baseline, and the delta versus the previous
iteration. A performance index (mean best score across models) is stored in a history file, so successive
pipeline runs show whether the program improved.

    python3 scripts/lab.py                # normal, ~10 s
    python3 scripts/lab.py --quick        # one game per pair, ~4 s
    python3 scripts/lab.py --verbose      # per-strategy scores for every iteration
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness import selfplay, tournament  # noqa: E402
from harness.candidates import Candidate  # noqa: E402

HISTORY = os.path.join(ROOT, "scripts", "lab", ".history.jsonl")

MODELS = [
    {
        "name": "connect5",
        "path": os.path.join(ROOT, "scripts", "lab", "models", "connect5.py"),
        "rules": "PLACEMENT - 9x7 board, five in a row wins.",
        "iterations": [["greedy"], ["greedy", "search"]],
    },
    {
        "name": "blotto",
        "path": os.path.join(ROOT, "scripts", "lab", "models", "blotto.py"),
        "rules": "COLONEL BLOTTO - 3 fields, 6 troops, 3 rounds, simultaneous allocations.",
        "iterations": [["greedy"], ["greedy", "search"], ["greedy", "search", "matrix"]],
    },
    {
        "name": "chance_pig",
        "path": os.path.join(ROOT, "scripts", "lab", "models", "chance_pig.py"),
        "rules": "PUSH YOUR LUCK - one die, target 30, roll or hold.",
        "iterations": [["greedy"], ["greedy", "search"], ["greedy", "search", "chance"]],
    },
]


def candidate(name: str) -> Candidate:
    return Candidate(kind="strategy", name=name, meta={"strategy": name})


def play_for(model_path: str, rules: str, seconds: float, max_plies: int):
    def play(a, b, seed: int) -> float:
        result = selfplay.play_game(
            model_path, rules,
            (a.meta["strategy"], b.meta["strategy"]),
            seed=seed, max_plies=max_plies, seconds_per_move=seconds,
        )
        if result.error or result.winner is None:
            return 0.5
        return 1.0 if result.winner == 0 else 0.0

    return play


def run_iterations(model: dict, *, games_per_pair: int, seconds: float, max_plies: int,
                   verbose: bool) -> dict:
    """One fixed gauntlet, then read the best candidate of each progressive pool from the same table.

    Scoring every strategy in the same run with the same seeds keeps iterations comparable: adding a
    candidate can only raise the best score, which is exactly the battle-royale improvement we want to show.
    """
    path = model["path"]
    rules = model["rules"]
    iterations = model["iterations"]
    pool: list[str] = []
    for strategies in iterations:
        for name in strategies:
            if name not in pool:
                pool.append(name)
    started = time.time()
    best, stats = tournament.run(
        candidates=[candidate(name) for name in pool],
        baselines=[candidate("random")],
        play=play_for(path, rules, seconds, max_plies),
        budget_s=120.0,
        seed=1000,
        games_per_pair=games_per_pair,
    )
    table = stats.get("scores") or {}
    scores = {str(name): float(row.get("score") or 0.0) for name, row in table.items()}
    rows = []
    previous = None
    for index, strategies in enumerate(iterations, start=1):
        eligible = [name for name in strategies if name in scores]
        winner = max(eligible, key=lambda name: scores.get(name, 0.0)) if eligible else "none"
        score = round(scores.get(winner, 0.0), 3)
        delta = None if previous is None else round(score - previous, 3)
        rows.append({
            "iteration": index,
            "candidates": list(strategies),
            "winner": winner,
            "score": score,
            "delta": delta,
            "games": int((table.get(winner) or {}).get("games") or 0),
            "timed_out": bool(stats.get("timed_out")),
            "elapsed": round(time.time() - started, 2),
            "scores": {name: round(scores.get(name, 0.0), 3) for name in strategies},
        })
        if verbose:
            shown = "  ".join(f"{name}={scores.get(name, 0):.2f}" for name in strategies)
            print(f"    iter {index}: winner={winner:<7} score={score:.3f} "
                  f"delta={'—' if delta is None else f'{delta:+.3f}'} "
                  f"games={(table.get(winner) or {}).get('games')} ({shown})")
        previous = score
    return {"model": model["name"], "iterations": rows}


def load_history() -> list[dict]:
    runs = []
    try:
        with open(HISTORY, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure decision strategies and battle-royale progress.")
    parser.add_argument("--quick", action="store_true", help="one game per pair, fewer seconds")
    parser.add_argument("--verbose", action="store_true", help="per-strategy scores per iteration")
    parser.add_argument("--json", default="", help="write this run to a JSON file")
    parser.add_argument("--no-history", action="store_true", help="do not append to the history file")
    args = parser.parse_args()

    mode = "quick" if args.quick else "normal"
    games_per_pair = 1 if args.quick else 2
    seconds = 0.04 if args.quick else 0.06
    max_plies = 40 if args.quick else 60

    previous = [run for run in load_history() if run.get("mode", "normal") == mode]
    baseline_index = float(previous[-1]["index"]) if previous else None

    print("=" * 92)
    print(" PREP LAB — battle royale iterations and current performance")
    print("=" * 92)
    results = []
    started = time.time()
    for model in MODELS:
        print(f"  {model['name']:<12} {model['rules'][:60]}")
        results.append(run_iterations(model, games_per_pair=games_per_pair, seconds=seconds,
                                      max_plies=max_plies, verbose=args.verbose))

    best_scores = [row["score"] for result in results for row in result["iterations"][-1:]]
    index = round(sum(best_scores) / max(1, len(best_scores)), 3)
    delta_index = None if baseline_index is None else round(index - baseline_index, 3)
    elapsed = round(time.time() - started, 1)

    print("-" * 92)
    print(f"  {'model':<12} {'winner':<10} {'score':>6} {'delta':>7} {'games':>6}")
    for result in results:
        last = result["iterations"][-1]
        delta = "—" if last["delta"] is None else f"{last['delta']:+.3f}"
        print(f"  {result['model']:<12} {last['winner']:<10} {last['score']:>6.3f} "
              f"{delta:>7} {last['games']:>6}")
    print("-" * 92)
    print(f"  PERFORMANCE INDEX {index:.3f}"
          + ("" if delta_index is None else f"  ({delta_index:+.3f} vs last run)")
          + f"   [{elapsed}s]")
    print("=" * 92)

    payload = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": mode, "index": index,
               "delta_index": delta_index, "elapsed_s": elapsed, "models": results}
    if not args.no_history:
        try:
            os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
            with open(HISTORY, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError:
            pass
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
