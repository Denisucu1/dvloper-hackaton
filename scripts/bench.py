#!/usr/bin/env python3
"""Benchmark the bot on the real arena-lite engine and show whether it improved.

It runs the bot through complete matches (prep included, real clocks, real chat), parses the official
replays, and prints one report card per run. Each run is stored as JSON, so the next run can be compared
against it automatically:

    python3 scripts/bench.py --label baseline
    ... change the harness ...
    python3 scripts/bench.py --label search-v1        # prints the delta vs baseline

What it measures (all from the replay, nothing self-reported):
- match/game score, wins, draws, losses;
- warnings, timeouts, crashes, illegal answers, forfeits  (robustness, rule 8.1/8.2);
- move time mean/p95/max and study time            (tie-break 10.4 + clock safety);
- chat reply rate, lines per move, length           (rule 6.2 + the automated chat check);
- gateway spend on the dev key                     (cost points, rule 9.3).

Exit code is 1 when a gate fails (any timeout/crash/forfeit, any warning, chat reply rate < 98%,
or a >10 point drop in score vs the baseline).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA = os.path.join(ROOT, "arena-lite")
RESULTS = os.path.join(ROOT, "bench", "results")
DEFAULT_BOT = "cmd:python3 ../main.py"
DEFAULT_OPPONENT = "cmd:python3 examples/random_bot.py"

# A move clock of 20 s: beyond this we are one slow answer away from losing a game to a timeout.
MOVE_WARN_MS = 12_000


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "run"


def sh(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def practice_games() -> list[str]:
    out = sh([sys.executable, "-m", "arena", "practice", "--list"], ARENA, 60)
    games = []
    for line in out.stdout.splitlines():
        name = line.split()[0] if line.split() else ""
        if name and not name.startswith("-") and " " not in name:
            games.append(name)
    return games


def key_spend(env_file: str) -> float | None:
    """Dev-key spend straight from the gateway (`/key/info`); prep and matches included."""
    values = read_env(env_file)
    base, key = values.get("LLM_BASE_URL", ""), values.get("LLM_API_KEY", "")
    if not base or not key:
        return None
    root = re.sub(r"/v1/?$", "", base.rstrip("/"))
    try:
        request = urllib.request.Request(f"{root}/key/info", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return float((json.load(response).get("info") or {}).get("spend") or 0.0)
    except Exception:  # noqa: BLE001 - offline benchmarking must still work
        return None


def read_env(path: str) -> dict:
    values = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip()
    except OSError:
        pass
    return values


# --------------------------------------------------------------------------- one match
def run_match(args, game: str, seed: int, opponent: str, name_b: str) -> dict:
    match_id = f"bench-{slug(game)}-{seed}"
    replay = os.path.join(ARENA, "results", "practice", "matches", f"{match_id}.json")
    if os.path.exists(replay):
        os.unlink(replay)  # never parse a stale replay from an earlier run

    cmd = [
        sys.executable, "-m", "arena", "practice",
        "--bot-a", args.bot, "--bot-b", opponent,
        "--game", game, "--seed", str(seed), "--best-of", str(args.best_of),
        "--id", match_id, "--name-a", "adn", "--name-b", name_b,
        "--quiet",
    ]
    if args.prep:
        cmd += ["--prep", "--prep-seconds", str(args.prep_seconds)]
    if os.path.exists(args.env_file):
        cmd += ["--env-file", os.path.abspath(args.env_file)]

    started = time.time()
    try:
        completed = sh(cmd, ARENA, args.match_timeout)
    except subprocess.TimeoutExpired:
        return {"game": game, "seed": seed, "match_id": match_id, "setup_error": "arena timeout"}
    duration = time.time() - started

    if not os.path.exists(replay):
        tail = (completed.stderr or completed.stdout or "")[-600:]
        return {"game": game, "seed": seed, "match_id": match_id, "setup_error": tail.strip() or "no replay"}
    with open(replay, encoding="utf-8") as handle:
        doc = json.load(handle)
    return {"game": game, "seed": seed, "match_id": match_id, "duration_s": duration, "doc": doc}


def analyse(run: dict) -> dict:
    """Turn one replay into numbers. Everything here is measured, not reported by the bot."""
    doc = run["doc"]
    name_a = "adn"
    games = doc.get("games") or []
    row = {
        "game": run["game"],
        "seed": run["seed"],
        "match_id": run["match_id"],
        "duration_s": round(run.get("duration_s") or 0, 1),
        "games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0.0,
        "warnings": 0,
        "timeouts": 0,
        "crashes": 0,
        "illegal": 0,
        "study_failures": 0,
        "forfeits": 0,
        "engine_errors": 0,
        "moves": 0,
        "move_ms": [],
        "study_ms": [],
        "chat_lines": 0,
        "chat_play_lines": 0,
        "chat_chars": [],
    }

    for game in games:
        seats = game.get("seats") or []
        if name_a not in seats:
            continue
        seat = seats.index(name_a)
        row["games"] += 1
        result = game.get("result") or {}
        scores = result.get("scores") or [0, 0]
        ours = float(scores[seat]) if len(scores) == 2 else 0.0
        theirs = float(scores[1 - seat]) if len(scores) == 2 else 0.0
        game_points = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
        row["points"] += game_points
        if game_points == 1.0:
            row["wins"] += 1
        elif game_points == 0.5:
            row["draws"] += 1
        else:
            row["losses"] += 1

        row["warnings"] += int((game.get("warnings") or {}).get(str(seat)) or 0)
        study = (game.get("study") or {}).get(str(seat)) or {}
        if isinstance(study.get("ms"), (int, float)):
            row["study_ms"].append(float(study["ms"]))

        # A failure during `prep`/`start` (or a crash before the first move) never reaches the timeline,
        # so the only trace is the reason text. Attribute it to us only when the reason names our team:
        # the opponent can forfeit too, and that must never be counted as our failure.
        reason = str(result.get("reason") or "")
        low = reason.lower()
        mine = re.search(rf"\b{re.escape(name_a.lower())}\b", low) is not None
        seen_kinds = set()
        for step in game.get("timeline") or []:
            entry = (step.get("actions") or {}).get(str(seat))
            if isinstance(entry, dict) and entry.get("error"):
                seen_kinds.add(str(entry["error"]).lower())
        if mine and game.get("forfeit"):
            row["forfeits"] += 1
        if mine and "during study" in low and not seen_kinds:
            row["study_failures"] += 1
        if mine and ("crashed" in low or "exited" in low) and not any("crash" in k or "exit" in k for k in seen_kinds):
            row["crashes"] += 1
        if mine and ("timeout" in low or "no answer" in low) and not any("timeout" in k for k in seen_kinds):
            row["timeouts"] += 1
        if mine and "illegal" in low and not any("illegal" in k for k in seen_kinds):
            row["illegal"] += 1
        if game.get("error"):
            row["engine_errors"] += 1

        for step in game.get("timeline") or []:
            entry = (step.get("actions") or {}).get(str(seat))
            if not isinstance(entry, dict):
                continue
            error = str(entry.get("error") or "")
            if entry.get("timeout") or error.startswith("timeout"):
                row["timeouts"] += 1
            elif "crashed" in error or "exited" in error:
                row["crashes"] += 1
            elif error:
                row["illegal"] += 1
            if isinstance(entry.get("ms"), (int, float)) and entry.get("action") and not entry.get("error"):
                row["moves"] += 1
                row["move_ms"].append(float(entry["ms"]))

    for line in doc.get("chat") or []:
        if line.get("team") != name_a:
            continue
        text = str(line.get("text") or "")
        row["chat_lines"] += 1
        row["chat_chars"].append(len(text))
        if (line.get("phase") or "play") == "play":
            row["chat_play_lines"] += 1

    moves = max(1, row["moves"])
    expected_lines = row["moves"] + row["games"]  # one chat with ready, one with every accepted move
    row["chat_reply_rate"] = round(min(1.0, row["chat_lines"] / max(1, expected_lines)), 3)
    row["chat_per_move"] = round(min(1.0, row["chat_play_lines"] / moves), 3)
    row["move_ms_mean"] = round(statistics.fmean(row["move_ms"]), 0) if row["move_ms"] else 0
    row["move_ms_p95"] = round(sorted(row["move_ms"])[max(0, int(0.95 * len(row["move_ms"])) - 1)], 0) if row["move_ms"] else 0
    row["move_ms_max"] = round(max(row["move_ms"]), 0) if row["move_ms"] else 0
    row["study_ms_mean"] = round(statistics.fmean(row["study_ms"]), 0) if row["study_ms"] else 0
    row["chat_chars_mean"] = round(statistics.fmean(row["chat_chars"]), 1) if row["chat_chars"] else 0
    row["chat_chars_max"] = max(row["chat_chars"]) if row["chat_chars"] else 0
    del row["move_ms"], row["study_ms"], row["chat_chars"]
    return row


# --------------------------------------------------------------------------- aggregate
def aggregate(rows: list[dict], spend: float | None, spent_before: float | None, args) -> dict:
    ok = [row for row in rows if not row.get("setup_error")]
    totals = {
        "matches": len(ok),
        "games": sum(r["games"] for r in ok),
        "wins": sum(r["wins"] for r in ok),
        "draws": sum(r["draws"] for r in ok),
        "losses": sum(r["losses"] for r in ok),
        "points": round(sum(r["points"] for r in ok), 1),
        "warnings": sum(r["warnings"] for r in ok),
        "timeouts": sum(r["timeouts"] for r in ok),
        "crashes": sum(r["crashes"] for r in ok),
        "illegal": sum(r["illegal"] for r in ok),
        "study_failures": sum(r["study_failures"] for r in ok),
        "forfeits": sum(r["forfeits"] for r in ok),
        "engine_errors": sum(r["engine_errors"] for r in ok),
        "chat_reply_rate": round(statistics.fmean([r["chat_reply_rate"] for r in ok]), 3) if ok else 0,
        "chat_chars_mean": round(statistics.fmean([r["chat_chars_mean"] for r in ok]), 1) if ok else 0,
        "move_ms_mean": round(statistics.fmean([r["move_ms_mean"] for r in ok]), 0) if ok else 0,
        "move_ms_p95": round(max((r["move_ms_p95"] for r in ok), default=0), 0),
        "move_ms_max": round(max((r["move_ms_max"] for r in ok), default=0), 0),
        "study_ms_mean": round(statistics.fmean([r["study_ms_mean"] for r in ok]), 0) if ok else 0,
        "duration_s": round(sum(r["duration_s"] for r in ok), 1),
        "setup_errors": len(rows) - len(ok),
    }
    games = totals["games"] or 1
    totals["score_pct"] = round(100.0 * totals["points"] / games, 1)
    totals["win_pct"] = round(100.0 * totals["wins"] / games, 1)
    totals["usd"] = round((spend - spent_before), 5) if (spend is not None and spent_before is not None) else None
    totals["usd_per_match"] = round(totals["usd"] / max(1, totals["matches"]), 5) if totals["usd"] is not None else None
    return {
        "label": args.label,
        "when": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "bot": args.bot,
            "opponents": args.opponent,
            "games": args.games,
            "seeds": args.seeds,
            "best_of": args.best_of,
            "prep": args.prep,
            "prep_seconds": args.prep_seconds,
        },
        "totals": totals,
        "matches": rows,
    }


# --------------------------------------------------------------------------- report
PRIMARY = [
    ("matches", "matches", "{:.0f}"),
    ("games", "games", "{:.0f}"),
    ("win_pct", "win %", "{:.1f}"),
    ("score_pct", "score % of max", "{:.1f}"),
    ("warnings", "warnings", "{:.0f}"),
    ("timeouts", "timeouts", "{:.0f}"),
    ("crashes", "crashes", "{:.0f}"),
    ("illegal", "illegal answers", "{:.0f}"),
    ("study_failures", "study failures", "{:.0f}"),
    ("forfeits", "forfeits", "{:.0f}"),
    ("chat_reply_rate", "chat reply rate", "{:.3f}"),
    ("chat_chars_mean", "chat chars (avg)", "{:.1f}"),
    ("move_ms_mean", "move ms (mean)", "{:.0f}"),
    ("move_ms_p95", "move ms (p95)", "{:.0f}"),
    ("move_ms_max", "move ms (max)", "{:.0f}"),
    ("usd_per_match", "USD / match", "{:.5f}"),
    ("duration_s", "wall seconds", "{:.0f}"),
]

LOWER_IS_BETTER = {"warnings", "timeouts", "crashes", "illegal", "study_failures", "forfeits",
                   "move_ms_mean", "move_ms_p95", "move_ms_max", "duration_s", "usd_per_match"}
# More chat characters is not "better"; it is just information (and a 200-char cap is the only rule).
NEUTRAL = {"chat_chars_mean"}


def fmt(value, pattern: str) -> str:
    if value is None:
        return "-"
    try:
        return pattern.format(value)
    except (ValueError, TypeError):
        return str(value)


def delta_text(key: str, now, before) -> str:
    if now is None or before is None:
        return ""
    diff = float(now) - float(before)
    if abs(diff) < 1e-9:
        return "="
    sign = "+" if diff > 0 else ""
    size = abs(diff)
    if size >= 10:
        shown = f"{sign}{diff:.0f}"
    elif size >= 1:
        shown = f"{sign}{diff:.1f}"
    else:
        shown = f"{sign}{diff:.3f}"
    if key in NEUTRAL:
        return f"changed {shown}"
    better = diff < 0 if key in LOWER_IS_BETTER else diff > 0
    return f"{'better' if better else 'WORSE'} {shown}"


def print_report(run: dict, previous: dict | None, history: list[dict]) -> None:
    totals = run["totals"]
    print()
    print(f"=== bench '{run['label']}' · {run['when']}")
    cfg = run["config"]
    print(f"    bot={cfg['bot']}  opponents={', '.join(cfg['opponents'])}  "
          f"games={','.join(cfg['games'])}  seeds={cfg['seeds']}  prep={cfg['prep']}")
    print()
    print(f"    {'metric':<20} {'now':>10} {'baseline':>10}   change")
    before = (previous or {}).get("totals") or {}
    for key, label, pattern in PRIMARY:
        now = totals.get(key)
        old = before.get(key)
        print(f"    {label:<20} {fmt(now, pattern):>10} {fmt(old, pattern):>10}   {delta_text(key, now, old)}")

    if run["matches"]:
        print()
        print(f"    {'game/seed':<28} {'pts':>4} {'W/D/L':>7} {'warn':>5} {'timeout':>8} {'p95ms':>7} {'chat':>6}")
        for row in run["matches"]:
            if row.get("setup_error"):
                print(f"    {row['game'] + '/' + str(row['seed']):<28} SETUP ERROR: {row['setup_error'][:60]}")
                continue
            wdl = f"{row['wins']}/{row['draws']}/{row['losses']}"
            print(f"    {row['game'] + '/' + str(row['seed']):<28} {row['points']:>4.1f} {wdl:>7} "
                  f"{row['warnings']:>5} {row['timeouts']:>8} {row['move_ms_p95']:>7.0f} "
                  f"{row['chat_reply_rate']:>6.2f}")

    if len(history) >= 2:
        print()
        print("    history (newest first):")
        for item in history[:8]:
            t = item["totals"]
            print(f"      {item['when'][5:16]}  {item['label'][:18]:<18} "
                  f"win {t.get('win_pct', 0):>5.1f}%  score {t.get('score_pct', 0):>5.1f}%  "
                  f"warn {t.get('warnings', 0):>2}  "
                  f"bad {t.get('timeouts', 0) + t.get('crashes', 0) + t.get('illegal', 0) + t.get('study_failures', 0):>2}  "
                  f"p95 {t.get('move_ms_p95', 0):>5.0f}ms  chat {t.get('chat_reply_rate', 0):.2f}")


def gate(run: dict, previous: dict | None) -> list[str]:
    """Failures that are never acceptable, plus a real regression versus the baseline."""
    problems = []
    t = run["totals"]
    if t["setup_errors"]:
        problems.append(f"{t['setup_errors']} match(es) could not run")
    for key, label in (("timeouts", "timeout"), ("crashes", "crash"),
                       ("illegal", "illegal answer"), ("study_failures", "study-phase failure"),
                       ("forfeits", "forfeit"), ("engine_errors", "engine error")):
        if t.get(key):
            problems.append(f"{t[key]} {label}(s): a lost game is -1.25 robustness and a lost match")
    if t.get("warnings"):
        problems.append(f"{t['warnings']} warning(s): first tie-break in a level knockout (10.4)")
    if t.get("chat_reply_rate", 1) < 0.98:
        problems.append(f"chat reply rate {t['chat_reply_rate']:.2f} < 0.98: chat rule 6.2 / chat check")
    if t.get("move_ms_max", 0) > MOVE_WARN_MS:
        problems.append(f"slowest answer {t['move_ms_max']:.0f} ms > {MOVE_WARN_MS} ms: timeout risk on a 20 s clock")
    if previous:
        before = previous.get("totals") or {}
        drop = (before.get("score_pct") or 0) - (t.get("score_pct") or 0)
        if drop > 10:
            problems.append(f"score dropped {drop:.1f} points vs baseline '{previous['label']}'")
    return problems


# --------------------------------------------------------------------------- history
def same_config(run: dict, args) -> bool:
    """Only compare runs of the same bot against the same opponents on the same games."""
    config = run.get("config") or {}
    return (
        config.get("bot") == args.bot
        and sorted(config.get("games") or []) == sorted(args.games)
        and sorted(config.get("opponents") or []) == sorted(args.opponent)
    )


def pick_baseline(history: list[dict], args) -> dict | None:
    for run in history:
        if same_config(run, args):
            return run
    return None


def load_history() -> list[dict]:
    if not os.path.isdir(RESULTS):
        return []
    runs = []
    for dirpath, _dirnames, filenames in os.walk(RESULTS):
        if "report.json" not in filenames:
            continue
        path = os.path.join(dirpath, "report.json")
        try:
            with open(path, encoding="utf-8") as handle:
                runs.append(json.load(handle))
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda item: item.get("when", ""), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the bot on arena-lite and track improvements.")
    parser.add_argument("--label", default="run", help="name of this run (shown in the history)")
    parser.add_argument("--bot", default=DEFAULT_BOT, help="arena-lite bot spec for our bot")
    parser.add_argument("--opponent", action="append", default=None,
                        help="opponent spec (repeatable); default: the random bot")
    parser.add_argument("--games", default="all", help="comma list or 'all' (practice games)")
    parser.add_argument("--seeds", type=int, default=2, help="matches per game (different deals)")
    parser.add_argument("--seed-start", type=int, default=1, help="first seed number")
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--prep", dest="prep", action="store_true", default=True,
                        help="prep every game first (the tournament flow); default on")
    parser.add_argument("--no-prep", dest="prep", action="store_false")
    parser.add_argument("--prep-seconds", type=float, default=600)
    parser.add_argument("--env-file", default=os.path.join(ROOT, "dev.env"))
    parser.add_argument("--match-timeout", type=float, default=600, help="seconds per match")
    parser.add_argument("--baseline", default="auto", help="'auto', 'none' or a JSON report path")
    parser.add_argument("--no-gate", action="store_true", help="do not fail the exit code")
    args = parser.parse_args()

    if not os.path.isdir(ARENA):
        print(f"arena-lite not found at {ARENA}", file=sys.stderr)
        return 2
    if args.games.strip() == "all":
        args.games = practice_games()
    else:
        args.games = [item.strip() for item in args.games.split(",") if item.strip()]
    if not args.games:
        print("no games to run", file=sys.stderr)
        return 2
    args.opponent = args.opponent or [DEFAULT_OPPONENT]

    history = load_history()
    previous = None
    if args.baseline == "auto" and history:
        previous = pick_baseline(history, args)
    elif args.baseline not in ("auto", "none"):
        try:
            with open(args.baseline, encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"cannot read baseline: {exc}", file=sys.stderr)
            return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS, f"{stamp}-{slug(args.label)}")
    os.makedirs(run_dir, exist_ok=True)

    spent_before = key_spend(args.env_file)
    rows = []
    for opponent in args.opponent:
        name_b = slug(os.path.basename(opponent.replace("cmd:", "").replace("image:", ""))) or "rival"
        for game in args.games:
            for offset in range(args.seeds):
                seed = args.seed_start + offset
                print(f"[bench] {game} seed={seed} vs {name_b} ...", end=" ", flush=True)
                run = run_match(args, game, seed, opponent, name_b)
                if run.get("setup_error"):
                    print(f"SETUP ERROR: {run['setup_error'][:120]}")
                    rows.append({"game": game, "seed": seed, "match_id": run["match_id"],
                                 "setup_error": run["setup_error"]})
                    continue
                row = analyse(run)
                rows.append(row)
                print(f"{row['points']:.1f} pts  W/D/L {row['wins']}/{row['draws']}/{row['losses']}  "
                      f"warn {row['warnings']}  timeout {row['timeouts']}  "
                      f"p95 {row['move_ms_p95']:.0f}ms  chat {row['chat_reply_rate']:.2f}")
                target = os.path.join(run_dir, "replays", os.path.basename(run["match_id"]) + ".json")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                try:
                    shutil.copy(replay_path(run["match_id"]), target)
                except OSError:
                    pass

    spend_after = key_spend(args.env_file)
    report = aggregate(rows, spend_after, spent_before, args)
    report["spend"] = {"before": spent_before, "after": spend_after}

    json_path = os.path.join(run_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown(report, previous))

    # Keep the "auto" baseline meaningful: newest first, and the current run must not compare to itself.
    history = [report] + [item for item in history if item.get("when") != report["when"]]
    print_report(report, previous, history)
    print()
    print(f"    report: {json_path}")

    problems = [] if args.no_gate else gate(report, previous)
    if problems:
        print()
        print("    GATE FAILED:")
        for problem in problems:
            print(f"      - {problem}")
        return 1
    print()
    print("    GATE PASSED")
    return 0


def replay_path(match_id: str) -> str:
    return os.path.join(ARENA, "results", "practice", "matches", f"{match_id}.json")


def markdown(report: dict, previous: dict | None) -> str:
    totals = report["totals"]
    lines = [
        f"# Bench `{report['label']}` — {report['when']}",
        "",
        f"bot `{report['config']['bot']}`, opponents `{', '.join(report['config']['opponents'])}`, "
        f"games `{', '.join(report['config']['games'])}`, seeds {report['config']['seeds']}, "
        f"prep {report['config']['prep']}",
        "",
        "| metric | now | baseline | change |",
        "|---|---|---|---|",
    ]
    before = (previous or {}).get("totals") or {}
    for key, label, pattern in PRIMARY:
        lines.append(f"| {label} | {fmt(totals.get(key), pattern)} | {fmt(before.get(key), pattern)} | "
                     f"{delta_text(key, totals.get(key), before.get(key))} |")
    lines += ["", "## Matches", "", "| game/seed | points | W/D/L | warnings | timeouts | p95 ms | chat |",
              "|---|---|---|---|---|---|---|"]
    for row in report["matches"]:
        if row.get("setup_error"):
            lines.append(f"| {row['game']}/{row['seed']} | - | - | - | - | - | setup error |")
            continue
        lines.append(f"| {row['game']}/{row['seed']} | {row['points']:.1f} | "
                     f"{row['wins']}/{row['draws']}/{row['losses']} | {row['warnings']} | "
                     f"{row['timeouts']} | {row['move_ms_p95']:.0f} | {row['chat_reply_rate']:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
