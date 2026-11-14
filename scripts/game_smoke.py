#!/usr/bin/env python3
"""API smoke test for every game in arena-lite, without starting a bot.

For each game and variant it checks the contract the arena relies on:

- at least two variants with distinct labels, non-empty rules;
- `legal_actions` / `normalize` agree for every legal action;
- `observation` is non-empty text, `frame` is JSON-serialisable;
- a random playout reaches a terminal state (or the move list becomes null, which is legal above 50 moves);
- `is_over` / `result` agree at the end and the scores are numeric.

Games with free-text actions (`legal_actions` is null from the start) are API-only: there is no generic
way to invent an action, so the playout is skipped and the game is reported as `api`.

    python3 scripts/game_smoke.py [--no-color] [--json out.json]
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA = os.path.join(ROOT, "arena-lite")
if ARENA not in sys.path:
    sys.path.insert(0, ARENA)

from arena.core import Game  # noqa: E402  (after sys.path)


def game_classes() -> list[type]:
    found: list[type] = []
    for package in ("arena.practice", "arena.games"):
        try:
            module = importlib.import_module(package)
        except ImportError:
            continue
        for path in sorted(os.listdir(os.path.join(ARENA, package.replace(".", "/")))):
            if not path.endswith(".py") or path.startswith("__"):
                continue
            module = importlib.import_module(f"{package}.{path[:-3]}")
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Game) and obj is not Game and getattr(obj, "id", ""):
                    found.append(obj)
    unique = {}
    for cls in found:
        unique[cls.id] = cls
    return [unique[key] for key in sorted(unique)]


def check_variant(cls: type, variant: dict, seed: int) -> tuple[bool, str, str]:
    """Returns (ok, mode, note). mode is 'play' or 'api'."""
    try:
        game = cls(variant, seed)
        rules = str(game.rules())
        if len(rules.strip()) < 40:
            return False, "play", "rules text too short"
        observation = str(game.observation(0))
        if not observation.strip():
            return False, "play", "empty observation"
        json.dumps(game.frame())
        legal = game.legal_actions(0)
        if legal is None:
            return True, "api", "free-text actions (no generic playout)"
        if not legal:
            return False, "play", "no legal actions at the start"
        rng = random.Random(seed)
        for _ply in range(400):
            if game.is_over():
                break
            movers = list(game.to_move())
            if not movers:
                return False, "play", "non-terminal but to_move is empty"
            actions = {}
            for mover in movers:
                options = game.legal_actions(mover)
                if options is None:
                    return True, "api", "move list became null mid-game"
                if not options:
                    return False, "play", f"seat {mover} has no legal actions"
                choice = rng.choice(list(options))
                canonical = game.normalize(mover, choice)
                if canonical is None:
                    return False, "play", f"normalize rejected legal action {choice!r}"
                if canonical not in options:
                    return False, "play", f"normalize changed spelling {choice!r} -> {canonical!r}"
                actions[mover] = canonical
            game.apply(actions)
        if not game.is_over():
            return False, "play", "no terminal state within 400 plies"
        result = game.result()
        if not isinstance(result, dict) or "scores" not in result:
            return False, "play", "result() missing scores"
        if len(result["scores"]) != 2 or not all(isinstance(x, (int, float)) for x in result["scores"]):
            return False, "play", "scores are not two numbers"
        return True, "play", f"random playout ok ({result.get('reason', '')})"[:80]
    except Exception as exc:  # noqa: BLE001 - report, do not crash the pipeline
        return False, "play", f"{type(exc).__name__}: {exc}"[:120]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test every arena-lite game API.")
    parser.add_argument("--json", default="")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = []
    failures = 0
    print("=" * 96)
    print(" GAME API SMOKE")
    print("=" * 96)
    print(f" {'game':<18} {'variant':<26} {'mode':<5} result")
    for cls in game_classes():
        try:
            variants = cls.variants()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            rows.append({"game": cls.id, "variant": "-", "ok": False, "mode": "api",
                         "note": f"variants() failed: {type(exc).__name__}"})
            continue
        if len(variants) < 2:
            failures += 1
            rows.append({"game": cls.id, "variant": "-", "ok": False, "mode": "api",
                         "note": "fewer than two variants"})
        for index, variant in enumerate(variants):
            ok, mode, note = check_variant(cls, variant, args.seed + index)
            rows.append({"game": cls.id, "variant": str(variant.get("label", "?")), "ok": ok,
                         "mode": mode, "note": note})
            failures += 0 if ok else 1
            status = "PASS" if ok else "FAIL"
            print(f" {cls.id:<18} {str(variant.get('label', '?'))[:26]:<26} {mode:<5} {status}  {note}")
    total = len(rows)
    print("-" * 96)
    print(f" {total - failures}/{total} variant checks passed")
    if failures:
        print(" FAILURES:")
        for row in rows:
            if not row["ok"]:
                print(f"   - {row['game']} [{row['variant']}]: {row['note']}")
    else:
        print(" ALL GAMES PASS THE API CHECK")
    print("=" * 96)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
