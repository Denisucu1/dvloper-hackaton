#!/usr/bin/env python3
"""Offline smoke test for the family solver layer, driven by real arena observations.

For every game and variant discovered in arena-lite this script plays one seeded random playout
(``random_action`` for each seat that has to move) and records every observation the arena would
send. Each observation is then handed directly to the solvers returned by
``harness.engine._solvers()``:

    solver.matches(game_id, spec)  ->  solver.propose(observation=..., legal=..., seat=..., turn=...)

The contract checked here is the one the engine relies on: a solver may decline (``None``) but it
must never raise, and any action it returns must normalize against the arena's legal list (or be a
non-empty string when the list is null, e.g. Wordle free text).

Solvers with modules that cannot be imported are skipped with ``skip <module>: <reason>`` so the
stage is green before a port lands. Solvers without a ``matches()`` hook are skipped the same way.

    python3 scripts/solver_smoke.py [--no-color] [--json out.json] [--seed 7]

Exit code: 0 only when there were no crashes, no illegal/empty proposals and at least one
observation was checked. No network is used.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA = os.path.join(ROOT, "arena-lite")
for path in (ROOT, ARENA):
    if path not in sys.path:
        sys.path.insert(0, path)

from harness.actions import normalize  # noqa: E402  (after sys.path)
from harness.spec import Spec  # noqa: E402

# Arena category -> harness spec family (the solver matchers understand the harness names).
CATEGORY_FAMILY = {
    "read": "read_opponent",
    "think": "think_ahead",
    "guess": "guessing",
    "guard": "guardrails",
    "practice": "unknown",
}

# Modules the engine knows about; used only to print a clear `skip` line when one is absent.
SOLVER_MODULES = (
    "solver_guess",
    "solver_pig",
    "solver_simul",
    "solver_leduc",
    "solver_luck",
    "solver_bluff",
    "solver_deals",
    "solver_grid",
)

MAX_PLIES = 120
TIMEOUT = 0.5


def game_classes() -> list[type]:
    """Every Game subclass with an id, via the arena registry (fallback: the game_smoke scan)."""
    try:
        from arena.registry import all_games

        games = all_games()
        return [games[key] for key in sorted(games)]
    except Exception:  # noqa: BLE001 - fall back to the scanner game_smoke.py uses
        return _scan_game_classes()


def _scan_game_classes() -> list[type]:
    import inspect

    from arena.core import Game

    found: list[type] = []
    for package in ("arena.practice", "arena.games"):
        try:
            importlib.import_module(package)
        except ImportError:
            continue
        directory = os.path.join(ARENA, package.replace(".", "/"))
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py") or name.startswith("__"):
                continue
            module = importlib.import_module(f"{package}.{name[:-3]}")
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Game) and obj is not Game and getattr(obj, "id", ""):
                    found.append(obj)
    unique = {cls.id: cls for cls in found}
    return [unique[key] for key in sorted(unique)]


def import_solvers() -> tuple[list, list[str]]:
    """Instances from ``harness.engine._solvers()`` plus the `skip` lines for missing modules."""
    skipped: list[str] = []
    for module_name in SOLVER_MODULES:
        try:
            importlib.import_module(f"harness.{module_name}")
        except Exception as exc:  # noqa: BLE001 - a missing port must not fail the stage
            skipped.append(f"skip {module_name}: {type(exc).__name__}: {exc}"[:160])
    try:
        from harness.engine import _solvers

        return list(_solvers()), skipped
    except Exception as exc:  # noqa: BLE001 - engine broken would be a real problem, report it
        skipped.append(f"skip _solvers: {type(exc).__name__}: {exc}"[:160])
        return [], skipped


def _random_action(game, seat: int, rng: random.Random) -> str | None:
    """The arena's own `random_action` when present, else a uniform legal pick."""
    generator = getattr(game, "random_action", None)
    if callable(generator):
        try:
            action = generator(seat, rng)
            if action is not None:
                return str(action)
        except Exception:  # noqa: BLE001 - try the generic fallback instead
            pass
    legal = game.legal_actions(seat)
    if legal:
        return str(rng.choice(list(legal)))
    return None


def playouts(seed: int):
    """Yield ``(game_id, variant, game, seat, observation, legal)`` for one seeded playout/variant."""
    for cls in game_classes():
        try:
            variants = cls.variants()
        except Exception as exc:  # noqa: BLE001 - an arena bug is not a solver failure
            print(f"skip {cls.__name__}.variants: {type(exc).__name__}: {exc}"[:160])
            continue
        for index, variant in enumerate(variants):
            label = str(variant.get("label", "?"))
            try:
                game = cls(dict(variant), seed + index)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {cls.id} [{label}]: {type(exc).__name__}: {exc}"[:160])
                continue
            yield from _play_variant(cls.id, label, game, seed + 101 * index + 17)


def _play_variant(game_id: str, label: str, game, rng_seed: int):
    """One seeded playout; `return` here only stops this variant, not the whole scan."""
    rng = random.Random(rng_seed)
    seen: set[tuple] = set()
    for _ply in range(MAX_PLIES):
        if game.is_over():
            return
        try:
            movers = list(game.to_move())
        except Exception:  # noqa: BLE001
            return
        if not movers:
            return
        actions: dict[int, str] = {}
        for seat in movers:
            try:
                observation = str(game.observation(seat))
                legal = game.legal_actions(seat)
            except Exception:  # noqa: BLE001 - stop this playout, keep the rest
                return
            key = (
                int(seat),
                observation,
                None if legal is None else tuple(str(item) for item in legal),
            )
            if observation and key not in seen:
                seen.add(key)
                yield game_id, label, game, int(seat), observation, legal
            action = _random_action(game, int(seat), rng)
            if action is None:
                return
            actions[int(seat)] = action
        try:
            game.apply(actions)
        except Exception:  # noqa: BLE001
            return


def validate_action(action, legal) -> tuple[bool, str]:
    """Check one solver proposal against the arena contract.

    Returns ``(ok, detail)``. When ``legal`` is a non-empty list the action must normalize onto the
    arena's exact spelling; when the list is null (or empty) the action only has to be non-empty.
    """
    text = "" if action is None else str(action).strip()
    if not text:
        return False, "empty action"
    if legal:
        normalized = normalize(text, legal)
        if not normalized:
            return False, f"illegal {text!r} (legal: {list(legal)[:6]})"
        return True, normalized
    return True, text


def _configure(solver, spec: Spec) -> None:
    """Best-effort configure: solvers that do not need it simply ignore the call."""
    configure = getattr(solver, "configure", None)
    if not callable(configure):
        return
    try:
        configure(spec, "", False)
    except TypeError:
        try:
            configure(spec)
        except Exception:  # noqa: BLE001 - never let configure break the smoke
            pass
    except Exception:  # noqa: BLE001
        pass


def _spec_for(game_id: str, category: str) -> Spec:
    return Spec(game_id=str(game_id or ""), family=CATEGORY_FAMILY.get(category, category or "unknown"))


def matched_solvers(solvers: list, spec: Spec, skipped_names: set[str]) -> tuple[list, int]:
    """Solvers that claim this game. A missing ``matches`` hook is a graceful skip.

    Returns ``(matched, crashes)``; a matcher that raises is a real crash and fails the stage.
    """
    matched = []
    crashes = 0
    for solver in solvers:
        name = str(getattr(solver, "name", type(solver).__name__))
        matches = getattr(solver, "matches", None)
        if not callable(matches):
            if name not in skipped_names:
                print(f"skip {name}: no matches() hook")
                skipped_names.add(name)
            continue
        try:
            if matches(spec.game_id, spec):
                _configure(solver, spec)
                matched.append(solver)
        except Exception as exc:  # noqa: BLE001 - report a matcher bug, do not crash the stage
            crashes += 1
            print(f"CRASH matches {name} {spec.game_id}: {type(exc).__name__}: {exc}"[:160])
    return matched, crashes


def run(seed: int) -> tuple[list[dict], int, int, int, float, int]:
    """Run the whole smoke. Returns (rows, proposals, declines, illegal, total_ms, crashes)."""
    solvers, skip_lines = import_solvers()
    for line in skip_lines:
        print(line)
    if not solvers:
        print("no family solvers available")

    rows: dict[tuple[str, str, str], dict] = {}
    proposals = declines = illegal = crashes = 0
    total_ms = 0.0
    skipped_names: set[str] = set()
    matched_by_game: dict[str, list] = {}

    for game_id, variant, game, seat, observation, legal in playouts(seed):
        if game_id not in matched_by_game:
            category = str(getattr(game, "category", "") or "")
            matched, matcher_crashes = matched_solvers(
                solvers, _spec_for(game_id, category), skipped_names
            )
            matched_by_game[game_id] = matched
            crashes += matcher_crashes
        for solver in matched_by_game[game_id]:
            name = str(getattr(solver, "name", type(solver).__name__))
            key = (game_id, variant, name)
            row = rows.setdefault(key, {"game": game_id, "variant": variant, "solver": name,
                                        "hits": 0, "illegal": 0, "ms": 0.0})
            started = time.perf_counter()
            try:
                try:
                    action = solver.propose(
                        observation=observation, legal=legal, seat=seat, turn=0, timeout=TIMEOUT
                    )
                except TypeError:
                    action = solver.propose(observation=observation, legal=legal)
            except Exception as exc:  # noqa: BLE001 - a solver bug is a failure, not a crash
                elapsed = (time.perf_counter() - started) * 1000.0
                row["ms"] += elapsed
                total_ms += elapsed
                crashes += 1
                print(f"CRASH {game_id} [{variant}] {name}: {type(exc).__name__}: {exc}"[:200])
                continue
            elapsed = (time.perf_counter() - started) * 1000.0
            row["ms"] += elapsed
            total_ms += elapsed

            if action is None:
                declines += 1
                continue
            ok, detail = validate_action(action, legal)
            if ok:
                proposals += 1
                row["hits"] += 1
            else:
                illegal += 1
                row["illegal"] += 1
                print(f"ILLEGAL {game_id} [{variant}] {name}: {detail}"[:200])
    return list(rows.values()), proposals, declines, illegal, total_ms, crashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the family solvers on real arena observations.")
    parser.add_argument("--json", default="")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("=" * 92)
    print(" SOLVER SMOKE")
    print("=" * 92)
    rows, proposals, declines, illegal, total_ms, crashes = run(args.seed)

    print(f" {'game':<20} {'variant':<26} {'solver':<10} {'hits':>4} {'illegal':>7} {'ms':>7}")
    print("-" * 92)
    for row in sorted(rows, key=lambda item: (item["game"], item["variant"], item["solver"])):
        print(
            f" {row['game']:<20} {row['variant'][:26]:<26} {row['solver']:<10} "
            f"{row['hits']:>4} {row['illegal']:>7} {row['ms']:>7.1f}"
        )
    print("-" * 92)

    checked = proposals + illegal + crashes
    bad = bool(crashes or illegal) or checked == 0
    status = "FAIL" if bad else "PASS"
    print(
        f" SOLVER SMOKE {status} ({proposals} proposals, {declines} declines, {illegal} illegal, "
        f"{total_ms:.1f} ms)"
    )
    if checked == 0:
        print(" no solver proposal was checked")
    print("=" * 92)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({
                "rows": rows, "proposals": proposals, "declines": declines,
                "illegal": illegal, "crashes": crashes, "ms": round(total_ms, 1),
            }, handle, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
