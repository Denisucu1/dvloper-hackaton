"""Self-play between decision strategies on one generated model.

The prep tournament needs to compare *strategies* (alpha-beta, matrix game, expectimax, greedy, random)
on the same validated simulator. Different generated model implementations cannot cross-play without a
shared referee, so the tournament varies the strategy, not the rules engine:

    for each ply: clone the model for each mover, let the strategy choose, apply on the canonical model

This is sound for perfect-information games (one shared state) and for simultaneous games (both seats act
on the same state). It is deliberately skipped for hidden-information families, where a deterministic
model cannot see the opponent's private state.

Everything here is deterministic given the seed. A broken strategy or model yields a draw for that game
(never an exception into prep).
"""
from __future__ import annotations

import importlib.util
import random
import time
from dataclasses import dataclass, field

from . import search

Strategy = str  # "search" | "matrix" | "chance" | "greedy" | "random"


@dataclass
class MatchResult:
    winner: int | None          # 0, 1, or None for a draw
    plies: int
    score: tuple[float, float]
    timed_out: bool = False
    error: str = ""
    notes: list[str] = field(default_factory=list)


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("selfplay_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(module, rules: str, seat: int = 0):
    cls = getattr(module, "Model", None) or getattr(module, "GameModel", None)
    if cls is None:
        raise RuntimeError("model has no class Model")
    try:
        return cls(rules, seat)
    except TypeError:
        try:
            return cls(seat)
        except TypeError:
            return cls()


def _legal(model, seat: int) -> list[str]:
    try:
        return [str(a) for a in (model.legal_actions(seat) or [])]
    except Exception:  # noqa: BLE001
        return []


def _movers(model) -> list[int]:
    try:
        return [int(s) for s in (model.to_move() or [])]
    except Exception:  # noqa: BLE001
        return []


def _terminal(model) -> bool:
    try:
        return bool(model.is_terminal())
    except Exception:  # noqa: BLE001
        return True


def _score(model, seat: int) -> float:
    try:
        return float(model.score(seat))
    except Exception:  # noqa: BLE001
        return 0.5


def _evaluate(model, seat: int) -> float:
    getter = getattr(model, "evaluate", None)
    if not callable(getter):
        return 0.5
    try:
        return float(getter(seat))
    except Exception:  # noqa: BLE001
        return 0.5


def _strategy_module(name: str, kind: str):
    try:
        if kind == "matrix":
            return importlib.import_module(f".solver_{name}", package=__package__)
    except Exception:  # noqa: BLE001
        return None
    return None


def choose(strategy: Strategy, model, seat: int, seconds: float, rng: random.Random) -> str | None:
    """One move from `strategy`, on a model the caller may mutate. Never raises."""
    try:
        if strategy == "search":
            # Depth/iteration-capped, not wall-clock capped: a self-play benchmark must be reproducible.
            try:
                joint = len(model.to_move() or []) > 1
            except Exception:  # noqa: BLE001
                joint = False
            if joint:
                engine = search.Mcts(model, seat, seconds=5.0, max_iters=800,
                                     seed=rng.randrange(2 ** 31))
                action, _ = engine.best()
                return action
            action, _ = search.best_action(model, seat, seconds=30.0, max_depth=2)
            return action
        if strategy == "matrix":
            module = _strategy_module("matrix", "matrix")
            if module is None:
                return None
            action, _ = module.MatrixSolver().propose_joint(model, seat, seconds=5.0, rng=rng)
            return action
        if strategy == "chance":
            module = _strategy_module("chance", "matrix")
            if module is None:
                return None
            action, _ = module.ChanceSolver().propose(model, seat, seconds=5.0, max_depth=2, rng=rng)
            return action
        if strategy == "greedy":
            actions = _legal(model, seat)
            if not actions:
                return None
            best, best_value = actions[0], -1.0
            for action in actions[:60]:
                clone = model.clone()
                clone.apply(seat, action)
                value = _score(clone, seat) if _terminal(clone) else _evaluate(clone, seat)
                if value > best_value:
                    best, best_value = action, value
            return best
        if strategy == "random":
            actions = _legal(model, seat)
            return rng.choice(actions) if actions else None
    except Exception:  # noqa: BLE001
        return None
    return None


def play_game(model_path: str, rules: str, strategies: tuple[str, str], seed: int,
              max_plies: int = 200, seconds_per_move: float = 0.12,
              deadline: float | None = None) -> MatchResult:
    """One self-play game. strategies[0] plays seat 0, strategies[1] plays seat 1."""
    rng = random.Random(seed)
    started = time.monotonic()
    try:
        module = _load_module(model_path)
        model = _build(module, rules, 0)
    except Exception as exc:  # noqa: BLE001
        return MatchResult(None, 0, (0.5, 0.5), error=f"model: {type(exc).__name__}")

    plies = 0
    while plies < max_plies:
        if deadline is not None and time.monotonic() > deadline:
            return MatchResult(None, plies, (0.5, 0.5), timed_out=True)
        if _terminal(model):
            break
        movers = _movers(model)
        if not movers:
            break
        actions: dict[int, str] = {}
        for mover in movers:
            strategy = strategies[mover] if mover < len(strategies) else "random"
            clone = model.clone()
            action = choose(strategy, clone, mover, seconds_per_move, rng)
            legal = _legal(model, mover)
            if action is None or (legal and action not in legal):
                action = rng.choice(legal) if legal else None
            if action is None:
                return MatchResult(None, plies, (0.5, 0.5), error="no legal action")
            actions[mover] = action
        try:
            # GameModel.apply(seat, action), not the arena's dict form: apply each mover on the canonical state.
            for mover, action in actions.items():
                model.apply(mover, action)
            # Resolve chance nodes that the move opened (dice, draws) before the next decision.
            for _ in range(20):
                getter = getattr(model, "chance_actions", None)
                if not callable(getter) or _terminal(model):
                    break
                chances = list(getter() or [])
                if not chances:
                    break
                outcomes = [str(outcome) for outcome, _ in chances]
                weights = [max(0.0, float(weight)) for _, weight in chances]
                if not outcomes:
                    break
                model.apply_chance(rng.choices(outcomes, weights=weights, k=1)[0])
        except Exception as exc:  # noqa: BLE001
            return MatchResult(None, plies, (0.5, 0.5), error=f"apply: {type(exc).__name__}")
        plies += 1

    if _terminal(model):
        score = (_score(model, 0), _score(model, 1))
    else:
        score = (_evaluate(model, 0), _evaluate(model, 1))
    if score[0] > score[1]:
        winner = 0
    elif score[1] > score[0]:
        winner = 1
    else:
        winner = None
    return MatchResult(winner, plies, score)


def head_to_head(model_path: str, rules: str, a: str, b: str, games: int = 20,
                 seed: int = 0, seconds_per_move: float = 0.12,
                 deadline: float | None = None) -> dict:
    """Paired-seed match: a plays seat 0 in even games and seat 1 in odd games."""
    wins = draws = losses = 0
    errors = 0
    for index in range(max(1, games)):
        seat_a = index % 2
        strategies = (a, b) if seat_a == 0 else (b, a)
        result = play_game(model_path, rules, strategies, seed=seed + index,
                           seconds_per_move=seconds_per_move, deadline=deadline)
        if result.error:
            errors += 1
            draws += 1
            continue
        winner = result.winner
        if winner is None:
            draws += 1
        elif (winner == 0) == (seat_a == 0):
            wins += 1
        else:
            losses += 1
    total = max(1, games)
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "errors": errors, "score": (wins + 0.5 * draws) / total}
