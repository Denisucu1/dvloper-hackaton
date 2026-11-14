"""Search over an explicit `GameModel`: negamax, alpha-beta, transposition table, iterative deepening.

DESIGN CONSTRAINTS (they matter more than depth):
- bounded by wall clock, never by node count: a turn has a 20 s clock and we also need a model call;
- never raises: a broken generated model must degrade to `None`, and the caller falls back to the model;
- stdlib only and no package-relative imports, because this file is also loaded *inside the sandbox*
  subprocess (next to a model the model wrote in prep).

Values are always in [0, 1] from the root seat's point of view (1 = win, 0.5 = draw, 0 = loss).
"""
from __future__ import annotations

import time
from typing import Any

MATE = 1.0
DRAW = 0.5


def _clip(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _score(model: Any, seat: int) -> float:
    try:
        return _clip(float(model.score(seat)))
    except Exception:  # noqa: BLE001
        return DRAW


def _heuristic(model: Any, seat: int) -> float:
    getter = getattr(model, "evaluate", None)
    if not callable(getter):
        return DRAW
    try:
        return _clip(float(getter(seat)))
    except Exception:  # noqa: BLE001
        return DRAW


class _Deadline(Exception):
    """Internal: the time budget for this turn is spent."""


class Searcher:
    def __init__(self, model: Any, seat: int, seconds: float = 0.8, max_depth: int = 8) -> None:
        self.root = model
        self.seat = seat
        self.seconds = max(0.05, float(seconds))
        self.max_depth = max(1, int(max_depth))
        self.deadline = time.monotonic() + self.seconds
        self.nodes = 0
        self.depth_reached = 0
        self.table: dict[str, tuple[float, int]] = {}
        self.use_table = callable(getattr(model, "key", None))

    # ------------------------------------------------------------------ public
    def best(self) -> tuple[str | None, float]:
        try:
            moves = self._moves(self.root)
        except Exception:  # noqa: BLE001
            return None, DRAW
        if not moves:
            return None, DRAW

        ordered = self._order(self.root, moves)
        best_move, best_value = ordered[0], DRAW
        for depth in range(1, self.max_depth + 1):
            try:
                value, move = self._root_search(depth, ordered)
            except _Deadline:
                break
            if move is not None:
                best_move, best_value = move, value
                self.depth_reached = depth
                if value >= MATE or value <= 0.0:
                    break  # a proven win or loss: deeper search cannot help
            # put the best move first for the next iteration
            if best_move in ordered:
                ordered.remove(best_move)
                ordered.insert(0, best_move)
        return best_move, best_value

    # ------------------------------------------------------------------ root
    def _root_search(self, depth: int, ordered: list[str]) -> tuple[float, str | None]:
        alpha, beta = 0.0, 1.0
        best_value, best_move = -1.0, None
        for action in ordered:
            child = self._child(self.root, self.seat, action)
            if child is None:
                continue
            value = self._negamax(child, depth - 1, alpha, beta)
            if value > best_value:
                best_value, best_move = value, action
            if value > alpha:
                alpha = value
        return best_value, best_move

    # ------------------------------------------------------------------ negamax
    def _negamax(self, model: Any, depth: int, alpha: float, beta: float) -> float:
        self._tick()
        if self._terminal(model):
            return _score(model, self.seat)
        if depth <= 0:
            return _heuristic(model, self.seat)

        key = ""
        if self.use_table:
            try:
                key = f"{model.key()}"
            except Exception:  # noqa: BLE001
                key = ""
            hit = self.table.get(key)
            if hit is not None and hit[1] >= depth:
                return hit[0]

        movers = self._movers(model)
        if not movers:
            return _heuristic(model, self.seat)
        if len(movers) > 1:
            # Simultaneous games are not a minimax tree; evaluate the position instead of guessing.
            return _heuristic(model, self.seat)

        mover = movers[0]
        maximizing = mover == self.seat
        moves = self._order(model, self._moves(model))
        if not moves:
            return _heuristic(model, self.seat)

        value = -1.0 if maximizing else 2.0
        for action in moves:
            child = self._child(model, mover, action)
            if child is None:
                continue
            raw = self._negamax(child, depth - 1, alpha, beta)
            # Values are always from the root seat's point of view, so no sign flip here: the node type
            # (maximising or minimising) decides how the same number is combined.
            child_value = raw
            value = max(value, child_value) if maximizing else min(value, child_value)
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if beta <= alpha:
                break
        value = _clip(value)
        if key:
            self.table[key] = (value, depth)
        return value

    # ------------------------------------------------------------------ helpers
    def _tick(self) -> None:
        self.nodes += 1
        if (self.nodes & 63) == 0 and time.monotonic() > self.deadline:
            raise _Deadline()
        if self.nodes > 400_000:
            raise _Deadline()

    def _moves(self, model: Any) -> list[str]:
        movers = self._movers(model)
        if len(movers) != 1:
            return []
        actions = model.legal_actions(movers[0])
        return [str(item) for item in (actions or [])]

    def _movers(self, model: Any) -> list[int]:
        try:
            return [int(seat) for seat in (model.to_move() or [])]
        except Exception:  # noqa: BLE001
            return []

    def _terminal(self, model: Any) -> bool:
        try:
            return bool(model.is_terminal())
        except Exception:  # noqa: BLE001
            return True

    def _child(self, model: Any, seat: int, action: str) -> Any | None:
        try:
            child = model.clone()
            child.apply(seat, action)
            return child
        except Exception:  # noqa: BLE001
            return None

    def _order(self, model: Any, moves: list[str]) -> list[str]:
        sorter = getattr(model, "move_score", None)
        if not callable(sorter):
            return list(moves)
        try:
            return sorted(moves, key=lambda action: -float(sorter(action)))
        except Exception:  # noqa: BLE001
            return list(moves)


def best_action(model: Any, seat: int, seconds: float = 0.8, max_depth: int = 8) -> tuple[str | None, float]:
    """Best move for `seat`, or (None, 0.5) if the model is unusable. Never raises."""
    if model is None:
        return None, DRAW
    try:
        return Searcher(model, int(seat), seconds, max_depth).best()
    except Exception:  # noqa: BLE001
        return None, DRAW


def rollout_value(model: Any, seat: int, plies: int = 40, rng: Any = None) -> float:
    """Cheap Monte-Carlo estimate used when the game is simultaneous or too wide for minimax."""
    import random

    generator = rng or random.Random(0)
    state = model
    try:
        for _ in range(int(plies)):
            if state.is_terminal():
                return _score(state, seat)
            movers = [int(s) for s in (state.to_move() or [])]
            if not movers:
                break
            for mover in movers:
                actions = state.legal_actions(mover) or []
                if not actions:
                    return _score(state, seat) if state.is_terminal() else DRAW
                state.apply(mover, generator.choice([str(a) for a in actions]))
        return _score(state, seat) if state.is_terminal() else _heuristic(state, seat)
    except Exception:  # noqa: BLE001
        return DRAW
