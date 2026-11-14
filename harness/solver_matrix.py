"""Mixed-strategy solver for simultaneous ("read the opponent") games.

Why this file exists
--------------------
`search.Mcts` only *samples* the opponents' moves. Against a repeated opponent in a round robin that is
enough to avoid being frozen on one action, but it is not an equilibrium: the sample can be exploited by
an opponent that learns our frequencies. When both seats move at once the position is a normal-form
(zero-sum) matrix game, and the right object to compute is a mixed strategy.

`MatrixSolver.propose_joint` builds the k x k payoff matrix directly from a validated `GameModel`:

    for our action a, for the opponent's action b:
        clone the model, apply(our seat, a), apply(opponent seat, b)
        value = terminal score(seat) or, when not terminal, evaluate(seat)

which is the same clone/apply order as `search.Mcts._child` / `Mcts._joint_state`. It then solves the
matrix with deterministic fictitious play (a regret-matching-family algorithm) and samples one action
from the average strategy. `random.SystemRandom` is the default generator so repeated turns are genuinely
unpredictable; pass `rng=` for reproducible tests.

Boundaries (deliberately conservative)
--------------------------------------
- The matrix is capped at `MAX_ACTIONS` (48) actions per seat. Larger action sets are reduced with a
  deterministic, evenly spaced subset, so the choice does not depend on a random seed.
- The solver never raises. A model that cannot be cloned/applied returns `(None, 0.5)` and the caller
  falls back to the generated model / search / policy / LLM path.
- `propose()` (the observation-only `Solver` protocol) returns `None`: a matrix game needs an explicit
  model, so the harness integrates through `propose_joint()` in the model-search path.
"""
from __future__ import annotations

import math
import random
import time
from itertools import product
from typing import Any, Iterable, Sequence

MAX_ACTIONS = 48

# Conservative id hints: a game id containing one of these is a simultaneous-game candidate. The spec
# flag `simultaneous` is the primary signal; the hints cover cold starts where prep has not filled it in.
_SIMULTANEOUS_HINTS = (
    "simul",
    "simultaneous",
    "rps",
    "rock_paper",
    "rock-paper",
    "scissor",
    "blotto",
    "matrix",
    "both_choose",
    "both-choose",
    "duel",
)

_SYSTEM_RNG = random.SystemRandom()


# ---------------------------------------------------------------------------- safe model access
def _clip(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _safe_actions(model: Any, seat: int) -> list[str]:
    try:
        actions = model.legal_actions(int(seat))
    except Exception:  # noqa: BLE001 - generated model code must never break a turn
        return []
    return _dedupe(str(action) for action in (actions or []))


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _safe_movers(model: Any) -> list[int]:
    try:
        return [int(seat) for seat in (model.to_move() or [])]
    except Exception:  # noqa: BLE001
        return []


def _value_of(model: Any, seat: int) -> float:
    """Terminal score when the game is over, otherwise the model's heuristic, always in [0, 1]."""
    try:
        terminal = bool(model.is_terminal())
    except Exception:  # noqa: BLE001 - a broken terminal test is treated as a dead position
        terminal = True
    if terminal:
        try:
            return _clip(float(model.score(int(seat))))
        except Exception:  # noqa: BLE001
            return 0.5
    getter = getattr(model, "evaluate", None)
    if not callable(getter):
        return 0.5
    try:
        return _clip(float(getter(int(seat))))
    except Exception:  # noqa: BLE001
        return 0.5


# ---------------------------------------------------------------------------- matrix reduction
def _evenly_spaced(items: Sequence[str], limit: int = MAX_ACTIONS) -> list[str]:
    """Deterministic subset that keeps the first and last action and spreads the rest evenly."""
    total = len(items)
    if total <= limit:
        return list(items)
    if limit <= 0:
        return []
    if limit == 1:
        return [items[0]]
    indices: list[int] = []
    for position in range(limit):
        index = int(round(position * (total - 1) / (limit - 1)))
        if not indices or index != indices[-1]:
            indices.append(index)
    return [items[index] for index in indices]


def _joint_profiles(action_lists: Sequence[Sequence[str]], limit: int = MAX_ACTIONS) -> list[tuple[str, ...]]:
    """Every opponent joint action, or a deterministic evenly spaced sample when the product is huge."""
    if not action_lists:
        return [()]
    sizes = [len(actions) for actions in action_lists]
    if any(size == 0 for size in sizes):
        return []
    total = 1
    for size in sizes:
        total *= size
    if total <= limit:
        return [tuple(profile) for profile in product(*action_lists)]

    profiles: list[tuple[str, ...]] = []
    for position in range(limit):
        flat = int(round(position * (total - 1) / (limit - 1))) if limit > 1 else 0
        digits = [0] * len(sizes)
        for index in range(len(sizes) - 1, -1, -1):
            digits[index] = flat % sizes[index]
            flat //= sizes[index]
        profiles.append(tuple(action_lists[index][digits[index]] for index in range(len(sizes))))
    return profiles


# ---------------------------------------------------------------------------- matrix solver
def solve_matrix(
    payoff: Sequence[Sequence[float]],
    iterations: int = 3000,
) -> tuple[list[float], float]:
    """Solve a zero-sum matrix game with fictitious play; return (row mixture, game value).

    `payoff[i][j]` is the row player's utility when the row player chooses action i and the column
    player chooses action j. The column player minimises it (zero-sum convention).

    Fictitious play keeps empirical counts of each player's best response and is updated incrementally:
    `row_value[i] = sum_j y_j * payoff[i][j]` and `column_cost[j] = sum_i x_i * payoff[i][j]`. Each
    iteration picks the current best response for both sides and adds one count, so one iteration is
    O(rows + columns) instead of O(rows * columns). The average of the row player's best-response counts
    is the returned mixture; in a zero-sum game it converges to a Nash equilibrium (regret matching
    family). Ties choose the lowest index, which keeps the result deterministic.

    Never raises: malformed input returns `([], 0.5)`.
    """
    try:
        rows = len(payoff)
        columns = len(payoff[0]) if rows else 0
        if rows == 0 or columns == 0:
            return [], 0.5
        matrix: list[list[float]] = []
        for row in payoff:
            if len(row) != columns:
                return [], 0.5
            matrix.append([_clip(float(value)) for value in row])
        rounds = max(1, int(iterations))

        # Empirical best-response counts, initialised at one so an action is never starved before its
        # first best-response turn (this also makes the final mixture strictly positive and safe to sample).
        row_counts = [1] * rows
        column_counts = [1] * columns
        row_value = [sum(row) for row in matrix]  # sum_j 1 * payoff[i][j]
        column_cost = [sum(matrix[i][j] for i in range(rows)) for j in range(columns)]  # sum_i 1 * payoff[i][j]

        row_range = range(rows)
        column_range = range(columns)
        for _ in range(rounds):
            best_row = max(row_range, key=row_value.__getitem__)
            best_column = min(column_range, key=column_cost.__getitem__)
            row_counts[best_row] += 1
            column_counts[best_column] += 1
            # Incremental update: the new row count adds one copy of the column's payoffs, and the new
            # column count adds one copy of the row's payoffs.
            for row in row_range:
                row_value[row] += matrix[row][best_column]
            for column in column_range:
                column_cost[column] += matrix[best_row][column]

        row_total = sum(row_counts)
        column_total = sum(column_counts)
        mixture = [count / row_total for count in row_counts]
        value = sum(mixture[row] * row_value[row] for row in row_range) / column_total
        return mixture, _clip(value)
    except Exception:  # noqa: BLE001 - a solver bug must not cost a turn
        return [], 0.5


def _sample(actions: Sequence[str], weights: Sequence[float], rng: Any | None = None) -> str | None:
    """Sample one action from a mixture. `rng=None` means fresh CSPRNG entropy, not a fixed seed."""
    if not actions:
        return None
    generator = rng if rng is not None else _SYSTEM_RNG
    cleaned = [max(0.0, float(weight)) if math.isfinite(float(weight)) else 0.0 for weight in weights]
    total = sum(cleaned)
    if total <= 0.0:
        try:
            return generator.choice(list(actions))
        except Exception:  # noqa: BLE001
            return actions[0]
    try:
        roll = generator.random() * total
    except Exception:  # noqa: BLE001 - a caller-supplied rng may be minimal
        return actions[0]
    running = 0.0
    for action, weight in zip(actions, cleaned):
        running += weight
        if roll < running:
            return action
    return actions[-1]


class MatrixSolver:
    """Nash-mixed solver for simultaneous finite-action games backed by a `GameModel`."""

    name = "matrix"

    def __init__(self, max_actions: int = MAX_ACTIONS, iterations: int = 3000) -> None:
        self.max_actions = max(2, int(max_actions))
        self.iterations = max(1, int(iterations))

    # ------------------------------------------------------------------ routing
    def matches(self, game_id: str, spec: Any) -> bool:
        """Conservative: the spec must say simultaneous, or the game id must clearly hint at it."""
        try:
            if bool(getattr(spec, "simultaneous", False)):
                return True
            spec_id = str(getattr(spec, "game_id", "") or "")
            identifier = f"{game_id or ''} {spec_id}".lower()
            return any(hint in identifier for hint in _SIMULTANEOUS_HINTS)
        except Exception:  # noqa: BLE001
            return False

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        """Observation-only protocol hook. A matrix game needs the model, so this always declines.

        `propose_joint(model, seat, ...)` is the real entry point; see the module docstring and the
        integration note in the report.
        """
        return None

    # ------------------------------------------------------------------ entry point
    def propose_joint(
        self,
        model: Any,
        seat: int,
        seconds: float = 1.0,
        rng: Any | None = None,
    ) -> tuple[str | None, float]:
        """Return (sampled action, equilibrium value) for a simultaneous node, or (None, 0.5).

        The matrix is built by cloning the model per cell and applying our action first, then every
        opponent action (the same order as `search.Mcts._joint_state`). Values come from `score()` at a
        terminal state and from `evaluate()` otherwise, clipped to [0, 1]. Failed clones/applies become a
        neutral 0.5 cell; if every cell fails the position is reported as unusable.
        """
        try:
            return self._propose_joint(model, seat, seconds, rng)
        except Exception:  # noqa: BLE001 - never raise into the turn loop
            return None, 0.5

    def _propose_joint(
        self,
        model: Any,
        seat: int,
        seconds: float,
        rng: Any | None,
    ) -> tuple[str | None, float]:
        if model is None:
            return None, 0.5
        seat = int(seat)
        movers = _safe_movers(model)
        if len(movers) < 2 or seat not in movers:
            # Not a simultaneous node: the matrix game is only defined when at least one opponent
            # moves at the same time as us.
            return None, 0.5
        if _safe_terminal(model):
            return None, 0.5

        ours = _safe_actions(model, seat)
        if not ours:
            return None, 0.5
        ours = _evenly_spaced(ours, self.max_actions)

        opponents = [mover for mover in movers if mover != seat]
        opponent_actions: list[list[str]] = []
        for opponent in opponents:
            actions = _safe_actions(model, opponent)
            if not actions:
                return None, 0.5
            opponent_actions.append(_evenly_spaced(actions, self.max_actions))
        profiles = _joint_profiles(opponent_actions, self.max_actions)
        if not profiles:
            return None, 0.5

        budget = max(0.05, float(seconds))
        deadline = time.monotonic() + budget
        matrix: list[list[float]] = []
        cells = 0
        for action in ours:
            row: list[float] = []
            for profile in profiles:
                try:
                    child = model.clone()
                    child.apply(seat, action)
                    for opponent, reply in zip(opponents, profile):
                        child.apply(opponent, reply)
                    value = _value_of(child, seat)
                    cells += 1
                except Exception:  # noqa: BLE001 - one bad cell is neutral, not fatal
                    value = 0.5
                row.append(value)
            matrix.append(row)
        if cells == 0:
            return None, 0.5
        # The matrix build is O(k^2) clones and cannot be interrupted usefully; the solve below is
        # O(iterations * 2k). If the build already ate the budget, spend fewer iterations rather than
        # skipping the solve: a coarse mixture still beats one frozen action.
        remaining = deadline - time.monotonic()
        rounds = self.iterations
        if remaining < 0.05:
            rounds = max(200, self.iterations // 10)
        elif remaining < 0.20:
            rounds = max(500, self.iterations // 3)
        mixture, value = solve_matrix(matrix, iterations=rounds)
        if not mixture:
            return None, 0.5
        return _sample(ours, mixture, rng), _clip(float(value))


def _safe_terminal(model: Any) -> bool:
    try:
        return bool(model.is_terminal())
    except Exception:  # noqa: BLE001
        return True


__all__ = ["MatrixSolver", "MAX_ACTIONS", "solve_matrix"]
