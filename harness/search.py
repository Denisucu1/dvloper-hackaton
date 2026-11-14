"""Search over an explicit `GameModel`: negamax, alpha-beta, transposition table, iterative deepening.

DESIGN CONSTRAINTS (they matter more than depth):
- bounded by wall clock, never by node count: a turn has a 20 s clock and we also need a model call;
- never raises: a broken generated model must degrade to `None`, and the caller falls back to the model;
- stdlib only and no package-relative imports, because this file is also loaded *inside the sandbox*
  subprocess (next to a model the model wrote in prep).

Values are always in [0, 1] from the root seat's point of view (1 = win, 0.5 = draw, 0 = loss).
"""
from __future__ import annotations

import math
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


WIDE_ACTIONS = 20


def _safe_terminal(model: Any) -> bool:
    try:
        return bool(model.is_terminal())
    except Exception:  # noqa: BLE001
        return True


def _safe_movers(model: Any) -> list[int]:
    try:
        return [int(seat) for seat in (model.to_move() or [])]
    except Exception:  # noqa: BLE001
        return []


def _safe_actions(model: Any, seat: int) -> list[str]:
    try:
        return [str(action) for action in (model.legal_actions(seat) or [])]
    except Exception:  # noqa: BLE001
        return []


class _MctsNode:
    """One position in the UCT tree. `total` is always from the root seat's point of view."""

    __slots__ = ("model", "movers", "children", "visits", "total", "terminal")

    def __init__(self, model: Any, movers: list[int]) -> None:
        self.model = model
        self.movers = list(movers)
        self.children: list[tuple[str, "_MctsNode"]] = []
        self.visits = 0
        self.total = 0.0
        self.terminal = _safe_terminal(model)


class Mcts:
    """UCT with UCB1 for the positions minimax cannot handle.

    Two cases route here from `best_action`:
    - wide nodes (`len(legal_actions) > 20`): alpha-beta would only look one ply deep;
    - simultaneous nodes (`len(to_move()) > 1`): a minimax tree does not exist, so we branch on OUR move
      and sample the opponents' moves (a best-response search against the current sample).
    Values are the root seat's [0, 1] score, and at opponent nodes the selection score is inverted, so the
    tree does not cooperate with itself. The budget is wall-clock, never node count.
    """

    def __init__(
        self,
        model: Any,
        seat: int,
        seconds: float = 1.0,
        exploration: float = 0.9,
        max_iters: int = 4000,
        rollout_plies: int = 40,
        seed: int = 12345,
        max_branch: int = 200,
    ) -> None:
        import random

        self.model = model
        self.seat = int(seat)
        self.seconds = max(0.05, float(seconds))
        self.exploration = float(exploration)
        self.max_iters = max(1, int(max_iters))
        self.rollout_plies = max(4, int(rollout_plies))
        self.max_branch = max(2, int(max_branch))
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ public
    def best(self) -> tuple[str | None, float]:
        movers = _safe_movers(self.model)
        if self.seat not in movers:
            return None, DRAW
        if len(movers) > 1:
            # Simultaneous move: there is no tree of alternating plies to build. Best response over our
            # actions, re-sampling the opponents on every visit so no single reply is frozen in.
            return self._best_joint(movers)
        root = _MctsNode(self.model.clone(), movers)
        if root.terminal:
            return None, DRAW
        self._expand(root, root=True)
        if not root.children:
            return None, DRAW

        deadline = time.monotonic() + self.seconds
        iterations = 0
        while iterations < self.max_iters and time.monotonic() < deadline:
            iterations += 1
            node = root
            path = [node]
            while node.children and not node.terminal:
                node = self._select(node)
                path.append(node)
            if not node.terminal and not node.children:
                self._expand(node, root=False)
                if node.children:
                    node = self._select(node)
                    path.append(node)
            reward = self._rollout(node)
            for item in path:
                item.visits += 1
                item.total += reward

        # Choose by estimated value, not by visit count: with a wide, flat game the mean is the better
        # estimator and a visit tie-break just returns whichever arm the exploration bonus favoured.
        action, child = max(
            root.children,
            key=lambda item: (item[1].total / max(1, item[1].visits), item[1].visits),
        )
        return action, _clip(child.total / max(1, child.visits))

    def _best_joint(self, movers: list[int]) -> tuple[str | None, float]:
        ours = _safe_actions(self.model, self.seat)[: self.max_branch]
        if not ours:
            return None, DRAW
        stats = {action: [0, 0.0] for action in ours}  # action -> [visits, total from our seat]
        deadline = time.monotonic() + self.seconds
        iterations = 0
        log_every = 0
        while iterations < self.max_iters and time.monotonic() < deadline:
            iterations += 1
            log_every = math.log(max(2, iterations))
            arm = max(
                ours,
                key=lambda candidate: (
                    float("inf") if stats[candidate][0] == 0
                    else stats[candidate][1] / stats[candidate][0]
                    + self.exploration * (log_every / stats[candidate][0]) ** 0.5
                ),
            )
            state = self._joint_state(arm, movers)
            if state is None:
                continue
            reward = _score(state, self.seat) if _safe_terminal(state) else self._rollout_state(state)
            stats[arm][0] += 1
            stats[arm][1] += reward
        action = max(ours, key=lambda candidate: (stats[candidate][1] / max(1, stats[candidate][0]),
                                                  stats[candidate][0]))
        return action, _clip(stats[action][1] / max(1, stats[action][0]))

    def _joint_state(self, action: str, movers: list[int]) -> Any | None:
        """Our action plus one uniformly sampled action per opponent, applied to a fresh clone."""
        try:
            state = self.model.clone()
            state.apply(self.seat, action)
            for mover in movers:
                if mover == self.seat:
                    continue
                options = _safe_actions(state, mover)
                if not options:
                    return None
                state.apply(mover, self.rng.choice(options))
            return state
        except Exception:  # noqa: BLE001
            return None

    def _rollout_state(self, state: Any) -> float:
        try:
            return rollout_value(state, self.seat, self.rollout_plies, self.rng)
        except Exception:  # noqa: BLE001
            return DRAW

    # ------------------------------------------------------------------ tree
    def _select(self, node: _MctsNode) -> _MctsNode:
        maximizing = not node.movers or len(node.movers) > 1 or node.movers[0] == self.seat
        unseen = [child for _, child in node.children if child.visits == 0]
        if unseen:
            # Random order, not list order: otherwise the first reply is always the same action and every
            # mean is computed over a biased sample of the opponent's moves.
            return self.rng.choice(unseen)
        log_parent = math.log(max(1, node.visits))
        best_child, best_score = node.children[0][1], -1.0
        for _, child in node.children:
            mean = child.total / child.visits
            bonus = self.exploration * (log_parent / child.visits) ** 0.5
            score = (mean + bonus) if maximizing else ((1.0 - mean) + bonus)
            if score > best_score:
                best_child, best_score = child, score
        return best_child

    def _expand(self, node: _MctsNode, root: bool = False) -> None:
        if node.terminal or node.children or not node.movers:
            return
        if len(node.movers) == 1:
            mover = node.movers[0]
            actions = _safe_actions(node.model, mover)
            if not root and (mover != self.seat or len(actions) > WIDE_ACTIONS):
                # Leave it to the rollout: sampling a fresh reply on every visit keeps the estimate
                # unbiased. Storing one sampled child would freeze a single opponent move per branch.
                return
            for action in actions[: self.max_branch]:
                child = self._child(node.model, [(mover, action)])
                if child is not None:
                    node.children.append((action, child))
            return

        # Simultaneous: branch on our own move, sample what the others play (best response to a sample).
        ours = _safe_actions(node.model, self.seat) if self.seat in node.movers else []
        if not ours:
            plays = []
            for mover in node.movers:
                options = _safe_actions(node.model, mover)
                if not options:
                    return
                plays.append((mover, self.rng.choice(options)))
            child = self._child(node.model, plays)
            if child is not None:
                node.children.append((plays[0][1], child))
            return

        for action in ours[: min(self.max_branch, 16)]:
            plays = [(self.seat, action)]
            for mover in node.movers:
                if mover == self.seat:
                    continue
                options = _safe_actions(node.model, mover)
                if not options:
                    plays = []
                    break
                plays.append((mover, self.rng.choice(options)))
            if not plays:
                break
            child = self._child(node.model, plays)
            if child is not None:
                node.children.append((action, child))

    def _child(self, model: Any, plays: list[tuple[int, str]]) -> _MctsNode | None:
        try:
            child_model = model.clone()
            for seat, action in plays:
                child_model.apply(seat, action)
            return _MctsNode(child_model, _safe_movers(child_model))
        except Exception:  # noqa: BLE001
            return None

    def _rollout(self, node: _MctsNode) -> float:
        if node.terminal:
            return _score(node.model, self.seat)
        try:
            return rollout_value(node.model.clone(), self.seat, self.rollout_plies, self.rng)
        except Exception:  # noqa: BLE001
            return DRAW


def best_action(model: Any, seat: int, seconds: float = 0.8, max_depth: int = 8) -> tuple[str | None, float]:
    """Best move for `seat`, or (None, 0.5) if the model is unusable. Never raises.

    Routing: wide or simultaneous -> MCTS/UCT; narrow alternating games -> alpha-beta (exact, cheap).
    """
    if model is None:
        return None, DRAW
    try:
        seat = int(seat)
        movers = _safe_movers(model)
        if seat not in movers:
            return None, DRAW
        actions = _safe_actions(model, seat)
        if not actions:
            return None, DRAW
        if len(movers) > 1 or len(actions) > WIDE_ACTIONS:
            return Mcts(model, seat, seconds=min(3.0, max(0.2, float(seconds)))).best()
        return Searcher(model, seat, seconds, max_depth).best()
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
