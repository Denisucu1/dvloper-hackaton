"""Expectimax expert for games with explicit chance nodes (dice, draws, shuffled decks).

Why this exists
---------------
The family solvers are hand-written for one game; `search.py` is an adversarial minimax and treats a
model without chance support as deterministic. A dice game modelled with `chance_actions()` would
therefore get a *wrong* answer: a roll is not a choice with one known result, it is a distribution.
This solver reads the optional chance surface from `game_model.py` and answers with an expected value:

    decision node where we move  -> maximize
    decision node where they move -> minimize
    chance node (no seat to move, `chance_actions()` non-empty) -> probability-weighted average

Values are always the root seat's win probability in [0, 1] (1 = win, 0.5 = draw, 0 = loss).
Everything is duck-typed, stdlib only, and bounded by wall clock *and* depth. `propose()` never
raises: a broken generated model degrades to (None, 0.5) or, when a legal decision exists, to a
cheap one-ply fallback.
"""
from __future__ import annotations

import re
import time
from typing import Any

try:  # normal package import (engine/tests)
    from .game_model import apply_chance as _apply_chance, chance_of as _chance_of
except ImportError:  # loaded standalone, like search.py inside the sandbox
    try:
        from game_model import apply_chance as _apply_chance, chance_of as _chance_of  # type: ignore
    except ImportError:  # sandbox: only this file is on disk, next to the generated model

        def _chance_of(model: Any) -> list[tuple[str, float]]:
            getter = getattr(model, "chance_actions", None)
            if not callable(getter):
                return []
            try:
                raw = getter()
            except Exception:  # noqa: BLE001
                return []
            outcomes: list[tuple[str, float]] = []
            for item in list(raw or []):
                try:
                    name, weight = item
                    weight = float(weight)
                except Exception:  # noqa: BLE001
                    continue
                if weight > 0.0 and weight == weight and weight != float("inf"):
                    outcomes.append((str(name), weight))
            return outcomes

        def _apply_chance(model: Any, outcome: str) -> bool:
            applier = getattr(model, "apply_chance", None)
            if not callable(applier):
                return False
            try:
                applier(str(outcome))
                return True
            except Exception:  # noqa: BLE001
                return False

DRAW = 0.5
TIE_EPS = 1e-12
NODE_CAP = 200_000

_CHANCE_ID = re.compile(
    r"\b(?:dice|die|d6|roll|rolling|shuffle|shuffled|deck|coin|coinflip|lottery|jackpot|"
    r"draw\s+(?:a\s+)?card|draw\s+cards?)\b",
    re.IGNORECASE,
)
_CHANCE_FAMILIES = {"chance", "dice", "stochastic", "push_your_luck", "luck"}
_DETERMINISTIC_FAMILIES = {"deals", "bluff", "truth", "guardrails", "read_opponent"}

_TURN_TOTAL = re.compile(r"turn\s+total\s*[:=]\s*(\d+)", re.IGNORECASE)
_BANKED_YOU = re.compile(r"(?:banked|score|total)\s*[:=]\s*you\s+(\d+)", re.IGNORECASE)
_TARGET = re.compile(r"\btarget\s*[:=]\s*(\d+)", re.IGNORECASE)


class _Deadline(Exception):
    """Internal: this turn's search budget is spent."""


def _clip(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


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


def _terminal(model: Any) -> bool:
    try:
        return bool(model.is_terminal())
    except Exception:  # noqa: BLE001
        return True


def _movers(model: Any) -> list[int]:
    try:
        return [int(seat) for seat in (model.to_move() or [])]
    except Exception:  # noqa: BLE001
        return []


def _actions(model: Any, seat: int) -> list[str]:
    try:
        return [str(action) for action in (model.legal_actions(seat) or [])]
    except Exception:  # noqa: BLE001
        return []


def _child(model: Any, seat: int, action: str) -> Any | None:
    try:
        child = model.clone()
        child.apply(seat, action)
        return child
    except Exception:  # noqa: BLE001
        return None


def _chance_child(model: Any, outcome: str) -> Any | None:
    try:
        child = model.clone()
    except Exception:  # noqa: BLE001
        return None
    if not _apply_chance(child, outcome):
        return None
    return child


def _state_key(model: Any) -> str | None:
    """Prefer `key()`; fall back to `info_state()`. No key means no transposition table."""
    for name in ("key", "info_state"):
        getter = getattr(model, name, None)
        if not callable(getter):
            continue
        try:
            value = getter()
        except Exception:  # noqa: BLE001
            continue
        if value is not None:
            return f"{name}:{value}"
    return None


def _legal_spelling(action: str, legal: list[str] | None) -> str | None:
    """Return the arena's exact spelling of `action`, or None when it is absent."""
    if legal is None:
        return action
    wanted = str(action).strip().lower()
    for item in legal:
        text = str(item)
        if text.strip().lower() == wanted:
            return text
    return None


class ChanceSolver:
    """Expected-value search over a `GameModel` that exposes `chance_actions()`/`apply_chance()`."""

    name = "chance"

    # ------------------------------------------------------------------ routing
    def matches(self, game_id: str, spec: object) -> bool:
        """Conservative claim: only games that explicitly talk about dice/draws/chance.

        This solver cannot inspect the model from `matches()` (the model lives in the sandbox), so a
        false positive would be expensive. A deterministic family or an unrecognised id is rejected;
        `propose()` then declines anyway when the model has no chance surface.
        """
        try:
            family = str(getattr(spec, "family", "") or "").strip().lower()
            if family in _CHANCE_FAMILIES:
                return True
            if family in _DETERMINISTIC_FAMILIES:
                return False
            text = " ".join(
                [
                    str(game_id or ""),
                    str(getattr(spec, "game_id", "") or ""),
                    str(getattr(spec, "action_format", "") or ""),
                    str(getattr(spec, "winning_idea", "") or ""),
                    str(getattr(spec, "notes", "") or ""),
                ]
            )
            # Ids are snake_case; treat '_' and '-' as word separators so `dice_race` matches.
            text = text.replace("_", " ").replace("-", " ")
            return bool(_CHANCE_ID.search(text))
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ public search
    def propose(
        self,
        model: Any,
        seat: int,
        seconds: float = 1.0,
        max_depth: int = 4,
        rng: Any = None,
    ) -> tuple[str | None, float]:
        """Best expected-value action for `seat`, or (None, 0.5) when there is nothing to decide.

        `rng` is used *only* to break exact ties; the returned value is always the expectimax value.
        Never raises.
        """
        try:
            return self._propose(model, int(seat), seconds, max_depth, rng)
        except Exception:  # noqa: BLE001 - a solver bug must cost at most one move
            return None, DRAW

    def _propose(
        self,
        model: Any,
        seat: int,
        seconds: float,
        max_depth: int,
        rng: Any,
    ) -> tuple[str | None, float]:
        if model is None:
            return None, DRAW
        if not callable(getattr(model, "chance_actions", None)):
            # Without the chance surface a deterministic search would silently play the wrong game.
            return None, DRAW

        self.seat = seat
        self.seconds = max(0.02, float(seconds))
        self.max_depth = max(1, int(max_depth))
        self.deadline = time.monotonic() + self.seconds
        self.nodes = 0
        self.table: dict[tuple[str, int], float] = {}

        if seat not in _movers(model):
            return None, DRAW
        actions = _actions(model, seat)
        if not actions:
            return None, DRAW

        ordered = list(actions)
        best_action: str | None = None
        best_value = DRAW
        for depth in range(1, self.max_depth + 1):
            try:
                scores: list[tuple[str, float]] = []
                for action in ordered:
                    child = _child(model, seat, action)
                    if child is None:
                        continue
                    scores.append((action, self._value(child, depth - 1)))
                if not scores:
                    break
                top = max(value for _, value in scores)
                ties = [action for action, value in scores if abs(value - top) <= TIE_EPS]
                best_action = self._tie_break(ties, rng)
                best_value = _clip(top)
                ordered = [action for action, _ in sorted(scores, key=lambda item: -item[1])]
                if best_value >= 1.0 - TIE_EPS or best_value <= TIE_EPS:
                    break  # proven win/loss: deeper search cannot improve the decision
            except _Deadline:
                break

        if best_action is None:
            # Budget spent before one full ply: a legal action beats no action.
            return ordered[0], DRAW
        return best_action, best_value

    def _tie_break(self, actions: list[str], rng: Any) -> str:
        if len(actions) == 1 or rng is None:
            return actions[0]
        try:
            return str(rng.choice(actions))
        except Exception:  # noqa: BLE001
            return actions[0]

    # ------------------------------------------------------------------ expectimax
    def _value(self, model: Any, depth: int) -> float:
        self._tick()
        if _terminal(model):
            return _score(model, self.seat)
        if depth < 0:
            return _heuristic(model, self.seat)

        outcomes = _chance_of(model)
        movers = _movers(model)
        if outcomes and not movers:
            return self._chance_value(model, outcomes, depth)
        if depth <= 0:
            return _heuristic(model, self.seat)
        if len(movers) != 1:
            # Simultaneous games are not a max/min tree; a heuristic is safer than inventing one.
            return _heuristic(model, self.seat)
        return self._decision_value(model, movers[0], depth)

    def _chance_value(
        self, model: Any, outcomes: list[tuple[str, float]], depth: int
    ) -> float:
        total = 0.0
        acc = 0.0
        weighted: list[tuple[Any, float]] = []
        for outcome, weight in outcomes:
            child = _chance_child(model, outcome)
            if child is None:
                continue
            weighted.append((child, weight))
            total += weight
        if total <= 0.0:
            return _heuristic(model, self.seat)
        for child, weight in weighted:
            acc += (weight / total) * self._value(child, depth - 1)
        return _clip(acc)

    def _decision_value(self, model: Any, mover: int, depth: int) -> float:
        actions = _actions(model, mover)
        if not actions:
            return _heuristic(model, self.seat)

        key = _state_key(model)
        table_key = f"{key}|{mover}" if key else None
        if table_key is not None:
            cached = self.table.get((table_key, depth))
            if cached is not None:
                return cached

        maximizing = mover == self.seat
        values: list[float] = []
        for action in actions:
            child = _child(model, mover, action)
            if child is None:
                continue
            values.append(self._value(child, depth - 1))
        if not values:
            return _heuristic(model, self.seat)

        value = _clip(max(values) if maximizing else min(values))
        if table_key is not None:
            self.table[(table_key, depth)] = value
        return value

    # ------------------------------------------------------------------ budget
    def _tick(self) -> None:
        self.nodes += 1
        if (self.nodes & 63) == 0 and time.monotonic() > self.deadline:
            raise _Deadline()
        if self.nodes > NODE_CAP:
            raise _Deadline()


# ---------------------------------------------------------------------- push-your-luck helper
def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except Exception:  # noqa: BLE001
        return None
    return number if number >= 0 else None


def hold_roll_policy(
    model: Any,
    *,
    seat: int = 0,
    legal: list[str] | None = None,
    threshold: float = 20.0,
    roll: str = "roll",
    hold: str = "hold",
) -> str | None:
    """Classic Pig-style threshold rule: hold when the turn total reaches `threshold`.

    This is the cheap one-roll expected-value boundary (the well-known ~20 for a single d6), not a
    game-theoretic optimum: a low target makes rolling correct even above it. Use it only as a
    fallback when `propose()` cannot enumerate chance outcomes. Never raises; None when unknown.

    Numbers are read from `target`, `turn_total`, `banked`/`bank`/`bank_total` attributes when the
    model provides them, otherwise from its `info_state()` text.
    """
    try:
        target = _positive_int(getattr(model, "target", None))
        turn_total = _positive_int(getattr(model, "turn_total", None))
        banked = None
        for name in ("banked", "bank", "bank_total", "scores"):
            value = getattr(model, name, None)
            if value is not None:
                banked = value
                break

        if turn_total is None or banked is None or target is None:
            getter = getattr(model, "info_state", None)
            text = str(getter() or "") if callable(getter) else ""
            if turn_total is None:
                match = _TURN_TOTAL.search(text)
                turn_total = int(match.group(1)) if match else None
            if target is None:
                match = _TARGET.search(text)
                target = int(match.group(1)) if match else None
            if banked is None:
                match = _BANKED_YOU.search(text)
                banked = int(match.group(1)) if match else None

        if turn_total is None or banked is None:
            return None
        if isinstance(banked, (list, tuple)):
            banked = banked[seat] if 0 <= int(seat) < len(banked) else banked[0]
        bank_total = _positive_int(banked)
        if bank_total is None:
            return None

        if target is not None and bank_total + turn_total >= target:
            return _legal_spelling(hold, legal)
        chosen = hold if float(turn_total) >= float(threshold) else roll
        return _legal_spelling(chosen, legal)
    except Exception:  # noqa: BLE001
        return None
