"""Contracts shared by deterministic solvers, generated models and the search engine.

Two roles, deliberately separate:

- `Solver`    a hand-written, family-specific expert (`solver_guess.py`, `solver_pig.py`). It reads the
              observation text and returns a move. Cheap, exact, no model needed.
- `GameModel` the machine-readable game a *generated* module exposes during prep: legal moves, apply,
              terminal test, score. `search.best_action()` plays it. This is the "model writes the
              simulator, code picks the move" pattern the judges score highest.

Everything is duck-typed on purpose: code written by the model in prep cannot import our package, so a
generated module just has to define the methods, and `check_model()` tells us whether it really does.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class ModelError(ValueError):
    """The generated module does not behave like a game model."""


# Generated artifacts live next to each other in /memory/<game_id>/; the fingerprint prefix keeps a model
# built for one rules text from ever being played against another.
ARTIFACT_VERSION = 3
MODEL_SUFFIX = "-model.py"
POLICY_SUFFIX = "-policy.py"


@runtime_checkable
class GameModel(Protocol):
    """A two-player game the solver can search. All methods must be cheap and side-effect free except apply()."""

    def update(self, observation: str) -> None:
        """Parse the arena's turn text into the model's state (called once per turn, before searching)."""

    def legal_actions(self, seat: int) -> list[str] | None:
        """Exact legal action strings for `seat`, or None when there are too many to enumerate."""

    def apply(self, seat: int, action: str) -> None:
        """Play `action` for `seat` and advance the state."""

    def is_terminal(self) -> bool: ...

    def to_move(self) -> list[int]:
        """Seats that must move now. One seat for alternating games, two for simultaneous games."""

    def score(self, seat: int) -> float:
        """At a terminal state: 1.0 win, 0.5 draw, 0.0 loss from `seat`'s point of view."""

    def evaluate(self, seat: int) -> float:
        """Heuristic value of a non-terminal state from `seat`'s point of view (0.5 = neutral)."""

    def clone(self) -> "GameModel":
        """A deep-enough copy: search mutates clones, never the live model."""


@runtime_checkable
class Solver(Protocol):
    """A hand-written expert for one game family."""

    name: str

    @classmethod
    def matches(cls, game_id: str, spec: Any) -> bool:
        """True when this solver is confident it understands the game."""

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        """A legal action string, or None when the observation is not understood."""


_REQUIRED = ("update", "legal_actions", "apply", "is_terminal", "to_move", "score", "clone")


def check_model(candidate: Any, *, need_evaluate: bool = False) -> None:
    """Raise ModelError unless `candidate` quacks like a GameModel. Used to reject generated code early."""
    if candidate is None:
        raise ModelError("no model object")
    missing = [name for name in _REQUIRED if not callable(getattr(candidate, name, None))]
    if need_evaluate and not callable(getattr(candidate, "evaluate", None)):
        missing.append("evaluate")
    if missing:
        raise ModelError("model is missing: " + ", ".join(missing))


def model_key(model: Any, seat: int) -> str:
    """Optional transposition key: models may expose `key()`; otherwise search runs without a TT."""
    getter = getattr(model, "key", None)
    if callable(getter):
        try:
            return f"{seat}:{getter()}"
        except Exception:  # noqa: BLE001 - a generated key() must never break a turn
            return ""
    return ""


def value_of(model: Any, seat: int) -> float:
    """Terminal score clipped to [0, 1]; a generated model returning garbage must not poison the search."""
    try:
        value = float(model.score(seat))
    except Exception:  # noqa: BLE001
        return 0.5
    return min(1.0, max(0.0, value))


def heuristic_of(model: Any, seat: int) -> float:
    getter = getattr(model, "evaluate", None)
    if not callable(getter):
        return 0.5
    try:
        value = float(getter(seat))
    except Exception:  # noqa: BLE001
        return 0.5
    return min(1.0, max(0.0, value))
