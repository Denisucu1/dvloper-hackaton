"""The code half of the decision: what the harness can work out without spending a token.

Order at play time (first hit wins):
1. a hand-written family solver (`solver_guess.py`, `solver_pig.py`) - exact, cheap, deterministic;
2. the generated game model from prep, searched here (`search.best_action`): the "the model writes the
   simulator, code picks the move" pattern. It runs inside the sandbox subprocess, so a broken or
   adversarial module can cost one move, never the game;
3. the generated `choose()` policy from prep (shallow, but better than nothing on unseen games);
4. nothing - the caller asks the model, and `actions.choose_fallback` is the last line of defense.

The opponent's chat claims are passed in here too, so their words genuinely reach the decision logic
(6.5), not just the prompt.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

from . import memory, sandbox
from .actions import normalize
from .game_model import ARTIFACT_VERSION
from .log import log
from .spec import Spec


_SEAT_IN_TEXT = re.compile(r"(?i)\byou are (?:seat|player)\s+(\d+)\b")


def seat_from_observation(observation: str, fallback: int) -> int:
    """The arena sends `seat` only in `start`; turn messages carry it inside the observation text.

    Reading it back here keeps every code path (solver, generated model, generated policy) on the right
    side even if the caller lost the seat. Falls back silently when the text does not say it.
    """
    match = _SEAT_IN_TEXT.search(observation or "")
    if not match:
        return int(fallback)
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return int(fallback)


def _solvers() -> list:
    """Import the family solvers lazily: the harness must keep working if one is missing."""
    found = []
    for module_name, class_name in (
        ("solver_guess", "GuessingSolver"),
        ("solver_pig", "PigSolver"),
        ("solver_simul", "SimultaneousSolver"),
        ("solver_bluff", "BluffSolver"),
    ):
        try:
            module = __import__(f"harness.{module_name}", fromlist=[class_name])
            found.append(getattr(module, class_name)())
        except (ImportError, AttributeError) as exc:
            log(f"solver {module_name} unavailable: {type(exc).__name__}")
    return found


class Engine:
    def __init__(self, spec: Spec | None = None, genpolicy_path: str | None = None, seed: str = "") -> None:
        self.spec = spec or Spec()
        self.seed = seed or self.spec.fingerprint or "0"
        self.generated_failures = 0
        self.model_failures = 0
        self.solver_failures = 0
        self._seat_logged = False
        self.genpolicy_path = genpolicy_path
        self.model_path: str | None = None
        self.strategy_hint = ""
        self.rules = ""
        self._load_generated()
        self._solvers = _solvers()

    # ------------------------------------------------------------------ artifacts
    def _load_generated(self) -> None:
        """Find the model/policy a prep (possibly in another container) left in /memory for this game."""
        try:
            directory = memory.game_dir(self.spec.game_id)
            manifest = memory.read_json(directory / "manifest.json") or {}
        except OSError:
            return
        if manifest.get("artifact_version") != ARTIFACT_VERSION:
            return
        name = str(manifest.get("genmodel") or "")
        candidate = directory / name
        if name and candidate.name == name and candidate.exists():
            self.model_path = str(candidate)
        self.strategy_hint = str(manifest.get("strategy") or "")
        if not self.genpolicy_path:
            policy_name = str(manifest.get("genpolicy") or "")
            policy = directory / policy_name
            if policy_name and policy.name == policy_name and policy.exists():
                self.genpolicy_path = str(policy)
        self.rules = memory.read_text(directory / "rules.txt")
        if self.model_path:
            log(f"engine: generated model {os.path.basename(self.model_path)}")

    # ------------------------------------------------------------------ public
    def propose(
        self,
        *,
        observation: str,
        legal: Iterable[str] | None,
        seat: int,
        turn: int,
        avoid: Iterable[str] | None = None,
        claims: Iterable[str] | None = None,
        timeout: float = 4.0,
    ) -> tuple[str | None, str]:
        """Returns (action or None, a short intent the chat line must be consistent with)."""
        legal_list = [str(item) for item in legal] if legal else None

        announced = seat_from_observation(observation, seat)
        if announced != seat:
            if not self._seat_logged:
                log(f"seat from observation: {announced} (caller said {seat})")
                self._seat_logged = True
            seat = announced

        action = self._ask_solver(observation, legal_list, seat, turn, timeout)
        if action:
            return action, self._intent(action, claims, source=f"{self._solver_name} solver")

        if self.model_path and self.model_failures < 3:
            budget = min(3.0, max(1.0, timeout * 0.7))
            # Strategy routing inside the sandbox: the generated model never runs in our process.
            strategy, matrix_path, chance_path = "search", "", ""
            if _flag("HARNESS_TOURNAMENT") and self.strategy_hint in ("matrix", "chance"):
                strategy = self.strategy_hint
                matrix_path = _sibling("solver_matrix.py") if strategy == "matrix" else ""
                chance_path = _sibling("solver_chance.py") if strategy == "chance" else ""
            if strategy == "search" and _flag("HARNESS_MATRIX") and (
                    self.spec.simultaneous or self.spec.family == "read_opponent"):
                matrix_path = _sibling("solver_matrix.py")
                if matrix_path:
                    strategy = "matrix"
            if strategy == "search" and _flag("HARNESS_CHANCE"):
                chance_path = _sibling("solver_chance.py")
                if chance_path:
                    strategy = "auto"
            meta = sandbox.call_model_search(
                self.model_path, _search_path(), self.rules, observation, legal_list, seat,
                timeout=budget, depth=8, strategy=strategy,
                matrix_path=matrix_path, chance_path=chance_path,
            )
            action = meta.get("action")
            if action:
                log(f"genmodel search: {action} route={meta.get('route')} "
                    f"value={float(meta.get('value') or 0.5):.2f} depth={meta.get('depth')}")
                self.model_failures = 0  # a success re-arms the model after a bad turn
                return action, self._intent(action, claims, source="generated model search")
            self.model_failures += 1
            log(f"generated model fallback #{self.model_failures}: route={meta.get('route')} "
                f"reason={meta.get('reason')}")

        if self.genpolicy_path and self.generated_failures < 3:
            action = sandbox.call_choose(
                self.genpolicy_path, observation, legal_list, seat, timeout=min(3.0, timeout)
            )
            if action:
                return action, self._intent(action, claims, source="generated policy")
            self.generated_failures += 1
            log(f"genpolicy fallback #{self.generated_failures}")

        # Only supply a code move proposal when a verified solver generated one;
        # avoid anchoring the model on a random legal move.
        return None, self._intent("", claims, source="model")

    # ------------------------------------------------------------------ solvers
    @property
    def _solver_name(self) -> str:
        return getattr(self, "_solver_label", "family")

    def _ask_solver(
        self, observation: str, legal: list[str] | None, seat: int, turn: int, timeout: float
    ) -> str | None:
        if self.solver_failures >= 3:
            return None
        game_id = str(self.spec.game_id or "")
        for solver in self._solvers:
            try:
                if not solver.matches(game_id, self.spec):
                    continue
                if hasattr(solver, "configure"):
                    solver.configure(
                        self.spec,
                        self.rules,
                        has_model=bool(self.model_path and self.model_failures < 3),
                    )
                action = solver.propose(
                    observation=observation, legal=legal, seat=seat, turn=turn,
                    timeout=max(0.2, min(1.0, timeout)),
                )
            except Exception as exc:  # noqa: BLE001 - an expert bug must not cost a move
                log(f"solver {getattr(solver, 'name', '?')} crashed: {type(exc).__name__}: {exc}")
                self.solver_failures += 1
                continue
            verified = normalize(action, legal) if action else None
            if verified:
                self._solver_label = getattr(solver, "name", "family")
                log(f"solver {self._solver_label}: {verified}")
                return verified
            if action:
                log(f"solver {getattr(solver, 'name', '?')} proposed {action!r}, not legal; ignored")
        return None

    # ------------------------------------------------------------------ hints
    def _intent(self, action: str, claims: Iterable[str] | None, source: str) -> str:
        claim_list = [str(item) for item in (claims or []) if str(item).strip()]
        parts = [f"{source}: {'play ' + action if action else 'no candidate computed'}"]
        if claim_list:
            # Concrete and private-state-free: the opponent's words become part of our reasoning.
            parts.append(f"react to their claim {claim_list[-1][:70]!r} without trusting it")
        if self.spec.family == "bluff":
            parts.append("sound confident either way")
        elif self.spec.family == "deals":
            parts.append("keep the negotiation open")
        elif self.spec.family == "truth":
            parts.append("offer only what the document supports")
        elif self.spec.family == "guardrails":
            parts.append("stay on allowed topics")
        return "; ".join(parts)[:240]


def _search_path() -> str:
    """`search.py` next to this file; it is loaded standalone inside the sandbox (no package imports)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "search.py")


def _sibling(name: str) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    return path if os.path.exists(path) else ""


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"
