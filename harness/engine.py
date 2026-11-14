"""The code half of the decision: what the harness can work out without spending a token.

Order at play time:
1. a policy module the model wrote in prep (validated, sandboxed) - the "model writes a solver, code picks
   the move" pattern that scores best on harness design;
2. a deterministic safe heuristic over `legal_actions` (never time out, never repeat forever);
3. nothing - the caller asks the model, and `actions.choose_fallback` is the last line of defense.

The opponent's chat claims are passed in here too, so the words genuinely reach the decision logic (6.5),
not just the prompt.
"""
from __future__ import annotations

import random
from typing import Iterable

from . import sandbox
from .actions import normalize
from .log import log
from .spec import Spec


class Engine:
    def __init__(self, spec: Spec | None = None, genpolicy_path: str | None = None, seed: str = "") -> None:
        self.spec = spec or Spec()
        self.genpolicy_path = genpolicy_path
        self.seed = seed or self.spec.fingerprint or "0"
        self.generated_failures = 0

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

        if self.genpolicy_path and self.generated_failures < 3:
            action = sandbox.call_choose(
                self.genpolicy_path, observation, legal_list, seat, timeout=min(4.0, timeout)
            )
            if action:
                return action, self._intent(action, claims, source="generated policy")
            self.generated_failures += 1
            log(f"genpolicy fallback #{self.generated_failures}")

        if legal_list:
            action = self._heuristic(legal_list, turn, seat, avoid)
            if action:
                return action, self._intent(action, claims, source="heuristic")
        return None, self._intent("", claims, source="model")

    # ------------------------------------------------------------------ hints
    def _heuristic(self, legal: list[str], turn: int, seat: int, avoid: Iterable[str] | None) -> str | None:
        if not legal:
            return None
        seen = {str(item) for item in (avoid or [])}
        fresh = [item for item in legal if item not in seen] or legal
        # Seeded by turn+seat: varied enough to look alive, stable enough to debug with a seed.
        rng = random.Random(f"{self.seed}:{turn}:{seat}")
        return normalize(rng.choice(fresh), legal)

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
