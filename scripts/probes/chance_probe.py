"""Probe: the expectimax chance solver (``harness/solver_chance.py``).

Fixture: a one-roll Pig-like decision. We hold with a banked total, or roll one d6: a 1 busts (loss),
2-6 either reach the target (win) or not. The exact one-step DP is computed inside the probe, so the
solver's decision and expected value are compared against the analytic optimum on five states that
cover roll and hold.

Also checked: a model without ``chance_actions()`` must return ``(None, 0.5)`` (safe fallback, no
deterministic search of the wrong game), and all five decisions together must take under 0.5 s.
Score is the fraction of boolean checks that pass.
"""
from __future__ import annotations

import time

from harness.solver_chance import ChanceSolver

from scripts.probes.base import Probe, collect, register, summarize

STATES = [(4, 3), (5, 5), (8, 7), (8, 8), (10, 6)]
BANKED = 0


class OneRollPig:
    """Push-your-luck turn with exactly one possible roll, then a terminal win/loss."""

    def __init__(self, target: int, turn_total: int, phase: str = "decision", result: int | None = None) -> None:
        self.target = int(target)
        self.turn_total = int(turn_total)
        self.banked = BANKED
        self.phase = phase
        self.result = result

    def update(self, observation: str) -> None:
        return None

    def legal_actions(self, seat: int) -> list[str]:
        return ["roll", "hold"] if self.phase == "decision" else []

    def apply(self, seat: int, action: str) -> None:
        if action == "hold":
            self.phase = "done"
            self.result = 1 if self.banked + self.turn_total >= self.target else 0
        elif action == "roll":
            self.phase = "chance"
        else:
            raise ValueError(f"unknown action {action!r}")

    def is_terminal(self) -> bool:
        return self.phase == "done"

    def to_move(self) -> list[int]:
        return [0] if self.phase == "decision" else []

    def chance_actions(self) -> list[tuple[str, float]]:
        if self.phase != "chance":
            return []
        return [(str(face), 1.0 / 6.0) for face in range(1, 7)]

    def apply_chance(self, outcome: str) -> None:
        face = int(outcome)
        self.phase = "done"
        if face == 1:
            self.result = 0  # bust: the turn total is lost
        else:
            self.result = 1 if self.banked + self.turn_total + face >= self.target else 0

    def score(self, seat: int) -> float:
        value = float(self.result or 0)
        return value if int(seat) == 0 else 1.0 - value

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "OneRollPig":
        return OneRollPig(self.target, self.turn_total, self.phase, self.result)

    def info_state(self) -> str:
        return f"target {self.target} turn {self.turn_total} phase {self.phase}"


class NoChanceGame:
    """A plain deterministic model: the chance solver must decline rather than mis-search it."""

    def legal_actions(self, seat: int) -> list[str]:
        return ["roll", "hold"]

    def apply(self, seat: int, action: str) -> None:
        return None

    def is_terminal(self) -> bool:
        return False

    def to_move(self) -> list[int]:
        return [0]

    def score(self, seat: int) -> float:
        return 0.5

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "NoChanceGame":
        return NoChanceGame()


def one_roll_dp(target: int, turn_total: int) -> tuple[str, float]:
    """Exact optimum for the fixture: hold value versus expected value of one roll."""
    hold = 1.0 if BANKED + turn_total >= target else 0.0
    roll = 0.0
    for face in range(1, 7):
        if face == 1:
            continue  # bust
        roll += (1.0 if BANKED + turn_total + face >= target else 0.0) / 6.0
    return ("roll" if roll > hold else "hold"), max(roll, hold)


@register
class ChanceProbe(Probe):
    name = "solvers.chance"
    component = "chance"

    def run(self) -> dict:
        solver = ChanceSolver()
        facts: dict = {}
        state_checks = []

        for target, turn_total in STATES:
            expected_action, expected_value = one_roll_dp(target, turn_total)

            def make_check(target=target, turn_total=turn_total,
                           expected_action=expected_action, expected_value=expected_value):
                def check() -> bool:
                    action, value = solver.propose(
                        OneRollPig(target, turn_total), 0, seconds=2.0, max_depth=2,
                    )
                    facts[f"state_{target}_{turn_total}"] = {
                        "action": action, "value": round(value, 6),
                        "dp_action": expected_action, "dp_value": round(expected_value, 6),
                    }
                    return action == expected_action and abs(value - expected_value) < 1e-6
                return check

            state_checks.append((f"state target={target} turn={turn_total} matches DP", make_check()))

        def no_chance_surface_falls_back() -> bool:
            result = solver.propose(NoChanceGame(), 0, seconds=1.0)
            facts["no_chance"] = result
            return result == (None, 0.5)

        def decisions_are_fast() -> bool:
            started = time.monotonic()
            for target, turn_total in STATES:
                solver.propose(OneRollPig(target, turn_total), 0, seconds=2.0, max_depth=2)
            elapsed = time.monotonic() - started
            facts["five_decisions_s"] = round(elapsed, 4)
            return elapsed < 0.5

        run = collect(state_checks + [
            ("missing chance surface returns (None, 0.5)", no_chance_surface_falls_back),
            ("five decisions take under 0.5 s", decisions_are_fast),
        ])
        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"{len(STATES)} DP states, no-chance {facts.get('no_chance')}, "
            f"{facts.get('five_decisions_s')} s"
        )
        return summarize(run, notes=notes, details=details)
