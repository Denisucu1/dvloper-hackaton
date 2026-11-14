"""Probe: the generic push-your-luck expert (``harness/solver_luck.py``).

Inline Pig-shaped observations (the same text shape the arena sends):

- one die, target 30: a low turn total rolls, a turn total that wins on hold holds;
- the exact one-die EV boundary at this target is checked explicitly: turn total 19 rolls, 20 holds;
- two dice with a snake-eyes reset: returns a legal action and preserves the arena spelling;
- opponent-turn text and unreadable text return ``None`` (never a guess);
- 200 decisions average well under the 5 ms per-move requirement.

Score is the fraction of eight boolean checks that pass.
"""
from __future__ import annotations

import time

from harness.solver_luck import LuckSolver

from scripts.probes.base import Probe, collect, register, summarize

TIMING_ITERATIONS = 20
TIMING_LIMIT_MS = 5.0


def one_die(banked: int, opponent: int, turn_total: int, target: int = 30) -> str:
    return (
        f"PIG - One die, first to {target}. You are seat 0. Target: {target}.\n"
        f"Banked: you {banked}, opponent {opponent}.\n"
        f"Turn 6: it is YOUR turn. Turn total: {turn_total} (rolls this turn: none yet).\n"
        f"If you hold now you will have {banked + turn_total}. Choose: roll | hold"
    )


def two_dice(banked: int, opponent: int, turn_total: int, target: int = 60) -> str:
    return (
        f"PIG - Two dice, first to {target} (snake eyes reset the bank). You are seat 0. Target: {target}.\n"
        f"Banked: you {banked}, opponent {opponent}.\n"
        f"Turn 9: it is YOUR turn. Turn total: {turn_total} (rolls this turn: none yet).\n"
        f"If you hold now you will have {banked + turn_total}. Choose: roll | hold"
    )


@register
class LuckProbe(Probe):
    name = "solvers.luck"
    component = "luck"

    def run(self) -> dict:
        solver = LuckSolver()
        facts: dict = {}

        def decide(observation: str, legal: list[str] | None = None, seat: int = 0):
            return solver.propose(observation=observation, legal=legal, seat=seat)

        def low_turn_total_rolls() -> bool:
            action = decide(one_die(0, 0, 5))
            facts["low_turn"] = action
            return action == "roll"

        def winning_hold_holds() -> bool:
            action = decide(one_die(25, 0, 5))  # 25 + 5 reaches target 30
            facts["winning_hold"] = action
            return action == "hold"

        def boundary_19_rolls() -> bool:
            action = decide(one_die(0, 0, 19))
            facts["boundary_19"] = action
            return action == "roll"

        def boundary_20_holds() -> bool:
            action = decide(one_die(0, 0, 20))
            facts["boundary_20"] = action
            return action == "hold"

        def two_dice_returns_legal_spelling() -> bool:
            legal = ["ROLL", "HOLD"]
            action = decide(two_dice(22, 8, 0), legal=legal)
            facts["two_dice"] = action
            return action in legal

        def opponent_turn_is_declined() -> bool:
            observation = one_die(0, 0, 5).replace(
                "it is YOUR turn", "it is the opponent's turn"
            )
            action = decide(observation)
            facts["opponent_turn"] = action
            return action is None

        def unreadable_text_is_declined() -> bool:
            samples = ["no idea what is happening here", "", "Turn total: 12, score 4, target 30"]
            actions = [decide(sample) for sample in samples]
            facts["unreadable"] = actions
            return all(action is None for action in actions)

        def decisions_are_under_five_ms() -> bool:
            observation = one_die(0, 0, 19)
            latencies = []
            for _ in range(TIMING_ITERATIONS):
                started = time.perf_counter()
                decide(observation, legal=["roll", "hold"])
                latencies.append((time.perf_counter() - started) * 1000.0)
            per_call_ms = sum(latencies) / len(latencies)
            slowest_ms = max(latencies)
            facts["per_call_ms"] = round(per_call_ms, 4)
            facts["slowest_call_ms"] = round(slowest_ms, 4)
            return slowest_ms < TIMING_LIMIT_MS

        run = collect([
            ("low turn total rolls", low_turn_total_rolls),
            ("holding for the win holds", winning_hold_holds),
            ("one-die boundary: 19 rolls", boundary_19_rolls),
            ("one-die boundary: 20 holds", boundary_20_holds),
            ("two-dice decision is legal", two_dice_returns_legal_spelling),
            ("opponent turn returns None", opponent_turn_is_declined),
            ("unreadable text returns None", unreadable_text_is_declined),
            (f"decision under {TIMING_LIMIT_MS:.0f} ms", decisions_are_under_five_ms),
        ])
        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"boundary {facts.get('boundary_19')}/{facts.get('boundary_20')}, "
            f"two-dice {facts.get('two_dice')}, "
            f"{facts.get('per_call_ms')} ms/call (max {facts.get('slowest_call_ms')})"
        )
        return summarize(run, notes=notes, details=details)
