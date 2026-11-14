"""Probe: engine routing and the legal-action verifier.

Uses a private ``MEMORY_DIR`` (patched to an empty temp dir) so no prep artifact from the real run
can leak into the result. Six checks:

1. the first matching solver wins (precedence);
2. a solver that proposes an illegal action is ignored, not emitted;
3. a real family solver (Pig) produces a verified move through the engine;
4. the seat is re-read from the observation and passed to the solver;
5. when nothing is available the engine returns no proposal and the fallback is legal;
6. three solver crashes disable the solver path (the failure budget), never raise.
"""
from __future__ import annotations

import shutil
import tempfile
from unittest import mock

from harness import actions, config
from harness.engine import Engine
from harness.spec import Spec

from scripts.probes.base import Probe, collect, register, summarize


class StubSolver:
    """A controllable solver: returns one action and records the seat it was asked about."""

    def __init__(self, name: str, action: str) -> None:
        self.name = name
        self.action = action
        self.seats: list[int] = []

    def matches(self, game_id: str, spec: object) -> bool:
        return True

    def propose(self, **kwargs) -> str:
        self.seats.append(int(kwargs.get("seat", -1)))
        return self.action


class CrashSolver(StubSolver):
    def __init__(self) -> None:
        super().__init__("crash", "")
        self.calls = 0

    def propose(self, **kwargs) -> str:
        self.calls += 1
        raise RuntimeError("solver bug")


PIG_TURN = (
    "PIG - one die. Target: 100\n"
    "Banked: you 0, opponent 20\n"
    "Turn total: 3\n"
    "It is your turn.\n"
)


@register
class EngineProbe(Probe):
    name = "engine.routing"
    component = "engine"

    def run(self) -> dict:
        facts: dict = {}
        root = tempfile.mkdtemp(prefix="adn-probe-engine-")
        try:
            with mock.patch.object(config, "MEMORY_DIR", root):
                def precedence() -> bool:
                    spec = Spec(game_id="probe_game")
                    engine = Engine(spec, seed="probe")
                    engine._solvers = [StubSolver("first", "A1"), StubSolver("second", "B2")]
                    action, intent = engine.propose(observation="board", legal=["A1", "B2"], seat=0, turn=1)
                    facts["precedence"] = {"action": action, "intent": intent}
                    return action == "A1" and "first solver" in intent

                def illegal_is_rejected() -> bool:
                    spec = Spec(game_id="probe_game")
                    engine = Engine(spec, seed="probe")
                    engine._solvers = [StubSolver("bad", "Z9")]
                    action, intent = engine.propose(observation="board", legal=["A1", "B2"], seat=0, turn=1)
                    facts["illegal"] = {"action": action, "intent": intent}
                    return action is None and "no candidate computed" in intent

                def family_solver_is_verified() -> bool:
                    spec = Spec(game_id="pig", family="pig")
                    engine = Engine(spec, seed="probe")
                    action, intent = engine.propose(observation=PIG_TURN, legal=["roll", "hold"], seat=0, turn=1)
                    facts["pig"] = {"action": action, "intent": intent}
                    return action == "roll" and "pig solver" in intent

                def seat_comes_from_observation() -> bool:
                    spec = Spec(game_id="probe_game")
                    engine = Engine(spec, seed="probe")
                    solver = StubSolver("seat-spy", "seat1")
                    engine._solvers = [solver]
                    action, _intent = engine.propose(
                        observation="You are seat 1. Your move.",
                        legal=["seat0", "seat1"], seat=0, turn=1,
                    )
                    facts["seat"] = {"action": action, "asked": solver.seats}
                    return action == "seat1" and bool(solver.seats) and solver.seats[-1] == 1

                def nothing_available_falls_back() -> bool:
                    spec = Spec(game_id="probe_game")
                    engine = Engine(spec, seed="probe")
                    engine._solvers = []
                    action, intent = engine.propose(observation="board", legal=["A1", "B2"], seat=0, turn=1)
                    fallback = actions.choose_fallback(["A1", "B2"], [])
                    facts["fallback"] = {"action": action, "intent": intent, "fallback": fallback}
                    return action is None and "no candidate computed" in intent and fallback == "A1"

                def crash_budget_stops_calling() -> bool:
                    spec = Spec(game_id="probe_game")
                    engine = Engine(spec, seed="probe")
                    solver = CrashSolver()
                    engine._solvers = [solver]
                    results = [
                        engine.propose(observation="board", legal=["A1", "B2"], seat=0, turn=turn)[0]
                        for turn in range(1, 5)
                    ]
                    facts["crash_budget"] = {"calls": solver.calls, "actions": results}
                    return solver.calls == 3 and all(action is None for action in results)

                checks = [
                    ("first matching solver wins", precedence),
                    ("illegal solver move is rejected", illegal_is_rejected),
                    ("family solver move is verified", family_solver_is_verified),
                    ("seat is re-read from the observation", seat_comes_from_observation),
                    ("nothing available returns no proposal, fallback is legal", nothing_available_falls_back),
                    ("three solver crashes disable the solver path", crash_budget_stops_calling),
                ]
                run = collect(checks)
        finally:
            shutil.rmtree(root, ignore_errors=True)

        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"pig verified {facts.get('pig', {}).get('action')}, "
            f"crash budget {facts.get('crash_budget', {}).get('calls', 0)} calls, "
            f"fallback {facts.get('fallback', {}).get('fallback')}"
        )
        return summarize(run, notes=notes, details=details)
