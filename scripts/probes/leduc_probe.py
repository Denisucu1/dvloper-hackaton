"""Probe: the Leduc poker expert (``harness/solver_leduc.py``).

Inline observations copy the arena shape used by ``tests/test_solver_leduc.py``:

- round 1 parses private card, pot and an empty current-round line;
- round 2 parses the community card and a ``check, bet`` line (facing a bet);
- the arena's exact legal spelling is preserved (``Check``/``Bet`` -> ``Bet``);
- round-1 equity enumerates the opponent card plus the community draw: private J calls
  a bet (0.30 equity against 0.25 pot odds), private K calls and bets (0.70);
- round 2 is exact over the known board: private J with a higher board card folds to a bet
  (0.125, the kicker loses the tie); private Q with board K is 0.625;
- Kuhn / one-card text is declined by both the parser and ``matches``;
- 20 decisions stay under the 5 ms per-move requirement.

Score is the fraction of nine boolean checks that pass.
"""
from __future__ import annotations

import time

from harness.solver_leduc import LeducSolver, parse_leduc_observation
from harness.spec import Spec

from scripts.probes.base import Probe, collect, register, summarize

TIMING_ITERATIONS = 20
TIMING_LIMIT_MS = 5.0


def round_one(*, card: str = "Q", seat: int = 0, pot: int = 2,
              line: str = "(no action yet)") -> str:
    return (
        f"LEDUC POKER (J, Q, K twice, 2 betting rounds). You are Player {seat}.\n"
        "Hand 1 of 1. Round 1 of 2.\n"
        f"Your private card: {card}\n"
        "Community card: (not dealt yet)\n"
        f"Pot: {pot}. Chips: you -1, opponent -1.\n"
        f"This round so far: {line}\n"
        'It is YOUR turn: "check" or "bet".'
    )


def round_two(*, card: str = "Q", community: str = "K", seat: int = 0, pot: int = 4,
              line: str = "(no action yet)") -> str:
    return (
        f"LEDUC POKER (J, Q, K twice, 2 betting rounds). You are Player {seat}.\n"
        "Hand 1 of 1. Round 2 of 2.\n"
        f"Your private card: {card}\n"
        f"Community card: {community}\n"
        "Earlier rounds: check, check\n"
        f"Pot: {pot}. Chips: you -2, opponent -2.\n"
        f"This round so far: {line}\n"
        'It is YOUR turn: "call" or "fold".'
    )


KUHN_TEXT = (
    "KUHN POKER - one card, one betting round. You are Player 0.\n"
    "Your card: K\n"
    "This round so far: (no action yet)\n"
    'It is YOUR turn: "check" or "bet".'
)


@register
class LeducProbe(Probe):
    name = "solvers.leduc"
    component = "leduc"

    def run(self) -> dict:
        solver = LeducSolver()
        facts: dict = {}

        def first_round_parses() -> bool:
            turn = parse_leduc_observation(round_one(card="K", pot=2), ["check", "bet"], 0)
            values = None if turn is None else (
                turn.seat, turn.private, turn.community, turn.round, turn.pot, turn.line,
            )
            facts["round_one"] = values
            return values == (0, "K", None, 1, 2, ())

        def second_round_parses_community_card() -> bool:
            turn = parse_leduc_observation(
                round_two(card="Q", community="Q", pot=3, line="check, bet"),
                ["call", "fold"], 1,
            )
            values = None if turn is None else (
                turn.private, turn.community, turn.round, turn.pot, turn.line, turn.facing_bet,
            )
            facts["round_two"] = values
            return values == ("Q", "Q", 2, 3, ("check", "bet"), True)

        def legal_spelling_is_preserved() -> bool:
            action = solver.propose(
                observation=round_one(card="K"), legal=["Check", "Bet"], seat=0,
            )
            facts["spelling"] = action
            return action == "Bet"

        def round_one_jack_calls_bet() -> bool:
            # Round-1 equity enumerates the opponent card and the community draw: J is 0.30,
            # which clears the 0.25 pot odds of a call.
            action = solver.propose(
                observation=round_one(card="J", pot=3, line="bet"),
                legal=["call", "fold"], seat=1,
            )
            facts["jack_call"] = action
            return action == "call"

        def strong_hand_calls_bet() -> bool:
            action = solver.propose(
                observation=round_one(card="K", pot=3, line="bet"),
                legal=["call", "fold"], seat=1,
            )
            facts["strong_call"] = action
            return action == "call"

        def strong_hand_bets_when_first() -> bool:
            action = solver.propose(
                observation=round_one(card="K"), legal=["check", "bet"], seat=0,
            )
            facts["strong_bet"] = action
            return action == "bet"

        def round_two_dominated_hand_folds() -> bool:
            # Private J with Q on the board is dominated by every Q and K; exact round-2 equity
            # is 0.125, below the pot odds, so it folds to a bet.
            action = solver.propose(
                observation=round_two(card="J", community="Q", pot=4, line="bet"),
                legal=["call", "fold"], seat=0,
            )
            facts["dominated_fold"] = action
            return action == "fold"

        def kuhn_text_is_declined() -> bool:
            action = solver.propose(observation=KUHN_TEXT, legal=["check", "bet"], seat=0)
            matched = solver.matches("tg_kuhn_probe", Spec(family="bluff"))
            facts["kuhn"] = {"action": action, "matched": matched}
            return action is None and not matched

        def decisions_are_under_five_ms() -> bool:
            observation = round_one(card="Q")
            latencies = []
            for turn in range(TIMING_ITERATIONS):
                started = time.perf_counter()
                solver.propose(
                    observation=observation, legal=["check", "bet"], seat=0, turn=turn,
                )
                latencies.append((time.perf_counter() - started) * 1000.0)
            per_call_ms = sum(latencies) / len(latencies)
            slowest_ms = max(latencies)
            facts["per_call_ms"] = round(per_call_ms, 4)
            facts["slowest_call_ms"] = round(slowest_ms, 4)
            return slowest_ms < TIMING_LIMIT_MS

        run = collect([
            ("round 1 parses private card and pot", first_round_parses),
            ("round 2 parses the community card", second_round_parses_community_card),
            ("exact legal spelling is preserved", legal_spelling_is_preserved),
            ("round-1 J calls a bet", round_one_jack_calls_bet),
            ("round-1 K calls a bet", strong_hand_calls_bet),
            ("round-1 K bets when first", strong_hand_bets_when_first),
            ("round-2 J with Q board folds to a bet", round_two_dominated_hand_folds),
            ("Kuhn / one-card text is declined", kuhn_text_is_declined),
            (f"decision under {TIMING_LIMIT_MS:.0f} ms", decisions_are_under_five_ms),
        ])
        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"r1 J {facts.get('jack_call')}, K {facts.get('strong_call')}/{facts.get('strong_bet')}, "
            f"r2 J/Q {facts.get('dominated_fold')}; {float(facts.get('per_call_ms') or 0):.3f} ms/call"
        )
        return summarize(run, notes=notes, details=details)
