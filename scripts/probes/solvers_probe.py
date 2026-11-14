"""Probe: the four deterministic family solvers (no arena, no model, no network).

Small, exact fixtures per solver:

- ``GuessingSolver``: binary-search midpoint after higher/lower feedback, interval pinning after a
  correct guess, refusal on the opponent's turn, and exact legal-action spelling;
- ``PigSolver``: expected-value roll, hold when holding wins, case-preserving legal spelling;
- ``BluffSolver``: pot-odds call with the best card, fold with the worst card facing a bet;
- ``SimultaneousSolver``: weighted-Nash distribution shape (deterministic), genuinely mixed sampling
  (statistical, 150 draws; this solver is allowed CSPRNG entropy by design) and refusal to randomize
  numeric bids.

Score is the fraction of boolean checks that pass. Expected simulator randomness is the only
non-deterministic input; its thresholds are far outside any realistic flake (each sign < 5 of 150).
"""
from __future__ import annotations

from collections import Counter

from harness.solver_bluff import BluffSolver
from harness.solver_guess import GuessingSolver
from harness.solver_pig import PigSolver
from harness.solver_simul import SimultaneousSolver, _rps_distribution
from harness.spec import Spec

from scripts.probes.base import Probe, collect, register, summarize

GUESS_LEGAL = [f"guess {number}" for number in range(1, 101)]
PIG_LEGAL = ["roll", "hold"]
BLUFF_LEGAL = ["check", "bet", "call", "fold"]


def _guess_obs(feedback_line: str, bounds: str = "1-100") -> str:
    return (
        f"GUESS THE NUMBER ({bounds}). It is your turn.\n"
        f"Your guesses at the opponent's number:\n{feedback_line}\n"
        "Opponent's guesses at your number:\n"
    )


def _pig_obs(banked: str, turn_total: int) -> str:
    return (
        "PIG - one die. Target: 100\n"
        f"Banked: you {banked}, opponent 20\n"
        f"Turn total: {turn_total}\n"
        "It is your turn.\n"
    )


def _bluff_obs(card: int) -> str:
    return (
        "BLUFF (cards 1-4).\n"
        f"Your card: {card}\n"
        "This hand so far: opponent bet\n"
        "Earlier hands:\n"
        "Hand 1: bet call -> showdown\n"
    )


@register
class SolversProbe(Probe):
    name = "solvers.family"
    component = "solvers"

    def run(self) -> dict:
        guess = GuessingSolver()
        pig = PigSolver()
        bluff = BluffSolver()
        simul = SimultaneousSolver()
        facts: dict = {}

        def guess_midpoint() -> bool:
            action = guess.propose(
                observation=_guess_obs("you guessed 50 -> higher"),
                legal=GUESS_LEGAL, seat=0,
            )
            facts["guess_midpoint"] = action
            return action == "guess 75"

        def guess_lower_half() -> bool:
            action = guess.propose(
                observation=_guess_obs("you guessed 75 -> lower"),
                legal=GUESS_LEGAL, seat=0,
            )
            facts["guess_lower"] = action
            return action == "guess 37"

        def guess_pinned() -> bool:
            action = guess.propose(
                observation=_guess_obs("you guessed 42 -> correct"),
                legal=GUESS_LEGAL, seat=0,
            )
            return action == "guess 42"

        def guess_respects_turn() -> bool:
            observation = _guess_obs("you guessed 50 -> higher") + "It is the opponent's turn.\n"
            return guess.propose(observation=observation, legal=GUESS_LEGAL, seat=0) is None

        def guess_legal_spelling() -> bool:
            action = guess.propose(
                observation=_guess_obs("you guessed 50 -> higher"),
                legal=["guess 74", "guess 75"], seat=0,
            )
            return action == "guess 75"

        def guess_matches() -> bool:
            return GuessingSolver.matches("guessing_number", Spec(family="guessing"))

        def pig_rolls_on_ev() -> bool:
            action = pig.propose(observation=_pig_obs("0", 3), legal=PIG_LEGAL, seat=0)
            facts["pig_roll"] = action
            return action == "roll"

        def pig_holds_a_win() -> bool:
            action = pig.propose(observation=_pig_obs("95", 10), legal=PIG_LEGAL, seat=0)
            facts["pig_hold"] = action
            return action == "hold"

        def pig_legal_spelling() -> bool:
            action = pig.propose(observation=_pig_obs("0", 3), legal=["Roll", "Hold"], seat=0)
            return action == "Roll"

        def pig_matches() -> bool:
            return PigSolver.matches("pig", Spec(family="pig"))

        def bluff_calls_best_card() -> bool:
            action = bluff.propose(observation=_bluff_obs(4), legal=BLUFF_LEGAL, seat=0)
            facts["bluff_call"] = action
            return action == "call"

        def bluff_folds_worst_card() -> bool:
            action = bluff.propose(observation=_bluff_obs(1), legal=BLUFF_LEGAL, seat=0)
            facts["bluff_fold"] = action
            return action == "fold"

        def bluff_matches() -> bool:
            return BluffSolver.matches("bluff_game", Spec(family="bluff"))

        def simul_matches() -> bool:
            return simul.matches("rps", Spec(family="read_opponent"))

        def simul_distribution_is_deterministic() -> bool:
            first = _rps_distribution("", "Choose your sign. Round 1")
            second = _rps_distribution("", "Choose your sign. Round 1")
            facts["simul_distribution"] = [round(value, 4) for value in first]
            return (
                len(first) == 3
                and all(value > 0 for value in first)
                and abs(sum(first) - 1.0) < 1e-9
                and first == second
            )

        def simul_mixes() -> bool:
            picks = Counter(
                simul.propose(
                    observation="Choose rock, paper or scissors. Round 1",
                    legal=["rock", "paper", "scissors"], seat=0,
                )
                for _ in range(150)
            )
            facts["simul_picks"] = dict(picks)
            return (
                set(picks) == {"rock", "paper", "scissors"}
                and min(picks.values()) >= 5
            )

        def simul_refuses_numeric_bids() -> bool:
            return simul.propose(
                observation="Submit your secret bid.",
                legal=["bid 1", "bid 2", "bid 3"], seat=0,
            ) is None

        checks = [
            ("guess: midpoint after higher", guess_midpoint),
            ("guess: midpoint after lower", guess_lower_half),
            ("guess: pinned interval returns the number", guess_pinned),
            ("guess: respects the opponent's turn", guess_respects_turn),
            ("guess: exact legal spelling", guess_legal_spelling),
            ("guess: matches its family", guess_matches),
            ("pig: rolls on positive EV", pig_rolls_on_ev),
            ("pig: holds when holding wins", pig_holds_a_win),
            ("pig: preserves legal spelling", pig_legal_spelling),
            ("pig: matches its family", pig_matches),
            ("bluff: calls with the best card", bluff_calls_best_card),
            ("bluff: folds the worst card facing a bet", bluff_folds_worst_card),
            ("bluff: matches its family", bluff_matches),
            ("simul: matches simultaneous games", simul_matches),
            ("simul: Nash distribution is stable", simul_distribution_is_deterministic),
            ("simul: samples all three signs", simul_mixes),
            ("simul: refuses numeric bids", simul_refuses_numeric_bids),
        ]
        run = collect(checks)
        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"guess {facts.get('guess_midpoint')}, pig {facts.get('pig_roll')}/{facts.get('pig_hold')}, "
            f"bluff {facts.get('bluff_call')}/{facts.get('bluff_fold')}, "
            f"simul {len(facts.get('simul_picks', {}))} signs"
        )
        return summarize(run, notes=notes, details=details)
