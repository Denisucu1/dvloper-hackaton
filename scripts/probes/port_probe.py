"""Probe: the newly ported deterministic pieces, exercised directly and offline.

Four independently-skippable groups (no arena, no model, no network):

- ``DealsSolver``: split-the-pot -> a legal offer/accept/walk, a shrinking pot -> an offer that fits
  the current pot, and a one-shot prisoner's dilemma -> defect;
- ``GuessingSolver`` + ``words_en``: a WORDLE DUEL observation with G/Y/X feedback -> a five-letter
  guess with no green mismatch, and ``words_of_length(5)`` -> a non-empty all-five-letter list;
- ``BluffSolver``: Liar's Dice hidden dice -> a legal bid or ``liar``;
- ``ConnectKSolver``: a ``CONNECT FOUR (7x6, connect 4)`` board with an immediate win -> that column.

A module that cannot be imported is reported as ``skipped`` and contributes no checks: a port that
has not landed yet must not turn the report red. Score is the fraction of the checks that could run.
"""
from __future__ import annotations

import importlib
import re

from scripts.probes.base import Probe, collect, make_result, register, summarize

FIVE = re.compile(r"^[a-z]{5}$")


def _load(module_name: str, *attrs: str) -> tuple[list | None, str]:
    """Import a module and its attributes. Returns ``(values, "")`` or ``(None, reason)``."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - a missing port is a skip, never an error
        return None, f"{module_name}: {type(exc).__name__}: {exc}"[:120]
    values = []
    for attr in attrs:
        value = getattr(module, attr, None)
        if value is None:
            return None, f"{module_name}.{attr} missing"
        values.append(value)
    return values, ""


def _deal_kind(action: str | None) -> str:
    """``offer`` / ``accept`` / ``walk`` / ``other`` for a Deals proposal."""
    text = str(action or "").strip().lower()
    if text.startswith("offer ") or text.startswith("offer"):
        return "offer"
    if text in ("accept", "walk"):
        return text
    return "other"


def _offer_amount(action: str | None) -> int | None:
    match = re.search(r"offer\s+(\d+)", str(action or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _wordle_safe(action: str | None, history: list[tuple[str, str]]) -> bool:
    """The guess is five lowercase letters and every green clue is respected."""
    text = str(action or "").strip().lower()
    if text.startswith("guess "):
        text = text[6:].strip()
    if not FIVE.match(text):
        return False
    for word, pattern in history:
        if len(pattern) != len(text):
            continue
        for index, mark in enumerate(pattern):
            if mark == "G" and text[index] != word[index]:
                return False
    return True


@register
class PortProbe(Probe):
    """Checks for the ported Deals / Wordle / Liar's Dice / Connect-K experts."""

    name = "port.checks"
    component = "port"

    def run(self) -> dict:
        checks: list[tuple[str, object]] = []
        skipped: list[str] = []
        facts: dict = {}

        # ---------------------------------------------------------------- deals
        deals_values, reason = _load("harness.solver_deals", "DealsSolver")
        if deals_values is None:
            skipped.append(f"deals: {reason}")
        else:
            deals = deals_values[0]()
            split_obs = (
                "SPLIT THE POT. You are Player 0.\n"
                "Your private walk-away value: 30\n"
                "Turn 1 of 8. Current pot: 100 coins.\n"
                "\nOffers so far:\n  (none yet)\n\n"
                'It is YOUR turn: "offer <n>" with n from 0 to 100, or "walk".'
            )
            decay_obs = (
                "SPLIT THE POT. You are Player 0.\n"
                "Your private walk-away value: 20\n"
                "Turn 3 of 8. Current pot: 60 coins.\n"
                "\nOffers so far:\n  (none yet)\n\n"
                'It is YOUR turn: "offer <n>" with n from 0 to 60, or "walk".'
            )
            decay_rules = "The pot shrinks by 5 coins after every turn (it is 100 on turn 1)."
            pd_obs = (
                "PRISONER'S DILEMMA - one shot. Round 1 of 1.\n"
                "It is YOUR turn: choose cooperate or defect."
            )

            def deals_split() -> bool:
                action = deals.propose(observation=split_obs, legal=None, seat=0, turn=1)
                facts["deals_split"] = action
                return _deal_kind(action) in ("offer", "accept", "walk")

            def deals_decay_fits() -> bool:
                from harness.spec import Spec

                solver = deals_values[0]()
                configure = getattr(solver, "configure", None)
                if callable(configure):
                    configure(Spec(game_id="tg_deals", family="deals"), decay_rules, False)
                action = solver.propose(observation=decay_obs, legal=None, seat=0, turn=3)
                amount = _offer_amount(action)
                facts["deals_decay"] = action
                return _deal_kind(action) == "offer" and amount is not None and amount <= 60

            def deals_one_shot_defects() -> bool:
                action = deals.propose(
                    observation=pd_obs, legal=["cooperate", "defect"], seat=0, turn=1
                )
                facts["deals_pd"] = action
                return str(action or "").strip().lower() == "defect"

            checks += [
                ("deals: split the pot returns offer/accept/walk", deals_split),
                ("deals: shrinking pot offer <= current pot", deals_decay_fits),
                ("deals: one-shot prisoner's dilemma defects", deals_one_shot_defects),
            ]

        # --------------------------------------------------------------- wordle
        guess_values, reason = _load("harness.solver_guess", "GuessingSolver")
        words_values, words_reason = _load("harness.words_en", "words_of_length")
        if guess_values is None or words_values is None:
            skipped.append(f"guessing: {reason or words_reason}")
        else:
            guess = guess_values[0]()
            words_of_length = words_values[0]
            history = [("slate", "X G Y X X")]
            wordle_obs = (
                "WORDLE DUEL (5 letters, 6 guesses). Round 2 of 6. It is your turn.\n"
                "Your guesses at the opponent's word:\n"
                "  slate -> X G Y X X\n"
                "Opponent's guesses at your word:\n"
            )

            def wordle_guess_safe() -> bool:
                action = guess.propose(observation=wordle_obs, legal=None, seat=0)
                facts["wordle_guess"] = action
                return _wordle_safe(action, history)

            def wordle_vocab() -> bool:
                words = list(words_of_length(5))
                facts["words5"] = len(words)
                return bool(words) and all(len(word) == 5 for word in words)

            checks += [
                ("wordle: five-letter guess respects green clues", wordle_guess_safe),
                ("wordle: words_of_length(5) is a non-empty 5-letter list", wordle_vocab),
            ]

        # ----------------------------------------------------------- liar's dice
        bluff_values, reason = _load("harness.solver_bluff", "BluffSolver")
        if bluff_values is None:
            skipped.append(f"bluff: {reason}")
        else:
            bluff = bluff_values[0]()
            liar_legal = ["1 1", "1 2", "1 3", "2 1", "2 2", "2 3", "liar"]
            liar_obs = (
                "LIAR'S DICE (2 dice each, faces 1-6). You are Player 0.\n"
                "Your dice: 3, 5 (hidden from the opponent).\n"
                "Chips: you +0, opponent +0.\n"
                "Current bid: none yet.\n"
                "Bid history:\n  (none yet)\n"
                'It is YOUR turn. Legal actions: "1 1", "1 2", "1 3", "2 1", "2 2", "2 3", "liar"'
            )

            def liar_bid_is_legal() -> bool:
                action = bluff.propose(observation=liar_obs, legal=liar_legal, seat=0)
                facts["liar_action"] = action
                return action is None or str(action).strip().lower() in liar_legal

            checks += [("bluff: liar's dice returns a legal bid or liar", liar_bid_is_legal)]

        # ------------------------------------------------------------ connect K
        grid_values, reason = _load("harness.solver_grid", "ConnectKSolver")
        if grid_values is None:
            skipped.append(f"grid: {reason}")
        else:
            grid = grid_values[0]()
            connect_obs = (
                "CONNECT FOUR (7x6, connect 4). You are Player 0 and play X; the opponent plays O.\n"
                "   A B C D E F\n"
                " 1 | . . . . . .\n"
                " 2 | . . . . . .\n"
                " 3 | . . . . . .\n"
                " 4 | . . . . . .\n"
                " 5 | . . . . . .\n"
                " 6 | . . . . . .\n"
                " 7 | X X X . O O\n"
                "Open columns: 0, 1, 2, 3, 4, 5\n"
                'It is YOUR turn. Answer with "drop <column letter>".'
            )
            winning = "drop 3"

            def connect_takes_the_win() -> bool:
                action = grid.propose(connect_obs, None)
                facts["connect_action"] = action
                return str(action or "").strip().lower() == winning

            checks += [("connect K: immediate win takes the winning column", connect_takes_the_win)]

        if not checks:
            return make_result(
                score=1.0,
                n=0,
                latencies=[],
                notes="skipped: " + "; ".join(skipped)[:160],
                details={"skipped": skipped, "checks": 0, "passed": 0},
            )

        run = collect(checks)
        details = dict(facts)
        details["skipped"] = skipped
        details["failures"] = run.failures
        notes = f"{run.passed}/{run.total} port checks"
        if skipped:
            notes += " · skipped " + ", ".join(item.split(":")[0] for item in skipped)
        return summarize(run, notes=notes, details=details)
