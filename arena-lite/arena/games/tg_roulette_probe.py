"""Team adn test game (chance family): simplified European roulette with simultaneous betting.

Not an official game. Both players start with the same bankroll and secretly place one bet per round
from a public wheel. Every random event (the spin) is revealed immediately and the rules are common
knowledge, so the only skill tested is chance modelling and bankroll management under a fixed horizon:
bet the colours for steady small gains, chase a number for a big swing, or pass and keep your chips.

    python -m arena practice --games-dir games --game tg_roulette_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

import re

from ..core import Game

_NUMBER_BET = re.compile(r"^(?:bet\s+)?(\d{1,2})$")


def _colour(spin: int) -> str:
    if spin == 0:
        return "green"
    return "red" if spin <= 18 else "black"


class TestRoulette(Game):
    id = "tg_roulette_probe"
    name = "Roulette Duel (test)"
    category = "guess"
    blurb = "One public wheel, ten secret bets, and a bankroll to keep alive."
    howto = ("Both players bet on the same European wheel at the same time each round. "
             "Colours pay small and often, a single number pays big, and passing costs nothing. "
             "Most chips after the last round wins.")
    duplicate_safe = False  # the spins come from the seed and the rival bet is hidden until the spin

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "100 chips, 10 rounds", "chips": 100, "rounds": 10},
            {"label": "100 chips, 5 rounds (short)", "chips": 100, "rounds": 5},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.start_chips = max(10, min(1000, int(self.variant.get("chips", 100))))
        self.rounds = max(1, min(30, int(self.variant.get("rounds", 10))))
        self.label = str(self.variant.get("label", ""))
        self.chips = [self.start_chips, self.start_chips]
        self.round = 1
        self.history: list[dict] = []  # one entry per finished round: bets, spin, colour, deltas
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""
        self.last_spin: int | None = None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _bets() -> list[str]:
        return ["bet red", "bet black"] + [f"bet {n}" for n in range(37)]

    def _legal(self, seat: int) -> list[str]:
        bets = ["pass"] if self.chips[seat] < 1 else self._bets() + ["pass"]
        return bets

    @staticmethod
    def _net(bet: str, spin: int) -> int:
        """Chips gained or lost by one settled bet (the 1-chip stake is already part of it)."""
        if bet == "pass":
            return 0
        if bet == "bet red":
            return 1 if 1 <= spin <= 18 else -1
        if bet == "bet black":
            return 1 if 19 <= spin <= 36 else -1
        number = int(bet.split()[-1])
        return 35 if spin == number else -1

    def _broke_note(self, seat: int) -> str:
        return (" You have 0 chips, so only \"pass\" is legal." if self.chips[seat] < 1 else "")

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""ROULETTE DUEL - {self.label}. 2 players, simultaneous secret bets, one public wheel.

GOAL
Finish with more chips than your opponent. Both players start with {self.start_chips} chips; the game
lasts exactly {self.rounds} rounds. Equal chips at the end is a draw.

THE WHEEL
One standard simplified European wheel: the numbers 0 to 36 are equally likely (probability 1/37 each).
  0 is GREEN, 1 to 18 are RED, 19 to 36 are BLACK.
The same single spin settles both players' bets in a round. Every spin is revealed immediately; there
is no hidden information and no hidden future randomness beyond the next public spin.

BETS AND PAYOUTS
Each round both players choose one bet at the same time, without seeing the other choice:
  "bet red"    costs 1 chip. Red wins: you receive 2 chips (net +1). Otherwise you lose the stake (net -1).
  "bet black"  costs 1 chip. Black wins: you receive 2 chips (net +1). Otherwise you lose the stake (net -1).
  "bet 0" to "bet 36"  costs 1 chip. That exact number wins: you receive 36 chips (net +35).
               Otherwise you lose the stake (net -1).
  "pass"       costs nothing and wins nothing (net 0).
A player with 0 chips cannot pay the stake, so that player may only "pass" until they have chips again
(passing never gives chips back, so at 0 chips every later round is a pass).
Example: with 50 chips you bet red and the spin is 17 (red): you now have 51. A bet black that round
would have left you with 49. Betting 17 that round would have left you with 85.

ROUNDS AND END
{self.rounds} rounds are played. In each round both bets are locked in, the wheel is spun once, both bets
are revealed together, and both bankrolls are updated. After the last spin, the player with more chips
wins; equal chips is a draw. There are no further payouts.

ACTION FORMAT
Reply with exactly one of:
  bet red | bet black | bet 0 | bet 1 | ... | bet 36 | pass
Case does not matter and extra spaces or trailing punctuation are ignored. Examples: "bet red",
"BET 17", "bet 36", "pass". A bet you cannot afford (1 chip or more is required) is illegal.
Any other answer is illegal (first time: a warning and the same turn again; second time: you lose the
game). Nothing is paid for an illegal answer.

WHAT YOU OBSERVE EACH TURN
Both bankrolls, the round number, every earlier round with both revealed bets and the spin, and the
full list of legal actions (40 of them while you can afford a bet, otherwise just "pass")."""

    def to_move(self) -> list[int]:
        return [] if self.over else [0, 1]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self.over or seat not in (0, 1):
            return []
        return self.cap_list(self._legal(seat))

    def normalize(self, seat: int, action: str) -> str | None:
        if self.over or seat not in (0, 1):
            return None
        text = self.clean(action).strip(" .!?\"'`")
        if not text:
            return None
        if text == "pass":
            return "pass"
        if self.chips[seat] < 1:
            return None  # the stake cannot be paid
        if text in ("bet red", "red"):
            return "bet red"
        if text in ("bet black", "black"):
            return "bet black"
        match = _NUMBER_BET.match(text)
        if not match:
            return None
        number = int(match.group(1))
        if not 0 <= number <= 36:
            return None
        return f"bet {number}"

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        first = self.normalize(0, actions.get(0, ""))
        second = self.normalize(1, actions.get(1, ""))
        if first is None or second is None:
            raise ValueError(f"illegal actions: {actions!r}")
        bets = [first, second]
        spin = self.rng.randint(0, 36)
        net = [self._net(bets[0], spin), self._net(bets[1], spin)]
        self.chips[0] += net[0]
        self.chips[1] += net[1]
        self.last_spin = spin
        self.history.append({
            "round": self.round,
            "bets": bets,
            "spin": spin,
            "colour": _colour(spin),
            "net": net,
            "chips": list(self.chips),
        })
        if self.round >= self.rounds:
            self.over = True
            if self.chips[0] > self.chips[1]:
                self.winner = 0
            elif self.chips[1] > self.chips[0]:
                self.winner = 1
            self.end_reason = (f"final chips {self.chips[0]} to {self.chips[1]}"
                               + (f", seat {self.winner} wins" if self.winner is not None else ", draw"))
            return
        self.round += 1

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        return {"winner": self.winner, "scores": list(self.chips),
                "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {
            "label": self.label, "chips": list(self.chips), "start_chips": self.start_chips,
            "round": self.round, "rounds": self.rounds, "history": [dict(h) for h in self.history],
            "last_spin": self.last_spin, "to_move": self.to_move(), "over": self.over,
            "result": self.result() if self.over else None,
        }

    def observation(self, seat: int) -> str:
        opponent = 1 - seat
        lines = [
            f"ROULETTE DUEL - {self.label}. You are Player {seat}.",
            f"Round {min(self.round, self.rounds)} of {self.rounds}. "
            f"Chips: you {self.chips[seat]}, opponent {self.chips[opponent]}.",
            "Wheel: 0 is green, 1-18 is red, 19-36 is black (37 equally likely numbers).",
            "Payouts: red/black wins net +1, an exact number wins net +35, a loss costs 1, pass is free.",
            "Previous rounds (your bet / opponent bet -> spin, your net / opponent net):",
        ]
        if not self.history:
            lines.append("  (none yet)")
        for entry in self.history:
            mine, theirs = entry["bets"][seat], entry["bets"][opponent]
            net = entry["net"]
            lines.append(
                f"  Round {entry['round']}: {mine} / {theirs} -> {entry['spin']} "
                f"({entry['colour']}), you {net[seat]:+d} / opponent {net[opponent]:+d}. "
                f"Chips after: you {entry['chips'][seat]}, opponent {entry['chips'][opponent]}."
            )
        if self.over:
            lines.append(f"The game is over: {self.end_reason}.")
            return "\n".join(lines)
        legal = self.legal_actions(seat) or []
        shown = ", ".join(f'"{a}"' for a in legal)
        lines.append(f"You have {self.chips[seat]} chips.{self._broke_note(seat)}")
        lines.append(f"Choose one bet now, at the same time as the opponent (legal actions: {len(legal)}): "
                     f"{shown}")
        return "\n".join(lines)
