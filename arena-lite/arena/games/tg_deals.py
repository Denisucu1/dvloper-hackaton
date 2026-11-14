"""Team adn test game (Deals family): split a pot, each side with a private walk-away value.

Not an official game. It tests negotiation in the chat and that our private walk-away value never
appears in a chat line.

    python -m arena practice --games-dir games --game tg_deals --bot-a ... --bot-b ...
"""
from __future__ import annotations

from ..core import Game


class TestDeals(Game):
    id = "tg_deals"
    name = "Split the Pot (test)"
    category = "deals"
    blurb = "Everyone wants more than half. Nobody knows how little the other side would take."
    howto = "Players trade offers for a pot of coins. Each has a secret walk-away value. The better deal for you, compared with walking away, wins."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "pot 100, 8 turns", "pot": 100, "turns": 8, "decay": 0},
            {"label": "pot 100, shrinking by 5 a turn", "pot": 100, "turns": 8, "decay": 5},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.pot0 = int(self.variant.get("pot", 100))
        self.turns = int(self.variant.get("turns", 8))
        self.decay = int(self.variant.get("decay", 0))
        self.reserve = [self.rng.randint(10, 45), self.rng.randint(10, 45)]
        self.turn = 1
        self.current = 0
        self.offers: list[tuple[int, int]] = []  # (seat, coins the proposer keeps)
        self._result: dict | None = None
        self.payout = [0, 0]

    def pot(self) -> int:
        return max(0, self.pot0 - self.decay * (self.turn - 1))

    def rules(self) -> str:
        shrink = (f"The pot shrinks by {self.decay} coins after every turn (it is {self.pot0} on turn 1)."
                  if self.decay else "The pot does not change.")
        return f"""SPLIT THE POT - {self.pot0} coins, at most {self.turns} turns in total.

SETUP
Two players must agree how to split a pot of {self.pot0} coins. {shrink}
Each player has a private walk-away value between 10 and 45 coins: what that player gets if there is no
deal. You know only your own. Player 0 moves first, then the players alternate.

ACTION FORMAT
    offer <n>    propose that YOU keep <n> coins and the opponent gets the rest of the current pot
    accept       accept the opponent's latest offer (only legal after they made one)
    walk         end the negotiation: both players get their own walk-away value
<n> is a whole number from 0 to the current pot. Example:  "offer 60"

HOW IT ENDS AND IS SCORED
The game ends on "accept", on "walk", or after turn {self.turns} without a deal (then both get their
walk-away value). Your result is your SURPLUS: the coins you got minus your walk-away value.
The player with the higher surplus wins; equal surplus (always the case without a deal) is a draw.
"""

    def to_move(self) -> list[int]:
        return [] if self._result else [self.current]

    def observation(self, seat: int) -> str:
        lines = [f"SPLIT THE POT. You are Player {seat}.",
                 f"Your private walk-away value: {self.reserve[seat]}",
                 f"Turn {min(self.turn, self.turns)} of {self.turns}. Current pot: {self.pot()} coins.", "", "Offers so far:"]
        if not self.offers:
            lines.append("  (none yet)")
        for index, (who, keep) in enumerate(self.offers, 1):
            name = "you" if who == seat else "opponent"
            lines.append(f"  {index}. {name} offered to keep {keep}")
        lines.append("")
        if self._result:
            lines.append(f"Game over: {self._result['reason']}.")
        elif seat == self.current:
            extra = ', or "accept"' if self.offers and self.offers[-1][0] != seat else ""
            lines.append(f'It is YOUR turn: "offer <n>" with n from 0 to {self.pot()}{extra}, or "walk".')
        else:
            lines.append("It is the opponent's turn.")
        return "\n".join(lines)

    def legal_actions(self, seat: int) -> list[str] | None:
        return [] if self._result or seat != self.current else None

    def normalize(self, seat: int, action: str) -> str | None:
        if self._result or seat != self.current:
            return None
        act = self.clean(action)
        if act == "walk":
            return act
        if act == "accept":
            return act if self.offers and self.offers[-1][0] != seat else None
        if act.startswith("offer "):
            value = act[6:].strip()
            if value.isdecimal() and 0 <= int(value) <= self.pot():
                return f"offer {int(value)}"
        return None

    def apply(self, actions: dict[int, str]) -> None:
        if self._result:
            return
        seat = self.current
        act = self.normalize(seat, actions.get(seat, ""))
        if act is None:
            raise ValueError(f"illegal action for seat {seat}: {actions.get(seat)!r}")
        if act == "walk":
            self._no_deal(f"seat {seat} walked away")
        elif act == "accept":
            proposer, keep = self.offers[-1]
            pot = self.pot()
            keep = min(keep, pot)
            self.payout[proposer], self.payout[1 - proposer] = keep, pot - keep
            surplus = [self.payout[s] - self.reserve[s] for s in (0, 1)]
            winner = None if surplus[0] == surplus[1] else (0 if surplus[0] > surplus[1] else 1)
            self._finish(winner, surplus, f"deal: seat {proposer} keeps {keep} of {pot}; surplus {surplus[0]:+d} vs {surplus[1]:+d}")
        else:
            self.offers.append((seat, int(act.split()[1])))
            if self.turn >= self.turns:
                self._no_deal("no deal after the last turn")
            else:
                self.turn += 1
                self.current = 1 - seat

    def _no_deal(self, why: str) -> None:
        self.payout = list(self.reserve)
        self._finish(None, [0, 0], f"{why}; both keep their walk-away value")

    def _finish(self, winner: int | None, scores: list[int], reason: str) -> None:
        self._result = {"winner": winner, "scores": list(scores), "reason": reason}

    def is_over(self) -> bool:
        return self._result is not None

    def result(self) -> dict:
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def frame(self) -> dict:
        return {"pot": self.pot(), "turn": self.turn, "reserve": list(self.reserve), "offers": [list(o) for o in self.offers],
                "payout": list(self.payout), "to_move": self.to_move(), "over": self.is_over(), "result": self._result}
