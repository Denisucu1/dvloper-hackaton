"""Team adn test game (Bluff family): Liar's Dice, hidden hands plus escalating bids.

Not an official game. Both players roll hidden dice and then bid on the total count of a face across
BOTH hands, raising the bid or calling the opponent a liar. It is the Bluff-family probe for hidden
information combined with luck: the dice are rolled once at the start, so every later decision is made
under uncertainty about the opponent's hand.

    python -m arena practice --games-dir games --game tg_liars_dice_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

import re

from ..core import Game

_BID = re.compile(r"(\d+)\s+(\d+)")
_FACES = 6  # six-sided dice


class TestLiarsDice(Game):
    id = "tg_liars_dice_probe"
    name = "Liar's Dice (test)"
    category = "bluff"
    blurb = "Hidden dice, rising bids: call the bluff before the count outgrows the table."
    howto = ("Both players secretly roll dice, then take turns raising a bid on how many dice show a face "
             "across both hands, or challenge the last bid. One chip changes hands at the showdown.")
    duplicate_safe = False  # each player's own dice are hidden from the other

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "2 dice each, one hand", "dice": 2},
            {"label": "3 dice each, one hand", "dice": 3},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.per_player = max(1, min(5, int(self.variant.get("dice", 2))))
        self.total = 2 * self.per_player
        self.label = str(self.variant.get("label", ""))
        self.chips = [0, 0]
        self.dice: list[list[int]] = [[], []]
        self.bid: tuple[int, int] | None = None
        self.bidder: int | None = None
        self.line: list[tuple[int, str]] = []
        self.current = 0
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""
        self.showdown: dict | None = None
        self._roll()

    def _roll(self) -> None:
        self.dice = [[self.rng.randint(1, _FACES) for _ in range(self.per_player)] for _ in range(2)]

    # ---------------------------------------------------------------- helpers
    def _beats(self, bid: tuple[int, int], floor: tuple[int, int] | None) -> bool:
        """Escalation: a higher quantity, or the same quantity with a higher face."""
        return floor is None or bid > floor

    def _bids(self) -> list[str]:
        return [f"{q} {f}" for q in range(1, self.total + 1) for f in range(1, _FACES + 1)
                if self._beats((q, f), self.bid)]

    def _legal(self) -> list[str]:
        if self.over:
            return []
        bids = self._bids()
        if self.bid is not None:
            bids.append("liar")
        return bids

    def _count(self, face: int) -> int:
        return sum(1 for hand in self.dice for die in hand if die == face)

    def _settle(self, challenger: int, bid: tuple[int, int]) -> None:
        quantity, face = bid
        bidder = self.bidder
        assert bidder is not None
        count = self._count(face)
        true_bid = count >= quantity
        loser = challenger if true_bid else bidder
        winner = 1 - loser
        self.chips[winner] += 1
        self.chips[loser] -= 1
        self.showdown = {"challenger": challenger, "bidder": bidder, "bid": list(bid),
                         "face": face, "count": count, "true": true_bid, "loser": loser}
        self.over = True
        self.winner = 0 if self.chips[0] > self.chips[1] else (1 if self.chips[1] > self.chips[0] else None)
        verdict = ("bid true" if true_bid else "bid false")
        self.end_reason = (f"{verdict}: {count} of {self.total} dice show {face} against bid {quantity} {face}; "
                           f"seat {winner} won 1 chip (chips {self.chips[0]:+d}/{self.chips[1]:+d})")

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""LIAR'S DICE - {self.label}. 2 players, {self.per_player} hidden dice each, one hand.

GOAL
Win 1 chip: the player who is right at the showdown ends at +1 chip, the other at -1.
You see only your own dice; the opponent's dice stay hidden until the showdown.

SETUP
Each player rolls {self.per_player} six-sided dice (faces 1-6) in secret, so there are {self.total} dice in
play across BOTH hands. Every hand of every die counts, not just each player's own.

PLAY
Player 0 opens with a bid and then players alternate. A bid is "quantity face", for example "2 4", and
claims that at least QUANTITY of the {self.total} dice in play show FACE. Each new bid must escalate over
the previous one: a higher quantity, or the same quantity with a higher face. So "2 4" may be raised to
"3 1" (more dice) or "2 5" (same quantity, higher face), but never to "2 3" or "1 6".
Instead of bidding, a player may say "liar" when at least one bid exists; that challenges the last bid.

SHOWDOWN
Both hands are revealed and FACE is counted across both hands. If the count is at least the bid quantity,
the bid was true and the CHALLENGER loses 1 chip. Otherwise the bid was false and the BIDDER loses 1 chip.
That single chip decides the game: the winner ends at +1 chip, the loser at -1 (equal is a draw, which
normal play cannot produce).

ACTION FORMAT
Exactly one of:
- a bid: two positive integers "QUANTITY FACE", for example "2 4" (quantity 1-{self.total}, face 1-6);
- "liar" - only when a bid already exists.
The legal actions are listed on every turn.

ILLEGAL ACTIONS
An action outside the legal list is rejected: the first bad action in the game earns a warning and the
same turn is re-sent; the second bad action loses the game.

WHAT YOU OBSERVE EACH TURN
Your own dice, the chip totals, every bid so far, whose turn it is and your legal actions."""

    def to_move(self) -> list[int]:
        return [] if self.over else [self.current]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self.over or seat != self.current:
            return []
        return self.cap_list(self._legal())

    def normalize(self, seat: int, action: str) -> str | None:
        if self.over or seat != self.current:
            return None
        act = self.clean(action).strip(" .!\"'`")
        if act == "liar":
            return act if self.bid is not None else None
        match = _BID.fullmatch(act)
        if not match:
            return None
        quantity, face = int(match.group(1)), int(match.group(2))
        if not (1 <= quantity <= self.total and 1 <= face <= _FACES):
            return None
        if not self._beats((quantity, face), self.bid):
            return None
        return f"{quantity} {face}"

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        seat = self.current
        act = self.normalize(seat, actions.get(seat, ""))
        if act is None:
            return
        self.line.append((seat, act))
        if act == "liar":
            assert self.bid is not None
            self._settle(seat, self.bid)
            return
        quantity, face = (int(n) for n in act.split())
        self.bid = (quantity, face)
        self.bidder = seat
        self.current = 1 - seat

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        return {"winner": self.winner, "scores": list(self.chips),
                "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {"label": self.label, "per_player": self.per_player, "total": self.total,
                "dice": [list(hand) for hand in self.dice], "chips": list(self.chips),
                "bid": list(self.bid) if self.bid else None, "bidder": self.bidder,
                "line": [[seat, act] for seat, act in self.line],
                "showdown": dict(self.showdown) if self.showdown else None,
                "to_move": self.to_move(), "over": self.over, "result": self.result() if self.over else None}

    def observation(self, seat: int) -> str:
        lines = [
            f"LIAR'S DICE ({self.per_player} dice each, faces 1-{_FACES}). You are Player {seat}.",
            f"Your dice: {', '.join(str(d) for d in self.dice[seat])} (hidden from the opponent).",
            f"Chips: you {self.chips[seat]:+d}, opponent {self.chips[1 - seat]:+d}.",
        ]
        if self.bid is None:
            lines.append("Current bid: none yet.")
        else:
            who = "you" if self.bidder == seat else "opponent"
            lines.append(f"Current bid: \"{self.bid[0]} {self.bid[1]}\" by {who} "
                         f"(at least {self.bid[0]} of {self.total} dice show {self.bid[1]}).")
        lines.append("Bid history:")
        if self.line:
            lines += [f"  {'you' if actor == seat else 'opponent'} {act}" for actor, act in self.line[-10:]]
        else:
            lines.append("  (none yet)")
        if self.over:
            lines.append(f"Game over: {self.end_reason}.")
            lines.append(f"Both hands: you [{', '.join(str(d) for d in self.dice[seat])}], "
                         f"opponent [{', '.join(str(d) for d in self.dice[1 - seat])}].")
        elif seat == self.current:
            legal = self.legal_actions(seat) or []
            lines.append("It is YOUR turn. Legal actions: " + ", ".join(f'"{a}"' for a in legal))
        else:
            lines.append("It is the opponent's turn.")
        return "\n".join(lines)
