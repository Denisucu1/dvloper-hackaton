"""Team adn test game (Bluff family): simplified Leduc poker.

Two betting rounds with one community card, six cards (J, Q, K twice), ante 1, fixed bet 1 and one bet
per round. Small enough to reason about by hand, big enough that a one-card heuristic (the Bluff solver)
is not enough: the second round and the pair/community card matter. It is the end-to-end probe for
`harness/solver_leduc.py`. Two variants differ only in the number of independent hands.

    python -m arena practice --games-dir games --game tg_leduc_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

from ..core import Game

_RANKS = ("J", "Q", "K")
_RANK = {card: rank for rank, card in enumerate(_RANKS)}


class TestLeduc(Game):
    id = "tg_leduc_probe"
    name = "Leduc Poker (test)"
    category = "bluff"
    blurb = "Two rounds, one community card: a pair can turn a check into a value bet."
    howto = ("Ante one chip, then check or bet in round one. A call ends the round and reveals one "
             "community card; bet again, then the best pair or high card wins the pot.")
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "J-Q-K twice, one hand", "hands": 1},
            {"label": "J-Q-K twice, three hands", "hands": 3},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.hands = max(1, min(3, int(self.variant.get("hands", 1))))
        self.label = str(self.variant.get("label", ""))
        self.chips = [0, 0]
        self.scores = [0, 0]
        self.hand = 1
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""
        self.history: list[str] = []
        self._deal()

    # ---------------------------------------------------------------- dealing
    def _deal(self) -> None:
        deck = [rank for rank in _RANKS for _ in range(2)]
        self.rng.shuffle(deck)
        self.cards = [deck[0], deck[1]]          # one private card each
        self._community = deck[2]                 # revealed after round one
        self.community: str | None = None
        self.round = 1
        self.lines: list[list[str]] = [[]]        # one action line per round
        self.current = 0                          # player 0 opens both rounds
        self.pot = 2                              # both antes
        self.chips[0] -= 1                        # net chips already include the ante
        self.chips[1] -= 1

    # ---------------------------------------------------------------- helpers
    def _legal(self) -> list[str]:
        if self.over:
            return []
        line = self.lines[-1]
        if not line or line == ["check"]:
            return ["check", "bet"]
        if line in (["bet"], ["check", "bet"]):
            return ["call", "fold"]
        return []

    def _line_text(self) -> str:
        return " | ".join(
            f"R{index + 1}: {', '.join(line) or '-'}" for index, line in enumerate(self.lines)
        )

    def _value(self, seat: int) -> tuple[int, int]:
        """(is_pair, rank): a pair with the community card beats any unpaired hand."""
        private = self.cards[seat]
        if self.community is None:
            return (0, _RANK[private])
        if private == self.community:
            return (1, _RANK[private])
        return (0, max(_RANK[private], _RANK[self.community]))

    def _end_round(self) -> None:
        if self.round == 1:
            self.round = 2
            self.community = self._community
            self.lines.append([])
            self.current = 0
            return
        self._showdown()

    def _showdown(self) -> None:
        value0, value1 = self._value(0), self._value(1)
        board = f"{self.cards[0]}+{self.community} vs {self.cards[1]}+{self.community}"
        if value0 > value1:
            self._settle(0, f"showdown {board}")
        elif value1 > value0:
            self._settle(1, f"showdown {board}")
        else:
            self._settle(None, f"showdown {board} (draw)")

    def _settle(self, winner: int | None, how: str) -> None:
        if winner is None:
            self.chips[0] += self.pot // 2
            self.chips[1] += self.pot - self.pot // 2
        else:
            self.chips[winner] += self.pot
        self.history.append(f"Hand {self.hand}: {self._line_text()} -> {how}")
        self._next_hand(how)

    def _next_hand(self, how: str) -> None:
        if self.hand >= self.hands:
            self.over = True
            a, b = self.chips
            self.winner = None if a == b else (0 if a > b else 1)
            self.scores = [0.5, 0.5] if self.winner is None else [
                int(self.winner == 0), int(self.winner == 1)
            ]
            tail = "draw" if self.winner is None else f"seat {self.winner} wins"
            self.end_reason = f"{how}; chips {a:+d} vs {b:+d}, {tail}"
            return
        self.hand += 1
        self._deal()

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""LEDUC POKER - {self.label}. 2 players, {self.hands} hand(s), two betting rounds.

SETUP
The deck is J, Q, K twice (J lowest, K highest). Both players ante 1 chip and receive one private card;
one more card is kept for the community. You see only your own card.

BETTING
Player 0 acts first in each round. Each round allows at most one bet of 1 chip:
- with no bet in front of you, choose "check" or "bet";
- after a check, the other player chooses "check" or "bet";
- facing a bet, choose "call" (1 chip) or "fold" (give up the pot).
A check-check or a bet-and-call ends the round. After round one the community card is revealed. After
round two the best hand wins: a pair (private card equal to the community card) beats any high card;
otherwise the higher of the private card and the community card decides. Equal hands split the pot.
Cards are revealed only at a showdown.

ACTION FORMAT
Exactly one of: check, bet, call, fold (the legal ones are listed on every turn).

SCORING
Chips are net for this match. After the last hand the higher total wins; equal is a draw."""

    def to_move(self) -> list[int]:
        return [] if self.over else [self.current]

    def legal_actions(self, seat: int) -> list[str] | None:
        if seat != self.current:
            return []
        return self.cap_list(self._legal())

    def normalize(self, seat: int, action: str) -> str | None:
        wanted = self.clean(action).strip(" .!\"'`")
        return wanted if wanted in self._legal() else None

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        seat = self.current
        action = self.normalize(seat, actions.get(seat, ""))
        if action is None:
            return
        line = self.lines[-1]
        line.append(action)
        if action == "fold":
            self._settle(1 - seat, f"seat {seat} folded")
            return
        if action == "bet":
            self.chips[seat] -= 1
            self.pot += 1
            self.current = 1 - seat
            return
        if action == "call":
            self.chips[seat] -= 1
            self.pot += 1
            self._end_round()
            return
        # check: a second check closes the round
        if len(line) == 2:
            self._end_round()
            return
        self.current = 1 - seat

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        if not self.over:
            return {"winner": None, "scores": [0, 0], "reason": "game in progress"}
        return {"winner": self.winner, "scores": list(self.scores),
                "reason": self.end_reason or "game over"}

    def frame(self) -> dict:
        return {"label": self.label, "hand": self.hand, "hands": self.hands, "round": self.round,
                "cards": list(self.cards), "community": self.community, "pot": self.pot,
                "chips": list(self.chips), "lines": [list(line) for line in self.lines],
                "to_move": self.to_move(), "over": self.over}

    def observation(self, seat: int) -> str:
        legal = self.legal_actions(seat)
        shown = " or ".join(f'"{action}"' for action in legal) if legal else "(not your turn)"
        lines = [
            f"LEDUC POKER (J, Q, K twice, 2 betting rounds). You are Player {seat}.",
            f"Hand {min(self.hand, self.hands)} of {self.hands}. Round {self.round} of 2.",
            f"Your private card: {self.cards[seat]}",
            f"Community card: {self.community or '(not dealt yet)'}",
            f"Pot: {self.pot}. Chips: you {self.chips[seat]:+d}, opponent {self.chips[1 - seat]:+d}.",
        ]
        if self.round == 2:
            lines.append("Earlier rounds: " + (", ".join(self.lines[0]) or "(none)"))
        lines.append("This round so far: " + (", ".join(self.lines[-1]) or "(no action yet)"))
        if self.over:
            lines.append(f"Game over: {self.end_reason}.")
        elif seat == self.current:
            lines.append(f"It is YOUR turn: {shown}.")
        else:
            lines.append("It is the opponent's turn.")
        return "\n".join(lines)
