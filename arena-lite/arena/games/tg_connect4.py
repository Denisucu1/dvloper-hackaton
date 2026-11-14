"""Team adn test game (Think ahead family): gravity Connect Four on a real board.

Not an official game. It exists to test the perfect-information pipeline on a game that is deeper than
Tic-Tac-Toe and has gravity: the observation parser must read a rectangular board, the action format is a
column, and the search must not be fooled by a column that is already full.

    python -m arena practice --games-dir games --game tg_connect4 --bot-a ... --bot-b ...
"""
from __future__ import annotations

import re

from ..core import Game

MARK = ("X", "O")
_DROP_RE = re.compile(r"^(?:drop|play|in|column)?\s*([a-h])$")


class TestConnect4(Game):
    id = "tg_connect4"
    name = "Connect Four (test)"
    category = "think"
    blurb = "Drop a disc, build a line of four, and mind the gravity."
    howto = "Players take turns dropping one disc into a column. The first to line up enough discs across, down or diagonally wins."
    duplicate_safe = True  # open board, no randomness

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "7x6, connect 4", "cols": 7, "rows": 6, "connect": 4},
            {"label": "6x5, connect 3", "cols": 6, "rows": 5, "connect": 3},
            # Mirrors the T1/Round-Robin shape that we beat with search: 9 columns, 7 rows, five in a row.
            {"label": "7x9, connect 5", "cols": 9, "rows": 7, "connect": 5},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.cols = max(4, min(9, int(self.variant.get("cols", 7))))
        self.rows = max(4, min(7, int(self.variant.get("rows", 6))))
        self.k = max(3, min(self.cols, int(self.variant.get("connect", 4))))
        self.label = str(self.variant.get("label", ""))
        self.board: list[list[int | None]] = [[None] * self.cols for _ in range(self.rows)]  # [row][col]
        self.current = 0
        self.history: list[str] = []
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""

    # ---------------------------------------------------------------- helpers
    def _column(self, c: int) -> str:
        return chr(ord("A") + c)

    def _landing_row(self, c: int) -> int | None:
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][c] is None:
                return r
        return None

    def _render(self) -> str:
        header = "   " + " ".join(self._column(c) for c in range(self.cols))
        lines = [header]
        for r in range(self.rows):
            cells = [MARK[v] if v is not None else "." for v in self.board[r]]
            lines.append(f"{r + 1:>2} " + " ".join(cells))
        return "\n".join(lines)

    def _find_winner(self) -> int | None:
        for r in range(self.rows):
            for c in range(self.cols):
                me = self.board[r][c]
                if me is None:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = []
                    ok = True
                    for i in range(self.k):
                        rr, cc = r + dr * i, c + dc * i
                        if not (0 <= rr < self.rows and 0 <= cc < self.cols) or self.board[rr][cc] != me:
                            ok = False
                            break
                        cells.append((rr, cc))
                    if ok:
                        return me
        return None

    def _open_columns(self) -> list[int]:
        return [c for c in range(self.cols) if self.board[0][c] is None]

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""CONNECT FOUR - {self.label}. 2 players, alternating turns.

GOAL
Be the first to line up {self.k} of your discs in an unbroken straight line: horizontally, vertically
or diagonally.

BOARD
A board with {self.cols} columns (letters A-{self._column(self.cols - 1)}, left to right) and {self.rows}
rows. A disc dropped into a column falls to the lowest empty cell of that column. Column letters are
shown above the board; rows are numbered 1-{self.rows} top to bottom:
{self._render()}
Seat 0 plays X and drops first; seat 1 plays O.

ACTION FORMAT
"drop <column letter>", for example "drop C" or "drop g". Case does not matter. A full column is
illegal (first time: a warning and the same turn again; second time: you lose the game).

END AND SCORING
The first player with {self.k} discs in a line wins (score 1, loser 0). If every column fills up with no
line, the game is a draw (0-0).

WHAT YOU OBSERVE EACH TURN
The board as a text grid (X, O and . for empty), your mark, the drop history, and the list of columns
that still have room."""

    def to_move(self) -> list[int]:
        return [] if self.over else [self.current]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self.over or seat != self.current:
            return []
        return self.cap_list([f"drop {self._column(c)}" for c in self._open_columns()])

    def normalize(self, seat: int, action: str) -> str | None:
        if self.over or seat != self.current:
            return None
        match = _DROP_RE.match(self.clean(action).strip(' .!?"\'`'))
        if not match:
            return None
        c = ord(match.group(1)) - ord("a")
        if not 0 <= c < self.cols or self._landing_row(c) is None:
            return None
        return f"drop {self._column(c)}"

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        action = actions.get(self.current)
        if action is None:
            return
        match = _DROP_RE.match(self.clean(action))
        if not match:
            return
        c = ord(match.group(1)) - ord("a")
        r = self._landing_row(c)
        if r is None:
            return
        self.board[r][c] = self.current
        self.history.append(f"{MARK[self.current]} {self._column(c)}")
        winner = self._find_winner()
        if winner is not None:
            self.over, self.winner = True, winner
            self.end_reason = f"seat {winner} ({MARK[winner]}) made {self.k} in a row"
            return
        if not self._open_columns():
            self.over = True
            self.end_reason = "board full, no line: draw"
            return
        self.current = 1 - self.current

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        scores = [0, 0]
        if self.winner is not None:
            scores[self.winner] = 1
        return {"winner": self.winner, "scores": scores,
                "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {
            "label": self.label,
            "cols": self.cols,
            "rows": self.rows,
            "connect": self.k,
            "board": [list(row) for row in self.board],
            "history": list(self.history),
            "to_move": self.to_move(),
        }

    def observation(self, seat: int) -> str:
        turn = "It is YOUR turn." if (not self.over and seat == self.current) else "The opponent moves."
        open_columns = ", ".join(f"drop {self._column(c)}" for c in self._open_columns()) or "(none)"
        return f"""CONNECT FOUR - {self.label}. You are Player {seat} and play {MARK[seat]}; the opponent plays {MARK[1 - seat]}.
{self._render()}
Drops so far: {", ".join(self.history) or "(none yet)"}
Open columns: {open_columns}
{turn} Answer with "drop <column letter>"."""
