"""Team adn test game (Think ahead family): simplified chess.

Deliberately small chess, not a full FIDE engine: standard piece movement, check and checkmate, but no
castling and no en passant, and a pawn that reaches the last rank becomes a queen automatically. That is
enough to stress the harness on a wide, deep, perfect-information game (~35 legal moves per turn, so the
router sends it to MCTS) without writing a full chess library.

    python -m arena practice --games-dir games --game tg_chess --bot-a ... --bot-b ...

Two variants: normal chess, and a short "first to 5 captures" race with the same movement.
"""
from __future__ import annotations

import re

from ..core import Game

WHITE, BLACK = 0, 1  # seat 0 plays white (uppercase), seat 1 plays black (lowercase)
_FILES = "abcdefgh"
_START = [
    "rnbqkbnr",
    "pppppppp",
    "........",
    "........",
    "........",
    "........",
    "PPPPPPPP",
    "RNBQKBNR",
]
_MOVE_RE = re.compile(r"^([a-h])([1-8])([a-h])([1-8])$")
_KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING = ((0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1))
_BISHOP = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ROOK = ((0, 1), (1, 0), (0, -1), (-1, 0))


class TestChess(Game):
    id = "tg_chess"
    name = "Chess (simplified, test)"
    category = "think"
    blurb = "Real pieces, real checkmate, minus the obscure rules."
    howto = "Move your pieces, give check, and mate the king. No castling, no en passant, auto-queen."
    duplicate_safe = True  # no hidden information, no randomness

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "Standard pieces, checkmate wins", "capture_to_win": 0},
            {"label": "First to 5 captures", "capture_to_win": 5},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.label = str(self.variant.get("label", ""))
        self.capture_target = int(self.variant.get("capture_to_win", 0) or 0)
        self.board = [list(row) for row in _START]
        self.current = WHITE
        self.history: list[str] = []
        self.captures = [0, 0]
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""

    # ---------------------------------------------------------------- geometry
    @staticmethod
    def _square(row: int, col: int) -> str:
        return f"{_FILES[col]}{8 - row}"

    @staticmethod
    def _coords(square: str) -> tuple[int, int] | None:
        square = square.strip().lower()
        if len(square) != 2 or square[0] not in _FILES or square[1] not in "12345678":
            return None
        return 8 - int(square[1]), _FILES.index(square[0])

    @staticmethod
    def _mine(piece: str, seat: int) -> bool:
        return piece.isupper() if seat == WHITE else piece.islower()

    @staticmethod
    def _enemy(piece: str, seat: int) -> bool:
        return piece.islower() if seat == WHITE else piece.isupper()

    # ---------------------------------------------------------------- moves
    def _pseudo(self, row: int, col: int) -> list[tuple[int, int]]:
        piece = self.board[row][col]
        if piece == ".":
            return []
        white = piece.isupper()
        kind = piece.upper()
        out: list[tuple[int, int]] = []

        def add(r: int, c: int) -> bool:
            if not (0 <= r < 8 and 0 <= c < 8):
                return False
            target = self.board[r][c]
            if target != "." and (target.isupper() == white):
                return False
            out.append((r, c))
            return target == "."

        if kind == "P":
            step = -1 if white else 1
            start_row = 6 if white else 1
            one = (row + step, col)
            if 0 <= one[0] < 8 and self.board[one[0]][one[1]] == ".":
                out.append(one)
                two = (row + 2 * step, col)
                if row == start_row and self.board[two[0]][two[1]] == ".":
                    out.append(two)
            for dc in (-1, 1):
                r, c = row + step, col + dc
                if 0 <= r < 8 and 0 <= c < 8 and self._enemy(self.board[r][c], WHITE if white else BLACK):
                    out.append((r, c))
        elif kind == "N":
            for dr, dc in _KNIGHT:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8 and not self._mine(self.board[r][c], WHITE if white else BLACK):
                    out.append((r, c))
        elif kind == "K":
            for dr, dc in _KING:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8 and not self._mine(self.board[r][c], WHITE if white else BLACK):
                    out.append((r, c))
        else:
            directions = ()
            if kind in ("B", "Q"):
                directions += _BISHOP
            if kind in ("R", "Q"):
                directions += _ROOK
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    target = self.board[r][c]
                    if target == ".":
                        out.append((r, c))
                    else:
                        if self._enemy(target, WHITE if white else BLACK):
                            out.append((r, c))
                        break
                    r, c = r + dr, c + dc
        return out

    def _king_square(self, seat: int) -> tuple[int, int] | None:
        wanted = "K" if seat == WHITE else "k"
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == wanted:
                    return r, c
        return None

    def _attacked(self, row: int, col: int, by: int) -> bool:
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece == "." or not self._mine(piece, by):
                    continue
                if (row, col) in self._pseudo(r, c):
                    return True
        return False

    def _in_check(self, seat: int) -> bool:
        king = self._king_square(seat)
        return bool(king) and self._attacked(king[0], king[1], 1 - seat)

    def _legal(self, seat: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == "." or not self._mine(self.board[r][c], seat):
                    continue
                for tr, tc in self._pseudo(r, c):
                    if self.board[tr][tc].upper() == "K":  # never capture the king itself
                        continue
                    mover = self.board[r][c]
                    target = self.board[tr][tc]
                    self.board[tr][tc] = mover.upper() if mover.isupper() and tr == 0 else \
                        mover.lower() if mover.islower() and tr == 7 else mover
                    self.board[r][c] = "."
                    ok = not self._in_check(seat)
                    self.board[r][c] = mover
                    self.board[tr][tc] = target
                    if ok:
                        moves.append(((r, c), (tr, tc)))
        return moves

    # ---------------------------------------------------------------- interface
    def rules(self) -> str:
        extra = (f"If a player makes {self.capture_target} captures, that player wins immediately."
                 if self.capture_target else "")
        return f"""CHESS (simplified) - {self.label}. 2 players, alternating turns.

GOAL
Checkmate the opponent's king. White (seat 0) moves first. {extra}

BOARD AND PIECES
A normal 8x8 board, files a-h, ranks 1-8. Uppercase pieces are White, lowercase are Black:
K/k king, Q/q queen, R/r rook, B/b bishop, N/n knight, P/p pawn.
Standard movement: pawns move one square forward (two from their starting square) and capture
diagonally; knights jump in an L; bishops move diagonally; rooks in straight lines; the queen both;
the king one square. A pawn reaching the last rank becomes a queen automatically.
There is NO castling and NO en passant in this variant. You may never leave your own king in check.

END AND SCORING
No legal move while in check is checkmate (win 1, loss 0). No legal move while not in check is
stalemate (draw 0-0).

ACTION FORMAT
The move is "<from><to>", for example "e2e4", "g1f3" or "e7e8" (the pawn promotes automatically).
Case does not matter. Illegal moves are refused (first time: warning and the same turn again;
second time: you lose the game).

WHAT YOU OBSERVE EACH TURN
The board with files and ranks, your colour, the move history, and every legal move in the format
above."""

    def to_move(self) -> list[int]:
        return [] if self.over else [self.current]

    def observation(self, seat: int) -> str:
        lines = [f"CHESS (simplified) - {self.label}. You are seat {seat} and play "
                 f"{'White (uppercase)' if seat == WHITE else 'Black (lowercase)'}."]
        lines.append("   " + " ".join(_FILES))
        for r in range(8):
            lines.append(f"{8 - r:>2} " + " ".join(self.board[r]))
        lines.append("Moves so far: " + (", ".join(self.history[-14:]) or "(none yet)"))
        moves = [f"{self._square(*a)}{self._square(*b)}" for a, b in self._legal(self.current)]
        if not self.over and seat == self.current:
            lines.append("It is YOUR turn. Answer with a move like e2e4.")
            lines.append("Legal moves: " + " ".join(moves[:60]))
        else:
            lines.append("The opponent moves.")
        return "\n".join(lines)

    def legal_actions(self, seat: int) -> list[str] | None:
        if self.over or seat != self.current:
            return []
        return self.cap_list([f"{self._square(*a)}{self._square(*b)}" for a, b in self._legal(seat)])

    def normalize(self, seat: int, action: str) -> str | None:
        if self.over or seat != self.current:
            return None
        text = self.clean(action).replace(" ", "").strip(" .!\"'`")
        match = _MOVE_RE.match(text)
        if not match:
            return None
        frm = self._coords(match.group(1) + match.group(2))
        to = self._coords(match.group(3) + match.group(4))
        if frm is None or to is None:
            return None
        if (frm, to) not in self._legal(seat):
            return None
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}"

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        action = actions.get(self.current)
        if action is None:
            return
        frm = self._coords(action[:2])
        to = self._coords(action[2:])
        if frm is None or to is None:
            return
        mover = self.board[frm[0]][frm[1]]
        target = self.board[to[0]][to[1]]
        captured = target != "."
        self.board[to[0]][to[1]] = mover
        self.board[frm[0]][frm[1]] = "."
        if mover.upper() == "P" and (to[0] == 0 or to[0] == 7):
            self.board[to[0]][to[1]] = "Q" if mover.isupper() else "q"
        self.history.append(f"{self._square(*frm)}{self._square(*to)}")
        if captured:
            self.captures[self.current] += 1
        mover_seat = self.current
        self.current = 1 - self.current
        opponent = self.current

        if self.capture_target and self.captures[mover_seat] >= self.capture_target:
            self.over, self.winner = True, mover_seat
            self.end_reason = f"seat {mover_seat} reached {self.capture_target} captures"
            return
        if not self._legal(opponent):
            self.over = True
            if self._in_check(opponent):
                self.winner = mover_seat
                self.end_reason = f"seat {mover_seat} delivered checkmate"
            else:
                self.winner = None
                self.end_reason = "stalemate: draw"

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        scores = [0, 0]
        if self.winner is not None:
            scores[self.winner] = 1
        return {"winner": self.winner, "scores": scores, "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {
            "label": self.label,
            "board": [list(row) for row in self.board],
            "history": list(self.history),
            "captures": list(self.captures),
            "to_move": self.to_move(),
        }
