"""Built-in Connect-K alpha-beta solver (`CONNECT FOUR (RxC, connect K)`).

Used as an automatic fallback when the prep-generated simulator is unavailable or
times out, guaranteeing tactical win/block and 4-ply lookahead on any Connect-K grid.
"""

from __future__ import annotations

import re
import time
from typing import Iterable

_CONNECT_HEADER_RE = re.compile(
    r"CONNECT\s+FOUR\s*\(\s*(\d+)x(\d+)\s*,\s*connect\s+(\d+)\s*\)",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(
    r"You\s+are\s+([XO])\s*\( opponent\s+is\s+([XO])\)|You\s+are\s+([XO])\b",
    re.IGNORECASE,
)
_OPEN_COLS_RE = re.compile(
    r"Open\s+columns:\s*([0-9,\s]+)",
    re.IGNORECASE,
)
_ROW_LINE_RE = re.compile(r"^\s*\d+\s*\|\s*([.XO\s]+)$")


def parse_connect_k(observation: str) -> dict | None:
    """Parse a Connect-K board observation into `(rows, cols, k, board, me, opp, open_cols)`."""
    text = observation or ""
    m_hdr = _CONNECT_HEADER_RE.search(text)
    if not m_hdr:
        return None
    rows = int(m_hdr.group(1))
    cols = int(m_hdr.group(2))
    k = int(m_hdr.group(3))
    if rows < 3 or cols < 3 or k < 3 or rows > 12 or cols > 12:
        return None

    me = "X"
    opp = "O"
    m_role = _ROLE_RE.search(text)
    if m_role:
        tok = (m_role.group(1) or m_role.group(3) or "X").upper()
        me = tok
        opp = "O" if me == "X" else "X"

    board: list[list[str]] = []
    for raw_line in text.splitlines():
        m_row = _ROW_LINE_RE.match(raw_line)
        if not m_row:
            continue
        cells = [c for c in m_row.group(1).split() if c in (".", "X", "O")]
        if len(cells) == cols:
            board.append(cells)

    if len(board) != rows:
        return None

    open_cols: list[int] = []
    m_open = _OPEN_COLS_RE.search(text)
    if m_open:
        for tok in re.findall(r"\d+", m_open.group(1)):
            c = int(tok)
            if 0 <= c < cols and board[0][c] == ".":
                open_cols.append(c)
    if not open_cols:
        open_cols = [c for c in range(cols) if board[0][c] == "."]

    return {
        "rows": rows,
        "cols": cols,
        "k": k,
        "board": board,
        "me": me,
        "opp": opp,
        "open_cols": open_cols,
    }


def _drop_row(board: list[list[str]], rows: int, col: int) -> int:
    for r in range(rows - 1, -1, -1):
        if board[r][col] == ".":
            return r
    return -1


def _check_win_from(
    board: list[list[str]],
    rows: int,
    cols: int,
    k: int,
    r: int,
    c: int,
    piece: str,
) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        nr, nc = r + dr, c + dc
        while 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == piece:
            count += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == piece:
            count += 1
            nr -= dr
            nc -= dc
        if count >= k:
            return True
    return False


def _eval_board(
    board: list[list[str]],
    rows: int,
    cols: int,
    k: int,
    me: str,
    opp: str,
) -> float:
    score = 0.0
    center = (cols - 1) / 2.0
    for r in range(rows):
        for c in range(cols):
            cell = board[r][c]
            if cell == ".":
                continue
            center_bonus = max(0.0, (cols / 2.0) - abs(c - center)) * 1.5
            if cell == me:
                score += center_bonus
            else:
                score -= center_bonus

    # Window evaluation along 4 directions
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_r = r + (k - 1) * dr
                end_c = c + (k - 1) * dc
                if not (0 <= end_r < rows and 0 <= end_c < cols):
                    continue
                me_cnt = 0
                opp_cnt = 0
                for step in range(k):
                    ch = board[r + step * dr][c + step * dc]
                    if ch == me:
                        me_cnt += 1
                    elif ch == opp:
                        opp_cnt += 1
                if me_cnt > 0 and opp_cnt > 0:
                    continue
                if me_cnt > 0:
                    if me_cnt == k - 1:
                        score += 80.0
                    elif me_cnt == k - 2 and k >= 4:
                        score += 12.0
                    else:
                        score += 2.0
                elif opp_cnt > 0:
                    if opp_cnt == k - 1:
                        score -= 90.0
                    elif opp_cnt == k - 2 and k >= 4:
                        score -= 14.0
                    else:
                        score -= 2.0
    return score


class ConnectKSolver:
    """Fast Connect-K alpha-beta solver with immediate win/block/trap detection."""

    name = "connect_k"

    def propose(
        self,
        observation: str = "",
        legal: list[str] | None = None,
        avoid: Iterable[str] | None = None,
        *,
        max_seconds: float = 1.2,
    ) -> str | None:
        state = parse_connect_k(observation)
        if state is None:
            return None

        rows: int = state["rows"]
        cols: int = state["cols"]
        k: int = state["k"]
        board: list[list[str]] = state["board"]
        me: str = state["me"]
        opp: str = state["opp"]
        open_cols: list[int] = state["open_cols"]

        avoid_set = {str(a).strip().lower() for a in (avoid or ()) if a}
        valid_cols = [c for c in open_cols if f"drop {c}" not in avoid_set]
        if not valid_cols:
            valid_cols = list(open_cols)
        if not valid_cols:
            return None

        center = (cols - 1) / 2.0
        ordered_cols = sorted(valid_cols, key=lambda c: abs(c - center))

        # 1. Immediate winning move
        for c in ordered_cols:
            r = _drop_row(board, rows, c)
            if r < 0:
                continue
            board[r][c] = me
            won = _check_win_from(board, rows, cols, k, r, c, me)
            board[r][c] = "."
            if won:
                return f"drop {c}"

        # 2. Immediate block of opponent's winning move
        must_block: list[int] = []
        for c in ordered_cols:
            r = _drop_row(board, rows, c)
            if r < 0:
                continue
            board[r][c] = opp
            opp_wins = _check_win_from(board, rows, cols, k, r, c, opp)
            board[r][c] = "."
            if opp_wins:
                must_block.append(c)
        if must_block:
            return f"drop {must_block[0]}"

        # 3. Filter out moves that directly gift the opponent a win on the cell above
        safe_cols: list[int] = []
        for c in ordered_cols:
            r = _drop_row(board, rows, c)
            if r <= 0:
                safe_cols.append(c)
                continue
            board[r][c] = me
            board[r - 1][c] = opp
            gives_win = _check_win_from(board, rows, cols, k, r - 1, c, opp)
            board[r - 1][c] = "."
            board[r][c] = "."
            if not gives_win:
                safe_cols.append(c)

        search_cols = safe_cols if safe_cols else ordered_cols
        if len(search_cols) == 1:
            return f"drop {search_cols[0]}"

        deadline = time.monotonic() + max(0.15, max_seconds)
        depth = 4 if cols <= 7 else 3

        def _alphabeta(d: int, alpha: float, beta: float, maximizing: bool) -> float:
            if d <= 0 or time.monotonic() >= deadline:
                return _eval_board(board, rows, cols, k, me, opp)
            avail = [c for c in range(cols) if board[0][c] == "."]
            if not avail:
                return 0.0
            avail.sort(key=lambda c: abs(c - center))
            if maximizing:
                val = -1e9
                for c in avail:
                    r = _drop_row(board, rows, c)
                    board[r][c] = me
                    if _check_win_from(board, rows, cols, k, r, c, me):
                        board[r][c] = "."
                        return 100000.0 + d * 100.0
                    sc = _alphabeta(d - 1, alpha, beta, False)
                    board[r][c] = "."
                    if sc > val:
                        val = sc
                    if val > alpha:
                        alpha = val
                    if alpha >= beta:
                        break
                return val
            else:
                val = 1e9
                for c in avail:
                    r = _drop_row(board, rows, c)
                    board[r][c] = opp
                    if _check_win_from(board, rows, cols, k, r, c, opp):
                        board[r][c] = "."
                        return -100000.0 - d * 100.0
                    sc = _alphabeta(d - 1, alpha, beta, True)
                    board[r][c] = "."
                    if sc < val:
                        val = sc
                    if val < beta:
                        beta = val
                    if alpha >= beta:
                        break
                return val

        best_col = search_cols[0]
        best_val = -1e18
        for c in search_cols:
            r = _drop_row(board, rows, c)
            board[r][c] = me
            sc = _alphabeta(depth - 1, -1e18, 1e18, False)
            board[r][c] = "."
            if sc > best_val:
                best_val = sc
                best_col = c

        return f"drop {best_col}"
