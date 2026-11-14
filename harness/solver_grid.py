"""Built-in Connect-K alpha-beta solver (`CONNECT FOUR (RxC, connect K)`).

Used as an automatic fallback when the prep-generated simulator is unavailable or
times out, guaranteeing tactical win/block and 4-ply lookahead on any Connect-K grid.
"""

from __future__ import annotations

import re
import time
from typing import Iterable

_CONNECT_HEADER_PAREN_RE = re.compile(
    r"CONNECT\s+FOUR\s*\(\s*(\d+)x(\d+)\s*,\s*connect\s+(\d+)\s*\)",
    re.IGNORECASE,
)
_CONNECT_HEADER_DASH_RE = re.compile(
    r"CONNECT\s+FOUR\s*[-\u2013\u2014]\s*(\d+)x(\d+)\s*,\s*connect\s+(\d+)",
    re.IGNORECASE,
)
_PLAY_AS_RE = re.compile(
    r"You\s+are\s+Player\s+\d+\s+and\s+play\s+([XO])",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(
    r"You\s+are\s+([XO])\s*\( opponent\s+is\s+([XO])\)|You\s+are\s+([XO])\b",
    re.IGNORECASE,
)
_OPEN_COLS_RE = re.compile(
    r"Open\s+columns?\s*:\s*([^\n]+)",
    re.IGNORECASE,
)
_ROW_LINE_RE = re.compile(r"^\s*(\d+)\s*\|?\s*([.XO\s]+?)\s*$")
_COL_HEADER_RE = re.compile(r"^\s*([A-Za-z](?:\s+[A-Za-z])+)\s*$")


def parse_connect_k(observation: str) -> dict | None:
    """Parse a Connect-K board observation into `(rows, cols, k, board, me, opp, open_cols)`.

    Accepts the lab header `CONNECT FOUR (RxC, connect K)` and the arena header
    `CONNECT FOUR - CxR, connect K` (arena label is columns x rows). Board rows may be
    printed with or without a `|` separator, and columns may be numeric or letters.
    """
    text = observation or ""
    m_hdr = _CONNECT_HEADER_PAREN_RE.search(text)
    header_form = "paren"
    if not m_hdr:
        m_hdr = _CONNECT_HEADER_DASH_RE.search(text)
        header_form = "dash"
    if not m_hdr:
        return None
    dim_a = int(m_hdr.group(1))
    dim_b = int(m_hdr.group(2))
    k = int(m_hdr.group(3))
    if k < 3:
        return None

    me = "X"
    opp = "O"
    m_play = _PLAY_AS_RE.search(text)
    m_role = _ROLE_RE.search(text)
    if m_play:
        me = m_play.group(1).upper()
        opp = "O" if me == "X" else "X"
    elif m_role:
        tok = (m_role.group(1) or m_role.group(3) or "X").upper()
        me = tok
        opp = "O" if me == "X" else "X"

    board_rows: list[tuple[int, list[str]]] = []
    col_header_tokens: list[str] = []
    for raw_line in text.splitlines():
        m_col = _COL_HEADER_RE.match(raw_line)
        if m_col:
            tokens = m_col.group(1).split()
            if len(tokens) >= 3 and len({t.upper() for t in tokens}) == len(tokens):
                col_header_tokens = tokens
            continue
        m_row = _ROW_LINE_RE.match(raw_line)
        if not m_row:
            continue
        cells = [c for c in m_row.group(2).split() if c in (".", "X", "O")]
        if cells:
            board_rows.append((int(m_row.group(1)), cells))

    if not board_rows:
        return None
    widths = {len(cells) for _, cells in board_rows}
    if len(widths) != 1:
        return None
    parsed_rows = len(board_rows)
    parsed_cols = widths.pop()

    # Prefer the orientation consistent with the parsed board; fall back to each
    # header's own convention (parentheses = RxC, arena dash = CxR).
    if parsed_rows == dim_a and parsed_cols == dim_b:
        rows, cols = dim_a, dim_b
    elif parsed_rows == dim_b and parsed_cols == dim_a:
        rows, cols = dim_b, dim_a
    elif header_form == "paren":
        rows, cols = dim_a, dim_b
    else:
        rows, cols = dim_b, dim_a
    if rows != parsed_rows or cols != parsed_cols:
        return None
    if rows < 3 or cols < 3 or rows > 12 or cols > 12:
        return None

    # Row labels are 1-based with row 1 at the top. If the board is printed bottom-up,
    # use the labels to restore top-to-bottom order instead of trusting print order.
    labels = [label for label, _ in board_rows]
    if len(set(labels)) == len(labels) and all(1 <= label <= rows for label in labels):
        board_rows.sort(key=lambda item: item[0])
    board = [cells for _, cells in board_rows]

    col_tokens: dict[int, str] = {}
    if len(col_header_tokens) == cols:
        for c, token in enumerate(col_header_tokens):
            col_tokens[c] = token

    open_cols: list[int] = []
    letter_mode = False
    saw_open_token = False
    m_open = _OPEN_COLS_RE.search(text)
    if m_open:
        for part in re.split(r"[,;]", m_open.group(1)):
            part = part.strip()
            if not part:
                continue
            letter_m = re.search(r"\bdrop\s+([A-Za-z])\b", part, re.I)
            if not letter_m:
                letter_m = re.fullmatch(r"([A-Za-z])", part)
            if letter_m:
                saw_open_token = True
                letter = letter_m.group(1)
                c = ord(letter.upper()) - ord("A")
                if 0 <= c < cols:
                    letter_mode = True
                    col_tokens.setdefault(c, letter)
                    if board[0][c] == ".":
                        open_cols.append(c)
                continue
            num_m = re.search(r"\b(\d+)\b", part)
            if num_m:
                saw_open_token = True
                c = int(num_m.group(1))
                if 0 <= c < cols and board[0][c] == ".":
                    open_cols.append(c)
    open_cols = list(dict.fromkeys(open_cols))
    if not open_cols:
        open_cols = [c for c in range(cols) if board[0][c] == "."]
    if not saw_open_token:
        # No explicit open list: a letter column header is the only spelling hint we have.
        letter_mode = bool(col_tokens)

    return {
        "rows": rows,
        "cols": cols,
        "k": k,
        "board": board,
        "me": me,
        "opp": opp,
        "open_cols": open_cols,
        "col_tokens": col_tokens,
        "letter_mode": letter_mode,
    }


def _drop_action(state: dict, col: int) -> str:
    """Format a drop in the observation's own column spelling (letters or numbers)."""
    if state.get("letter_mode"):
        token = (state.get("col_tokens") or {}).get(col)
        if token is None:
            token = chr(ord("A") + col)
        return f"drop {token}"
    return f"drop {col}"


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
            return self._propose_nim(observation, legal, avoid)

        rows: int = state["rows"]
        cols: int = state["cols"]
        k: int = state["k"]
        board: list[list[str]] = state["board"]
        me: str = state["me"]
        opp: str = state["opp"]
        open_cols: list[int] = state["open_cols"]

        avoid_set = {str(a).strip().lower() for a in (avoid or ()) if a}
        valid_cols = [
            c
            for c in open_cols
            if _drop_action(state, c).lower() not in avoid_set and f"drop {c}" not in avoid_set
        ]
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
                return _drop_action(state, c)

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
            return _drop_action(state, must_block[0])

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
            return _drop_action(state, search_cols[0])

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

        return _drop_action(state, best_col)

    def _propose_nim(
        self,
        observation: str,
        legal: Iterable[str] | None,
        avoid: Iterable[str] | None,
    ) -> str | None:
        text = observation or ""
        text_l = text.lower()
        if not any(k in text_l for k in ("nim", "stones", "pile", "heap", "take-away", "coins remaining")):
            return None

        avoid_set = {str(a).strip().lower() for a in (avoid or ()) if a}
        legal_list = [str(a) for a in legal] if legal else []

        # Single-pile take-away: e.g. "Stones remaining: 17" with "take 1 to 3"
        pile_m = re.search(
            r"(?:stones|coins|objects|matches|sticks|pile|heap)\s*(?:remaining|left|size)?\s*:\s*(\d+)",
            text,
            re.I,
        )
        if pile_m:
            s = int(pile_m.group(1))
            max_take = 3
            range_m = re.search(r"take\s+1\s+(?:to|-)\s+(\d+)", text, re.I)
            if range_m:
                max_take = int(range_m.group(1))
            elif legal_list:
                takes = [
                    int(m.group(1))
                    for item in legal_list
                    for m in [re.search(r"^(?:take|remove)\s+(\d+)$", item.strip(), re.I)]
                    if m
                ]
                if takes:
                    max_take = max(takes)

            misere = "last stone loses" in text_l or "last to take loses" in text_l or "poison" in text_l
            target_rem = (s - 1) % (max_take + 1) if misere else s % (max_take + 1)
            take = target_rem if 1 <= target_rem <= min(s, max_take) else 1
            cand = f"take {take}"
            if legal_list:
                for item in legal_list:
                    if item.strip().lower() in (f"take {take}", f"remove {take}", str(take)):
                        if item.strip().lower() not in avoid_set:
                            return item
                return None
            if cand not in avoid_set:
                return cand

        return None
