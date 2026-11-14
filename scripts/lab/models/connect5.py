"""Fixture model: 9x7 placement, five in a row (the Connect-5 shape, no gravity).

Used by scripts/lab.py to measure how the decision strategies improve over iterations. It implements the
GameModel contract and starts from a valid empty board in __init__, so self-play needs no observation text.
"""


class Model:
    def __init__(self, rules, seat):
        self.seat = int(seat)
        self.cols = 9
        self.rows = 7
        self.need = 5
        self.board = ["."] * (self.cols * self.rows)
        self.current = 0
        self.winner = None
        self.over = False

    # ------------------------------------------------------------------ state
    def update(self, observation):
        return None

    def _name(self, index):
        return "ABCDEFGHI"[index % self.cols] + str(index // self.cols + 1)

    def _index(self, action):
        return (int(action[1:]) - 1) * self.cols + "ABCDEFGHI".index(action[0].upper())

    def legal_actions(self, seat):
        if self.over or seat != self.current:
            return []
        return [self._name(i) for i, cell in enumerate(self.board) if cell == "."]

    def apply(self, seat, action):
        index = self._index(action)
        if self.board[index] != ".":
            raise ValueError("occupied")
        self.board[index] = "X" if int(seat) == 0 else "O"
        self.winner = self._winner()
        self.over = self.winner is not None or "." not in self.board
        if not self.over:
            self.current = 1 - int(seat)

    def is_terminal(self):
        return self.over

    def to_move(self):
        return [] if self.over else [self.current]

    def score(self, seat):
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == int(seat) else 0.0

    def evaluate(self, seat):
        me, opp = ("X", "O") if int(seat) == 0 else ("O", "X")
        mine = self._windows(me)
        theirs = self._windows(opp)
        total = max(1, mine + theirs)
        return max(0.0, min(1.0, 0.5 + 0.4 * (mine - theirs) / total))

    def clone(self):
        other = Model("", self.seat)
        other.board = list(self.board)
        other.current = self.current
        other.winner = self.winner
        other.over = self.over
        return other

    def key(self):
        return "".join(self.board) + str(self.current)

    # ------------------------------------------------------------------ lines
    def _windows(self, mark):
        count = 0
        n, k = self.cols, self.need
        for r in range(self.rows):
            for c in range(n):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = []
                    for i in range(k):
                        rr, cc = r + dr * i, c + dc * i
                        if not (0 <= rr < self.rows and 0 <= cc < n):
                            cells = []
                            break
                        cells.append(rr * n + cc)
                    if cells and all(self.board[i] != ("O" if mark == "X" else "X") for i in cells):
                        count += 1
        return count

    def _winner(self):
        n, k = self.cols, self.need
        for r in range(self.rows):
            for c in range(n):
                mark = self.board[r * n + c]
                if mark == ".":
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = []
                    for i in range(k):
                        rr, cc = r + dr * i, c + dc * i
                        if not (0 <= rr < self.rows and 0 <= cc < n):
                            cells = []
                            break
                        cells.append(rr * n + cc)
                    if cells and all(self.board[i] == mark for i in cells):
                        return 0 if mark == "X" else 1
        return None
