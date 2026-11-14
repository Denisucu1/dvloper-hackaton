"""Fixture model: Colonel Blotto (3 fields, 6 troops, 3 rounds), simultaneous moves.

Used by scripts/lab.py to show that the matrix solver enters the tournament and improves over the
heuristic baseline. Starts from a valid round-1 state in __init__.
"""


def _compositions(total, parts):
    if parts == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            out.append((first, *rest))
    return out


class Model:
    def __init__(self, rules, seat):
        self.seat = int(seat)
        self.fields = 3
        self.troops = 6
        self.rounds = 3
        self.points = [0, 0]
        self.round = 1
        self.pending = {}
        self.over = False
        self.winner = None

    def update(self, observation):
        return None

    def _allocations(self):
        return [" ".join(map(str, combo)) for combo in _compositions(self.troops, self.fields)]

    def legal_actions(self, seat):
        return [] if self.over else self._allocations()

    def apply(self, seat, action):
        numbers = [int(n) for n in str(action).split()]
        if len(numbers) != self.fields or sum(numbers) != self.troops:
            raise ValueError("bad allocation")
        self.pending[int(seat)] = numbers
        if len(self.pending) == 2:
            first, second = self.pending[0], self.pending[1]
            for field in range(self.fields):
                if first[field] > second[field]:
                    self.points[0] += 1
                elif second[field] > first[field]:
                    self.points[1] += 1
            self.pending = {}
            if self.round >= self.rounds:
                self.over = True
                self.winner = 0 if self.points[0] > self.points[1] else (1 if self.points[1] > self.points[0] else None)
            else:
                self.round += 1

    def is_terminal(self):
        return self.over

    def to_move(self):
        return [] if self.over else [0, 1]

    def score(self, seat):
        if not self.over:
            return 0.5
        if self.points[0] == self.points[1]:
            return 0.5
        return 1.0 if (self.points[0] > self.points[1]) == (int(seat) == 0) else 0.0

    def evaluate(self, seat):
        mine, theirs = (self.points[0], self.points[1]) if int(seat) == 0 else (self.points[1], self.points[0])
        return max(0.0, min(1.0, 0.5 + (mine - theirs) / 20.0))

    def clone(self):
        other = Model("", self.seat)
        other.points = list(self.points)
        other.round = self.round
        other.pending = dict(self.pending)
        other.over = self.over
        other.winner = self.winner
        return other

    def key(self):
        return f"{self.round}:{self.points}:{sorted(self.pending.items())}"
