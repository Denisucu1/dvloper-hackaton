"""Fixture model: one-die push-your-luck (target 30) with an explicit chance node.

Used by scripts/lab.py to show that the chance solver enters the tournament. The model exposes
`chance_actions`/`apply_chance`: applying "roll" moves to a chance state, then the die outcome is applied.
"""


class Model:
    def __init__(self, rules, seat):
        self.seat = int(seat)
        self.banked = [0, 0]
        self.turn = 0
        self.current = 0
        self.pending_chance = False
        self.target = 30
        self.over = False
        self.winner = None

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        if self.over or self.pending_chance or int(seat) != self.current:
            return []
        return ["roll", "hold"]

    def apply(self, seat, action):
        action = str(action).lower()
        if action == "hold":
            self.banked[self.current] += self.turn
            self.turn = 0
            self.current = 1 - self.current
            self._check()
        elif action == "roll":
            self.pending_chance = True
        else:
            raise ValueError("bad action")

    def chance_actions(self):
        return [(str(face), 1.0 / 6.0) for face in range(1, 7)]

    def apply_chance(self, outcome):
        face = int(outcome)
        self.pending_chance = False
        if face == 1:
            self.turn = 0
            self.current = 1 - self.current
        else:
            self.turn += face
        self._check()

    def _check(self):
        if self.banked[0] >= self.target:
            self.over, self.winner = True, 0
        elif self.banked[1] >= self.target:
            self.over, self.winner = True, 1

    def is_terminal(self):
        return self.over

    def to_move(self):
        if self.over or self.pending_chance:
            return []
        return [self.current]

    def score(self, seat):
        if not self.over:
            return 0.5
        return 1.0 if self.winner == int(seat) else 0.0

    def evaluate(self, seat):
        mine = self.banked[int(seat)] + (self.turn if self.current == int(seat) else 0)
        theirs = self.banked[1 - int(seat)] + (self.turn if self.current != int(seat) else 0)
        return max(0.0, min(1.0, 0.5 + (mine - theirs) / (2.0 * self.target)))

    def clone(self):
        other = Model("", self.seat)
        other.banked = list(self.banked)
        other.turn = self.turn
        other.current = self.current
        other.pending_chance = self.pending_chance
        other.over = self.over
        other.winner = self.winner
        return other
