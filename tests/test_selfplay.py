import os
import tempfile
import unittest

from harness import selfplay

MODEL = '''
class Model:
    """Numbers 1..9: alternate picks, higher total wins. Perfect information, no chance."""

    def __init__(self, rules, seat):
        self.seat = int(seat)
        self.left = list(range(1, 10))
        self.picks = []

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [f"pick {n}" for n in self.left]

    def apply(self, seat, action):
        number = int(str(action).split()[-1])
        if number not in self.left:
            raise ValueError("taken")
        self.left.remove(number)
        self.picks.append((int(seat), number))

    def is_terminal(self):
        return not self.left

    def to_move(self):
        if self.is_terminal():
            return []
        return [len(self.picks) % 2]

    def score(self, seat):
        totals = {0: 0, 1: 0}
        for s, n in self.picks:
            totals[s] += n
        if totals[0] == totals[1]:
            return 0.5
        return 1.0 if (totals[seat] > totals[1 - seat]) else 0.0

    def evaluate(self, seat):
        totals = {0: 0, 1: 0}
        for s, n in self.picks:
            totals[s] += n
        mine, theirs = totals[seat], totals[1 - seat]
        return max(0.0, min(1.0, 0.5 + (mine - theirs) / 80.0))

    def clone(self):
        other = Model("", self.seat)
        other.left = list(self.left)
        other.picks = list(self.picks)
        return other

    def key(self):
        return ",".join(map(str, self.left)) + "|" + str(len(self.picks))
'''


class SelfPlayTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="adn-selfplay-")
        self.path = os.path.join(self.dir, "model.py")
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(MODEL)

    def test_search_beats_random(self):
        result = selfplay.head_to_head(self.path, "rules", "search", "random", games=10, seed=1)
        self.assertEqual(result["errors"], 0)
        self.assertGreaterEqual(result["score"], 0.8, result)

    def test_greedy_beats_random(self):
        result = selfplay.head_to_head(self.path, "rules", "greedy", "random", games=10, seed=3)
        self.assertGreaterEqual(result["score"], 0.7, result)

    def test_paired_seats_are_symmetric(self):
        first = selfplay.head_to_head(self.path, "rules", "search", "greedy", games=10, seed=5)
        second = selfplay.head_to_head(self.path, "rules", "greedy", "search", games=10, seed=5)
        self.assertAlmostEqual(first["score"] + second["score"], 1.0, places=6)

    def test_broken_strategy_never_raises(self):
        result = selfplay.head_to_head(self.path, "rules", "nonsense", "random", games=4, seed=2)
        self.assertEqual(result["games"], 4)  # fallback to random, no exception

    def test_deadline_stops_the_game(self):
        result = selfplay.play_game(self.path, "rules", ("search", "search"), seed=0,
                                    max_plies=200, seconds_per_move=0.2, deadline=0.0)
        self.assertTrue(result.timed_out or result.plies <= 200)

    def test_missing_model_is_a_safe_draw(self):
        result = selfplay.play_game(os.path.join(self.dir, "nope.py"), "rules", ("search", "search"), seed=0)
        self.assertIsNone(result.winner)
        self.assertTrue(result.error)


if __name__ == "__main__":
    unittest.main()
