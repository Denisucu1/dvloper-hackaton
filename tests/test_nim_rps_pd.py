import unittest
from harness.solver_nim import NimSolver, parse_nim_state, solve_nim
from harness.solver_deals import DealsSolver
from harness.solver_simul import SimultaneousSolver, _rps_distribution, _opponent_counts


class TestNimSolver(unittest.TestCase):
    def setUp(self):
        self.solver = NimSolver()

    def test_normal_nim_winning_move(self):
        # Piles: 3, 4, 5 -> XOR sum: 3 ^ 4 ^ 5 = 2.
        # Target for pile 3 (size 5): 5 ^ 2 = 7 (> 5, no)
        # Target for pile 2 (size 4): 4 ^ 2 = 6 (> 4, no)
        # Target for pile 1 (size 3): 3 ^ 2 = 1 (< 3, yes! take 2 from pile 1)
        piles = {1: 3, 2: 4, 3: 5}
        take, p = solve_nim(piles, misere=False, max_take=None)
        self.assertEqual(p, 1)
        self.assertEqual(take, 2)
        # After move, piles are 1, 4, 5 -> 1 ^ 4 ^ 5 = 0 (winning state!)
        self.assertEqual(1 ^ 4 ^ 5, 0)

    def test_subtraction_take_1_3(self):
        # Piles: 1=9, 2=3, 3=4, 4=7, max_take=3 (mod 4: 9%4=1, 3%4=3, 4%4=0, 7%4=3)
        # Grundy sum: 1 ^ 3 ^ 0 ^ 3 = 1.
        # Pile 1 has 9 (g=1). If we take 1, pile becomes 8 (g=0). Target g = 1 ^ 1 = 0!
        # So taking 1 from pile 1 leaves G = 0!
        piles = {1: 9, 2: 3, 3: 4, 4: 7}
        take, p = solve_nim(piles, misere=False, max_take=3)
        self.assertEqual(p, 1)
        self.assertEqual(take, 1)

    def test_misere_nim_endgame(self):
        # Misere: piles are {1: 1, 2: 1, 3: 4}. Exactly one pile > 1.
        # Ones count = 2 (even). To leave an ODD number of 1s, we should reduce pile 3 to 1 (take 3).
        piles = {1: 1, 2: 1, 3: 4}
        take, p = solve_nim(piles, misere=True, max_take=None)
        self.assertEqual(p, 3)
        self.assertEqual(take, 3)

    def test_observation_propose(self):
        obs = """NIM (Take 1-3, 4 piles). You are Player 0.
Rule: On your turn you remove 1, 2, ... or 3 tokens (never more than 3) from exactly ONE pile.
NORMAL PLAY: the player who takes the LAST token WINS.

Starting piles: pile 1 = 9, pile 2 = 3, pile 3 = 4, pile 4 = 7   (total 23)
Piles now: pile 1 = 9, pile 2 = 3, pile 3 = 4, pile 4 = 7   (total 23)
It is YOUR turn. Answer with "take <count> from pile <p>".
"""
        action = self.solver.propose(observation=obs)
        self.assertEqual(action, "take 1 from pile 1")


class TestRPSAntiSpam(unittest.TestCase):
    def test_paper_spam_gets_countered_by_scissors(self):
        # Opponent threw paper 5 times in a row!
        obs = """ROCK PAPER SCISSORS - 15 throws, Rock wins count double. You are seat 0.
Throw 5.
Throw history (oldest first):
  1. you rock - opponent paper -> opponent +1
  2. you rock - opponent paper -> opponent +1
  3. you rock - opponent paper -> opponent +1
  4. you rock - opponent paper -> opponent +1
Opponent's signs so far: rock 0, paper 4, scissors 0.
Throw now: rock | paper | scissors"""
        dist = _rps_distribution("Rock wins count double", obs)
        # Order: rock, paper, scissors
        # Scissors (index 2) must have overwhelming weight >= 0.80!
        self.assertGreater(dist[2], 0.80)


class TestPrisonersDilemmaTieBreaker(unittest.TestCase):
    def setUp(self):
        self.solver = DealsSolver()

    def test_long_game_tiebreaker_protection(self):
        # Long game (quota 40). Opponent has cooperated. We must NOT defect blindly and accumulate defections!
        obs = """PRISONER'S DILEMMA - Long game: qualify at 40, at least 12 rounds. You are seat 0. Qualifying score: 40.
Round 5 (the game lasts at least until round 12; never beyond round 20).
You: 12 points (needs 28 more to qualify), 0 defections.
Opponent: 12 points (needs 28 more to qualify), 0 defections.
Rounds so far:
  1. you cooperate, opponent cooperate -> you +3, opponent +3
  2. you cooperate, opponent cooperate -> you +3, opponent +3
  3. you cooperate, opponent cooperate -> you +3, opponent +3
  4. you cooperate, opponent cooperate -> you +3, opponent +3
Answer: cooperate | defect"""
        action = self.solver.propose(observation=obs, legal=["cooperate", "defect"], seat=0, turn=1)
        self.assertEqual(action, "cooperate")

    def test_clinch_threshold(self):
        # We have 36 points (needs 4 more to qualify at 40). Defecting gives 5 points (total 41 >= 40 QUALIFIED!).
        # Opponent has 10 points. We MUST defect to clinch!
        obs = """PRISONER'S DILEMMA - Long game: qualify at 40, at least 12 rounds. You are seat 0. Qualifying score: 40.
Round 9 (the game lasts at least until round 12; never beyond round 20).
You: 36 points (needs 4 more to qualify), 0 defections.
Opponent: 10 points (needs 30 more to qualify), 0 defections.
Answer: cooperate | defect"""
        action = self.solver.propose(observation=obs, legal=["cooperate", "defect"], seat=0, turn=1)
        self.assertEqual(action, "defect")


if __name__ == "__main__":
    unittest.main()
