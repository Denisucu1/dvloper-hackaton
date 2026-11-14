"""Unit tests for the T2 & Round-Robin upgrades:
- Leduc / Kuhn / Liar's Dice (`solver_bluff.py`)
- Colonel Blotto fixed-budget, weighted & reserves (`solver_simul.py`)
- Wordle Duel 4/5/6-letter + clue verification (`solver_guess.py`, `words_en.py`)
- Connect-K tactical & alpha-beta fallback (`solver_grid.py`)
- Universal Format Guard (`actions.enforce_format`)
- Subagent architecture (`subagents.py`: TruthMeter, StrategistSubagent, PsychologistSubagent, PrepGuardrailsSubagent)
- Extended thinking policy (`turnpolicy.thinking_for`)
"""

from __future__ import annotations

import unittest

from harness import actions, turnpolicy
from harness.solver_bluff import BluffSolver
from harness.solver_grid import ConnectKSolver
from harness.solver_guess import (
    GuessingSolver,
    verify_or_repair_wordle,
    wordle_candidates,
    wordle_feedback,
)
from harness.solver_simul import SimultaneousSolver
from harness.spec import Spec
from harness.subagents import (
    PsychologistSubagent,
    StrategistSubagent,
    TruthMeter,
)


class FakeGateway:
    def __init__(self, reply: str = '{"intent": "verified from doc", "action": "unknown"}') -> None:
        self.affordable = True
        self.reply = reply
        self.last_system = ""
        self.last_user = ""

    def ask(self, system: str, user: str, **kwargs) -> str:
        self.last_system = system
        self.last_user = user
        return self.reply


class T2UpgradesTest(unittest.TestCase):
    def test_leduc_hand4_and_hand6_from_t1(self) -> None:
        solver = BluffSolver()
        spec = Spec(game_id="mini_poker", family="bluff")
        self.assertTrue(solver.matches("mini_poker", spec))

        # T1 Hand 4 R2: we hold K, community card is Q, opponent raised to 8 -> must FOLD unpaired K
        obs_hand4_raise = """MINI POKER (Leduc poker, 8 hands)
Hand 4 of 8. Net chips: You +3, Opponent -3.
Your card: K
Community card: Q
Pot: 18 chips (your contribution: 7, opponent's contribution: 11). To call: 4.
Betting round 2 of 2.
History this hand:
  R1: opponent: bet 2, you: call
  R2: you: bet 4, opponent: raise 8"""
        move = solver.propose(
            observation=obs_hand4_raise,
            legal=["fold", "call"],
            seat=1,
            turn=14,
        )
        self.assertEqual(move, "fold")

        # T1 Hand 4 R2 before betting: we hold K on a Q flop after opponent bet R1 -> check (pot control)
        obs_hand4_first = """MINI POKER (Leduc poker, 8 hands)
Hand 4 of 8. Net chips: You +3, Opponent -3.
Your card: K
Community card: Q
Pot: 6 chips (your contribution: 3, opponent's contribution: 3). To call: 0.
Betting round 2 of 2.
History this hand:
  R1: opponent: bet 2, you: call"""
        move_check = solver.propose(
            observation=obs_hand4_first,
            legal=["check", "bet 4"],
            seat=1,
            turn=13,
        )
        self.assertEqual(move_check, "check")

        # T1 Hand 6 R2: we hold Q, community card is J, opponent bet 2 in R1 and bets 4 in R2 -> FOLD
        obs_hand6 = """MINI POKER (Leduc poker, 8 hands)
Hand 6 of 8. Net chips: You -13, Opponent +13.
Your card: Q
Community card: J
Pot: 10 chips (your contribution: 3, opponent's contribution: 7). To call: 4.
Betting round 2 of 2.
History this hand:
  R1: opponent: bet 2, you: call
  R2: you: check, opponent: bet 4"""
        move_h6 = solver.propose(
            observation=obs_hand6,
            legal=["fold", "call", "raise 8"],
            seat=1,
            turn=21,
        )
        self.assertEqual(move_h6, "fold")

        # Paired board nuts (holding Q with community Q) -> always raise/bet
        obs_paired = """MINI POKER (Leduc poker, 8 hands)
Hand 5 of 8. Net chips: You 0, Opponent 0.
Your card: Q
Community card: Q
Pot: 10 chips (your contribution: 3, opponent's contribution: 7). To call: 4.
Betting round 2 of 2.
History this hand:
  R1: you: bet 2, opponent: call
  R2: opponent: bet 4"""
        move_pair = solver.propose(
            observation=obs_paired,
            legal=["fold", "call", "raise 8"],
            seat=0,
            turn=17,
        )
        self.assertEqual(move_pair, "raise 8")

    def test_kuhn_poker_and_liars_dice(self) -> None:
        solver = BluffSolver()
        obs_kuhn_k = """MINI POKER (Kuhn poker, 12 hands)
Hand 1 of 12. Net chips: You 0, Opponent 0.
Your card: K
Pot: 3 chips (your contribution: 1, opponent's contribution: 2). To call: 1.
Betting round 1 of 1."""
        self.assertEqual(
            solver.propose(observation=obs_kuhn_k, legal=["fold", "call"], seat=0, turn=1),
            "call",
        )

        obs_dice = """LIAR'S DICE (10 dice total, 1s are wild)
Your dice: 2 3 4 5 6
Opponent bids: 8 fours"""
        self.assertEqual(
            solver.propose(
                observation=obs_dice,
                legal=["liar", "8 fives", "9 fours"],
                seat=0,
                turn=3,
            ),
            "liar",
        )

    def test_colonel_blotto_reserves_and_fixed(self) -> None:
        solver = SimultaneousSolver()
        spec = Spec(game_id="colonel_blotto", family="read_opponent")
        self.assertTrue(solver.matches("colonel_blotto", spec))

        # T1 Match 7 Game 3 Round 4: leading 2-1 with 43 reserve vs 48 reserve
        obs_r4 = """COLONEL BLOTTO (Reserves: +30 per round, 3 fields)
Round 4 of 5. Score: You 2 - 1 Opponent (first to 3 wins).
Battlefield values this round: 1 1 1
Your reserve: 43 soldiers (you may send 0 to 43 in total; unused soldiers stay for next round).
Opponent reserve: 48 soldiers.

Past rounds:
  Round 1 (values 1 1 1): You: 16 14 0 | Opponent: 10 10 10 -> You won 2-1
  Round 2 (values 1 1 1): You: 12 10 10 | Opponent: 0 18 18 -> Opponent won 1-2
  Round 3 (values 1 1 1): You: 13 0 12 | Opponent: 12 12 0 -> You won 2-1

It is your turn. Answer with 3 numbers separated by spaces (sum 0..43)."""
        alloc_r4 = solver.propose(observation=obs_r4, legal=None, seat=0, turn=4)
        self.assertIsNotNone(alloc_r4)
        nums_r4 = [int(x) for x in alloc_r4.split()]
        self.assertEqual(len(nums_r4), 3)
        self.assertLessEqual(sum(nums_r4), 43)
        # Must concentrate on 2 fields (leaving 0 on the sacrificed field) instead of flat 15 15 13
        self.assertIn(0, nums_r4)

        # Fixed budget: 100 soldiers, 5 equal fields
        obs_100 = """COLONEL BLOTTO (100 soldiers, 5 equal fields)
Round 1 of 5. Score: You 0 - 0 Opponent.
Battlefield values this round: 1 1 1 1 1
You must place exactly 100 soldiers on 5 battlefields."""
        alloc_100 = solver.propose(observation=obs_100, legal=None, seat=0, turn=1)
        self.assertIsNotNone(alloc_100)
        nums_100 = [int(x) for x in alloc_100.split()]
        self.assertEqual(len(nums_100), 5)
        self.assertEqual(sum(nums_100), 100)

        # Weighted fields: 60 soldiers, 6 fields worth 1-6
        obs_60 = """COLONEL BLOTTO (60 soldiers, 6 fields worth 1-6)
Round 1 of 5. Score: You 0 - 0 Opponent.
Battlefield values this round: 4 1 6 2 5 3
You must place exactly 60 soldiers on 6 battlefields."""
        alloc_60 = solver.propose(observation=obs_60, legal=None, seat=0, turn=1)
        self.assertIsNotNone(alloc_60)
        nums_60 = [int(x) for x in alloc_60.split()]
        self.assertEqual(len(nums_60), 6)
        self.assertEqual(sum(nums_60), 60)

    def test_wordle_duel_feedback_and_solver(self) -> None:
        self.assertEqual(
            wordle_feedback("garden", "danger"),
            ("Y", "G", "Y", "Y", "G", "Y"),
        )
        solver = GuessingSolver()
        spec = Spec(game_id="wordle_duel", family="guessing")
        self.assertTrue(solver.matches("wordle_duel", spec))

        # Secret is "slight": after "salter -> G Y Y X X X", "slight" must be among candidates
        pat = wordle_feedback("slight", "salter")
        cands = wordle_candidates(6, [("salter", pat)])
        self.assertIn("slight", cands)

        obs_wordle = """WORDLE DUEL (5 letters, 6 guesses)
Round 3 of 6.
Your guesses (G=green, Y=yellow, X=gray):
  slate -> X X X Y X
  crony -> X X G G X
Opponent's guesses (letters hidden):
  ??? -> X Y X X X
  ??? -> G G X X X
It is your turn. Answer "guess <word>" with exactly 5 letters a-z."""
        move = solver.propose(observation=obs_wordle, legal=None, seat=0, turn=3)
        self.assertIsNotNone(move)
        self.assertTrue(move.startswith("guess "))
        word = move.split()[1]
        self.assertEqual(len(word), 5)
        self.assertEqual(wordle_feedback(word, "slate"), ("X", "X", "X", "Y", "X"))
        self.assertEqual(wordle_feedback(word, "crony"), ("X", "X", "G", "G", "X"))

        # Verify that an invalid 6-letter or clue-violating LLM guess is repaired
        repaired = verify_or_repair_wordle("guess slate", obs_wordle)
        self.assertNotEqual(repaired, "guess slate")
        self.assertEqual(len(repaired.split()[1]), 5)

    def test_connect_k_solver_win_and_block(self) -> None:
        solver = ConnectKSolver()
        obs = """CONNECT FOUR (6x7, connect 4)
Move 7. You are X (opponent is O).
  0 | . . . . . . .
  1 | . . . . . . .
  2 | . . . . . . .
  3 | . . . . . . .
  4 | . . O O O . .
  5 | . X X X . . .
Open columns: 0, 1, 2, 3, 4, 5, 6
It is your turn. Answer "drop <column>"."""
        # X has 3 in a row on row 5 (cols 1, 2, 3) -> can win immediately with drop 4 or drop 0
        move = solver.propose(observation=obs)
        self.assertIn(move, ("drop 4", "drop 0"))

    def test_universal_format_guard(self) -> None:
        obs_blotto = """COLONEL BLOTTO (100 soldiers, 5 equal fields)
Battlefield values this round: 1 1 1 1 1
You must place exactly 100 soldiers on 5 battlefields."""
        fixed = actions.enforce_format("50 50", observation=obs_blotto)
        nums = [int(x) for x in fixed.split()]
        self.assertEqual(len(nums), 5)
        self.assertEqual(sum(nums), 100)

        obs_c4 = """CONNECT FOUR (6x7, connect 4)
Open columns: 0, 2, 4, 6
Answer "drop <column>"."""
        self.assertIn(actions.enforce_format("drop 3", observation=obs_c4), ("drop 2", "drop 4"))

    def test_subagents_and_thinking_policy(self) -> None:
        tm = TruthMeter()
        tm.observe_turn("Round 1", ["I will send 30 soldiers to field 1"])
        self.assertIn(30, tm.pending_claimed_numbers)
        tm.observe_turn("Round 1: You: 20 20 0 | Opponent: 0 15 15", [])
        self.assertEqual(tm.verified_lies, 1)
        self.assertLess(tm.trust_score, 0.35)

        psych = PsychologistSubagent()
        guidance = psych.mirror_guidance(
            family="read_opponent",
            action="28 15 0",
            observation="",
            truth_meter=tm,
        )
        self.assertIn("Battlefield 3", guidance)

        gw = FakeGateway()
        spec = Spec(game_id="doc_qa", family="truth")
        act, intent = StrategistSubagent().propose_action(
            gw,
            spec=spec,
            rules="Answer from doc or say unknown.",
            playbook="Verify quotes.",
            observation="Doc: The treaty was signed in 1920. Question: Who signed it?",
            legal=None,
            engine_hint="no candidate",
            truth_summary=tm.summary_for_strategist(),
        )
        self.assertEqual(act, "unknown")
        self.assertNotIn("UNTRUSTED-CHAT", gw.last_user)

        for fam in ("bluff", "deals", "read_opponent", "think_ahead", "truth"):
            s = Spec(game_id="g", family=fam)
            self.assertIsNotNone(turnpolicy.thinking_for(s, has_code_move=False, legal=None))
            self.assertIsNone(turnpolicy.thinking_for(s, has_code_move=True, legal=None))


    def test_deals_solver_split_the_pot_and_trust_and_nim(self):
        from harness.solver_deals import DealsSolver
        from harness.solver_grid import ConnectKSolver
        from harness.actions import enforce_format

        deals = DealsSolver()
        spec = Spec(game_id="tg_deals", family="deals", hidden_information=True)
        rules = (
            "SPLIT THE POT\nYou and your opponent are splitting a pot of 100 coins, shrinking by 5 a turn.\n"
            "Each player has a private walk-away value (10 to 45)."
        )
        deals.configure(spec, rules)
        obs = (
            "SPLIT THE POT. You are Player 0.\nYour private walk-away value: 23\n"
            "Turn 1 of 8. Current pot: 100 coins.\nOffers so far: none.\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 100, or "walk".'
        )
        # Turn 1 pot=100, decay=5 -> next_pot=95, low_w=10 -> k_win=floor((95+23-10)/2)+2=56
        act = deals.propose(observation=obs, legal=None, seat=0, turn=1)
        self.assertEqual(act, "offer 56")

        # Non-shrinking pot=100, walk-away=28: must not assume decay=5, and must beat low_w=10 (>= 60)
        no_shrink_rules = (
            "SPLIT THE POT\nYou and your opponent are splitting a pot of 100 coins.\n"
            "Each player has a private walk-away value (10 to 45)."
        )
        deals.configure(spec, no_shrink_rules)
        obs_28 = (
            "SPLIT THE POT. You are Player 0.\nYour private walk-away value: 28\n"
            "Turn 1 of 8. Current pot: 100 coins.\nOffers so far: none.\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 100, or "walk".'
        )
        self.assertEqual(deals.propose(observation=obs_28, legal=None, seat=0, turn=1), "offer 61")

        # Final turn 8 of 8 with unfavorable opponent offer -> walk
        obs_t8 = (
            "SPLIT THE POT. You are Player 1.\nYour private walk-away value: 30\n"
            "Turn 8 of 8. Current pot: 65 coins.\nOffers so far:\n  1. opponent offered to keep 55\n"
            'It is YOUR turn: "offer <n>" with n from 0 to 65, or "accept", or "walk".'
        )
        self.assertEqual(deals.propose(observation=obs_t8, legal=None, seat=1, turn=8), "walk")

        # Iterated Prisoner's Dilemma: cooperate in Round 1, defect in final round 5 of 5
        pd_spec = Spec(game_id="prisoners_dilemma", family="deals")
        deals.configure(pd_spec, "Iterated Prisoner's Dilemma.")
        self.assertEqual(
            deals.propose(
                observation="Round 1 of 5. Choose cooperate or defect.",
                legal=["cooperate", "defect"],
                seat=0,
                turn=1,
            ),
            "cooperate",
        )
        self.assertEqual(
            deals.propose(
                observation="Round 5 of 5. Choose cooperate or defect.",
                legal=["cooperate", "defect"],
                seat=0,
                turn=5,
            ),
            "defect",
        )

        # Nim / Take-away: 15 stones, take 1 to 3 -> 15 % 4 = 3 -> take 3
        grid = ConnectKSolver()
        self.assertEqual(
            grid.propose(
                observation="NIM. Stones remaining: 15. You may take 1 to 3 stones.",
                legal=["take 1", "take 2", "take 3"],
            ),
            "take 3",
        )

        # Truth / Guardrails format guard prefixes "answer "
        self.assertEqual(
            enforce_format("unknown", observation='It is YOUR turn: "answer <text>".'),
            "answer unknown",
        )


if __name__ == "__main__":
    unittest.main()
