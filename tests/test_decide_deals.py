"""Deals: the acceptance audit recomputes the split, a refused acceptance never stands, and the
deterministic deals solver is only a last resort."""

import importlib.util
import json
import re
import unittest
from unittest import mock

from harness import decide
from harness.chatpolicy import ChatPolicy
from harness.engine import Engine
from harness.spec import Spec

RULES = "SPLIT THE POT. offer <n>: YOU keep n. accept ends it. Higher surplus wins."
OBS = ("SPLIT THE POT. You are Player 0.\nYour private walk-away value: 30\nCurrent pot: 100 coins.\n"
       "Offers so far:\n  1. you offered to keep 60\n  2. opponent offered to keep 80\n"
       'It is YOUR turn: "offer <n>", "accept", or "walk".')

SOLVER_LANDED = importlib.util.find_spec("harness.solver_deals") is not None


class FakeGateway:
    """Returns the audit JSON for audit prompts and the model JSON for everything else."""

    def __init__(self, move="", audit=None, text=""):
        self.move = move
        self.audit = audit
        self.text = text
        self.affordable = True
        self.calls = 0
        self.prompts = []

    def ask(self, system, user, **kwargs):
        self.calls += 1
        self.prompts.append(system + "\n---\n" + user)
        if "audit one move" in system:
            return json.dumps(self.audit) if self.audit is not None else ""
        if self.text:
            return self.text
        if not self.move:
            return ""
        return json.dumps({"intent": "close it", "action": self.move, "chat": "Deal, let us lock it in."})


def play(gateway, *, spec=None, observation=OBS, legal=None, avoid=None):
    spec = spec or Spec(game_id="g", family="deals", hidden_information=True)
    return decide.decide_turn(
        gateway=gateway,
        spec=spec,
        playbook="",
        engine=Engine(spec, seed="t"),
        chat_policy=ChatPolicy(spec, seed="t"),
        msg={"observation": observation, "legal_actions": legal, "chat": [], "turn": 3, "seat": 0,
             "_avoid": avoid or []},
        rules=RULES,
        deadline=15.0,
    )


class AcceptanceAuditRecomputesTheSplitTests(unittest.TestCase):
    def test_misattributed_number_is_recomputed_and_the_acceptance_refused(self):
        # The model trusts its own theirs=50/mine=50 and answers "accept". The standing offer is the
        # opponent keeping 80, so the audit must recompute theirs=80/mine=20 and reject on walk-away 30.
        gateway = FakeGateway("accept", {
            "offer_number": 80, "offer_number_is": "theirs", "total": 100,
            "theirs": 50, "mine": 50, "walk_away": 30,
            "higher_gain_wins": False, "verdict": "ok", "better_action": "offer 55",
        })
        decision = play(gateway)
        self.assertEqual(decision.action, "offer 55")
        self.assertEqual(decision.source, "audit")
        self.assertNotEqual(decision.action, "accept")

    def _facts_seen_by_losing(self, audit):
        seen = {}

        def capture(facts):
            seen.update(facts)
            return False  # stop before any replacement is built

        with mock.patch.object(decide, "_losing", side_effect=capture):
            decide.audit_commitment(FakeGateway(audit=audit), rules=RULES, observation=OBS,
                                    action="accept", timeout=10.0)
        return seen

    def test_offer_number_meaning_the_opponent_keeps_it_is_theirs(self):
        facts = self._facts_seen_by_losing({
            "offer_number": 80, "offer_number_is": "theirs", "total": 100,
            "theirs": 50, "mine": 50, "walk_away": 0, "higher_gain_wins": False, "verdict": "ok",
        })
        self.assertEqual(facts["theirs"], 80)
        self.assertEqual(facts["mine"], 20)

    def test_offer_number_meaning_we_receive_it_is_mine(self):
        facts = self._facts_seen_by_losing({
            "offer_number": 80, "offer_number_is": "mine", "total": 100,
            "theirs": 50, "mine": 50, "walk_away": 0, "higher_gain_wins": False, "verdict": "ok",
        })
        self.assertEqual(facts["theirs"], 20)
        self.assertEqual(facts["mine"], 80)


class RejectedAcceptanceFallbackTests(unittest.TestCase):
    def test_repeats_our_last_demand_when_the_correction_is_not_legal(self):
        gateway = FakeGateway("accept", {
            "offer_number": 80, "offer_number_is": "theirs", "total": 100,
            "theirs": 80, "mine": 20, "walk_away": 30,
            "higher_gain_wins": False, "verdict": "reject", "better_action": "offer 70",
        })
        legal = ["accept", "offer 30", "offer 40", "walk"]
        decision = play(gateway, legal=legal, avoid=["offer 20", "offer 40"])
        self.assertEqual(decision.action, "offer 40")  # our last demand, not the refused acceptance
        self.assertTrue(decision.action.strip())

    def test_falls_back_to_a_concrete_example_never_an_empty_action(self):
        # The audit's replacement is another acceptance, so nothing usable comes back from it.
        gateway = FakeGateway("accept", {
            "offer_number": 80, "offer_number_is": "theirs", "total": 100,
            "theirs": 80, "mine": 20, "walk_away": 30,
            "higher_gain_wins": False, "verdict": "reject", "better_action": "accept",
        })
        spec = Spec(game_id="g", family="deals", action_examples=["offer <n>", "offer 65"])
        decision = play(gateway, spec=spec, legal=None, avoid=[])
        self.assertEqual(decision.action, "offer 65")


@unittest.skipUnless(SOLVER_LANDED, "harness/solver_deals.py has not landed yet")
class DealsSolverLastResortTests(unittest.TestCase):
    def test_model_failure_without_a_legal_list_gets_a_legal_offer(self):
        gateway = FakeGateway(text="")  # the model produced nothing usable
        decision = play(gateway, legal=None)
        self.assertRegex(decision.action, r"(?i)^offer\s+\d+$")
        self.assertEqual(decision.source, "solver")


if __name__ == "__main__":
    unittest.main()
