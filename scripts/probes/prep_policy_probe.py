"""Probe: prep builds a valid policy when ``legal_actions`` is null (free text).

Offline end-to-end prep with a scripted gateway: the spec says free-text guessing, the sample turn
carries ``legal_actions=None``, and the model's ``choose()`` answer is a concrete five-letter word.
The probe checks that prep accepts the policy, that the sandbox validates it against the null-legal
sample, that the manifest records it, and that the sandboxed call returns the expected concrete move.

No network: the fake gateway is a deterministic object. Score is the fraction of four checks.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
from unittest import mock

from harness import config, memory, prep, sandbox

from scripts.probes.base import Probe, collect, register, summarize

RULES = """WORD DUEL. Two players guess a hidden five-letter word. A guess is any five-letter word.
Feedback marks exact letters, present letters and absent letters. The first to solve it wins."""

SAMPLE = {
    "observation": (
        "WORD DUEL. You are Player 0. Feedback so far: none.\n"
        "It is YOUR turn: guess a five-letter word."
    ),
    "legal_actions": None,
    "seat": 0,
    "turn": 1,
}

SPEC_PAYLOAD = {
    "family": "guessing",
    "action_format": "a five-letter word",
    "action_examples": ["crane", "slate"],
    "winning_idea": "guess words that split the remaining word list",
    "hidden_information": True,
    "simultaneous": False,
    "scoring": "first to solve wins",
    "speech_rules": "",
    "forbidden_subjects": [],
    "traps": [],
    "notes": "free text five-letter guesses",
    "playbook": ["Open with a word with many distinct letters."],
}

POLICY_SOURCE = 'def choose(observation, legal_actions, seat):\n    return "crane"\n'


class ScriptedGateway:
    """Minimal Gateway surface used by prep; no network, deterministic replies."""

    def __init__(self) -> None:
        self.calls = 0
        self.spend = 0.0
        self.cap = 0.75

    def configure(self, info: dict | None = None, cap: float | None = None) -> None:
        return None

    @property
    def available(self) -> bool:
        return True

    @property
    def affordable(self) -> bool:
        return True

    def ask_json(self, system: str, user: str, **kwargs):
        self.calls += 1
        return dict(SPEC_PAYLOAD)

    def ask(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        return POLICY_SOURCE

    def summary(self) -> dict:
        return {"calls": self.calls, "spend_usd": 0.0, "cap_usd": self.cap}


@register
class PrepPolicyProbe(Probe):
    name = "prep.policy"
    component = "prep"

    def run(self) -> dict:
        root = tempfile.mkdtemp(prefix="adn-probe-prep-")
        facts: dict = {}
        message = {
            "type": "prep",
            "game_id": "word_duel_probe",
            "game_name": "Word Duel",
            "variants": [{"label": "5 letters", "rules": RULES, "sample_turn": dict(SAMPLE), "size": 5}],
            "seconds": 60,
            "you": "adn",
            "study_seconds": 30,
            "move_seconds": 20,
            "llm": {"base_url": "scripted://", "api_key": "sk-test", "model": "haiku"},
        }
        try:
            with (
                mock.patch.object(config, "MEMORY_DIR", root),
                mock.patch.object(config, "GENPOLICY", True),
                mock.patch.object(prep, "GENMODEL", False),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = prep.run_prep(message, ScriptedGateway())
                directory = memory.game_dir("word_duel_probe")
                manifest = memory.read_json(directory / "manifest.json") or {}
                valid, reason = (
                    sandbox.validate_module(result.genpolicy, [SAMPLE])
                    if result.genpolicy
                    else (False, "no policy built")
                )
                chosen = (
                    sandbox.call_choose(result.genpolicy, SAMPLE["observation"], None, 0, timeout=2.0)
                    if result.genpolicy
                    else None
                )
                facts["genpolicy"] = result.genpolicy
                facts["validation"] = {"ok": bool(valid), "reason": str(reason)[:160]}
                facts["manifest_genpolicy"] = manifest.get("genpolicy")
                facts["chosen"] = chosen
                facts["notes"] = result.notes

            def policy_was_built() -> bool:
                return bool(result.genpolicy) and os.path.exists(str(result.genpolicy))

            def policy_validates_with_null_legal() -> bool:
                return bool(valid)

            def manifest_records_policy() -> bool:
                name = str(manifest.get("genpolicy") or "")
                return bool(name) and name.endswith("-policy.py")

            def sandbox_returns_concrete_answer() -> bool:
                return chosen == "crane"

            run = collect([
                ("prep built a policy file", policy_was_built),
                ("policy validates on legal_actions=null", policy_validates_with_null_legal),
                ("manifest records the policy artifact", manifest_records_policy),
                ("sandboxed choose() returns a concrete answer", sandbox_returns_concrete_answer),
            ])
        finally:
            shutil.rmtree(root, ignore_errors=True)

        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"policy {os.path.basename(str(facts.get('genpolicy')))}; "
            f"valid {facts.get('validation', {}).get('ok')}; choose {facts.get('chosen')!r}"
        )
        return summarize(run, notes=notes, details=details)
