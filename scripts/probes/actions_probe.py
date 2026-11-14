"""Probe: ``actions.normalize`` and the guaranteed-legal fallbacks.

The verifier is the cheapest insurance against rule 8.1 (first illegal answer is a warning, the
second loses the game). This probe exercises the four things the arena actually sends back:
a legal action, a differently-cased action, an action embedded in prose, and garbage that must be
rejected rather than emitted (the caller then falls back).
"""
from __future__ import annotations

from harness import actions

from scripts.probes.base import Probe, collect, register, summarize

LEGAL = ["A1", "B2", "C3"]


@register
class ActionsProbe(Probe):
    name = "actions.normalize"
    component = "actions"

    def run(self) -> dict:
        checks = [
            ("exact legal spelling is kept", lambda: actions.normalize("B2", LEGAL) == "B2"),
            ("case and quotes map onto the legal spelling", lambda: actions.normalize(' "b2" ', LEGAL) == "B2"),
            ("embedded action is recovered", lambda: actions.normalize("I will play C3 now", LEGAL) == "C3"),
            ("punctuation around an action is trimmed", lambda: actions.normalize("(a1).", LEGAL) == "A1"),
            ("illegal action is rejected", lambda: actions.normalize("Z9", LEGAL) is None),
            ("no legal list keeps the cleaned action", lambda: actions.normalize(" guess 42 ", None) == "guess 42"),
            ("fallback avoids a repeated action", lambda: actions.choose_fallback(LEGAL, ["A1"]) == "B2"),
            ("fallback with every action seen still answers", lambda: actions.choose_fallback(LEGAL, LEGAL) == "A1"),
            ("plausibility rejects a JSON blob", lambda: not actions.is_plausible('{"action":"A1"}')),
            ("plausibility accepts the declared action shape", lambda: actions.is_plausible("guess 42", ["guess <n>"])),
        ]
        run = collect(checks)
        details = {
            "legal_fixture": LEGAL,
            "failures": run.failures,
        }
        notes = "verifier accepts/rewrites legal moves and rejects garbage" if not run.failures else (
            f"{len(run.failures)} verifier check(s) failed"
        )
        return summarize(run, notes=notes, details=details)
