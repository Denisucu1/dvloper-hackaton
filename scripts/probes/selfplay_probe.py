"""Probe: self-play between decision strategies (``harness/selfplay.py``).

Writes one tiny model file to a private temp directory — a two-ply "take a higher unused number"
game — then runs ``head_to_head(search vs random, games=2)``. The two games swap seats, so the score
is a paired-seat result. Requirements checked: both games complete, zero errors, the reported score
is a finite number in [0, 1], and the whole match takes under 2 s.

Score is the fraction of those four checks that pass; ``n`` is the number of games played.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import textwrap
import time

from harness import selfplay

from scripts.probes.base import Probe, collect, register, summarize

MODEL_SOURCE = textwrap.dedent('''
    class Model:
        """Each seat takes one unused number 1-3; the higher number wins."""

        def __init__(self, rules="", seat=0):
            self.picks = {}

        def update(self, observation):
            return None

        def legal_actions(self, seat):
            used = set(self.picks.values())
            return [str(n) for n in (1, 2, 3) if str(n) not in used]

        def apply(self, seat, action):
            if seat in self.picks or str(action) in self.picks.values():
                raise ValueError("illegal pick")
            self.picks[int(seat)] = str(action)

        def is_terminal(self):
            return len(self.picks) >= 2

        def to_move(self):
            if self.is_terminal():
                return []
            return [0] if 0 not in self.picks else [1]

        def score(self, seat):
            if not self.is_terminal():
                return 0.5
            mine = int(self.picks.get(seat, 0))
            theirs = int(self.picks.get(1 - seat, 0))
            if mine > theirs:
                return 1.0
            if mine < theirs:
                return 0.0
            return 0.5

        def evaluate(self, seat):
            return 0.5

        def clone(self):
            other = Model()
            other.picks = dict(self.picks)
            return other

        def key(self):
            return ",".join(f"{s}:{a}" for s, a in sorted(self.picks.items()))
''').strip() + "\n"


@register
class SelfPlayProbe(Probe):
    name = "selfplay.paired"
    component = "selfplay"

    def run(self) -> dict:
        root = tempfile.mkdtemp(prefix="adn-probe-selfplay-")
        path = os.path.join(root, "model.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(MODEL_SOURCE)
        facts: dict = {"stats": {"games": 0, "wins": 0, "draws": 0, "losses": 0,
                                 "errors": 0, "score": 0.0}}
        try:
            started = time.monotonic()
            stats = selfplay.head_to_head(
                path,
                "Two players take one unused number; the higher number wins.",
                "search",
                "random",
                games=2,
                seed=1234,
                seconds_per_move=0.05,
            )
            elapsed = time.monotonic() - started
            facts["stats"] = dict(stats)
            facts["elapsed_s"] = round(elapsed, 4)

            def two_paired_games() -> bool:
                return int(stats.get("games") or 0) == 2

            def zero_errors() -> bool:
                return int(stats.get("errors") or 0) == 0

            def score_in_unit_interval() -> bool:
                score = stats.get("score")
                return (
                    isinstance(score, (int, float))
                    and math.isfinite(float(score))
                    and 0.0 <= float(score) <= 1.0
                )

            def match_is_fast() -> bool:
                return elapsed < 2.0

            run = collect([
                ("two paired games complete", two_paired_games),
                ("zero self-play errors", zero_errors),
                ("score is in [0, 1]", score_in_unit_interval),
                ("search vs random finishes under 2 s", match_is_fast),
            ])
        finally:
            shutil.rmtree(root, ignore_errors=True)

        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"score {facts['stats'].get('score')}, "
            f"{facts['stats'].get('wins')}/{facts['stats'].get('draws')}/{facts['stats'].get('losses')} "
            f"W/D/L, {facts.get('elapsed_s')} s"
        )
        result = summarize(run, notes=notes, details=details)
        result["n"] = max(1, int(facts["stats"].get("games") or 0))
        return result
