"""Probe: the timeboxed prep tournament (``harness/tournament.py``).

Two fake candidates (``alpha`` always beats everything, ``beta`` beats only the baseline) and one
baseline, with a deterministic ``play(a, b, seed)`` callback and ``games_per_pair=2``:

- the round robin actually plays games (internal + baseline, both orientations);
- a ranking is present and contains both candidates;
- the run finishes inside its budget without ``timed_out``;
- the winner is one of the input candidates;
- repeating the run gives the same ranking (deterministic seeds/callback);
- a zero budget returns safely with a valid candidate and a ranking, rather than raising.

Score is the fraction of boolean checks that pass; ``n`` is the number of games in the full run.
"""
from __future__ import annotations

from harness import tournament
from harness.candidates import Candidate

from scripts.probes.base import Probe, collect, register, summarize


def _candidates() -> tuple[list[Candidate], list[Candidate]]:
    return (
        [Candidate(kind="strategy", name="alpha"), Candidate(kind="strategy", name="beta")],
        [Candidate(kind="strategy", name="base")],
    )


def _play(a: Candidate, b: Candidate, seed: int) -> float:
    """Deterministic, seed-independent result from ``a``'s perspective."""
    if a.name == b.name:
        return 0.5
    if a.name == "alpha":
        return 1.0
    if b.name == "alpha":
        return 0.0
    if a.name == "beta" and b.name == "base":
        return 1.0
    if a.name == "base" and b.name == "beta":
        return 0.0
    return 0.5


@register
class TournamentProbe(Probe):
    name = "tournament.roundrobin"
    component = "tournament"

    def run(self) -> dict:
        facts: dict = {}
        candidates, baselines = _candidates()
        best, stats = tournament.run(
            candidates=candidates,
            baselines=baselines,
            play=_play,
            budget_s=5.0,
            seed=7,
            games_per_pair=2,
        )
        facts["games"] = stats.get("games")
        facts["ranking"] = list(stats.get("ranking") or [])
        facts["winner"] = best.name
        facts["timed_out"] = bool(stats.get("timed_out"))
        facts["scores"] = {
            name: round(float(view.get("score") or 0.0), 4)
            for name, view in (stats.get("scores") or {}).items()
        }

        def games_were_played() -> bool:
            return int(stats.get("games") or 0) > 0

        def ranking_is_present() -> bool:
            ranking = list(stats.get("ranking") or [])
            return len(ranking) >= 2 and {"alpha", "beta"} <= set(ranking)

        def did_not_time_out() -> bool:
            return not bool(stats.get("timed_out"))

        def winner_is_a_candidate() -> bool:
            return isinstance(best, Candidate) and best.name in {"alpha", "beta"}

        def repeat_is_deterministic() -> bool:
            again_candidates, again_baselines = _candidates()
            again_best, again_stats = tournament.run(
                candidates=again_candidates,
                baselines=again_baselines,
                play=_play,
                budget_s=5.0,
                seed=7,
                games_per_pair=2,
            )
            facts["repeat"] = {
                "winner": again_best.name,
                "ranking": list(again_stats.get("ranking") or []),
            }
            return (
                again_best.name == best.name
                and list(again_stats.get("ranking") or []) == facts["ranking"]
            )

        def tiny_budget_returns_safely() -> bool:
            tiny_candidates, tiny_baselines = _candidates()
            tiny_best, tiny_stats = tournament.run(
                candidates=tiny_candidates,
                baselines=tiny_baselines,
                play=_play,
                budget_s=0.0,
                seed=7,
                games_per_pair=2,
            )
            facts["tiny"] = {
                "winner": getattr(tiny_best, "name", None),
                "games": tiny_stats.get("games"),
                "timed_out": bool(tiny_stats.get("timed_out")),
                "ranking": list(tiny_stats.get("ranking") or []),
            }
            return (
                isinstance(tiny_best, Candidate)
                and isinstance(tiny_stats, dict)
                and bool(tiny_stats.get("ranking"))
            )

        run = collect([
            ("tournament played games", games_were_played),
            ("ranking contains both candidates", ranking_is_present),
            ("full-budget run is not timed out", did_not_time_out),
            ("winner is an input candidate", winner_is_a_candidate),
            ("repeated run is deterministic", repeat_is_deterministic),
            ("zero budget returns safely", tiny_budget_returns_safely),
        ])
        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"winner {facts['winner']}, {facts['games']} games, "
            f"ranking {facts['ranking']}, tiny {facts.get('tiny', {}).get('timed_out')}"
        )
        result = summarize(run, notes=notes, details=details)
        result["n"] = max(1, int(facts.get("games") or 0))
        return result
