"""Offline, deterministic tests for the timeboxed prep tournament.

All play functions are pure Python and instant except the explicit slow-play deadline test, so the whole
module runs in well under a second.
"""
import time
import unittest

from harness import tournament
from harness.candidates import Candidate


def make(name: str, power: float = 0, kind: str = "strategy") -> Candidate:
    return Candidate(kind=kind, name=name, meta={"power": power})


def power_play(a: Candidate, b: Candidate, seed: int) -> float:
    """Transitive strength: the stronger candidate always wins."""
    pa = float(a.meta.get("power", 0))
    pb = float(b.meta.get("power", 0))
    if pa > pb:
        return 1.0
    if pa < pb:
        return 0.0
    return 0.5


def matchup_play(a: Candidate, b: Candidate, seed: int) -> float:
    """Explicit, non-transitive relation, so we can isolate baseline-only wins."""
    beats = {("A", "B"), ("B", "wall")}
    if (a.name, b.name) in beats:
        return 1.0
    if (b.name, a.name) in beats:
        return 0.0
    return 0.5


class TournamentTests(unittest.TestCase):
    def test_transitive_round_robin_selects_the_strongest(self):
        candidates = [make("A", 3), make("B", 2), make("C", 1)]
        best, stats = tournament.run(
            candidates=candidates,
            baselines=[],
            play=power_play,
            budget_s=5.0,
            seed=11,
        )

        self.assertEqual(best.name, "A")
        self.assertEqual(stats["ranking"], ["A", "B", "C"])
        # 3 pairs, both orientations, one repetition (default games_per_pair).
        self.assertEqual(stats["games_played"], 6)
        self.assertFalse(stats["timed_out"])
        self.assertFalse(stats["mutation_attempted"])
        self.assertGreater(stats["winner_margin"], 0.0)

        scores = stats["scores"]
        self.assertGreater(scores["A"]["score"], scores["B"]["score"])
        self.assertGreater(scores["B"]["score"], scores["C"]["score"])
        self.assertEqual(scores["A"]["internal"]["win_rate"], 1.0)
        self.assertEqual(scores["C"]["internal"]["win_rate"], 0.0)

        # Wilson interval wraps the observed rate and keeps the raw counts.
        for view in scores.values():
            low, high = view["wilson"]
            self.assertLessEqual(low, view["win_rate"])
            self.assertGreaterEqual(high, view["win_rate"])
            self.assertEqual(
                view["games"], view["wins"] + view["draws"] + view["losses"]
            )
        self.assertEqual(candidates[0].score, scores["A"]["score"])

    def test_baseline_only_winner_is_ranked_below_the_internal_winner(self):
        candidates = [make("A", 2), make("B", 1)]
        best, stats = tournament.run(
            candidates=candidates,
            baselines=[make("wall", 0)],
            play=matchup_play,
            budget_s=5.0,
            seed=3,
        )

        # B beats the baseline but loses to A; A wins internally. Internal games dominate.
        self.assertEqual(best.name, "A")
        self.assertEqual(stats["ranking"], ["A", "B"])
        scores = stats["scores"]
        self.assertEqual(scores["A"]["internal"]["win_rate"], 1.0)
        self.assertEqual(scores["B"]["internal"]["win_rate"], 0.0)
        self.assertEqual(scores["B"]["baseline"]["win_rate"], 1.0)
        self.assertGreater(scores["A"]["score"], scores["B"]["score"])
        # Baseline scores are reported separately and kept as raw counts.
        self.assertIn("wall", stats["baselines"])
        self.assertEqual(stats["baselines"]["wall"]["games"], 4)

    def test_paired_seeds_and_repeat_runs_match(self):
        candidates = [make("alpha"), make("beta"), make("gamma")]
        seen: list[tuple[str, str, int]] = []

        def seeded_play(a: Candidate, b: Candidate, seed: int) -> float:
            seen.append((a.name, b.name, seed))
            value = (seed * 2_654_435_761 + len(a.name) * 97 + len(b.name) * 31) % 3
            return (0.0, 0.5, 1.0)[int(value)]

        first_best, first = tournament.run(
            candidates=candidates,
            baselines=[],
            play=seeded_play,
            budget_s=5.0,
            seed=42,
            games_per_pair=2,
        )

        # Each repetition has one root seed; its two orientations are root*2 and root*2+1.
        by_pair: dict[frozenset, list[int]] = {}
        for a, b, seed in seen:
            by_pair.setdefault(frozenset((a, b)), []).append(seed)
        self.assertEqual(len(by_pair), 3)
        for seeds in by_pair.values():
            self.assertEqual(len(seeds), 4)  # 2 repetitions x 2 orientations
            roots: dict[int, list[int]] = {}
            for seed in seeds:
                roots.setdefault(seed // 2, []).append(seed % 2)
            self.assertEqual(len(roots), 2)
            for orientations in roots.values():
                self.assertEqual(sorted(orientations), [0, 1])

        second_best, second = tournament.run(
            candidates=[make("alpha"), make("beta"), make("gamma")],
            baselines=[],
            play=seeded_play,
            budget_s=5.0,
            seed=42,
            games_per_pair=2,
        )
        self.assertEqual(first_best.name, second_best.name)
        self.assertEqual(
            {k: v for k, v in first.items() if k not in ("elapsed_s", "log")},
            {k: v for k, v in second.items() if k not in ("elapsed_s", "log")},
        )
        self.assertEqual(first["games_played"], second["games_played"])
        self.assertEqual(first["scores"], second["scores"])

    def test_games_per_pair_multiplies_play_calls(self):
        calls = {"n": 0}

        def counting_play(a: Candidate, b: Candidate, seed: int) -> float:
            calls["n"] += 1
            return power_play(a, b, seed)

        best, stats = tournament.run(
            candidates=[make("A", 2), make("B", 1)],
            baselines=[make("wall", 0)],
            play=counting_play,
            budget_s=5.0,
            seed=9,
            games_per_pair=3,
        )

        # 3 pairs (A-B + A-wall + B-wall) x 3 repetitions x 2 orientations.
        self.assertEqual(calls["n"], 18)
        self.assertEqual(stats["games_per_pair"], 3)
        self.assertEqual(stats["games"], 18)
        self.assertEqual(stats["games_played"], 18)
        self.assertEqual(best.name, "A")
        for view in stats["scores"].values():
            self.assertEqual(view["games"], 12)
            self.assertEqual(view["games"], view["wins"] + view["draws"] + view["losses"])
            for key in ("wins", "draws", "losses", "errors", "score"):
                self.assertIn(key, view)

    def test_games_per_pair_is_clamped_and_non_ints_default_to_one(self):
        cases = [
            (0, 1),
            (20, 12),
            (3, 3),
            ("3", 1),
            (2.5, 1),
            (None, 1),
            (True, 1),
        ]
        for requested, expected in cases:
            with self.subTest(requested=requested):
                _, stats = tournament.run(
                    candidates=[make("solo", 1)],
                    baselines=[],
                    play=power_play,
                    budget_s=5.0,
                    seed=1,
                    games_per_pair=requested,
                )
                self.assertEqual(stats["games_per_pair"], expected)
                self.assertEqual(stats["games"], 0)

    def test_always_raising_play_still_yields_a_safe_winner(self):
        candidates = [make("A", 9), make("B", 1)]

        def boom(a: Candidate, b: Candidate, seed: int) -> float:
            raise RuntimeError("this candidate is broken")

        best, stats = tournament.run(
            candidates=candidates,
            baselines=[make("wall", 0)],
            play=boom,
            budget_s=5.0,
            seed=1,
        )

        self.assertIsInstance(best, Candidate)
        self.assertIn(best.name, {"A", "B"})
        self.assertEqual(stats["games_played"], 6)
        self.assertFalse(stats["timed_out"])
        self.assertTrue(all(view["errors"] > 0 for view in stats["scores"].values()))
        self.assertTrue(all(view["score"] >= 0.0 for view in stats["scores"].values()))

    def test_slow_play_hits_the_deadline_and_returns_best_so_far(self):
        candidates = [make("A", 3), make("B", 2), make("C", 1)]

        def slow(a: Candidate, b: Candidate, seed: int) -> float:
            time.sleep(0.08)
            return power_play(a, b, seed)

        best, stats = tournament.run(
            candidates=candidates,
            baselines=[],
            play=slow,
            budget_s=0.2,
            seed=5,
            games_per_pair=3,
        )

        self.assertTrue(stats["timed_out"])
        self.assertIsInstance(best, Candidate)
        self.assertEqual(stats["games_per_pair"], 3)
        self.assertGreater(stats["games_played"], 0)
        self.assertLess(stats["games_played"], 18)  # 3 pairs x 3 reps x 2 orientations
        self.assertLess(stats["elapsed_s"], 1.0)
        self.assertIn(best.name, {"A", "B", "C"})

    def test_mutation_can_win(self):
        calls = {"mutate": 0}

        def mutate(best: Candidate, stats: dict, log: list) -> Candidate:
            calls["mutate"] += 1
            self.assertIn("scores", stats)
            self.assertEqual(best.name, "A")
            return make("mutant", 9, kind="model")

        candidates = [make("A", 2), make("B", 1)]
        best, stats = tournament.run(
            candidates=candidates,
            baselines=[make("wall", 3)],
            play=power_play,
            budget_s=5.0,
            seed=8,
            mutate=mutate,
            games_per_pair=2,
        )

        self.assertEqual(calls["mutate"], 1)
        self.assertTrue(stats["mutation_attempted"])
        self.assertTrue(stats["mutation_accepted"])
        self.assertEqual(stats["mutation"]["candidate"]["name"], "mutant")
        self.assertEqual(stats["mutation"]["head_to_head"], {"games": 4, "points": 4.0})
        self.assertEqual(best.name, "mutant")
        self.assertIn("mutant", stats["ranking"])
        self.assertIn("mutant", stats["scores"])

    def test_invalid_mutation_is_ignored(self):
        for returned in ({"kind": "not-a-candidate"}, object()):
            calls = {"n": 0}

            def mutate(best: Candidate, stats: dict, log: list, value=returned) -> object:
                calls["n"] += 1
                return value

            with self.subTest(returned=type(returned).__name__):
                best, stats = tournament.run(
                    candidates=[make("A", 2), make("B", 1)],
                    baselines=[],
                    play=power_play,
                    budget_s=5.0,
                    seed=2,
                    mutate=mutate,
                )
                self.assertEqual(calls["n"], 1)
                self.assertTrue(stats["mutation_attempted"])
                self.assertFalse(stats["mutation_accepted"])
                self.assertIsNone(stats["mutation"]["candidate"])
                self.assertEqual(best.name, "A")
                self.assertEqual(stats["games_played"], 2)

    def test_raising_mutation_is_ignored(self):
        def mutate(best: Candidate, stats: dict, log: list) -> Candidate:
            raise RuntimeError("mutation exploded")

        best, stats = tournament.run(
            candidates=[make("A", 2), make("B", 1)],
            baselines=[],
            play=power_play,
            budget_s=5.0,
            seed=2,
            mutate=mutate,
        )

        self.assertTrue(stats["mutation_attempted"])
        self.assertFalse(stats["mutation_accepted"])
        self.assertIsNone(stats["mutation"]["candidate"])
        self.assertEqual(best.name, "A")


if __name__ == "__main__":
    unittest.main()
