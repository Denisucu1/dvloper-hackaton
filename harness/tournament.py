"""Timeboxed candidate tournament for prep.

Scheduling policy
-----------------
Round-robin, not a bracket. Every pair of candidates plays ``games_per_pair`` games in each orientation
(default one, clamped to 1..12), then every candidate plays every baseline the same way. With the
handful of candidates prep builds (1-4) the round robin is cheap, gives every candidate the same
internal opposition, and ranks candidates that never reach a bracket final. A bracket would save games
but make the score depend on the draw, which is the opposite of what a prep choice needs.

Seeds are derived from ``(base seed, pair index, repetition index)`` to a repetition root, then the
orientation bit is appended: ``game_seed = repetition_root * 2 + orientation``. Both orientations of
one repetition therefore share the repetition root, while every individual game gets a distinct,
reproducible seed.

Internal games run first because head-to-head is the primary signal; the baseline gauntlet is the
calibration pass and is what gets truncated when the clock is tight. All games stop at the first
``time.monotonic()`` check past the deadline, so a partial tournament still returns the best-so-far.

Combined score
--------------
``score = 0.7 * internal_win_rate + 0.3 * baseline_win_rate - 0.25 * error_rate`` (clamped to 0..1),
using Wilson's interval for the reported confidence. Internal games dominate so that a candidate which
only farms a weak baseline cannot outrank a candidate that wins the head-to-head games. When one of the
two buckets has no games (e.g. no baselines supplied) the other bucket carries the full weight.

Mutation
--------
``mutate`` is called at most once, only when at least 30% of the wall-clock budget is still left. A
valid ``Candidate`` is played head-to-head against the incumbent and through the baseline gauntlet; it
is kept when it wins the head-to-head or scores strictly higher than the incumbent. Anything else is
ignored. Invalid returns and exceptions are swallowed.
"""
from __future__ import annotations

import math
import time

from .candidates import Candidate

_INTERNAL_WEIGHT = 0.7
_BASELINE_WEIGHT = 0.3
_ERROR_PENALTY = 0.25
_MUTATION_MIN_FRACTION = 0.30
_MIN_GAMES_PER_PAIR = 1
_MAX_GAMES_PER_PAIR = 12
_Z = 1.96


# --- small statistics helpers -------------------------------------------------

def _blank() -> dict:
    return {"games": 0, "wins": 0, "draws": 0, "losses": 0, "errors": 0, "points": 0.0}


def _add(record: dict, result: float, error: bool = False) -> None:
    record["games"] += 1
    record["points"] += float(result)
    if result >= 0.75:
        record["wins"] += 1
    elif result <= 0.25:
        record["losses"] += 1
    else:
        record["draws"] += 1
    if error:
        record["errors"] += 1


def _rate(record: dict) -> float:
    return record["points"] / record["games"] if record["games"] else 0.0


def _wilson(points: float, games: int) -> tuple[float, float]:
    """Wilson score interval for a win proportion; (0, 1) when there is no evidence."""
    if games <= 0:
        return 0.0, 1.0
    p = max(0.0, min(1.0, points / games))
    denominator = 1.0 + _Z * _Z / games
    centre = (p + _Z * _Z / (2.0 * games)) / denominator
    half = _Z * math.sqrt(p * (1.0 - p) / games + _Z * _Z / (4.0 * games * games)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _combined_rate(internal: dict, baseline: dict) -> float:
    internal_games = internal["games"]
    baseline_games = baseline["games"]
    if internal_games and baseline_games:
        return _INTERNAL_WEIGHT * _rate(internal) + _BASELINE_WEIGHT * _rate(baseline)
    if internal_games:
        return _rate(internal)
    if baseline_games:
        return _rate(baseline)
    return 0.0


def _record_view(record: dict) -> dict:
    low, high = _wilson(record["points"], record["games"])
    return {
        "games": record["games"],
        "wins": record["wins"],
        "draws": record["draws"],
        "losses": record["losses"],
        "errors": record["errors"],
        "points": record["points"],
        "win_rate": _rate(record),
        "wilson": [low, high],
    }


def _unique(stem: object, used: set[str]) -> str:
    name = str(stem or "candidate")
    candidate = name
    suffix = 2
    while candidate in used:
        candidate = f"{name}#{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _unique_keys(names: list[str]) -> list[str]:
    used: set[str] = set()
    return [_unique(name, used) for name in names]


def _derive_seed(seed: int, index: int) -> int:
    """Deterministic 31-bit mix used to build pair/repetition roots."""
    return (int(seed) * 1_000_003 + int(index) * 7_919 + 17) & 0x7FFF_FFFF


def _repetition_seed(seed: int, pair_index: int, repetition: int) -> int:
    """Root seed shared by both orientations of one repetition of one pair."""
    return _derive_seed(_derive_seed(seed, pair_index), repetition)


def _game_seed(repetition_root: int, orientation: int) -> int:
    """Distinct seed per game; the two orientations are the two children of the repetition root."""
    return int(repetition_root) * 2 + int(orientation)


def _normalise_games_per_pair(value: object) -> int:
    """Clamp an int to 1..12; any non-int gets the safe default of one game per orientation."""
    if isinstance(value, bool) or not isinstance(value, int):
        return _MIN_GAMES_PER_PAIR
    return max(_MIN_GAMES_PER_PAIR, min(_MAX_GAMES_PER_PAIR, value))


def _valid_result(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _minimal_stats(
    best: Candidate,
    started: float,
    budget: float,
    games_per_pair: int,
    reason: str,
) -> dict:
    return {
        "games": 0,
        "games_played": 0,
        "games_per_pair": games_per_pair,
        "elapsed_s": time.monotonic() - started,
        "timed_out": False,
        "mutation_attempted": False,
        "mutation_accepted": False,
        "mutation": {
            "attempted": False,
            "accepted": False,
            "incumbent": None,
            "head_to_head": {"games": 0, "points": 0.0},
            "candidate": None,
        },
        "scores": {},
        "baselines": {},
        "ranking": [best.name],
        "winner": best.name,
        "winner_margin": 0.0,
        "budget_s": budget,
        "log": [],
        "note": reason,
    }


# --- the scheduler ------------------------------------------------------------

class _Run:
    def __init__(
        self,
        candidates: list[Candidate],
        baselines: list[Candidate],
        play,
        budget_s: float,
        seed: object,
        mutate,
        games_per_pair: int,
    ) -> None:
        self.started = time.monotonic()
        try:
            self.budget = max(0.0, float(budget_s))
        except (TypeError, ValueError):
            self.budget = 0.0
        self.deadline = self.started + self.budget
        try:
            self.seed = int(seed)
        except (TypeError, ValueError):
            self.seed = 0

        self.play = play
        self.mutate = mutate
        self.games_per_pair = games_per_pair
        self.cand = list(candidates)
        self.base = list(baselines)
        self.n_original = len(self.cand)

        self.keys = _unique_keys([candidate.name for candidate in self.cand])
        self.base_keys = _unique_keys([baseline.name for baseline in self.base])
        self.recs = [_blank() for _ in self.cand]
        self.irecs = [_blank() for _ in self.cand]
        self.brecs = [_blank() for _ in self.cand]
        self.base_recs = [_blank() for _ in self.base]

        self.games = 0
        self.aborted = False
        self.log: list[dict] = []

        self.mutation_attempted = False
        self.mutation_accepted = False
        self.mutation_index: int | None = None
        self.mutation_view: dict | None = None
        self.mutation_incumbent: str | None = None
        self.mutation_h2h = {"games": 0, "points": 0.0}

        self.schedule = self._build_schedule()
        # Mutation pairs continue after the scheduled pair indices, so their seeds stay deterministic.
        self.next_pair_index = len(self.schedule)

    # --- scheduling -------------------------------------------------------

    def _build_schedule(self) -> list[tuple[str, int, int, int]]:
        """One entry per pair; each pair expands to games_per_pair repetitions in both orientations."""
        schedule: list[tuple[str, int, int, int]] = []
        pair_index = 0
        for i in range(len(self.cand)):
            for j in range(i + 1, len(self.cand)):
                schedule.append(("internal", i, j, pair_index))
                pair_index += 1
        for i in range(len(self.cand)):
            for k in range(len(self.base)):
                schedule.append(("baseline", i, k, pair_index))
                pair_index += 1
        return schedule

    def _out_of_time(self) -> bool:
        return time.monotonic() >= self.deadline

    # --- one game ---------------------------------------------------------

    def _call(self, a: Candidate, b: Candidate, seed_value: int):
        """Play one orientation. Returns (result, error) or None when the clock stopped us."""
        if self._out_of_time():
            self.aborted = True
            return None
        raw = None
        try:
            raw = self.play(a, b, seed_value)
        except Exception:
            raw = None
        error = not _valid_result(raw)
        result = 0.0 if error else float(raw)
        self.games += 1
        return result, error

    def _internal_pair(self, i: int, j: int, pair_index: int) -> None:
        for repetition in range(self.games_per_pair):
            root = _repetition_seed(self.seed, pair_index, repetition)
            first = self._call(self.cand[i], self.cand[j], _game_seed(root, 0))
            if first is None:
                return
            self._record_internal(i, j, first[0], first[1])
            second = self._call(self.cand[j], self.cand[i], _game_seed(root, 1))
            if second is None:
                return
            self._record_internal(j, i, second[0], second[1])

    def _record_internal(self, actor: int, opponent: int, result: float, error: bool) -> None:
        _add(self.recs[actor], result, error)
        _add(self.irecs[actor], result)
        _add(self.recs[opponent], 1.0 - result)
        _add(self.irecs[opponent], 1.0 - result)

    def _baseline_pair(self, i: int, k: int, pair_index: int) -> None:
        for repetition in range(self.games_per_pair):
            root = _repetition_seed(self.seed, pair_index, repetition)
            first = self._call(self.cand[i], self.base[k], _game_seed(root, 0))
            if first is not None:
                result, error = first
                self._record_candidate_baseline(i, result, error)
                _add(self.base_recs[k], 1.0 - result)
            second = self._call(self.base[k], self.cand[i], _game_seed(root, 1))
            if second is None:
                return
            result, error = second
            self._record_candidate_baseline(i, 1.0 - result, False)
            _add(self.base_recs[k], result, error)

    def _record_candidate_baseline(self, i: int, result: float, error: bool) -> None:
        _add(self.recs[i], result, error)
        _add(self.brecs[i], result, error)

    # --- scoring ----------------------------------------------------------

    def _score_for(self, i: int) -> float:
        record = self.recs[i]
        penalty = _ERROR_PENALTY * record["errors"] / record["games"] if record["games"] else 0.0
        return _clamp(_combined_rate(self.irecs[i], self.brecs[i]) - penalty)

    def _eligible(self) -> list[int]:
        indices = list(range(self.n_original))
        if self.mutation_accepted and self.mutation_index is not None:
            indices.append(self.mutation_index)
        return indices

    def _ranking(self) -> list[int]:
        return sorted(
            self._eligible(),
            key=lambda i: (-self._score_for(i), -_rate(self.irecs[i]), i),
        )

    def _candidate_view(self, i: int) -> dict:
        record = self.recs[i]
        view = _record_view(record)
        view.update(
            {
                "name": self.cand[i].name,
                "kind": self.cand[i].kind,
                "internal": _record_view(self.irecs[i]),
                "baseline": _record_view(self.brecs[i]),
                "score": self._score_for(i),
                "error_rate": record["errors"] / record["games"] if record["games"] else 0.0,
            }
        )
        return view

    def _baseline_view(self, k: int) -> dict:
        record = self.base_recs[k]
        view = _record_view(record)
        penalty = _ERROR_PENALTY * record["errors"] / record["games"] if record["games"] else 0.0
        view.update(
            {
                "name": self.base[k].name,
                "kind": self.base[k].kind,
                "score": _clamp(_rate(record) - penalty),
                "error_rate": record["errors"] / record["games"] if record["games"] else 0.0,
            }
        )
        return view

    # --- mutation ---------------------------------------------------------

    def _maybe_mutate(self) -> None:
        if self.mutate is None or self.budget <= 0.0 or self.aborted:
            return
        remaining = self.deadline - time.monotonic()
        if remaining < _MUTATION_MIN_FRACTION * self.budget:
            return

        self.mutation_attempted = True
        incumbent_index = self._ranking()[0]
        incumbent = self.cand[incumbent_index]
        self.mutation_incumbent = incumbent.name
        self.log.append(
            {
                "event": "mutation_request",
                "incumbent": incumbent.name,
                "remaining_s": round(max(0.0, remaining), 3),
            }
        )

        proposed = None
        try:
            proposed = self.mutate(incumbent, self._snapshot(), self.log)
        except Exception:
            proposed = None

        if not isinstance(proposed, Candidate):
            self.log.append({"event": "mutation_ignored", "reason": "not a Candidate"})
            return

        mutant_index = len(self.cand)
        self.cand.append(proposed)
        self.recs.append(_blank())
        self.irecs.append(_blank())
        self.brecs.append(_blank())
        used = set(self.keys)
        self.keys.append(_unique(proposed.name, used))

        # Head-to-head against the incumbent, then the baseline gauntlet, all under the same clock.
        self._internal_pair(mutant_index, incumbent_index, self.next_pair_index)
        self.next_pair_index += 1
        for k in range(len(self.base)):
            if self._out_of_time():
                self.aborted = True
                break
            self._baseline_pair(mutant_index, k, self.next_pair_index)
            self.next_pair_index += 1

        head_to_head = self.irecs[mutant_index]
        self.mutation_h2h = {"games": head_to_head["games"], "points": head_to_head["points"]}
        self.mutation_view = self._candidate_view(mutant_index)

        wins_head_to_head = (
            head_to_head["games"] > 0
            and head_to_head["points"] > 0.5 * head_to_head["games"]
        )
        beats_score = self._score_for(mutant_index) > self._score_for(incumbent_index)
        accepted = bool(wins_head_to_head or beats_score)
        self.mutation_accepted = accepted
        if accepted:
            self.mutation_index = mutant_index
        self.log.append(
            {
                "event": "mutation_decision",
                "accepted": accepted,
                "h2h_games": head_to_head["games"],
                "h2h_points": head_to_head["points"],
            }
        )

    def _snapshot(self) -> dict:
        return {
            "games_played": self.games,
            "games": self.games,
            "games_per_pair": self.games_per_pair,
            "elapsed_s": time.monotonic() - self.started,
            "timed_out": self.aborted,
            "mutation_attempted": self.mutation_attempted,
            "scores": {self.keys[i]: self._candidate_view(i) for i in self._eligible()},
            "baselines": {
                self.base_keys[k]: self._baseline_view(k) for k in range(len(self.base))
            },
            "ranking": [self.keys[i] for i in self._ranking()],
        }

    # --- main loop / result ----------------------------------------------

    def execute(self) -> tuple[Candidate, dict]:
        try:
            for kind, i, j, pair_index in self.schedule:
                if self._out_of_time():
                    self.aborted = True
                    break
                if kind == "internal":
                    self._internal_pair(i, j, pair_index)
                else:
                    self._baseline_pair(i, j, pair_index)
            self._maybe_mutate()
        except Exception:
            # A bad play/mutate callback must never take prep down.
            self.aborted = True
            self.log.append({"event": "internal_error"})
        return self._finalize()

    def _finalize(self) -> tuple[Candidate, dict]:
        now = time.monotonic()
        elapsed = now - self.started
        timed_out = self.aborted or (self.games > 0 and now > self.deadline)

        eligible = self._eligible()
        ranking = self._ranking()
        scores = {self.keys[i]: self._candidate_view(i) for i in eligible}
        baselines = {
            self.base_keys[k]: self._baseline_view(k) for k in range(len(self.base))
        }

        for i in eligible:
            self.cand[i].score = scores[self.keys[i]]["score"]
        for k in range(len(self.base)):
            self.base[k].score = baselines[self.base_keys[k]]["score"]

        winner_index = ranking[0]
        winner = self.cand[winner_index]
        if len(ranking) > 1:
            margin = (
                scores[self.keys[ranking[0]]]["score"]
                - scores[self.keys[ranking[1]]]["score"]
            )
        else:
            baseline_scores = [
                view["score"] for view in baselines.values() if view["games"]
            ]
            margin = (
                scores[self.keys[winner_index]]["score"] - max(baseline_scores)
                if baseline_scores
                else 0.0
            )

        stats = {
            "games_played": self.games,
            "games": self.games,
            "games_per_pair": self.games_per_pair,
            "elapsed_s": elapsed,
            "timed_out": timed_out,
            "mutation_attempted": self.mutation_attempted,
            "mutation_accepted": self.mutation_accepted,
            "mutation": {
                "attempted": self.mutation_attempted,
                "accepted": self.mutation_accepted,
                "incumbent": self.mutation_incumbent,
                "head_to_head": dict(self.mutation_h2h),
                "candidate": self.mutation_view,
            },
            "scores": scores,
            "baselines": baselines,
            "ranking": [self.keys[i] for i in ranking],
            "winner": self.keys[winner_index],
            "winner_margin": margin,
            "budget_s": self.budget,
            "log": self.log,
        }
        return winner, stats


# --- public API ---------------------------------------------------------------

def run(
    *,
    candidates: list[Candidate],
    baselines: list[Candidate],
    play,
    budget_s: float,
    seed: int,
    mutate=None,
    games_per_pair: int = 1,
) -> tuple[Candidate, dict]:
    """Rank ``candidates`` by playing them, timeboxed, and return ``(best, stats)``.

    ``play(a, b, seed) -> float`` returns 1.0/0.5/0.0 from ``a``'s perspective. Each pair plays
    ``games_per_pair`` games in each orientation (clamped to 1..12; non-int values mean 1). Every game
    has a distinct seed derived from ``(seed, pair index, repetition index, orientation)``, and both
    orientations of one repetition share the repetition root. ``mutate(best, stats, log)`` is attempted
    at most once when at least 30% of the budget remains. Never raises: on a deadline, a bad callback or
    empty input it returns the best candidate seen so far plus a stats dict with ``timed_out`` set.
    """
    started = time.monotonic()
    try:
        budget = max(0.0, float(budget_s))
    except (TypeError, ValueError):
        budget = 0.0
    games_per_pair = _normalise_games_per_pair(games_per_pair)

    valid_candidates = [c for c in (candidates or []) if isinstance(c, Candidate)]
    valid_baselines = [b for b in (baselines or []) if isinstance(b, Candidate)]

    if not valid_candidates or not callable(play):
        best = (
            valid_candidates[0]
            if valid_candidates
            else (valid_baselines[0] if valid_baselines else Candidate(kind="strategy", name="fallback"))
        )
        reason = "no candidates" if not valid_candidates else "play is not callable"
        return best, _minimal_stats(best, started, budget, games_per_pair, reason)

    try:
        return _Run(
            valid_candidates,
            valid_baselines,
            play,
            budget,
            seed,
            mutate,
            games_per_pair,
        ).execute()
    except Exception:
        best = valid_candidates[0]
        return best, _minimal_stats(best, started, budget, games_per_pair, "internal error")
