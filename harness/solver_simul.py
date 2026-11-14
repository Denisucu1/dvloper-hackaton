"""Adaptive mixed-strategy expert for simultaneous ("read the opponent") games.

Models cannot produce genuine randomness across repeated prompts and quickly collapse onto a single
deterministic favourite (such as playing 'rock' six rounds in a row). This solver restores true
CSPRNG entropy, solves weighted Rock-Paper-Scissors cycles via exact Nash equilibrium blended with
light empirical adaptation, and safely randomizes across finite simultaneous choices.
"""
from __future__ import annotations

import math
import re
import secrets

_RNG = secrets.SystemRandom()

_RPS = ("rock", "paper", "scissors")
_RPS_BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
_GAME_OVER = re.compile(r"\bgame\s+(?:is\s+)?over\b", re.IGNORECASE)
_NOT_SIMUL_FAMILY = {"deals", "bluff", "truth", "guardrails"}

_ROUND_PAIR = re.compile(r"Round\s+\d+\s*:\s*([a-z0-9_-]+)\s*/\s*([a-z0-9_-]+)", re.IGNORECASE)
_OPP_PLAYED = re.compile(
    r"\bopponent(?:'s)?\s+(?:played|chose|picked|used)\s+([a-z0-9_-]+)", re.IGNORECASE
)
_NUMERIC_LIKE = re.compile(r"^(?:\d+|(?:offer|bid|bet|raise|guess|answer)\s+.+)$", re.IGNORECASE)


def _rps_map(legal: list[str]) -> dict[str, str] | None:
    """Return {canonical: legal_spelling} when legal options are exactly Rock-Paper-Scissors."""
    by_lower = {item.strip().lower(): item for item in legal}
    if set(by_lower) == set(_RPS):
        return {name: by_lower[name] for name in _RPS}
    return None


def _parse_rps_weights(rules_text: str) -> dict[str, float]:
    """Read per-sign win points from rules (e.g. 'won with rock scores 3'); default 1 each."""
    weights = {sign: 1.0 for sign in _RPS}
    if not rules_text:
        return weights
    for sign in _RPS:
        match = re.search(
            rf"(?:with\s+{sign}\s+(?:scores?|wins?|gives?|gets?)|{sign}\s+wins?)\s+(\d+)",
            rules_text,
            re.IGNORECASE,
        )
        if match:
            value = int(match.group(1))
            if value >= 1:
                weights[sign] = float(value)
    return weights


def _opponent_counts(observation: str, signs: tuple[str, ...]) -> dict[str, int]:
    """Count opponent's past choices from history lines in the observation."""
    counts = {sign: 0 for sign in signs}
    valid = set(signs)
    for line in (observation or "").splitlines():
        m_pair = _ROUND_PAIR.search(line)
        if m_pair:
            theirs = m_pair.group(2).strip().lower()
            if theirs in valid:
                counts[theirs] += 1
            continue
        m_opp = _OPP_PLAYED.search(line)
        if m_opp:
            theirs = m_opp.group(1).strip().lower()
            if theirs in valid:
                counts[theirs] += 1
    return counts


def _rps_distribution(rules_text: str, observation: str) -> list[float]:
    """Blend exact weighted Nash (unexploitable) with soft empirical best-response."""
    w = _parse_rps_weights(rules_text)
    # Exact zero-sum mixed Nash equilibrium makes opponent indifferent across all signs:
    # (p_rock, p_paper, p_scissors) proportional to (w_scissors, w_rock, w_paper).
    nash_raw = [w["scissors"], w["rock"], w["paper"]]
    nash_total = sum(nash_raw) or 3.0
    nash = [v / nash_total for v in nash_raw]

    counts = _opponent_counts(observation, _RPS)
    prior = 2.0
    denom = sum(counts.values()) + prior * len(_RPS)
    q = {s: (counts[s] + prior) / denom for s in _RPS}

    evs = []
    for sign in _RPS:
        beats = _RPS_BEATS[sign]
        loses_to = next(k for k, v in _RPS_BEATS.items() if v == sign)
        evs.append(q[beats] * w[sign] - q[loses_to] * w[loses_to])

    temp = max(0.35, 0.25 * max(w.values()))
    max_ev = max(evs)
    exp_evs = [math.exp((ev - max_ev) / temp) for ev in evs]
    exp_sum = sum(exp_evs) or 3.0
    adaptive = [v / exp_sum for v in exp_evs]

    return [0.60 * n + 0.40 * a for n, a in zip(nash, adaptive)]


class SimultaneousSolver:
    """Nash-mixed and entropy-backed solver for simultaneous finite-choice games."""

    name = "simul"

    def __init__(self) -> None:
        self.rules_text = ""

    def configure(self, spec: object, rules: str = "") -> None:
        scoring = str(getattr(spec, "scoring", "") or "")
        notes = str(getattr(spec, "winning_idea", "") or "")
        self.rules_text = "\n".join(part for part in (rules, scoring, notes) if part)

    def matches(self, game_id: str, spec: object) -> bool:
        family = str(getattr(spec, "family", "") or "").lower()
        if family == "read_opponent":
            return True
        return bool(getattr(spec, "simultaneous", False)) and family not in _NOT_SIMUL_FAMILY

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        """Sample a genuinely unpredictable legal move appropriate for simultaneous play."""
        if not observation or _GAME_OVER.search(observation) or not legal:
            return None
        clean_legal = [str(item) for item in legal]
        if not (2 <= len(clean_legal) <= 12):
            return None
        # Do not randomize numeric bids, offers, or sequential free-form expressions.
        if sum(1 for item in clean_legal if _NUMERIC_LIKE.match(item.strip())) > len(clean_legal) // 2:
            return None

        rps = _rps_map(clean_legal)
        if rps is not None:
            weights = _rps_distribution(f"{self.rules_text}\n{observation}", observation)
            chosen_key = _RNG.choices(list(_RPS), weights=weights, k=1)[0]
            return rps[chosen_key]

        return _RNG.choice(clean_legal)
