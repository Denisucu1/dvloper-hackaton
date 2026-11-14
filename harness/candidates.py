"""Candidates: the unit of the prep tournament.

A candidate is one way to play the game: a generated simulator, a generated policy, or a hand-written
strategy (matrix solver, CFR table, expectimax, the current alpha-beta path). The prep tournament ranks
them by playing, not by reading prompts, then keeps the winner.

This module is intentionally tiny and dependency-free: everything else imports it, so the shape must not
change without updating the tournament, the probes and prep integration together.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Candidate:
    kind: str                      # "model" | "policy" | "strategy"
    name: str                      # short stable id, e.g. "model-v2", "matrix", "cfr-leduc"
    path: str | None = None        # file the candidate lives in, if any
    meta: dict = field(default_factory=dict)   # family, variant fingerprints, notes, params
    score: float = 0.0             # filled by the tournament (higher is better)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        return cls(
            kind=str(data.get("kind") or "strategy"),
            name=str(data.get("name") or "candidate"),
            path=data.get("path"),
            meta=dict(data.get("meta") or {}),
            score=float(data.get("score") or 0.0),
        )


def save(path: str | Path, candidates: list[Candidate]) -> None:
    Path(path).write_text(json.dumps([c.to_dict() for c in candidates], indent=1), encoding="utf-8")


def load(path: str | Path) -> list[Candidate]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Candidate.from_dict(row) for row in data]
