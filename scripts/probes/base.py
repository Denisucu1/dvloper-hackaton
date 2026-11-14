"""Probe contract, registry and tiny dependency-free statistics helpers.

A probe is one self-contained, offline check of one harness component. It must:

- return ``score`` as a normalized number in ``[0.0, 1.0]`` where ``1.0`` means every check in the
  probe passed (``report.py`` prints it multiplied by 100). A probe with a continuous metric may
  document its own normalization in its module docstring, but the range never changes;
- return ``n`` (the number of observations/checks behind the score), ``p50_ms`` and ``p95_ms``
  (nearest-rank percentiles of the per-check wall times), a one-line ``notes`` string and a
  ``details`` dict with the measured values and any failures;
- never touch the network, the gateway, arena-lite or another probe's files;
- use deterministic inputs (fixed seeds / fixtures) and finish quickly. The whole registry must run
  in well under 60 seconds.

``scripts/report.py`` executes the registry, aggregates by component and stores a JSONL history.
The helpers here exist so every probe reports the same shape with the same statistics rules.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:  # probes import `harness.*`; make that work from any cwd
    sys.path.insert(0, ROOT)

SCORE_MIN = 0.0
SCORE_MAX = 1.0
# Fixed in one place so two runs on the same machine compare like with like.
DEFAULT_SEED = 20261114

REQUIRED_KEYS = ("score", "n", "p50_ms", "p95_ms", "notes", "details")


def clamp_score(value: Any) -> float:
    """Coerce a score into [0, 1]; a broken probe must not poison the aggregate."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return SCORE_MIN
    if math.isnan(number):
        return SCORE_MIN
    return min(SCORE_MAX, max(SCORE_MIN, number))


def percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile: no interpolation, no numpy, stable for tiny samples.

    ``p50`` of four samples is the lower median (rank 2), ``p95`` is the maximum. Every probe uses
    this function, so the report's percentile columns mean the same thing everywhere.
    """
    sample = sorted(float(value) for value in values)
    if not sample:
        return 0.0
    if len(sample) == 1:
        return round(sample[0], 3)
    rank = max(1, math.ceil(float(q) * len(sample)))
    return round(sample[min(rank, len(sample)) - 1], 3)


def make_result(
    *,
    score: Any,
    latencies: Iterable[float] = (),
    n: int | None = None,
    notes: str = "",
    details: dict | None = None,
) -> dict:
    """Build the canonical probe result dict. Probes should use this instead of hand-writing keys."""
    times = [float(value) for value in latencies]
    return {
        "score": clamp_score(score),
        "n": int(n if n is not None else len(times)),
        "p50_ms": percentile(times, 0.50),
        "p95_ms": percentile(times, 0.95),
        "notes": str(notes or "").strip(),
        "details": dict(details or {}),
    }


@dataclass
class CheckRun:
    """Outcome of a list of timed boolean checks."""

    passed: int
    total: int
    latencies: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return clamp_score(self.passed / self.total) if self.total else 0.0


def collect(checks: Iterable[tuple[str, Callable[[], Any]]]) -> CheckRun:
    """Run ``(label, callable)`` checks, timing each one.

    A truthy return is a pass. An exception is a failure with the exception text recorded, so one
    broken fixture never aborts the whole probe. Failures are labelled, which is what the report
    shows in its FAILURES section.
    """
    run = CheckRun(passed=0, total=0)
    for label, check in checks:
        run.total += 1
        started = time.perf_counter()
        failure = ""
        try:
            ok = bool(check())
        except Exception as exc:  # noqa: BLE001 - a probe must report, not raise, on a bad fixture
            ok = False
            failure = f"{label}: {type(exc).__name__}: {exc}"
        finally:
            run.latencies.append((time.perf_counter() - started) * 1000.0)
        if ok:
            run.passed += 1
        else:
            run.failures.append(failure or label)
    return run


def summarize(run: CheckRun, *, notes: str = "", details: dict | None = None) -> dict:
    """Turn a ``CheckRun`` into a result dict with the standard check counts in ``details``."""
    payload = dict(details or {})
    payload.setdefault("checks", run.total)
    payload.setdefault("passed", run.passed)
    if run.failures:
        payload.setdefault("failures", list(run.failures))
    return make_result(
        score=run.score,
        latencies=run.latencies,
        n=run.total,
        notes=notes or ("all checks passed" if not run.failures else "; ".join(run.failures)[:160]),
        details=payload,
    )


class Probe:
    """Base class. Subclasses set ``name``/``component`` and implement ``run``."""

    name = "unnamed"
    component = "unknown"
    # Slow probes are skipped by ``report.py --fast`` (subprocesses, wide searches, ...).
    slow = False

    def run(self) -> dict:
        raise NotImplementedError


_REGISTRY: list[Probe] = []


def register(cls: type[Probe]) -> type[Probe]:
    """Class decorator that instantiates a probe once and adds it to the registry.

    Re-importing a module (``scripts.probes.x`` and ``probes.x``) is idempotent by name so the CLI
    and the test runner can import the package in different ways without duplicating rows.
    """
    probe = cls()
    if any(existing.name == probe.name for existing in _REGISTRY):
        return cls
    _REGISTRY.append(probe)
    return cls


def registry() -> list[Probe]:
    """All registered probes, in import order (which is also the report's row order)."""
    return list(_REGISTRY)


def selected(component: str | None = None) -> list[Probe]:
    """Registry filtered by ``--component``: exact component or probe name, or a name prefix."""
    if not component:
        return registry()
    wanted = component.strip().lower()
    found = []
    for probe in _REGISTRY:
        names = (probe.component.lower(), probe.name.lower())
        if wanted in names or probe.name.lower().startswith(wanted + "."):
            found.append(probe)
    return found
