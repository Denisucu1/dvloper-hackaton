"""Probe package for the T2 testing interface.

Importing this package registers every probe exactly once. ``scripts/report.py`` uses
``helpers``/``base``; tests import the same package, so registry order (and therefore table order)
is always: actions, chat, memory, search, solvers, engine, bench, protocol.
"""
from __future__ import annotations

from scripts.probes.base import (
    DEFAULT_SEED,
    SCORE_MAX,
    SCORE_MIN,
    CheckRun,
    Probe,
    clamp_score,
    collect,
    make_result,
    percentile,
    register,
    registry,
    selected,
    summarize,
)

# Import order is the report's row order. Keep it stable across runs.
from scripts.probes import actions_probe  # noqa: F401,E402
from scripts.probes import chat_gate_probe  # noqa: F401,E402
from scripts.probes import memory_probe  # noqa: F401,E402
from scripts.probes import search_probe  # noqa: F401,E402
from scripts.probes import solvers_probe  # noqa: F401,E402
from scripts.probes import matrix_probe  # noqa: F401,E402
from scripts.probes import chance_probe  # noqa: F401,E402
from scripts.probes import cfr_probe  # noqa: F401,E402
from scripts.probes import engine_probe  # noqa: F401,E402
from scripts.probes import selfplay_probe  # noqa: F401,E402
from scripts.probes import tournament_probe  # noqa: F401,E402
from scripts.probes import bench_probe  # noqa: F401,E402
from scripts.probes import prep_policy_probe  # noqa: F401,E402
from scripts.probes import protocol_probe  # noqa: F401,E402

__all__ = [
    "DEFAULT_SEED",
    "SCORE_MAX",
    "SCORE_MIN",
    "CheckRun",
    "Probe",
    "clamp_score",
    "collect",
    "make_result",
    "percentile",
    "register",
    "registry",
    "selected",
    "summarize",
]
