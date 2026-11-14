"""RECONSTRUCTION of the T1 game `mini_poker` - not the official engine.

This module is our best reading of the T1 brief plus what our probes saw, exposed under the
round-robin id `mini_poker`. It is not a copy of, and makes no claim to match, the organizers'
engine: betting details, deck composition and wording may differ. All game logic is the shared
`tg_leduc_probe.TestLeduc` implementation (simplified Leduc: J, Q, K twice, two betting rounds,
one community card); only the metadata, the id and the variant labels are pinned here.

    python -m arena practice --games-dir games --game mini_poker --bot-a ... --bot-b ...
"""
from __future__ import annotations

from .tg_leduc_probe import TestLeduc


class MiniPoker(TestLeduc):
    id = "mini_poker"
    name = "Mini Poker"
    category = "bluff"
    # Rules, observations and scoring are inherited from TestLeduc; the observation keeps the
    # exact Leduc layout that harness/solver_leduc.py parses.
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "Single hand, two betting rounds", "hands": 1},
            {"label": "Three hands, two betting rounds", "hands": 3},
        ]
