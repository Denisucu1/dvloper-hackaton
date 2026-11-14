"""RECONSTRUCTION of the T1 game `connect_four` - not the official engine.

This module is our best reading of the T1 brief plus what our probes saw, exposed under the
round-robin id `connect_four`. It is not a copy of, and makes no claim to match, the organizers'
engine: rules wording, board rendering and edge-case handling may differ. All game logic is the
shared `tg_connect4.TestConnect4` implementation; only the metadata, the id and the variant list
are pinned here so the pipeline and bench can run against the T1 id and shape.

The T1 shape is the reason this file exists: a 9-column, 7-row board needing five in a row
(the "7x9, connect 5" variant), alongside the familiar 7x6 connect-four board.

    python -m arena practice --games-dir games --game connect_four --bot-a ... --bot-b ...
"""
from __future__ import annotations

from .tg_connect4 import TestConnect4


class ConnectFour(TestConnect4):
    id = "connect_four"
    name = "Connect Four"
    category = "think"
    # Shape-neutral: the parent blurb says "line of four", wrong for the connect-5 variant.
    blurb = "Drop a disc, build an unbroken line, and mind the gravity."
    # Rules, observations and all game logic are inherited from TestConnect4; they render
    # self.cols/self.rows/self.k, so the text always names the real shape.
    duplicate_safe = True

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            # The T1 shape from the brief: 9 columns, 7 rows, five in a row.
            {"label": "7x9, connect 5", "cols": 9, "rows": 7, "connect": 5},
            {"label": "7x6, connect 4", "cols": 7, "rows": 6, "connect": 4},
        ]
