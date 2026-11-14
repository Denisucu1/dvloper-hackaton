"""Team adn test game (chance family): two push-your-luck variants under one dice id.

Reuses the practice Pig engine, but with its own id and different targets, so the harness meets a game
whose outcome depends on dice rather than on a board. Used to check that chance handling and the
hold/roll threshold logic do not depend on the exact practice-game id.

    python -m arena practice --games-dir games --game tg_dice_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

from arena.practice.pig import Pig


class TestDice(Pig):
    id = "tg_dice_probe"
    name = "Dice Push (test)"
    category = "guess"
    blurb = "Roll, build a turn total, hold before a one wipes it."
    howto = "On your turn roll the dice and bank the points, or hold. Rolling a one loses the turn."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "One die, target 30", "dice": 1, "target": 30},
            {"label": "Two dice, target 40 (snake eyes reset)", "dice": 2, "target": 40},
        ]
