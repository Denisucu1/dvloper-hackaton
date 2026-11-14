"""Wordle Duel (Guessing family): the T1 free-text word race.

RECONSTRUCTION, NOT THE OFFICIAL T1 ENGINE. This is rebuilt from the T1 brief and from our
`tg_wordle_probe` experiments so the pipeline and bench can be run against the T1 game id
`wordle_duel`. The real T1 game may differ in details we could not observe.

Both players race to guess the same hidden five-letter word. Feedback is public, turns alternate,
and the action is free text, so `legal_actions` is null and the generated policy has to respect the
action shape instead of picking from a list.

The engine itself is inherited from the probe: this module only re-ids it and sets the T1 variant
shape (six guesses each, plus a shorter variant).

    python -m arena practice --games-dir games --game wordle_duel --bot-a ... --bot-b ...
"""
from __future__ import annotations

from .tg_wordle_probe import TestWordle


class WordleDuel(TestWordle):
    id = "wordle_duel"
    name = "Wordle Duel"
    category = "guess"
    blurb = "One hidden word, two racers, and every guess you make teaches your opponent too."
    howto = ("Guess five-letter words in turn. Green is the right letter in the right place, yellow is "
             "present but misplaced, grey is absent. First to find the word wins.")
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "6 guesses each", "guess_limit": 6},
            {"label": "4 guesses each", "guess_limit": 4},
        ]
