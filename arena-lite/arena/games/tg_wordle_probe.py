"""Team adn test game (Guessing family): a two-player Wordle duel with free-text guesses.

Each side races to guess the same hidden five-letter word; feedback is public and turns alternate. It
exists to test the free-text path: `legal_actions` is null, the action is a word, and the generated code
must respect the action shape instead of assuming a list of moves.

    python -m arena practice --games-dir games --game tg_wordle_probe --bot-a ... --bot-b ...
"""
from __future__ import annotations

import re

from ..core import Game

_WORDS = (
    "crane slate trace crate plant stone honey mount share brave ghost light night water "
    "world music power river smile green black white cloud dream flame frost grape house "
    "juice knife lemon mango north ocean peach queen radio sugar tiger"
).split()
_WORD_RE = re.compile(r"^[a-z]{5}$")


class TestWordle(Game):
    id = "tg_wordle_probe"
    name = "Word Duel (test)"
    category = "guess"
    blurb = "Five letters, six guesses, and feedback that tells you exactly enough."
    howto = "Guess five-letter words. Green is the right letter in the right place, yellow is present but misplaced, grey is absent. First to find the word wins."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "6 guesses each", "guess_limit": 6},
            {"label": "4 guesses each", "guess_limit": 4},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.guess_limit = max(2, min(8, int(self.variant.get("guess_limit", 6))))
        self.label = str(self.variant.get("label", ""))
        self.secret = _WORDS[self.rng.randrange(len(_WORDS))]
        self.current = 0
        self.made = [0, 0]
        self.history: list[str] = []
        self.over = False
        self.winner: int | None = None
        self.end_reason = ""

    # ---------------------------------------------------------------- helpers
    def _feedback(self, guess: str) -> str:
        marks = []
        remaining = list(self.secret)
        for index, letter in enumerate(guess):
            if self.secret[index] == letter:
                marks.append("G")
                remaining[index] = "?"
            else:
                marks.append(".")
        for index, letter in enumerate(guess):
            if marks[index] == "." and letter in remaining:
                marks[index] = "Y"
                remaining[remaining.index(letter)] = "?"
        return "".join(marks)

    # --------------------------------------------------------------- interface
    def rules(self) -> str:
        return f"""WORD DUEL - {self.label}. 2 players, alternating guesses.

GOAL
Guess the same hidden five-letter word before your opponent. The word is one of {len(_WORDS)} common
English words. Both players see every guess and its feedback.

PLAY
Players alternate guesses, Player 0 first. Each player has at most {self.guess_limit} guesses. Feedback per
letter: G = right letter in the right place, Y = the letter is in the word but elsewhere, . = absent.
Only five-letter alphabetic guesses are allowed. The first player to type the secret word wins. If both
run out of guesses, the game is a draw.

ACTION FORMAT
Exactly one five-letter word, for example "crane". Case does not matter. Any other string is illegal
(first time: a warning and the same turn again; second time: you lose the game).

WHAT YOU OBSERVE EACH TURN
The previous guesses with their feedback, whose turn it is, and how many guesses each side has left."""

    def to_move(self) -> list[int]:
        return [] if self.over else [self.current]

    def legal_actions(self, seat: int) -> list[str] | None:
        # Any five-letter word is legal: more than 50 actions, so the arena sends null.
        return None if not self.over and seat == self.current else []

    def normalize(self, seat: int, action: str) -> str | None:
        word = self.clean(action).strip(" .!\"'`")
        if not _WORD_RE.match(word):
            return None
        return word if word in _WORDS else (word if len(word) == 5 else None)

    def apply(self, actions: dict[int, str]) -> None:
        if self.over:
            return
        actor = self.current
        guess = self.normalize(actor, actions.get(actor, ""))
        if guess is None:
            return
        feedback = self._feedback(guess)
        self.made[actor] += 1
        self.history.append(f"Player {actor} guessed {guess} -> {feedback}")
        if guess == self.secret:
            self.over = True
            self.winner = actor
            self.end_reason = f"seat {actor} found the word in {self.made[actor]} guess(es)"
            return
        self.current = 1 - actor
        if self.made[0] >= self.guess_limit and self.made[1] >= self.guess_limit:
            self.over = True
            self.end_reason = "both players ran out of guesses: draw"

    def is_over(self) -> bool:
        return self.over

    def result(self) -> dict:
        scores = [0.5, 0.5] if self.over and self.winner is None else [0, 0]
        if self.winner is not None:
            scores[self.winner] = 1
        return {"winner": self.winner, "scores": scores,
                "reason": self.end_reason or "game in progress"}

    def frame(self) -> dict:
        return {"label": self.label, "history": list(self.history), "made": list(self.made),
                "guess_limit": self.guess_limit, "to_move": self.to_move(), "over": self.over}

    def observation(self, seat: int) -> str:
        left = self.guess_limit - self.made[seat]
        return f"""WORD DUEL - {self.label}. You are Player {seat}.
Guesses left: you {left}, opponent {self.guess_limit - self.made[1 - seat]}.
Previous guesses:
{chr(10).join("  " + line for line in self.history[-10:]) or "  (none yet)"}
It is YOUR turn: guess a five-letter word."""
