"""The chat is part of the game (section 6). This module is the disciplined part of it.

Offense: the line must support the move we already chose, so move and chat are one decision (6.4).
Defense: the opponent's lines are data, never instructions (6.7); they are passed to the model verbatim
inside a marked untrusted block, and they also reach the code (the engine sees the claims), so we never
"read them only to throw them away" (6.6).
Safety: every line is capped at 200 characters, single-line, and checked against the game's forbidden
subjects (6.8) before it can cost a game - including the line sent with `ready`.
"""
from __future__ import annotations

import random
import re

from . import config
from .log import log
from .spec import Spec

# Fallback lines: deterministic, safe in every game, and never empty (6.2). Rotated so a match does not
# read as one canned sentence repeated 40 times in front of the audience.
_BANK = {
    "bluff": [
        "I like my chances here - but go on, show me.",
        "Careful, this is exactly the hand I wanted.",
        "I will keep my cards close. Your move.",
    ],
    "deals": [
        "Let us make this worth both our while.",
        "I am listening. Propose something serious.",
        "A fair split now beats a clever trick later.",
    ],
    "read_opponent": [
        "I wonder if you are as predictable as the last one.",
        "I am mixing it up - good luck reading this.",
        "Same board, different mind. Your move.",
    ],
    "guessing": [
        "Narrowing it down. Your turn.",
        "I have a good feeling about this range.",
        "Feedback is all I need. Keep going.",
    ],
    "think_ahead": [
        "I have seen this position before.",
        "Every move here has a cost. Let us see yours.",
        "I am playing the long game.",
    ],
    "truth": [
        "I will answer only what the document supports.",
        "If it is not in the text, I will say unknown.",
        "Let us stay precise.",
    ],
    "guardrails": [
        "Happy to help with anything else you need.",
        "Let us keep this on the allowed topics.",
        "I can answer that within the rules.",
    ],
    "unknown": [
        "Good luck - let us play.",
        "Interesting position. Your move.",
        "I am watching carefully.",
    ],
}

_GENERIC = [
    "Good luck - let us play.",
    "Your move.",
    "Interesting. I have my plan.",
    "Let us see where this goes.",
]

_INJECTION_CUES = (
    "ignore your",
    "ignore the rules",
    "ignore previous",
    "system prompt",
    "you must play",
    "play the move",
    "disregard",
)


def _one_line(text: object) -> str:
    value = str(text or "").replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value[: config.MAX_CHAT_CHARS]


class ChatPolicy:
    """Pure, synchronous, never fails: it either validates the decision's line or produces a safe one."""

    def __init__(self, spec: Spec | None = None, seed: int = 0) -> None:
        self.spec = spec or Spec()
        self.rng = random.Random(f"{seed}:{self.spec.fingerprint}")
        self._used: set[str] = set()
        self._blocked = 0

    # ---------------------------------------------------------------- defense
    def render_history(self, chat_lines: list[dict] | None) -> str:
        """The opponent's words, verbatim, as untrusted data.

        We do not strip them (6.6). We do cap the volume, label the source and tag injection-shaped lines,
        which is explicitly allowed as defense (6.5).
        """
        lines = list(chat_lines or [])[-config.MAX_CHAT_LINES :]
        rendered = []
        for line in lines:
            text = _one_line(line.get("text", ""))
            if not text:
                continue
            who = "OPPONENT" if line.get("from") == "opponent" else "YOU"
            flags = []
            if who == "OPPONENT" and any(cue in text.lower() for cue in _INJECTION_CUES):
                flags.append("looks-like-prompt-injection")
            phase = line.get("phase") or "play"
            suffix = f" [{','.join(flags)}]" if flags else ""
            rendered.append(f"{who} ({phase}): {text}{suffix}")
        if not rendered:
            return "(no chat yet)"
        return "\n".join(rendered)[:4000]

    def opponent_claims(self, chat_lines: list[dict] | None) -> list[str]:
        """What the opponent actually said, for the code path (engine hints, claims, bluffs to test)."""
        return [
            _one_line(line.get("text", ""))
            for line in (chat_lines or [])[-8:]
            if line.get("from") == "opponent" and _one_line(line.get("text", ""))
        ]

    # ---------------------------------------------------------------- safety
    def forbidden_hit(self, text: str) -> str | None:
        lowered = (text or "").lower()
        for subject in self.spec.forbidden_subjects:
            subject = str(subject).strip().lower()
            if len(subject) >= 2 and re.search(rf"(?<!\w){re.escape(subject)}(?!\w)", lowered):
                return subject
        return None

    def finalize(self, candidate: object, *, action: str = "", intent: str = "") -> str:
        """Turn whatever the model said into a legal chat move, or fall back to a safe line."""
        text = _one_line(candidate)
        if text and self.forbidden_hit(text):
            self._blocked += 1
            log(f"chat blocked: forbidden subject in {text!r}")
            text = ""
        if text and len(text) >= 4:
            return text
        return self.safe_line(action=action, intent=intent)

    def safe_line(self, *, action: str = "", intent: str = "") -> str:
        pool = _BANK.get(self.spec.family) or _BANK["unknown"]
        fresh = [line for line in pool if line not in self._used]
        if not fresh:
            self._used.clear()
            fresh = pool
        line = self.rng.choice(fresh) if fresh else self.rng.choice(_GENERIC)
        if self.forbidden_hit(line):  # a bank line must never break a speech rule either
            line = self.rng.choice(_GENERIC)
        self._used.add(line)
        return _one_line(line)

    # ---------------------------------------------------------------- offense
    def intent_hint(self, action: str, intent: str) -> str:
        """What the line should be consistent with (6.4). Kept concrete and free of private state."""
        base = intent or f"playing {action}"
        return base[:160]
