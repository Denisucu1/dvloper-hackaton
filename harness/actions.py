"""Action safety: nothing reaches stdout unless it is a plausible, legal action.

Rulebook 8.1: the first bad turn answer is a warning, the second loses the game. A verifier is the
cheapest insurance in the whole harness, because it also protects the warning tie-break (10.4).
"""
from __future__ import annotations

import re
from typing import Iterable

_STRIP = " \t\r\n"


def clean(action: object) -> str:
    """A protocol-safe action string: one line, no quotes added by a chatty model, no JSON braces."""
    if action is None:
        return ""
    text = str(action).replace("\r", " ").replace("\n", " ")
    text = text.strip(_STRIP)
    text = text.strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    return text[:120]


def normalize(candidate: object, legal: Iterable[str] | None) -> str | None:
    """Map a candidate onto the arena's exact legal-action spelling, or return None."""
    action = clean(candidate)
    if not action:
        return None
    if not legal:
        return action
    legal_list = list(legal)
    if action in legal_list:
        return action
    lowered = {str(item).strip().lower(): str(item) for item in legal_list}
    key = action.lower()
    if key in lowered:
        return lowered[key]
    # "guess 36." / "C1 (center)" / "play C1" -> try the first legal token inside the answer
    for token in re.findall(r"[^\s,;]+", key):
        token = token.strip("\"'`()[]{}:.")
        if token in lowered:
            return lowered[token]
    return None


def choose_fallback(legal: Iterable[str] | None, avoid: Iterable[str] | None = None) -> str | None:
    """A guaranteed-legal move when the model and the engine both failed.

    Prefers an action not played before, so a fallback does not look like a frozen bot to the judges.
    Returns None only when the game gave us no legal list (then the caller needs a format guess).
    """
    if not legal:
        return None
    legal_list = [str(item) for item in legal]
    if not legal_list:
        return None
    seen = {str(item) for item in (avoid or [])}
    for action in legal_list:
        if action not in seen:
            return action
    return legal_list[0]


def is_plausible(action: str, examples: Iterable[str] | None = None, allow_unknown: bool = False) -> bool:
    """When `legal_actions` is null we cannot verify; reject obvious garbage instead.

    Rulebook 5.8: with more than 50 moves the list is null and the rules define the format, so the
    spec's examples are the only shape check available before the arena answers with a warning.
    `allow_unknown` is set for Truth, where "unknown" is a real, preferred answer.
    """
    if not action or len(action) > 120:
        return False
    if any(token in action for token in ("{", "}", "[", "]", "\x00")):
        return False
    if action.lower() in {"none", "null", "n/a"} or (action.lower() == "unknown" and not allow_unknown):
        return False
    patterns = [str(example) for example in (examples or []) if example]
    if not patterns:
        return True
    shape = action.lower()
    for pattern in patterns:
        template = re.sub(r"<[^>]+>", "", str(pattern).lower())
        template = re.sub(r"\d+", "#", template).strip()
        candidate = re.sub(r"\d+", "#", shape).strip()
        if template and (template in candidate or candidate.startswith(template.split()[0])):
            return True
    # Any short, single-line, digit-bearing or word action is still worth trying once.
    return bool(re.fullmatch(r"[\w .:+/#-]{1,60}", action))
