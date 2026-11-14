"""What we learned from a game's rules, in a shape code can use.

`prep` fills this in with the model (structured JSON), and a regex-only fallback is always available so
a cold container (no prep, 120 s study clock) still gets a family, an action shape and a speech filter.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

FAMILIES = (
    "bluff",
    "deals",
    "read_opponent",
    "guessing",
    "think_ahead",
    "truth",
    "guardrails",
    "unknown",
)

_FAMILY_HINTS = {
    "bluff": ("bluff", "bet", "raise", "fold", "wager", "pot", "chips", "hidden card", "odds"),
    "deals": ("offer", "accept", "reject", "negotiat", "auction", "bid", "price", "split", "deal"),
    "read_opponent": ("simultaneous", "at the same time", "both players choose", "mixed strateg"),
    "guessing": ("guess", "higher", "lower", "feedback", "round", "number"),
    "think_ahead": ("board", "row", "column", "diagonal", "capture", "piece", "move", "legal"),
    "truth": ("document", "passage", "question", "unknown", "dossier", "quote", "source"),
    "guardrails": ("forbidden", "never mention", "must not say", "do not mention", "banned"),
}

_SPEECH_CUES = (
    "forbidden",
    "never mention",
    "must not mention",
    "must never say",
    "do not mention",
    "never say",
    "not allowed to say",
    "may not say",
)


@dataclass
class Spec:
    game_id: str = ""
    family: str = "unknown"
    action_format: str = ""
    action_examples: list[str] = field(default_factory=list)
    fallback_action: str = ""
    winning_idea: str = ""
    hidden_information: bool = False
    simultaneous: bool = False
    scoring: str = ""
    speech_rules: str = ""
    forbidden_subjects: list[str] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)
    notes: str = ""
    playbook: str = ""
    fingerprint: str = ""
    source: str = "fallback"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None, fp: str = "") -> "Spec":
        data = dict(data or {})
        spec = cls()
        for key in ("game_id", "family", "action_format", "winning_idea", "scoring", "speech_rules", "notes", "playbook"):
            value = data.get(key)
            if isinstance(value, str):
                setattr(spec, key, value.strip())
        fb = str(data.get("fallback_action") or "").strip()
        if (
            fb
            and len(fb) <= 40
            and "\n" not in fb
            and "<" not in fb
            and ">" not in fb
            and "legal action" not in fb.lower()
        ):
            spec.fallback_action = fb
        for key in ("hidden_information", "simultaneous"):
            setattr(spec, key, bool(data.get(key)))
        for key in ("action_examples", "forbidden_subjects", "traps"):
            value = data.get(key)
            if isinstance(value, list):
                setattr(spec, key, [str(item).strip() for item in value if str(item).strip()][:8])
        if not spec.fallback_action:
            for ex in spec.action_examples:
                if ex and len(ex) <= 40 and "<" not in ex and ">" not in ex:
                    spec.fallback_action = ex
                    break
        if spec.family not in FAMILIES:
            spec.family = guess_family(f"{spec.action_format} {spec.notes}")
        if not spec.family:
            spec.family = "unknown"
        spec.fingerprint = fp
        spec.source = "prep" if data else "fallback"
        return spec

    def prompt_block(self, limit: int = 1200) -> str:
        """Compact text for a per-turn prompt; keeps the prompt small enough to stay cheap."""
        lines = [
            f"family: {self.family}",
            f"action format: {self.action_format or '(see rules)'}",
        ]
        if self.action_examples:
            lines.append("action examples: " + " | ".join(self.action_examples[:5]))
        if self.winning_idea:
            lines.append(f"winning idea: {self.winning_idea}")
        if self.hidden_information:
            lines.append("hidden information: yes (never reveal your private state)")
        if self.simultaneous:
            lines.append("moves are simultaneous: do not assume the opponent's move is known")
        if self.forbidden_subjects:
            lines.append("FORBIDDEN TOPICS (never mention, not even in chat): " + ", ".join(self.forbidden_subjects))
        if self.speech_rules:
            lines.append(f"speech rules: {self.speech_rules}")
        if self.traps:
            lines.append("traps: " + " | ".join(self.traps[:4]))
        text = "\n".join(lines)
        return text[:limit]


def guess_family(text: str) -> str:
    lowered = (text or "").lower()
    scores = {family: 0 for family in _FAMILY_HINTS}
    for family, words in _FAMILY_HINTS.items():
        for word in words:
            scores[family] += lowered.count(word)
    best = max(scores, key=lambda name: scores[name])
    return best if scores[best] > 0 else "unknown"


def extract_action_examples(rules: str) -> list[str]:
    """'guess <n>', "hold", ACTION: C1 -> the shapes the arena accepts when legal_actions is null."""
    examples: list[str] = []
    for match in re.finditer(r"[\"“']([^\"“”']{1,40})[\"”']", rules or ""):
        candidate = match.group(1).strip()
        if any(char in candidate for char in "<>{}") and len(candidate) <= 40:
            examples.append(candidate)
    for match in re.finditer(r"(?im)^\s*(?:action|move|answer)\s*[:=]\s*(.{1,40})$", rules or ""):
        examples.append(match.group(1).strip())
    for match in re.finditer(r"(?im)\b(?:exactly|format)\s*[:=]?\s*[\"“']?([a-z]+ <[^>]{1,20}>[^\"”'\n]{0,20})", rules or ""):
        examples.append(match.group(1).strip())
    seen, unique = set(), []
    for example in examples:
        key = example.lower()
        if key not in seen and len(example) <= 60:
            seen.add(key)
            unique.append(example)
    return unique[:8]


def extract_forbidden(rules: str) -> list[str]:
    """Best-effort denylist for the Guardrails family. The model does this better in prep; this is the net."""
    found: list[str] = []
    lowered = (rules or "").lower()
    for cue in _SPEECH_CUES:
        start = 0
        while True:
            index = lowered.find(cue, start)
            if index < 0:
                break
            window = (rules or "")[index : index + 220]
            for match in re.finditer(r"[\"“']([^\"“”'\n]{1,40})[\"”']", window):
                found.append(match.group(1).strip())
            for match in re.finditer(r"(?i)(?:mention|say|discuss|refer to|name)\s+(?:the\s+)?(?:word\s+|subject\s+|topic\s+)?([a-z][\w -]{2,30})", window):
                found.append(match.group(1).strip())
            start = index + len(cue)
    cleaned, seen = [], set()
    for item in found:
        key = item.lower().strip()
        if key and key not in seen and key not in {"the", "it", "this", "that"}:
            seen.add(key)
            cleaned.append(item[:40])
    return cleaned[:6]


def fallback_spec(game_id: str, rules: str, samples: list[dict] | None = None, fp: str = "") -> Spec:
    """Cold-container spec: no model call needed, always something to work with."""
    sample_actions: list[str] = []
    for sample in samples or []:
        for action in (sample.get("legal_actions") or [])[:4]:
            sample_actions.append(str(action))
    action_format = ""
    match = re.search(r"(?i)(?:exactly|format|type)\s*[:=]?\s*[\"“']([^\"”'\n]{1,60})", rules or "")
    if match:
        action_format = match.group(1).strip()
    return Spec(
        game_id=str(game_id or ""),
        family=guess_family(rules),
        action_format=action_format,
        action_examples=(extract_action_examples(rules or "") + sample_actions)[:8],
        hidden_information=bool(re.search(r"(?i)\b(hidden|secret|private|your hand|your cards|face down)\b", rules or "")),
        simultaneous=bool(re.search(r"(?i)\b(simultaneous|at the same time|both players choose)\b", rules or "")),
        speech_rules="",
        forbidden_subjects=extract_forbidden(rules or ""),
        fingerprint=fp,
        source="fallback",
    )
