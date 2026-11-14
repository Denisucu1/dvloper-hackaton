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
    "guessing": ("guess", "higher", "lower", "feedback", "secret number"),
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
        if spec.simultaneous and spec.family in ("unknown", "guessing"):
            spec.family = "read_opponent"
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
    if any(cue in lowered for cue in _SPEECH_CUES) and (
        re.search(r"forbidden\s+(?:subject|topic|word|term)", lowered) or extract_forbidden(text)
    ):
        return "guardrails"
    scores = {family: 0 for family in _FAMILY_HINTS}
    for family, words in _FAMILY_HINTS.items():
        for word in words:
            weight = 3 if family == "read_opponent" and (" " in word or word == "simultaneous") else 1
            scores[family] += lowered.count(word) * weight
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


_FORBIDDEN_NAMED = [
    # "The forbidden subject of this game is: pineapple" / "Banned word: moon" / "forbidden topic (...): X"
    re.compile(r"(?i)(?:forbidden|banned|prohibited|taboo)\s+(?:subject|word|topic|term|thing|name)s?\b[^.\n:]{0,60}?(?:\bis\b|\bare\b|:)\s*:?\s*[\"'“]?([^\"'”\n.;:()]{2,40})"),
    # never mention "pineapple" / must not say the word 'moon' / not allowed to say "pineapple"
    re.compile(r"(?i)(?:(?:never|not|n't|cannot)\s+(?:(?:allowed|permitted)\s+to\s+)?(?:mention|say|name|use|utter|discuss|write|reveal)|avoid\s+(?:mentioning|saying|naming|using))\b[^.;,\n\"'“]{0,25}[\"'“]([^\"'”\n]{2,40})[\"'”]"),
]
_META_WORDS = {"forbidden", "subject", "mention", "answer", "cannot"}
_PRONOUNS = {"it", "this", "that", "them", "anything", "something", "the", "a"}


def is_meta_forbidden(subject: str) -> bool:
    """'forbidden subject', 'answer I cannot say that': phrases about the rule, not the subject itself."""
    if "<" in subject or ">" in subject:
        return True
    words = re.findall(r"\w+", subject.casefold())
    if not words or len(words) > 3 or set(words) <= _PRONOUNS:
        return True
    return bool(_META_WORDS & set(words))


def extract_forbidden(rules: str) -> list[str]:
    """Forbidden subjects only where text explicitly names them, so quoted answers ('Paris') are not captured."""
    found: list[str] = []
    seen: set[str] = set()

    for rx in _FORBIDDEN_NAMED:
        for match in rx.finditer(rules or ""):
            item = match.group(1).strip(" \t-*_")
            key = item.casefold()
            if len(item) >= 2 and key not in seen and not is_meta_forbidden(item):
                seen.add(key)
                found.append(item[:40])
    return found[:6]


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
