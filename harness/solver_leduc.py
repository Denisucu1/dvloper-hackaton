"""Deterministic expert for Leduc-shaped poker (two betting rounds, one community card).

Why: the Bluff solver only understands one-card, one-round poker, so a Leduc game (the arena's
``mini_poker``) fell through to a language model. Leduc needs two extra facts the text already carries:
which betting round it is and, in round two, the community card that can pair our private card.

What it reads from the observation (no hidden state invented):
- our private card ("Your private card: Q" / "Your card: Q", J/Q/K or jack/queen/king);
- the community card in round two ("Community card: K");
- the round ("Round 2 of 2") and the pot ("Pot: 4");
- this round's action line ("This round so far: check, bet"), which must be a legal round prefix.

Policy: a hand-strength estimate over the cards still unseen, compared with the pot odds a call needs.
Strong hands (a pair, or the top card) bet; weak hands check, fold to a bet, and bluff at a fixed
rate seeded from the observation so the choice is deterministic and reproducible. It never raises,
never invents an action outside ``legal`` and returns None whenever the text is not clearly Leduc.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_ACTION_WORDS = {
    "check": "check",
    "checks": "check",
    "checked": "check",
    "pass": "check",
    "passes": "check",
    "bet": "bet",
    "bets": "bet",
    "betting": "bet",
    "raise": "bet",
    "raises": "bet",
    "call": "call",
    "calls": "call",
    "called": "call",
    "fold": "fold",
    "folds": "fold",
    "folded": "fold",
}
_CARD_WORDS = {"j": "J", "jack": "J", "q": "Q", "queen": "Q", "k": "K", "king": "K"}
_RANK = {"J": 0, "Q": 1, "K": 2}
_DECK = {"J": 2, "Q": 2, "K": 2}

_CARD = r"(jack|queen|king|[jqk])"
_PRIVATE_PATTERNS = (
    re.compile(rf"(?i)\byour\s+(?:private|hidden)\s+card\s*(?:is|:|=|-)?\s*(?:the\s+)?{_CARD}\b"),
    re.compile(rf"(?i)\byour\s+card\s*(?:is|:|=|-)?\s*(?:the\s+)?{_CARD}\b"),
)
_COMMUNITY_PATTERN = re.compile(
    rf"(?i)\b(?:community|board|shared)\s+card\s*(?:is|:|=|-)?\s*(?:the\s+)?{_CARD}\b"
)
_ROUND_PATTERN = re.compile(r"(?i)\bround\s+(\d+)\s*(?:of|/)\s*\d+\b")
_ROUND_LOOSE = re.compile(r"(?i)\bround\s+(\d+)\b")
_POT_PATTERN = re.compile(r"(?i)\bpot\s*(?:is|:|=|-)?\s*(\d+)\b")
_LINE_PATTERN = re.compile(r"(?i)\b(?:this|current)\s+(?:betting\s+)?round\s+so\s+far\s*:\s*(.*)")
_SEAT_PATTERN = re.compile(r"(?i)\byou\s+are\s+(?:player|seat)\s*([01])\b")
_OPPONENT_TURN = re.compile(
    r"(?i)\bit\s+is\s+the\s+opponent(?:['\u2019]s)?\s+turn\b|\bnot\s+your\s+turn\b"
)
_GAME_OVER = re.compile(r"(?i)\bgame\s+(?:is\s+)?over\b")
_ACTION_TOKEN = re.compile(
    r"(?i)\b(checks?|checked|pass(?:es)?|bets?|betting|raises?|calls?|called|folds?|folded)\b"
)
_VALID_PREFIXES = ([], ["check"], ["bet"], ["check", "bet"])


@dataclass(frozen=True)
class LeducTurn:
    """A parsed Leduc position: our seat, both cards, round, pot and the current round's line."""

    seat: int
    private: str
    community: str | None
    round: int
    pot: int
    line: tuple[str, ...]

    @property
    def facing_bet(self) -> bool:
        return bool(self.line) and self.line[-1] == "bet"


def _card(patterns, text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            card = _CARD_WORDS.get(match.group(1).lower())
            if card:
                return card
    return None


def _parse_line(raw: str) -> list[str] | None:
    """This round's actions, or None when the words are not a non-terminal round prefix."""
    raw = str(raw or "").strip().rstrip(".")
    if not raw or "no action" in raw.lower() or "none yet" in raw.lower():
        return []
    actions = [_ACTION_WORDS[match.group(1).lower()] for match in _ACTION_TOKEN.finditer(raw)]
    if not actions or actions not in _VALID_PREFIXES:
        return None
    return actions


def _legal_map(legal: list[str] | None) -> dict[str, str]:
    """Canonical action -> the arena's exact spelling, preserving its capitalisation."""
    mapping: dict[str, str] = {}
    for item in legal or []:
        text = str(item).strip()
        words = re.findall(r"[a-z]+", text.lower())
        canonical = _ACTION_WORDS.get(words[0]) if words else None
        if canonical and canonical not in mapping:
            mapping[canonical] = text
    return mapping


def parse_leduc_observation(
    observation: str,
    legal: list[str] | None = None,
    seat: int = 0,
) -> LeducTurn | None:
    """Parse a Leduc-shaped observation, or return None when any part is uncertain."""
    text = observation or ""
    if not text.strip() or _GAME_OVER.search(text) or _OPPONENT_TURN.search(text):
        return None

    private = _card(_PRIVATE_PATTERNS, text)
    if private is None:
        return None

    seat_match = _SEAT_PATTERN.search(text)
    if seat_match:
        try:
            seat = int(seat_match.group(1))
        except (TypeError, ValueError):
            pass

    round_match = _ROUND_PATTERN.search(text) or _ROUND_LOOSE.search(text)
    if not round_match:
        return None
    betting_round = int(round_match.group(1))
    if betting_round not in (1, 2):
        return None

    community = _card((_COMMUNITY_PATTERN,), text)
    if betting_round == 2 and community is None:
        return None
    if betting_round == 1 and community is not None:
        return None

    pot_match = _POT_PATTERN.search(text)
    if not pot_match:
        return None
    pot = int(pot_match.group(1))
    if pot < 2:
        return None

    line_match = _LINE_PATTERN.search(text)
    if line_match is None:
        return None
    line = _parse_line(line_match.group(1))
    if line is None:
        return None

    facing_bet = bool(line) and line[-1] == "bet"
    if legal is not None:
        available = _legal_map(legal)
        if not available:
            return None
        offered = set(available)
        if facing_bet and offered & {"check", "bet"}:
            return None
        if not facing_bet and offered & {"call", "fold"}:
            return None

    return LeducTurn(
        seat=int(seat),
        private=private,
        community=community,
        round=betting_round,
        pot=pot,
        line=tuple(line),
    )


def _hand_value(private: str, community: str | None) -> tuple[int, int]:
    """(is_pair, rank): with one community card a pair beats any unpaired hand."""
    if community is None:
        return (0, _RANK[private])
    if private == community:
        return (1, _RANK[private])
    return (0, _RANK[private])


def _equity(private: str, community: str | None) -> float:
    """Win + half-tie probability against one uniformly random unseen private card."""
    counts = dict(_DECK)
    counts[private] -= 1
    if community is not None:
        counts[community] -= 1
    total = sum(counts.values())
    if total <= 0:
        return 0.5
    ours = _hand_value(private, community)
    score = 0.0
    for rank, copies in counts.items():
        if copies <= 0:
            continue
        theirs = _hand_value(rank, community)
        if ours > theirs:
            score += copies
        elif ours == theirs:
            score += 0.5 * copies
    return score / total


def _bluff_roll(observation: str, seat: int, turn: int) -> float:
    raw = f"{observation}|{seat}|{turn}|leduc".encode("utf-8")
    return int.from_bytes(hashlib.sha1(raw).digest()[:4], "big") / float(1 << 32)


def _choose(turn: LeducTurn, legal: list[str] | None, seat: int, turn_number: int, text: str) -> str | None:
    mapping = _legal_map(legal) if legal is not None else {
        action: action for action in ("check", "bet", "call", "fold")
    }
    if not mapping:
        return None

    equity = _equity(turn.private, turn.community)
    pair = turn.community is not None and turn.private == turn.community

    if turn.facing_bet:
        # A call costs one chip, so it must win more than pot / (pot + 1); add a small margin because
        # a bettor's range is stronger than a random hand's.
        required = 1.0 / (turn.pot + 1.0)
        candidate = "call" if equity + 0.02 >= required else "fold"
        if pair:
            candidate = "call"
    else:
        strong = pair or equity >= 0.62
        bluff = equity < 0.35 and _bluff_roll(text, seat, turn_number) < 0.2
        if "bet" in mapping and (strong or bluff):
            candidate = "bet"
        elif "check" in mapping:
            candidate = "check"
        elif "bet" in mapping:
            candidate = "bet"
        else:
            return None

    return mapping.get(candidate)


class LeducSolver:
    """Conservative Leduc expert: reads the text, plays a pot-odds heuristic, declines anything odd."""

    name = "leduc"

    def __init__(self) -> None:
        self.rules_text = ""

    def configure(self, spec: object, rules: str = "", has_model: bool = False) -> None:
        self.rules_text = "\n".join(
            part
            for part in (
                str(rules or ""),
                str(getattr(spec, "winning_idea", "") or ""),
                str(getattr(spec, "notes", "") or ""),
                str(getattr(spec, "playbook", "") or ""),
            )
            if part
        )

    def matches(self, game_id: str, spec: object) -> bool:
        """Only claim Leduc / mini-poker / explicit community-card poker; never plain Kuhn."""
        try:
            identifier = " ".join(
                (
                    str(game_id or ""),
                    str(getattr(spec, "game_id", "") or ""),
                    str(getattr(spec, "winning_idea", "") or ""),
                    str(getattr(spec, "notes", "") or ""),
                    str(getattr(spec, "playbook", "") or ""),
                    str(getattr(spec, "action_format", "") or ""),
                    self.rules_text,
                )
            ).lower()
            if "leduc" in identifier:
                return True
            if any(cue in identifier for cue in ("mini poker", "mini-poker", "mini_poker")):
                return True
            if any(cue in identifier for cue in ("community card", "community-card", "community_card")):
                return True
            return False
        except Exception:  # noqa: BLE001 - a matcher must never cost a move
            return False

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        """A legal Leduc action, or None when the text is not understood; never raises."""
        try:
            if legal is not None and not legal:
                return None
            parsed = parse_leduc_observation(observation, legal, seat)
            if parsed is None:
                return None
            return _choose(parsed, legal, int(seat), int(turn), str(observation or ""))
        except Exception:  # noqa: BLE001 - an expert bug must not cost a move
            return None


__all__ = ["LeducSolver", "LeducTurn", "parse_leduc_observation"]
