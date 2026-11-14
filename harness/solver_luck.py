"""Generic push-your-luck expert that decides from the observation text alone.

The family solvers in this package each know one game. This one is the runtime safety net for the
long tail of dice/luck games: it needs no generated model and no game-specific code, only the text
the arena already sends. From that text it reads the race target, our banked total, the current turn
total, whether the game uses one or two dice, and whose turn it is, then compares the exact expected
value of one more roll with the guaranteed value of holding.

The model is deliberately small and exact:

* one die: a 1 loses the turn total (and, only when the text says so, the bank);
* two dice: all 36 ordered outcomes are enumerated; a single 1 loses the turn total and snake eyes
  lose the bank only when the text says the bank is reset;
* holding is chosen when it reaches the target immediately or when the roll's expected banked score
  does not beat the banked score from holding (ties go to holding, which is the safe side).

The one-die expected-value boundary therefore lands at the well-known turn total of 20 without that
number being written anywhere: ``E[roll] = bank + (5 * turn_total + 20) / 6``.

`propose()` never raises, performs no I/O, uses no randomness and needs well under a millisecond.
"""
from __future__ import annotations

import re
from typing import NamedTuple

# --------------------------------------------------------------------------- observation patterns
_LABEL = r"(?:banked|bank|score|scores|total|points)(?:\s+(?:total|score|points?))?"

_MY_SEAT = re.compile(r"\byou\s+are\s+(?:seat|player)\s+(\d+)\b", re.IGNORECASE)
_SEAT_TURN = re.compile(
    r"\b(?:seat|player)\s+(\d+)\s*(?:\([^)]*\)\s*)?(?:is\s+)?"
    r"(?:to\s+move|['\u2019]s\s+turn|has\s+the\s+turn)\b",
    re.IGNORECASE,
)
_OPPONENT_TURN = re.compile(
    r"\b(?:it\s+is\s+(?:now\s+)?(?:the\s+)?(?:opponent|other\s+player)(?:['\u2019]s)?\s+"
    r"(?:turn|move)|(?:opponent|other\s+player)(?:['\u2019]s)?\s+(?:turn|to\s+move))\b",
    re.IGNORECASE,
)
_MY_TURN = re.compile(
    r"\b(?:your|yours)\s+(?:turn|move)\b(?!\s*(?:total|points?|score))"
    r"|\byou\s+to\s+move\b|\byou\s+have\s+the\s+(?:turn|move)\b",
    re.IGNORECASE,
)
_NOT_MY_TURN = re.compile(
    r"\b(?:not|no\s+longer)\s+(?:your|yours)\s+(?:turn|move)\b"
    r"|\b(?:your|yours)\s+(?:turn|move)\s+(?:is\s+)?(?:over|done|finished)\b",
    re.IGNORECASE,
)
_GAME_OVER = re.compile(
    r"\bgame\s+(?:is\s+)?over\b|\bgame\s+(?:is\s+)?(?:finished|complete|done)\b"
    r"|\b(?:has|have)\s+won\b|\b(?:you|opponent|other\s+player)\s+wins?\s+the\s+game\b"
    r"|\b(?:seat|player)\s+\d+\s+wins?\b|\bfinal\s+scores?\b|\bno\s+more\s+(?:turns|moves)\b"
    r"|^\s*(?:you|opponent|other\s+player)\s+(?:wins?|loses?|lost)[!.]",
    re.IGNORECASE | re.MULTILINE,
)

_TARGET = re.compile(
    r"\b(?:target|goal|winning\s+score|to\s+win|win\s+at|race\s+to)\s*"
    r"(?:score|points?)?\s*(?:is|of|at|:|=)?\s*(\d+)\b",
    re.IGNORECASE,
)
_TARGET_FIRST_TO = re.compile(r"\bfirst\s+to\s+(\d+)\b", re.IGNORECASE)
_TARGET_REACH = re.compile(r"\b(?:reach|hit|get\s+to)\s+(\d+)\b", re.IGNORECASE)
_TARGET_POINTS_TO_WIN = re.compile(r"\b(\d+)\s+(?:points?|pts)\s+to\s+win\b", re.IGNORECASE)

_PAIR = re.compile(
    rf"\b{_LABEL}\s*[:=]\s*you\s*(\d+)\s*[,;/|]\s*(?:opponent|other|them)\s*(\d+)",
    re.IGNORECASE,
)
_PAIR_ALT = re.compile(
    r"\byou\s*(?:have|are\s+at|banked)?\s*[:=]?\s*(\d+)\s*(?:points?)?\s*[,;/|]\s*"
    r"(?:opponent|other|them)\s*(?:has|is\s+at|has\s+banked)?\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_BANK_YOU_LABEL = re.compile(rf"\b{_LABEL}\s*[:=]\s*you\s*[:=]?\s*(\d+)\b", re.IGNORECASE)
_YOUR_BANK = re.compile(
    rf"\b(?:your|you)\s+(?:current\s+)?{_LABEL}\s*(?:is|are|at|=|:)?\s*(\d+)\b",
    re.IGNORECASE,
)
_BANK_SIMPLE = re.compile(rf"^\s*{_LABEL}\s*[:=]\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_YOU_SIMPLE = re.compile(r"^\s*you\s*[:=]\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_OPPONENT_BANK = re.compile(
    r"\b(?:opponent|other|them)(?:['\u2019]s)?\s+(?:current\s+)?"
    r"(?:banked|bank|score|total|points)(?:\s+(?:total|score|points?))?"
    r"\s*(?:is|at|=|:)?\s*(\d+)\b",
    re.IGNORECASE,
)
_OPPONENT_SIMPLE = re.compile(
    r"^\s*(?:opponent|other)(?:['\u2019]s)?\s*(?:banked|bank|score|total|points)?\s*[:=]\s*(\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_TURN_TOTAL = re.compile(
    r"\b(?:turn\s+total|turn\s+points?|turn\s+score|current\s+turn|unbanked|at\s+risk|pending|"
    r"(?<!rolls\s)this\s+turn)\s*(?:is|of|at|:|=)?\s*(\d+)\b",
    re.IGNORECASE,
)
_TURN_TOTAL_REVERSED = re.compile(r"\b(\d+)\s+(?:unbanked|at\s+risk|pending)\b", re.IGNORECASE)
_HOLD_TOTAL = re.compile(r"\bif\s+you\s+hold[^.\n]{0,40}?(\d+)\b", re.IGNORECASE)
_HOLD_TOTAL_ALT = re.compile(
    r"\b(?:holding|banking)(?:\s+now)?[^.\n]{0,30}?"
    r"(?:gives?|scores?|banks?|leaves?\s+you\s+(?:with|at)|puts?\s+you\s+at)\s+(?:you\s+)?(\d+)\b",
    re.IGNORECASE,
)
_BANK_NOW_FOR = re.compile(r"\bbank(?:ing)?\s+now\s+(?:gives|scores|for)\s+(\d+)\b", re.IGNORECASE)
_NO_TURN_POINTS = re.compile(
    r"\b(?:rolls?\s+this\s+turn|this\s+turn|no\s+rolls?)\s*[:=]?\s*(?:none|no\s+rolls?|0)\b",
    re.IGNORECASE,
)

_TWO_DICE = re.compile(
    r"\b(?:two|2|pair\s+of|both)\s+(?:dice|die|d6s?)\b|\b2d6\b"
    r"|\bsnake\s*eyes\b|\bdouble\s+ones\b",
    re.IGNORECASE,
)
_ONE_DIE = re.compile(
    r"\b(?:one|1|single)\s+(?:die|d6)\b|\b1d6\b|\broll(?:ing)?\s+a\s+die\b",
    re.IGNORECASE,
)

_BANK_RESET_PATTERNS = (
    re.compile(
        r"(?:snake\s*eyes|double\s+ones|bust|rolling\s+a\s+1|roll(?:ing)?\s+1)"
        r"[^.\n]{0,60}?(?:reset|resets|lose|loses|lost|clear|clears|wipe|wipes|zero|zeroes|forfeit)"
        r"[^.\n]{0,40}?(?:bank|score|total|points|everything)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:reset|resets|lose|loses|lost|clear|clears|wipe|wipes|zero|zeroes|forfeit)"
        r"[^.\n]{0,30}?(?:bank|score|total|points|everything)"
        r"[^.\n]{0,60}?(?:snake\s*eyes|double\s+ones|bust)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:lose|loses|lost|forfeit)\s+(?:your\s+|the\s+)?"
        r"(?:entire\s+|whole\s+)?(?:bank|score|points|everything)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:bank|score)\s+(?:is\s+)?(?:reset|cleared|zeroed|wiped|gone)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsnake\s*eyes\s+reset\b|\breset\s+on\s+snake\s*eyes\b", re.IGNORECASE),
)
_NEGATED = re.compile(r"(?:\bnot\b|\bnever\b|\bno\b|n't)\s+(?:\w+\s+){0,4}$", re.IGNORECASE)

# --------------------------------------------------------------------------- routing patterns
_REJECT_FAMILIES = {"deals", "bluff", "truth", "guardrails", "think_ahead", "read_opponent"}
_LUCK_FAMILIES = {"chance", "dice", "luck", "push your luck"}
_REJECT_WORDS = re.compile(
    r"\b(?:auction|auctions|bid|bids|bidding|poker|blackjack|board|chess|checkers|draughts|"
    r"connect\s*4|connect\s*four|tic\s*tac\s*toe|sudoku|shogi|go\s+game|monopoly)\b",
    re.IGNORECASE,
)
_LUCK_WORDS = re.compile(
    r"\b(?:pig|luck|lucky|dice|die|d6|roll|rolls|rolling|hold|holding|bank|banked|bust|"
    r"snake\s*eyes|push\s+your\s+luck|chance)\b",
    re.IGNORECASE,
)
_ROLL_ACTION = re.compile(r"\broll(?:s|ing)?\b", re.IGNORECASE)
_HOLD_ACTION = re.compile(r"\b(?:hold(?:s|ing)?|bank(?:s|ing)?|stop|stay|stand)\b", re.IGNORECASE)


class _LuckState(NamedTuple):
    """The numbers the decision needs, already parsed out of the observation."""

    target: int
    banked: int
    opponent: int | None
    turn_total: int
    dice: int
    bank_reset: bool


# --------------------------------------------------------------------------- parsing helpers
def _first_int(*matches: re.Match[str] | None) -> int | None:
    for match in matches:
        if match is not None:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _search_int(pattern: re.Pattern[str], text: str) -> int | None:
    try:
        match = pattern.search(text)
    except (TypeError, ValueError):
        return None
    return _first_int(match)


def _negated(text: str, start: int) -> bool:
    """True when the matched phrase is preceded by a negation ('not', 'never', ...)."""
    window = text[max(0, start - 60):start]
    return bool(_NEGATED.search(window))


def _bank_resets_on_bust(observation: str) -> bool:
    """Only claim a bank reset when the text plainly says the bank is lost."""
    for pattern in _BANK_RESET_PATTERNS:
        for match in pattern.finditer(observation):
            if not _negated(observation, match.start()):
                return True
    return False


def _parse_state(observation: str) -> _LuckState | None:
    """Read target, dice count, banked scores and turn total from one observation."""
    target = _search_int(_TARGET, observation)
    if target is None:
        target = _search_int(_TARGET_FIRST_TO, observation)
    if target is None:
        target = _search_int(_TARGET_POINTS_TO_WIN, observation)
    if target is None:
        target = _search_int(_TARGET_REACH, observation)
    if target is None or target <= 0:
        return None

    pair = _PAIR.search(observation) or _PAIR_ALT.search(observation)
    banked: int | None = None
    opponent: int | None = None
    if pair is not None:
        try:
            banked, opponent = int(pair.group(1)), int(pair.group(2))
        except (TypeError, ValueError):
            banked, opponent = None, None
    if banked is None:
        banked = _search_int(_BANK_YOU_LABEL, observation)
    if banked is None:
        banked = _search_int(_YOUR_BANK, observation)
    if banked is None:
        banked = _search_int(_BANK_SIMPLE, observation)
    if banked is None:
        banked = _search_int(_YOU_SIMPLE, observation)
    if opponent is None:
        opponent = _search_int(_OPPONENT_BANK, observation)
    if opponent is None:
        opponent = _search_int(_OPPONENT_SIMPLE, observation)

    turn_total = _search_int(_TURN_TOTAL, observation)
    if turn_total is None:
        turn_total = _search_int(_TURN_TOTAL_REVERSED, observation)

    hold_total = _search_int(_HOLD_TOTAL, observation)
    if hold_total is None:
        hold_total = _search_int(_HOLD_TOTAL_ALT, observation)
    if hold_total is None:
        hold_total = _search_int(_BANK_NOW_FOR, observation)

    # The "if you hold now you will have N" line lets either field fill in the other.
    if banked is None and hold_total is not None and turn_total is not None:
        banked = max(0, hold_total - turn_total)
    if turn_total is None and hold_total is not None and banked is not None:
        turn_total = max(0, hold_total - banked)
    if turn_total is None and _NO_TURN_POINTS.search(observation):
        turn_total = 0

    if banked is None or turn_total is None:
        return None

    dice = 2 if _TWO_DICE.search(observation) else 1
    return _LuckState(
        target=target,
        banked=banked,
        opponent=opponent,
        turn_total=turn_total,
        dice=dice,
        bank_reset=_bank_resets_on_bust(observation),
    )


# --------------------------------------------------------------------------- turn ownership
def _turn_status(observation: str, seat: int) -> bool | None:
    """True = our turn, False = theirs, None = the text does not say."""
    my_seat = int(seat)
    mine = _MY_SEAT.search(observation)
    if mine is not None:
        try:
            my_seat = int(mine.group(1))
        except (TypeError, ValueError):
            pass
    if _NOT_MY_TURN.search(observation):
        return False
    actor = _SEAT_TURN.search(observation)
    if actor is not None:
        try:
            return int(actor.group(1)) == my_seat
        except (TypeError, ValueError):
            pass
    if _OPPONENT_TURN.search(observation):
        return False
    if _MY_TURN.search(observation):
        return True
    return None


# --------------------------------------------------------------------------- expected value
def _roll_value(state: _LuckState) -> float:
    """Expected banked score after one roll (then holding), in points.

    A bust keeps the bank unless the text says it resets; the two-dice branch enumerates all 36
    equally likely ordered outcomes.
    """
    if state.dice >= 2:
        total = 0.0
        for first in range(1, 7):
            for second in range(1, 7):
                if first == 1 and second == 1:
                    if not state.bank_reset:
                        total += state.banked
                elif first == 1 or second == 1:
                    total += state.banked
                else:
                    total += state.banked + state.turn_total + first + second
        return total / 36.0

    total = 0.0 if state.bank_reset else float(state.banked)
    for face in range(2, 7):
        total += state.banked + state.turn_total + face
    return total / 6.0


def _should_hold(state: _LuckState) -> bool:
    """Hold when it wins now, or when rolling does not beat the guaranteed banked total."""
    if state.target > 0 and state.banked + state.turn_total >= state.target:
        return True
    hold_value = float(state.banked + state.turn_total)
    return not (_roll_value(state) > hold_value)


# --------------------------------------------------------------------------- legal spelling
_ROLL_WORDS = {"roll", "roll again", "roll dice", "roll the dice"}
_HOLD_WORDS = {"hold", "bank", "stop", "stay", "stand", "hold and bank"}


def _classify_action(text: object) -> str | None:
    """Map one legal action onto the canonical 'roll'/'hold' family, case-insensitively."""
    lowered = re.sub(r"\s+", " ", str(text).strip().strip("\"'`").lower())
    if lowered in _ROLL_WORDS:
        return "roll"
    if lowered in _HOLD_WORDS:
        return "hold"
    if re.fullmatch(r"roll(?:\s+(?:again|dice|the\s+dice))?", lowered):
        return "roll"
    if re.fullmatch(r"(?:hold|bank|stop|stay|stand)(?:\s+(?:and\s+)?(?:bank|hold))?", lowered):
        return "hold"
    return None


def _legal_spelling(action: str, legal: list[str] | None) -> str | None:
    """Return the arena's exact spelling of `action`, or None when it is absent."""
    if legal is None:
        return action
    for item in legal:
        if _classify_action(item) == action:
            return str(item)
    return None


def _has_roll_hold_shape(text: str) -> bool:
    return bool(_ROLL_ACTION.search(text) and _HOLD_ACTION.search(text))


def _field(spec: object, name: str, default: object = None) -> object:
    """Read a spec field from an object or a plain dict."""
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


# --------------------------------------------------------------------------- public solver
class LuckSolver:
    """Exact one-ply expected-value solver for generic one-die and two-dice push-your-luck games."""

    name = "luck"

    @classmethod
    def matches(cls, game_id: str, spec: object) -> bool:
        """Conservative claim based on the id/spec text, never on the observation.

        A true answer only costs one cheap `propose()` call: the observation must also contain a
        target, a banked total and a turn total before any action is returned.
        """
        try:
            parts = [str(game_id or "")]
            for name in ("game_id", "family", "notes", "winning_idea", "action_format", "scoring", "playbook"):
                value = _field(spec, name)
                if value is not None:
                    parts.append(str(value))
            for name in ("action_examples", "traps"):
                for item in _field(spec, name, None) or []:
                    parts.append(str(item))
            text = " ".join(parts).replace("_", " ").replace("-", " ")

            if _REJECT_WORDS.search(text):
                return False
            identifier = f"{game_id or ''} {_field(spec, 'game_id', '') or ''}".lower()
            if re.search(r"\bpig\b", identifier):
                return True  # the practice id is authoritative even if prep mislabels the family

            family = str(_field(spec, "family", "") or "").strip().lower()
            family = family.replace("_", " ").replace("-", " ")
            if family in _REJECT_FAMILIES:
                return False
            if family in _LUCK_FAMILIES:
                return True
            if _LUCK_WORDS.search(text):
                return True
            return _has_roll_hold_shape(text)
        except Exception:  # noqa: BLE001 - routing must never break a match
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
        """Return the exact legal ``roll``/``hold`` spelling, or None when undecided."""
        try:
            return self._propose(observation, legal, seat)
        except Exception:  # noqa: BLE001 - an expert bug must cost at most one move
            return None

    def _propose(self, observation: str, legal: list[str] | None, seat: int) -> str | None:
        if not observation or not isinstance(observation, str):
            return None
        if _GAME_OVER.search(observation):
            return None

        legal_list: list[str] | None = None
        if legal is not None:
            try:
                legal_list = [str(item) for item in legal]
            except TypeError:
                return None
            if not legal_list:
                return None
            if not any(_classify_action(item) is not None for item in legal_list):
                return None

        status = _turn_status(observation, seat)
        if status is False:
            return None

        state = _parse_state(observation)
        if state is None:
            return None

        action = "hold" if _should_hold(state) else "roll"
        return _legal_spelling(action, legal_list)


__all__ = ["LuckSolver"]
