"""Deterministic expert for Guess the Number and Wordle Duel.

- Numeric interval bisection ("GUESS THE NUMBER (low-high)" with higher/lower feedback)
- Wordle Duel (4, 5, 6+ letter secret words with G/Y/X feedback and opponent progress tracking)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .words_en import words_of_length

_RANGE = re.compile(
    r"GUESS\s+THE\s+NUMBER\s*\(\s*(\d+)\s*(?:[-\u2013\u2014]|to|and)\s*(\d+)\s*\)",
    re.IGNORECASE,
)
_TURN_RANGE = re.compile(r"\bwith\s+n\s+from\s+(\d+)\s+to\s+(\d+)", re.IGNORECASE)
_RANGE_ALT = re.compile(
    r"(?:range|between)\s+(\d+)\s*(?:[-\u2013\u2014]|to|and)\s*(\d+)",
    re.IGNORECASE,
)
_OURS = re.compile(
    r"Your\s+guesses(?:\s+at\s+the\s+opponent['\u2019]s\s+(?:number|word)|\s*\([^)]*\))?\s*:",
    re.IGNORECASE,
)
_THEIRS = re.compile(
    r"Opponent['\u2019]s\s+guesses(?:\s+at\s+your\s+(?:number|word)|\s*\([^)]*\))?\s*:",
    re.IGNORECASE,
)
_GUESS = re.compile(
    r"(?:you\s+guessed\s+)?(-?\d+)\s*(?:->|=>|\u2192)\s*(higher|lower|correct|too\s+low|too\s+high)",
    re.IGNORECASE,
)
_GAME_OVER = re.compile(r"\bgame\s+(?:is\s+)?over\b", re.IGNORECASE)
_OPPONENT_TURN = re.compile(r"\bit\s+is\s+the\s+opponent['\u2019]s\s+turn\b", re.IGNORECASE)

_WORDLE_HEADER_RE = re.compile(
    r"WORDLE\s+DUEL\s*\(\s*(\d+)\s+letters(?:\s*,\s*(\d+)\s+guesses)?",
    re.IGNORECASE,
)
_WORDLE_EXACT_LEN_RE = re.compile(
    r"exactly\s+(\d+)\s+letters\s+a-z",
    re.IGNORECASE,
)
_WORDLE_ROUND_RE = re.compile(
    r"Round\s+(\d+)\s+of\s+(\d+)",
    re.IGNORECASE,
)
_WORDLE_GUESS_LINE_RE = re.compile(
    r"^\s*([a-z]{3,10})\s*->\s*((?:[GYX]\s*){3,10})\s*$",
    re.IGNORECASE,
)
_OPP_PAT_LINE_RE = re.compile(
    r"^\s*\?\?\?\s*->\s*((?:[GYX]\s*){3,10})\s*$",
    re.IGNORECASE,
)

_FREQ_LETTERS = "eariotnslcudpmhgbfywkvxzjq"
_DEFAULT_OPENERS = {
    4: ("tale", "rate", "sore", "line", "cart"),
    5: ("slate", "crane", "trace", "stare", "arise"),
    6: ("orates", "salter", "planet", "strang", "castle"),
}


def _parse_range(observation: str) -> tuple[int, int] | None:
    """Return the inclusive hint range announced by the observation, if any."""
    for pattern in (_RANGE, _TURN_RANGE, _RANGE_ALT):
        match = pattern.search(observation)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            if lo <= hi:
                return lo, hi
    return None


def _our_guess_lines(observation: str) -> list[str]:
    """The lines describing guesses *we* made, never the opponent's guesses."""
    ours = _OURS.search(observation)
    start = ours.end() if ours else 0
    theirs = _THEIRS.search(observation, start)
    end = theirs.start() if theirs else len(observation)
    return observation[start:end].splitlines()


def _remaining_interval(observation: str) -> tuple[int, int] | None:
    """Recompute the still-possible interval for the opponent's number from the text alone."""
    bounds = _parse_range(observation)
    if bounds is None:
        return None
    lo, hi = bounds
    for line in _our_guess_lines(observation):
        match = _GUESS.search(line)
        if not match:
            continue
        number = int(match.group(1))
        feedback = match.group(2).lower()
        if "higher" in feedback or "too low" in feedback:
            lo = max(lo, number + 1)
        elif "lower" in feedback or "too high" in feedback:
            hi = min(hi, number - 1)
        else:  # correct: the secret is pinned down
            lo = hi = number
    return (lo, hi) if lo <= hi else None


def _from_legal(candidate: str, legal: list[str]) -> str | None:
    """Map ``candidate`` onto the arena's exact legal spelling, or give up."""
    items = [str(item) for item in legal]
    for item in items:
        if item == candidate:
            return item
    parts = candidate.split()
    if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
        number = int(parts[1])
        for item in items:
            digits = re.findall(r"-?\d+", item)
            if len(digits) == 1 and int(digits[0]) == number:
                return item
    return None


def wordle_feedback(secret: str, guess: str) -> tuple[str, ...]:
    """Compute exact Wordle feedback ('G', 'Y', 'X') per position."""
    n = len(secret)
    if len(guess) != n:
        return tuple("X" for _ in range(len(guess)))
    res = ["X"] * n
    remaining: Counter[str] = Counter()
    for i in range(n):
        if guess[i] == secret[i]:
            res[i] = "G"
        else:
            remaining[secret[i]] += 1
    for i in range(n):
        if res[i] == "G":
            continue
        ch = guess[i]
        if remaining[ch] > 0:
            res[i] = "Y"
            remaining[ch] -= 1
    return tuple(res)


def parse_wordle_state(observation: str, rules: str = "") -> dict | None:
    """Parse a Wordle Duel observation into structured state, or None if not Wordle."""
    combined = f"{observation or ''}\n{rules or ''}"
    m_len = _WORDLE_HEADER_RE.search(combined)
    word_len: int | None = None
    max_rounds: int = 6
    if m_len:
        word_len = int(m_len.group(1))
        if m_len.group(2):
            max_rounds = int(m_len.group(2))
    else:
        m_ex = _WORDLE_EXACT_LEN_RE.search(combined)
        if m_ex:
            word_len = int(m_ex.group(1))
    if word_len is None or word_len < 3 or word_len > 10:
        return None

    cur_round = 1
    m_rnd = _WORDLE_ROUND_RE.search(observation or "")
    if m_rnd:
        cur_round = int(m_rnd.group(1))
        max_rounds = int(m_rnd.group(2))

    my_guesses: list[tuple[str, tuple[str, ...]]] = []
    opp_patterns: list[tuple[str, ...]] = []
    section = "none"
    for raw_line in (observation or "").splitlines():
        line = raw_line.strip()
        low = line.lower()
        if low.startswith("your guesses"):
            section = "me"
            continue
        if low.startswith("opponent's guesses") or low.startswith("opponent guesses"):
            section = "opp"
            continue
        if section == "me":
            m_g = _WORDLE_GUESS_LINE_RE.match(line)
            if m_g:
                w = m_g.group(1).lower()
                pat = tuple(tok.upper() for tok in m_g.group(2).split())
                if len(w) == word_len and len(pat) == word_len:
                    my_guesses.append((w, pat))
        elif section == "opp":
            m_o = _OPP_PAT_LINE_RE.match(line)
            if m_o:
                pat = tuple(tok.upper() for tok in m_o.group(1).split())
                if len(pat) == word_len:
                    opp_patterns.append(pat)

    opp_best_greens = max((sum(1 for c in p if c == "G") for p in opp_patterns), default=0)
    return {
        "length": word_len,
        "round": cur_round,
        "max_rounds": max_rounds,
        "guesses": my_guesses,
        "opp_patterns": opp_patterns,
        "opp_best_greens": opp_best_greens,
    }


def is_consistent_with_history(
    candidate: str,
    guesses: list[tuple[str, tuple[str, ...]]],
) -> bool:
    """Return True iff ``candidate`` produces exact feedback ``pat`` for every past guess."""
    for guess_word, pat in guesses:
        if wordle_feedback(candidate, guess_word) != pat:
            return False
    return True


def wordle_candidates(
    word_len: int,
    guesses: list[tuple[str, tuple[str, ...]]],
    avoid_words: set[str] | None = None,
) -> list[str]:
    """Filter the built-in English word list by all observed (guess, pattern) clues."""
    avoid = avoid_words or set()
    used = {g for g, _ in guesses} | avoid
    vocab = words_of_length(word_len)
    return [
        w
        for w in vocab
        if len(w) == word_len and w not in used and is_consistent_with_history(w, guesses)
    ]


def _synthesize_fallback_word(
    word_len: int,
    guesses: list[tuple[str, tuple[str, ...]]],
    avoid_words: set[str],
) -> str:
    """Construct a valid N-letter lowercase string satisfying G/Y/X constraints when out-of-vocab."""
    green_at: dict[int, str] = {}
    banned_at: dict[int, set[str]] = {i: set() for i in range(word_len)}
    min_count: Counter[str] = Counter()
    max_count: dict[str, int] = {}

    for word, pat in guesses:
        gy_in_word: Counter[str] = Counter()
        x_in_word: Counter[str] = Counter()
        for idx, (ch, mark) in enumerate(zip(word, pat)):
            if mark == "G":
                green_at[idx] = ch
                gy_in_word[ch] += 1
            elif mark == "Y":
                banned_at[idx].add(ch)
                gy_in_word[ch] += 1
            else:
                banned_at[idx].add(ch)
                x_in_word[ch] += 1
        for ch, cnt in gy_in_word.items():
            if cnt > min_count[ch]:
                min_count[ch] = cnt
        for ch in x_in_word:
            max_count[ch] = gy_in_word[ch]

    chars: list[str | None] = [green_at.get(i) for i in range(word_len)]
    current_counts: Counter[str] = Counter(c for c in chars if c is not None)

    for ch, req in min_count.items():
        while current_counts[ch] < req:
            placed = False
            for i in range(word_len):
                if chars[i] is None and ch not in banned_at[i]:
                    chars[i] = ch
                    current_counts[ch] += 1
                    placed = True
                    break
            if not placed:
                break

    for i in range(word_len):
        if chars[i] is not None:
            continue
        for ch in _FREQ_LETTERS:
            limit = max_count.get(ch, word_len)
            if ch not in banned_at[i] and current_counts[ch] < limit:
                chars[i] = ch
                current_counts[ch] += 1
                break
        if chars[i] is None:
            for ch in _FREQ_LETTERS:
                if current_counts[ch] < max_count.get(ch, word_len):
                    chars[i] = ch
                    current_counts[ch] += 1
                    break
        if chars[i] is None:
            chars[i] = "e"

    word = "".join(c or "e" for c in chars)
    used = {g for g, _ in guesses} | avoid_words
    if word in used:
        for ch in _FREQ_LETTERS:
            alt = word[:-1] + ch
            if alt not in used:
                return alt
    return word


def _score_wordle_candidates(
    candidates: list[str],
    word_len: int,
    tested_letters: set[str],
    prefer_solution: bool,
) -> str:
    """Rank consistent candidates by information gain and positional frequency."""
    if len(candidates) == 1:
        return candidates[0]

    pos_freq: list[Counter[str]] = [Counter() for _ in range(word_len)]
    letter_freq: Counter[str] = Counter()
    for w in candidates:
        for i, ch in enumerate(w):
            pos_freq[i][ch] += 1
        for ch in set(w):
            letter_freq[ch] += 1

    best_word = candidates[0]
    best_score = -1e18
    total = max(1, len(candidates))

    for rank_idx, w in enumerate(candidates):
        unique_chars = set(w)
        coverage = sum(
            letter_freq[ch] * (1.35 if ch not in tested_letters else 1.0)
            for ch in unique_chars
        ) / total
        positional = sum(pos_freq[i][ch] for i, ch in enumerate(w)) / total
        prior_bonus = max(0.0, 1.0 - (rank_idx / max(50.0, float(len(candidates)))))
        dup_penalty = (len(w) - len(unique_chars)) * 0.35
        if prefer_solution:
            score = positional * 1.4 + coverage * 1.0 + prior_bonus * 0.8 - dup_penalty
        else:
            score = coverage * 1.6 + positional * 1.0 + prior_bonus * 0.5 - dup_penalty
        if score > best_score:
            best_score = score
            best_word = w

    return best_word


def _propose_wordle_from_state(state: dict, avoid_words: set[str]) -> str:
    word_len: int = state["length"]
    guesses: list[tuple[str, tuple[str, ...]]] = state["guesses"]
    cur_round: int = state["round"]
    max_rounds: int = state["max_rounds"]
    opp_best_greens: int = state["opp_best_greens"]

    if not guesses:
        for opener in _DEFAULT_OPENERS.get(word_len, ()):
            if len(opener) == word_len and opener not in avoid_words:
                return f"guess {opener}"
        cands = wordle_candidates(word_len, [], avoid_words)
        if cands:
            return f"guess {cands[0]}"

    cands = wordle_candidates(word_len, guesses, avoid_words)
    if cands:
        tested_letters = {ch for g, _ in guesses for ch in g}
        urgent = (
            len(cands) <= 4
            or cur_round >= max_rounds - 1
            or opp_best_greens >= word_len - 1
        )
        chosen = _score_wordle_candidates(
            cands,
            word_len=word_len,
            tested_letters=tested_letters,
            prefer_solution=urgent,
        )
        return f"guess {chosen}"

    fallback_word = _synthesize_fallback_word(word_len, guesses, avoid_words)
    return f"guess {fallback_word}"


def verify_or_repair_wordle(
    action: str | None,
    observation: str,
    rules: str = "",
    avoid: Iterable[str] | None = None,
) -> str | None:
    """Verify that ``action`` is a valid `"guess <word>"` consistent with all Wordle clues."""
    state = parse_wordle_state(observation, rules)
    if state is None:
        return action

    word_len = state["length"]
    guesses = state["guesses"]
    avoid_words: set[str] = set()
    for a in avoid or ():
        s = str(a).strip().lower()
        if s.startswith("guess "):
            s = s[6:].strip()
        if len(s) == word_len and s.isalpha():
            avoid_words.add(s)

    used_words = {g for g, _ in guesses} | avoid_words

    if action:
        text = str(action).strip().lower()
        cand = text[6:].strip() if text.startswith("guess ") else text
        if (
            len(cand) == word_len
            and cand.isalpha()
            and cand.isascii()
            and cand not in used_words
            and is_consistent_with_history(cand, guesses)
        ):
            return f"guess {cand}"

    return _propose_wordle_from_state(state, avoid_words)


class GuessingSolver:
    """Binary-search solver for number-guessing and constraint solver for Wordle Duel."""

    name = "guessing"

    @classmethod
    def matches(cls, game_id: str, spec: object) -> bool:
        """Claim games whose id or action shape says 'guess' or 'wordle'."""
        spec_id = str(getattr(spec, "game_id", "") or "")
        identifier = f"{game_id or ''} {spec_id}".lower()
        if "guess" in identifier or "wordle" in identifier:
            return True
        family = str(getattr(spec, "family", "") or "").lower()
        if family != "guessing":
            return False
        if any(
            word in identifier
            for word in ("number", "deduc", "higher", "lower", "hot-cold", "hot_cold", "word")
        ):
            return True
        action_format = str(getattr(spec, "action_format", "") or "").lower()
        return "guess" in action_format

    def propose(
        self,
        *,
        observation: str = "",
        legal: list[str] | None = None,
        seat: int = 0,
        turn: int = 0,
        timeout: float = 2.0,
        rules: str = "",
        avoid: Iterable[str] | None = None,
        **kwargs,
    ) -> str | None:
        """Guess the Wordle candidate or midpoint of the remaining numeric interval."""
        if not observation or _GAME_OVER.search(observation) or _OPPONENT_TURN.search(observation):
            return None

        wordle_state = parse_wordle_state(observation, rules)
        if wordle_state is not None:
            avoid_words: set[str] = set()
            for a in avoid or ():
                s = str(a).strip().lower()
                if s.startswith("guess "):
                    s = s[6:].strip()
                if len(s) == wordle_state["length"] and s.isalpha():
                    avoid_words.add(s)
            proposal = _propose_wordle_from_state(wordle_state, avoid_words)
            if legal is None:
                return proposal
            return _from_legal(proposal, legal) or proposal

        interval = _remaining_interval(observation)
        if interval is None:
            return None
        lo, hi = interval
        candidate = f"guess {(lo + hi) // 2}"
        if legal is None:
            return candidate
        return _from_legal(candidate, legal)
