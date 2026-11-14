"""Action safety: nothing reaches stdout unless it is a plausible, legal action.

Rulebook 8.1: the first bad turn answer is a warning, the second loses the game. A verifier is the
cheapest insurance in the whole harness, because it also protects the warning tie-break (10.4).
"""
from __future__ import annotations

import re
from typing import Iterable

_STRIP = " \t\r\n"

_EXACT_SOLDIERS_RE = re.compile(
    r"place\s+exactly\s+(\d+)\s+soldiers\s+on\s+(\d+)\s+battlefields",
    re.IGNORECASE,
)
_RESERVE_SOLDIERS_RE = re.compile(
    r"Your\s+reserve:\s*(\d+)\s+soldiers",
    re.IGNORECASE,
)
_K_NUMBERS_RE = re.compile(
    r"(\d+)\s+numbers\s+separated\s+by\s+spaces",
    re.IGNORECASE,
)
_BFIELD_VALUES_RE = re.compile(
    r"Battlefield\s+values\s+this\s+round:\s*([0-9\s]+)",
    re.IGNORECASE,
)
_OPEN_COLS_RE = re.compile(
    r"Open\s+columns:\s*([0-9,\s]+)",
    re.IGNORECASE,
)
_DROP_COL_HINT_RE = re.compile(
    r'Answer\s+"drop\s+<column>"',
    re.IGNORECASE,
)
_GUESS_NUM_RANGE_RE = re.compile(
    r'Answer\s+"guess\s+<n>"\s+with\s+n\s+from\s+(-?\d+)\s+to\s+(-?\d+)',
    re.IGNORECASE,
)
_INT_TOKEN_RE = re.compile(r"-?\d+")


def clean(action: object) -> str:
    """A protocol-safe action string: one line, no quotes added by a chatty model, no JSON braces."""
    if action is None:
        return ""
    text = str(action).replace("\r", " ").replace("\n", " ")
    text = text.strip(_STRIP)
    text = text.strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    return text[:120]


def clean_legal_actions(legal_actions: Iterable[str] | None) -> list[str]:
    if not legal_actions:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in legal_actions:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


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


normalize_action = normalize


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


def _enforce_blotto_format(
    candidate: str | None,
    observation: str,
    rules: str,
    avoid_set: set[str],
) -> str | None:
    combined = f"{observation or ''}\n{rules or ''}"
    m_exact = _EXACT_SOLDIERS_RE.search(combined)
    m_res = _RESERVE_SOLDIERS_RE.search(observation or "")
    m_k = _K_NUMBERS_RE.search(combined)
    m_bf = _BFIELD_VALUES_RE.search(observation or "")
    f_eq = re.findall(r"\bF(\d+)\s*=\s*(\d+)", observation or "", re.IGNORECASE)

    if not m_exact and not m_res and not (m_k and m_bf) and not f_eq:
        return None

    k_fields = 0
    if m_exact:
        k_fields = int(m_exact.group(2))
    elif m_k:
        k_fields = int(m_k.group(1))
    if f_eq:
        k_fields = len(f_eq)
    elif m_bf:
        bf_vals = [int(x) for x in m_bf.group(1).split() if x.strip().isdigit()]
        if bf_vals:
            k_fields = len(bf_vals)
    if k_fields <= 0:
        return None

    if not candidate:
        from .solver_simul import _propose_blotto

        solved = _propose_blotto(observation or "", rules or "", 0, 1)
        if solved:
            return solved

    exact_budget: int | None = int(m_exact.group(1)) if m_exact and not m_res else None
    max_reserve: int | None = int(m_res.group(1)) if m_res else None

    nums: list[int] = []
    if candidate:
        raw_ints = [int(t) for t in _INT_TOKEN_RE.findall(str(candidate))]
        nums = [max(0, x) for x in raw_ints[:k_fields]]

    if len(nums) < k_fields:
        nums.extend([0] * (k_fields - len(nums)))

    if exact_budget is not None:
        total = sum(nums)
        if total <= 0:
            base = exact_budget // k_fields
            rem = exact_budget - base * k_fields
            nums = [base + (1 if i < rem else 0) for i in range(k_fields)]
        elif total != exact_budget:
            scaled = [int(round(x * exact_budget / total)) for x in nums]
            diff = exact_budget - sum(scaled)
            idx = 0
            while diff != 0 and idx < 200:
                pos = idx % k_fields
                if diff > 0:
                    scaled[pos] += 1
                    diff -= 1
                elif scaled[pos] > 0:
                    scaled[pos] -= 1
                    diff += 1
                idx += 1
            nums = scaled
    elif max_reserve is not None:
        total = sum(nums)
        if total > max_reserve:
            if max_reserve <= 0:
                nums = [0] * k_fields
            else:
                scaled = [int(x * max_reserve // total) for x in nums]
                rem = max_reserve - sum(scaled)
                for i in range(k_fields):
                    if rem <= 0:
                        break
                    if nums[i] > 0:
                        scaled[i] += 1
                        rem -= 1
                nums = scaled

    out = " ".join(str(max(0, x)) for x in nums[:k_fields])
    if out.lower() in avoid_set and k_fields >= 2:
        alt = list(nums[:k_fields])
        if alt[0] > 0:
            alt[0] -= 1
            alt[1] += 1
            out = " ".join(str(x) for x in alt)
    return out


def enforce_format(
    action: str | None,
    *,
    observation: str = "",
    rules: str = "",
    legal: Iterable[str] | None = None,
    avoid: Iterable[str] | None = None,
) -> str | None:
    """Universal Format Guard: ensure action strictly satisfies protocol & game syntax.

    Prevents illegal-action retry loops when ``legal_actions`` is None (open action
    space games such as Colonel Blotto, Wordle Duel, Connect Four, Guess the Number).
    """
    cleaned_legal = clean_legal_actions(legal)
    avoid_set = {clean(a).lower() for a in (avoid or ()) if isinstance(a, str)}

    if cleaned_legal:
        norm = normalize(action, cleaned_legal)
        if norm and clean(norm).lower() not in avoid_set:
            return norm
        return choose_fallback(cleaned_legal, avoid)

    obs = observation or ""
    rls = rules or ""

    # 1. Wordle Duel ("guess <word>" with exact length N and G/Y/X clue consistency)
    if "wordle" in obs.lower() or "letters a-z" in obs.lower():
        from .solver_guess import verify_or_repair_wordle

        repaired = verify_or_repair_wordle(action, obs, rls, avoid=avoid)
        if repaired:
            return repaired

    # 2. Colonel Blotto / K integers separated by spaces
    blotto_out = _enforce_blotto_format(action, obs, rls, avoid_set)
    if blotto_out is not None:
        return blotto_out

    # 3. Connect-K ("Open columns: ..." and 'Answer "drop <column>"')
    m_cols = _OPEN_COLS_RE.search(obs)
    if m_cols and (_DROP_COL_HINT_RE.search(obs) or "connect" in obs.lower()):
        open_cols = [int(tok) for tok in re.findall(r"\d+", m_cols.group(1))]
        if open_cols:
            chosen_col: int | None = None
            if action:
                m_int = _INT_TOKEN_RE.search(str(action))
                if m_int:
                    c = int(m_int.group(0))
                    if c in open_cols and f"drop {c}" not in avoid_set:
                        chosen_col = c
            if chosen_col is None:
                center = open_cols[len(open_cols) // 2]
                ordered = sorted(open_cols, key=lambda c: abs(c - center))
                for c in ordered:
                    if f"drop {c}" not in avoid_set:
                        chosen_col = c
                        break
                if chosen_col is None:
                    chosen_col = ordered[0]
            return f"drop {chosen_col}"

    # 4. Guess the Number ('Answer "guess <n>" with n from L to H')
    m_gnum = _GUESS_NUM_RANGE_RE.search(obs)
    if m_gnum:
        low_n = int(m_gnum.group(1))
        high_n = int(m_gnum.group(2))
        if low_n > high_n:
            low_n, high_n = high_n, low_n
        if action:
            m_int = _INT_TOKEN_RE.search(str(action))
            if m_int:
                val = max(low_n, min(high_n, int(m_int.group(0))))
                cand = f"guess {val}"
                if cand.lower() not in avoid_set:
                    return cand
        mid = (low_n + high_n) // 2
        return f"guess {mid}"

    # 5. Truth & Guardrails ('It is YOUR turn: "answer <text>"' or 'Answer "answer <text>"')
    if '"answer <text>"' in obs.lower() or "'answer <text>'" in obs.lower():
        cleaned_ans = clean(action)
        if cleaned_ans:
            if not cleaned_ans.lower().startswith("answer "):
                cleaned_ans = f"answer {cleaned_ans}"
            return cleaned_ans
        return "answer unknown"

    cleaned = clean(action)
    return cleaned or None
