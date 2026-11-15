"""Analytic mathematical expert for Nim.

Solves all 4 arena variants with 100% mathematical optimality:
1. Normal play (any tokens from one pile): exact Nim-sum (XOR) winning strategy.
2. Misere play (taking the last token loses): normal play until end-game, then leave an odd
   number of size-1 piles to force opponent to take the last token.
3. Subtraction Nim ("Take 1-3" or "Take 1-M"): Grundy values g(x) = x % (M + 1).
4. Multi-pile / up to two piles.
"""
from __future__ import annotations

import re
from typing import Sequence

_GAME_OVER = re.compile(r"\bgame\s+(?:is\s+)?over\b", re.IGNORECASE)
_PILE_PAIR = re.compile(r"pile\s+(\d+)\s*=\s*(\d+)", re.IGNORECASE)
_PILES_NOW = re.compile(r"Piles\s+now\s*:\s*([^\n]+)", re.IGNORECASE)
_STARTING_PILES = re.compile(r"Starting\s+piles\s*:\s*([^\n]+)", re.IGNORECASE)
_MAX_TAKE = re.compile(
    r"(?:take\s+1\s*-\s*(\d+)|never\s+more\s+than\s+(\d+)|or\s+(\d+)\s+tokens|at\s+most\s+(\d+)\s+tokens)",
    re.IGNORECASE,
)
_MISERE = re.compile(r"\bmisere\b", re.IGNORECASE)


def parse_nim_state(observation: str, rules: str = "") -> dict | None:
    """Extract pile sizes, misere flag, and max_take constraint from text."""
    combined = f"{observation or ''}\n{rules or ''}"
    m_now = _PILES_NOW.search(observation or "")
    if not m_now:
        m_now = _STARTING_PILES.search(observation or "")
    if not m_now:
        return None

    pairs = _PILE_PAIR.findall(m_now.group(1))
    if not pairs:
        # Fallback: search anywhere in observation
        pairs = _PILE_PAIR.findall(observation or "")
    if not pairs:
        return None

    piles: dict[int, int] = {}
    for p_str, count_str in pairs:
        piles[int(p_str)] = int(count_str)

    if not piles or all(c == 0 for c in piles.values()):
        return None

    misere = bool(_MISERE.search(combined))

    m_take = _MAX_TAKE.search(combined)
    max_take = int(m_take.group(1) or m_take.group(2) or m_take.group(3) or m_take.group(4)) if m_take else None

    return {
        "piles": piles,
        "misere": misere,
        "max_take": max_take,
    }


def solve_nim(piles: dict[int, int], misere: bool, max_take: int | None) -> tuple[int, int]:
    """Return (count_to_take, pile_index) for optimal move."""
    # Active non-empty piles: [(pile_idx, count)]
    active = [(p, c) for p, c in sorted(piles.items()) if c > 0]
    if not active:
        return 1, 1

    # ---------------- CASE 1: Subtraction Nim (Take 1..M) ----------------
    if max_take is not None and max_take >= 1:
        mod = max_take + 1
        # Grundy value of pile of size x is x % (M + 1)
        g_sum = 0
        for _, c in active:
            g_sum ^= (c % mod)

        if g_sum != 0:
            for p, c in active:
                current_g = c % mod
                target_g = current_g ^ g_sum
                if target_g < current_g:
                    take = current_g - target_g
                    if 1 <= take <= min(max_take, c):
                        return take, p
                else:
                    # In subtraction game with mod, can we reach target_g?
                    needed_g = target_g
                    for take in range(1, min(max_take, c) + 1):
                        if (c - take) % mod == needed_g:
                            return take, p

        # Losing position or no clean transition: take min(max_take, 1) from largest pile
        best_p, best_c = max(active, key=lambda item: item[1])
        return 1, best_p

    # ---------------- CASE 2: Misere Nim ----------------
    if misere:
        piles_gt_1 = [item for item in active if item[1] > 1]
        if len(piles_gt_1) == 1:
            # End-game: exactly one pile > 1, all others have size 1
            big_p, big_c = piles_gt_1[0]
            ones_count = sum(1 for _, c in active if c == 1 and _ != big_p)
            # If ones_count is even, we want to leave an ODD number of 1s -> leave 1
            # If ones_count is odd, we want to leave an ODD number of 1s -> leave 0 (take all)
            take = (big_c - 1) if (ones_count % 2 == 0) else big_c
            return max(1, take), big_p
        elif not piles_gt_1:
            # All remaining piles have size 1
            # Just take 1 from any pile
            return 1, active[0][0]

    # ---------------- CASE 3: Standard Nim (Normal or Misere before end-game) ----------------
    xor_sum = 0
    for _, c in active:
        xor_sum ^= c

    if xor_sum != 0:
        for p, c in active:
            target = c ^ xor_sum
            if target < c:
                take = c - target
                return take, p

    # Position is losing (xor_sum == 0): take 1 from largest pile
    best_p, _ = max(active, key=lambda item: item[1])
    return 1, best_p


class NimSolver:
    """Exact closed-form mathematical solver for Nim."""

    name = "nim"

    def __init__(self) -> None:
        self.rules_text = ""

    def configure(self, spec: object, rules: str = "", has_model: bool = False) -> None:
        scoring = str(getattr(spec, "scoring", "") or "")
        notes = str(getattr(spec, "winning_idea", "") or "")
        self.rules_text = "\n".join(part for part in (rules, scoring, notes) if part)

    def matches(self, game_id: str, spec: object) -> bool:
        gid = str(game_id or "").lower()
        if "nim" in gid:
            return True
        spec_id = str(getattr(spec, "game_id", "") or "").lower()
        return "nim" in spec_id

    def propose(
        self,
        *,
        observation: str,
        legal: Sequence[str] | None = None,
        seat: int = 0,
        turn: int = 0,
        timeout: float = 2.0,
        **kwargs,
    ) -> str | None:
        if not observation or _GAME_OVER.search(observation):
            return None

        state = parse_nim_state(observation, self.rules_text)
        if not state:
            return None

        take, pile_idx = solve_nim(state["piles"], state["misere"], state["max_take"])
        action = f"take {take} from pile {pile_idx}"

        if legal:
            # Snap to canonical spelling if legal list exists
            clean_legal = list(legal)
            action_low = action.lower()
            for cand in clean_legal:
                if cand.strip().lower() == action_low:
                    return cand

        return action
