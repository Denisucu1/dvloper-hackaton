"""Deterministic solver for the `deals` family:
- Alternating-Offers / Split-the-Pot Bargaining with shrinking pot & private walk-away
- Iterated Prisoner's Dilemma / Trust / Split-or-Steal (Tit-for-Tat + final-round defection)
- Second-Price (Vickrey) and First-Price Sealed-Bid Auctions
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

from .actions import normalize
from .spec import Spec

_COOP_WORDS = ("cooperate", "trust", "split", "share", "ally", "invest", "contribute")
_DEFECT_WORDS = ("defect", "betray", "steal", "take", "cheat", "withhold", "keep")
_GAME_HINTS = ("deals", "pot", "bargain", "ultimatum", "negotiation", "dilemma", "trust", "auction")


def _nearest_numeric_action(raw: str, legal: Sequence[str]) -> str | None:
    """Return the legal item whose first number is closest to the number in `raw`."""
    target_m = re.search(r"-?\d+", raw)
    if not target_m:
        return None
    target = int(target_m.group(0))
    best: str | None = None
    best_distance: int | None = None
    for item in legal:
        item_m = re.search(r"-?\d+", str(item))
        if not item_m:
            continue
        distance = abs(int(item_m.group(0)) - target)
        if best_distance is None or distance < best_distance:
            best = str(item)
            best_distance = distance
    return best


class DealsSolver:
    def __init__(self) -> None:
        self.rules: str = ""
        self.spec: Spec | None = None

    def matches(self, game_id: str, spec: Spec | None = None) -> bool:
        family = (spec.family if spec else "").lower()
        if family == "deals":
            return True
        gid = (game_id or "").lower().strip()
        return any(tok in gid for tok in _GAME_HINTS)

    def configure(self, spec: Spec, rules: str = "", has_model: bool = False) -> None:
        self.spec = spec
        if rules:
            self.rules = rules

    def propose(
        self,
        *,
        observation: str,
        legal: Sequence[str] | None,
        seat: int,
        turn: int,
        timeout: float = 1.0,
        avoid: Iterable[str] | None = None,
    ) -> str | None:
        obs_l = (observation or "").lower()
        rules_l = (self.rules or "").lower()

        if self._is_bargaining(obs_l):
            act = self._propose_bargaining(observation, legal, self.rules)
            if act:
                return act

        if self._is_trust_dilemma(obs_l, legal):
            act = self._propose_trust_dilemma(observation, legal)
            if act:
                return act

        if self._is_explicit_auction(obs_l, rules_l):
            act = self._propose_auction(observation, legal, self.rules)
            if act:
                return act

        return None

    def _is_bargaining(self, obs_l: str) -> bool:
        has_pot = "current pot:" in obs_l or "pot:" in obs_l
        has_walk = "walk-away value:" in obs_l or "reserve value:" in obs_l
        return has_pot and has_walk

    def _propose_bargaining(
        self,
        observation: str,
        legal: Sequence[str] | None,
        rules: str,
    ) -> str | None:
        walk_m = re.search(r"(?:walk-away|reserve)\s+value\s*:\s*(\d+)", observation, re.I)
        pot_m = re.search(r"current\s+pot\s*:\s*(\d+)", observation, re.I)
        if not walk_m or not pot_m:
            return None

        my_walk = int(walk_m.group(1))
        cur_pot = int(pot_m.group(1))

        turn_m = re.search(r"turn\s+(\d+)\s+of\s+(\d+)", observation, re.I)
        obs_turn = int(turn_m.group(1)) if turn_m else 1
        max_turns = int(turn_m.group(2)) if turn_m else 8

        # Parse pot decay per turn from rules ("shrinking by 5 a turn"). When the rules only
        # say the pot shrinks, infer the decay from the *parsed* initial pot. Never assume 100.
        decay_m = re.search(r"shrink(?:ing|s)\s+by\s+(\d+)", rules or "", re.I)
        initial_pot: int | None = None
        for pattern in (
            r"initial\s+pot\s*(?:of|:)?\s*(\d+)",
            r"pot\s+of\s+(\d+)",
            r"starts?\s+at\s+(\d+)",
        ):
            init_m = re.search(pattern, rules or "", re.I)
            if init_m:
                initial_pot = int(init_m.group(1))
                break
        rules_l = (rules or "").lower()
        shrinks = "shrink" in rules_l or "decay" in rules_l
        if decay_m:
            decay = int(decay_m.group(1))
        elif shrinks and initial_pot is not None and obs_turn > 1:
            decay = max(0, round((initial_pot - cur_pot) / (obs_turn - 1)))
        elif shrinks and obs_turn > 1 and cur_pot < 100:
            decay = max(0, round((100 - cur_pot) / (obs_turn - 1)))
        else:
            decay = 0

        # Parse walk-away distribution range from rules e.g. "(10 to 45)"
        range_m = re.search(r"walk-away\s+value\s*\((\d+)\s+to\s+(\d+)\)", rules or "", re.I)
        low_w = int(range_m.group(1)) if range_m else 10
        high_w = int(range_m.group(2)) if range_m else 45

        offers = re.findall(
            r"(you|opponent)\s+offered\s+to\s+keep\s+(\d+)",
            observation,
            re.I,
        )
        opp_offers = [int(k) for who, k in offers if who.lower() == "opponent"]
        last_opp_keep = opp_offers[-1] if opp_offers else None

        # 1. Check whether accepting opponent's standing offer is winning right now at cur_pot
        # Must be safe against low_w so accepting never produces a 0.0 loss instead of a 0.5 no-deal draw.
        if last_opp_keep is not None:
            my_share = cur_pot - last_opp_keep
            my_surplus = my_share - my_walk
            opp_worst_surplus = last_opp_keep - low_w
            if my_share >= my_walk + 2 and my_surplus > opp_worst_surplus:
                act = self._fmt_deal_action("accept", legal)
                if act:
                    return act

        # 2. On the final turn, any new offer expires into no-deal anyway; walk cleanly if we didn't accept
        if obs_turn >= max_turns:
            act = self._fmt_deal_action("walk", legal)
            if act:
                return act

        # 3. Compute optimal counter-offer evaluated at NEXT turn's pot (when opponent decides to accept)
        next_pot = max(1, cur_pot - decay)
        # k_win must be safe against the lowest opponent walk-away in the rules (low_w):
        # K - my_walk > (next_pot - K) - low_w  <=>  2*K > next_pot + my_walk - low_w
        # Remember: No-deal is 0.5 pts, losing a deal is 0.0 pts! Never offer below k_win.
        k_win = max(
            my_walk + 2,
            int(math.floor((next_pot + my_walk - low_w) / 2.0)) + 2,
        )
        # Maximum keep that still leaves opponent >= int(0.45 * next_pot) on next turn
        opp_min_45 = int(0.45 * next_pot)
        k_45 = next_pot - opp_min_45

        if obs_turn <= 2:
            target_keep = max(k_win, k_45)
        else:
            concession = min(4, obs_turn - 2)
            target_keep = max(k_win, k_45 - concession)

        max_reasonable = max(my_walk + 1, cur_pot - low_w)
        target_keep = max(my_walk + 1, min(cur_pot, max_reasonable, target_keep))

        return self._fmt_deal_action(f"offer {target_keep}", legal)

    def _fmt_deal_action(self, raw: str, legal: Sequence[str] | None) -> str | None:
        if not legal:
            return raw
        normalized = normalize(raw, list(legal))
        if normalized:
            return normalized
        # Fall back to the legal offer whose numeric value is closest to the target
        # (e.g. raw "offer 56" with legal ["offer 50", "offer 60"] -> "offer 60").
        nearest = _nearest_numeric_action(raw, legal)
        if nearest is not None:
            return nearest
        # Nothing numeric to snap to: keep the raw action only if it normalizes.
        return normalize(raw, list(legal))

    def _is_trust_dilemma(self, obs_l: str, legal: Sequence[str] | None) -> bool:
        if legal and len(legal) == 2:
            a, b = legal[0].lower().strip(), legal[1].lower().strip()
            has_coop = any(w in a or w in b for w in _COOP_WORDS)
            has_def = any(w in a or w in b for w in _DEFECT_WORDS)
            if has_coop and has_def:
                return True
        if ("cooperate" in obs_l and "defect" in obs_l) or ("split" in obs_l and "steal" in obs_l):
            if "your turn" in obs_l or "choose" in obs_l:
                return True
        return False

    def _propose_trust_dilemma(
        self,
        observation: str,
        legal: Sequence[str] | None,
    ) -> str | None:
        obs_l = observation.lower()
        coop_act, defect_act = "cooperate", "defect"
        if legal and len(legal) == 2:
            for item in legal:
                il = item.lower()
                if any(w in il for w in _COOP_WORDS):
                    coop_act = item
                elif any(w in il for w in _DEFECT_WORDS):
                    defect_act = item
        elif "split" in obs_l and "steal" in obs_l:
            coop_act, defect_act = "split", "steal"
        elif "trust" in obs_l and "betray" in obs_l:
            coop_act, defect_act = "trust", "betray"

        round_m = re.search(r"(?:round|turn)\s+(\d+)\s+of\s+(\d+)", observation, re.I)
        cur_r = int(round_m.group(1)) if round_m else 1
        max_r = int(round_m.group(2)) if round_m else 1

        # Backward induction: on the final round (or a one-shot game with no round marker),
        # defecting strictly dominates.
        if not round_m or cur_r >= max_r:
            return defect_act

        opp_moves = re.findall(
            r"opponent\s+(?:played|chose|picked|selected)\s+['\"]?([a-zA-Z_]+)['\"]?",
            observation,
            re.I,
        )
        if opp_moves:
            last_opp = opp_moves[-1].lower()
            if any(w in last_opp for w in _DEFECT_WORDS):
                return defect_act

        return coop_act

    def _is_explicit_auction(self, obs_l: str, rules_l: str) -> bool:
        if "auction" not in obs_l and "auction" not in rules_l:
            return False
        combined = f"{obs_l}\n{rules_l}"
        return (
            "second-price" in combined
            or "second highest" in combined
            or "second-highest" in combined
            or "vickrey" in combined
            or "first-price" in combined
            or "sealed-bid" in combined
            or "pays its bid" in combined
        )

    def _propose_auction(
        self,
        observation: str,
        legal: Sequence[str] | None,
        rules: str,
    ) -> str | None:
        vals = re.findall(r"your\s+(?:private\s+)?(?:value|valuation)[^:\n]*:\s*(\d+)", observation, re.I)
        if len(vals) != 1:
            return None
        val = int(vals)[0] if isinstance(vals, int) else int(vals[0])
        combined_l = f"{(rules or '').lower()}\n{observation.lower()}"
        if "second-price" in combined_l or "second highest" in combined_l or "second-highest" in combined_l or "vickrey" in combined_l:
            bid = val
        else:
            range_m = re.search(
                r"(?:(\d+)\s+to\s+(\d+)|values\s+(\d+)\s*-\s*(\d+))",
                f"{rules or ''}\n{observation}",
                re.I,
            )
            low_v = int(range_m.group(1) or range_m.group(3)) if range_m else 0
            bid = max(low_v, int(round(low_v + 0.55 * (val - low_v))))

        raw = f"bid {bid}"
        if legal:
            normalized = normalize(raw, list(legal))
            if normalized:
                return normalized
            return _nearest_numeric_action(raw, legal)
        return raw
