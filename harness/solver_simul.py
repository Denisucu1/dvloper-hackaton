"""Adaptive mixed-strategy and resource-allocation expert for simultaneous ("read the opponent") games.

Covers two major game classes in the `read_opponent` family:
1. Finite simultaneous choice games (e.g. weighted/standard Rock-Paper-Scissors): exact mixed Nash
   equilibrium blended with empirical opponent adaptation and CSPRNG entropy.
2. Colonel Blotto / Simultaneous Resource Allocation games (`legal_actions` is null, players allocate
   soldiers/resources across K battlefields with equal or unequal values, fixed budgets or carry-over reserves):
   asymmetric concentration + opponent history best-response + reserve endgame management.
"""
from __future__ import annotations

import math
import re
import secrets

_RNG = secrets.SystemRandom()

_RPS = ("rock", "paper", "scissors")
_RPS_BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
_GAME_OVER = re.compile(r"\bgame\s+(?:is\s+)?over\b", re.IGNORECASE)
_NOT_SIMUL_FAMILY = {"deals", "bluff", "truth", "guardrails"}

_ROUND_PAIR = re.compile(r"Round\s+\d+\s*:\s*([a-z0-9_-]+)\s*/\s*([a-z0-9_-]+)", re.IGNORECASE)
_OPP_PLAYED = re.compile(
    r"\bopponent(?:'s)?\s+(?:played|chose|picked|used)\s+([a-z0-9_-]+)", re.IGNORECASE
)
_NUMERIC_LIKE = re.compile(
    r"^(?:\d+|(?:offer|bid|bet|raise|guess|answer|pick)\s+.+|.*\s\d+)$", re.IGNORECASE
)

# Colonel Blotto / Battlefield allocation patterns
_FIELD_VAL = re.compile(r"\bF(\d+)\s*=\s*(\d+)", re.IGNORECASE)
_PLAIN_BFIELD_VALS = re.compile(r"Battlefield\s+values\s+this\s+round\s*:\s*([0-9\s]+)", re.IGNORECASE)
_EXACT_BUDGET = re.compile(
    r"place\s+exactly\s+(\d+)\s+(?:soldiers|troops|units|points)\s+on\s+(\d+)\s+(?:battlefields|fields)",
    re.IGNORECASE,
)
_RESERVE_BUDGET = re.compile(
    r"your\s+reserve\s*:\s*(\d+)\s*(?:soldiers|troops|units)?.*?(?:opponent(?:'s)?\s+reserve\s*:\s*(\d+))?",
    re.IGNORECASE | re.DOTALL,
)
_ROUND_OF = re.compile(r"\bRound\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
_ROUNDS_WON = re.compile(
    r"(?:Rounds\s+won\s*:\s*you\s+(\d+)\s*,\s*opponent\s+(\d+)|Score\s*:\s*You\s+(\d+)\s*-\s*(\d+)\s*Opponent)",
    re.IGNORECASE,
)
_HIST_FIELD = re.compile(r"F(\d+)\s*\(worth\s+(\d+)\)\s*:\s*you\s+(\d+)\s+vs\s+(\d+)", re.IGNORECASE)
_ALT_HIST_ROUND = re.compile(
    r"Round\s+\d+\s*(?:\(values\s+([0-9\s]+)\))?\s*:\s*You\s*:\s*([0-9\s]+)\|\s*Opponent\s*:\s*([0-9\s]+)",
    re.IGNORECASE,
)
_MOVE_K_NUMS = re.compile(r"(\d+)\s+numbers\s+separated\s+by\s+spaces", re.IGNORECASE)


def _rps_map(legal: list[str]) -> dict[str, str] | None:
    """Return {canonical: legal_spelling} when legal options are exactly Rock-Paper-Scissors."""
    by_lower = {item.strip().lower(): item for item in legal}
    if set(by_lower) == set(_RPS):
        return {name: by_lower[name] for name in _RPS}
    return None


def _parse_rps_weights(rules_text: str) -> dict[str, float]:
    """Read per-sign win points from rules (e.g. 'won with rock scores 3'); default 1 each."""
    weights = {sign: 1.0 for sign in _RPS}
    if not rules_text:
        return weights
    for sign in _RPS:
        match = re.search(
            rf"(?:with\s+{sign}\s+(?:scores?|wins?|gives?|gets?)|{sign}\s+wins?)\s+(\d+)",
            rules_text,
            re.IGNORECASE,
        )
        if match:
            value = int(match.group(1))
            if value >= 1:
                weights[sign] = float(value)
    return weights


def _opponent_counts(observation: str, signs: tuple[str, ...]) -> dict[str, int]:
    """Count opponent's past choices from history lines in the observation."""
    counts = {sign: 0 for sign in signs}
    valid = set(signs)
    for line in (observation or "").splitlines():
        m_pair = _ROUND_PAIR.search(line)
        if m_pair:
            theirs = m_pair.group(2).strip().lower()
            if theirs in valid:
                counts[theirs] += 1
            continue
        m_opp = _OPP_PLAYED.search(line)
        if m_opp:
            theirs = m_opp.group(1).strip().lower()
            if theirs in valid:
                counts[theirs] += 1
    return counts


def _rps_distribution(rules_text: str, observation: str) -> list[float]:
    """Blend exact weighted Nash (unexploitable) with soft empirical best-response."""
    w = _parse_rps_weights(rules_text)
    nash_raw = [w["scissors"], w["rock"], w["paper"]]
    nash_total = sum(nash_raw) or 3.0
    nash = [v / nash_total for v in nash_raw]

    counts = _opponent_counts(observation, _RPS)
    prior = 2.0
    denom = sum(counts.values()) + prior * len(_RPS)
    q = {s: (counts[s] + prior) / denom for s in _RPS}

    evs = []
    for sign in _RPS:
        beats = _RPS_BEATS[sign]
        loses_to = next(k for k, v in _RPS_BEATS.items() if v == sign)
        evs.append(q[beats] * w[sign] - q[loses_to] * w[loses_to])

    temp = max(0.35, 0.25 * max(w.values()))
    max_ev = max(evs)
    exp_evs = [math.exp((ev - max_ev) / temp) for ev in evs]
    exp_sum = sum(exp_evs) or 3.0
    adaptive = [v / exp_sum for v in exp_evs]

    return [0.60 * n + 0.40 * a for n, a in zip(nash, adaptive)]


# ------------------------------------------------------------------ Colonel Blotto / Allocation Solver
def _parse_blotto_history(text: str, k: int) -> list[tuple[list[int], list[int], list[int]]]:
    """Return list of (values, my_alloc, opp_alloc) for each completed round in History."""
    history: list[tuple[list[int], list[int], list[int]]] = []
    for line in text.splitlines():
        if "Round " not in line:
            continue
        matches = _HIST_FIELD.findall(line)
        if len(matches) == k:
            vals = [int(m[1]) for m in matches]
            mine = [int(m[2]) for m in matches]
            theirs = [int(m[3]) for m in matches]
            history.append((vals, mine, theirs))
            continue
        m_alt = _ALT_HIST_ROUND.search(line)
        if m_alt:
            mine = [int(x) for x in m_alt.group(2).split() if x.isdigit()]
            theirs = [int(x) for x in m_alt.group(3).split() if x.isdigit()]
            vals = (
                [int(x) for x in m_alt.group(1).split() if x.isdigit()]
                if m_alt.group(1)
                else [1] * k
            )
            if len(mine) == k and len(theirs) == k:
                if len(vals) != k:
                    vals = [1] * k
                history.append((vals, mine, theirs))
    return history


def _propose_blotto(observation: str, rules_text: str, seat: int, turn: int) -> str | None:
    """Solve Colonel Blotto across equal-field, weighted-field, and carry-over reserve variants."""
    field_matches = _FIELD_VAL.findall(observation)
    if field_matches:
        values = [int(val) for _, val in field_matches]
    else:
        m_plain = _PLAIN_BFIELD_VALS.search(observation)
        if not m_plain:
            return None
        values = [int(x) for x in m_plain.group(1).split() if x.isdigit()]
    k = len(values)
    if k < 2:
        return None

    m_round = _ROUND_OF.search(observation)
    round_no = int(m_round.group(1)) if m_round else max(1, turn)
    total_rounds = int(m_round.group(2)) if m_round else 5

    m_won = _ROUNDS_WON.search(observation)
    if m_won:
        my_wins = int(m_won.group(1) or m_won.group(3) or 0)
        opp_wins = int(m_won.group(2) or m_won.group(4) or 0)
    else:
        my_wins = 0
        opp_wins = 0

    history = _parse_blotto_history(observation, k)

    # Check whether this is a carry-over Reserve variant or a fixed-budget variant
    m_res = _RESERVE_BUDGET.search(observation)
    if m_res:
        my_res = int(m_res.group(1))
        opp_res = int(m_res.group(2)) if m_res.group(2) else my_res
        alloc = _solve_blotto_reserves(
            k=k,
            values=values,
            my_res=my_res,
            opp_res=opp_res,
            round_no=round_no,
            total_rounds=total_rounds,
            my_wins=my_wins,
            opp_wins=opp_wins,
            history=history,
            seat=seat,
        )
        return " ".join(str(x) for x in alloc)

    m_exact = _EXACT_BUDGET.search(observation)
    if not m_exact:
        m_exact = _EXACT_BUDGET.search(rules_text)
    if not m_exact:
        return None
    budget = int(m_exact.group(1))
    alloc = _solve_blotto_fixed(
        k=k,
        values=values,
        budget=budget,
        round_no=round_no,
        history=history,
        seat=seat,
    )
    return " ".join(str(x) for x in alloc)


def _solve_blotto_reserves(
    *,
    k: int,
    values: list[int],
    my_res: int,
    opp_res: int,
    round_no: int,
    total_rounds: int,
    my_wins: int,
    opp_wins: int,
    history: list[tuple[list[int], list[int], list[int]]],
    seat: int,
) -> list[int]:
    """Optimal reserve management + 2-of-3 asymmetric field targeting."""
    if my_res <= 0:
        return [0] * k

    wins_to_clinch = (total_rounds // 2) + 1
    need_me = max(1, wins_to_clinch - my_wins)

    # 1. Decide total soldiers to deploy this round (`spend` <= my_res)
    if round_no >= total_rounds:
        # Final round: leftover reserve has zero value, spend everything!
        spend = my_res
    elif need_me == 1 and my_res >= opp_res - 3:
        # Match point for us with comparable or superior army: go all-in on 2 fields to close out 3-1/3-0!
        spend = my_res
    elif my_res <= opp_res - 10 and round_no < total_rounds:
        # Badly outnumbered this round: bank virtually all soldiers (send 1 token soldier in case of 0-0)
        # so the opponent wastes their army and we enter next round with +30 income and a big lead.
        alloc = [0] * k
        if my_res >= 1:
            alloc[(round_no + seat) % k] = 1
        return alloc
    elif round_no == 1 and my_res == opp_res:
        # Round 1: deploy ~73% (e.g. 22 of 30) on 2 fields so we beat even splits (10,10,10) AND save 8 reserve!
        spend = max(2, int(round(my_res * 0.74)))
    else:
        # Normal middle round: if we have a reserve lead, press it (~78%); otherwise spend ~70%.
        ratio = 0.78 if my_res > opp_res else 0.70
        spend = max(2, min(my_res, int(round(my_res * ratio))))

    # 2. Choose WHICH fields to contest (for k=3 equal fields, we only ever need 2 fields!)
    target_count = (k // 2) + 1
    if round_no >= total_rounds and my_res >= 2 * opp_res + k:
        # Crushing final-round lead: dominate all k fields
        base = my_res // k
        rem = my_res - base * k
        return [base + (1 if i < rem else 0) for i in range(k)]

    # Analyze opponent's field habits from history
    if history:
        opp_totals = [sum(h[2][i] for h in history) / len(history) for i in range(k)]
        last_opp = history[-1][2]
        # Rank fields by where opponent puts the FEWEST soldiers (easiest to win cheap!)
        field_order = sorted(range(k), key=lambda i: (last_opp[i] * 0.6 + opp_totals[i] * 0.4, -values[i]))
        weakest_field = field_order[0]
        # Pick the weakest field + one of the other fields (rotating so we are unpredictable)
        others = [i for i in range(k) if i != weakest_field]
        second_field = others[(round_no + seat) % len(others)]
        chosen = [second_field, weakest_field]
    else:
        # Round 1: avoid leaving F3=0 like every textbook bot; target F2 and F3 (or F1 and F3)
        start_idx = (round_no + seat + 1) % k
        chosen = [start_idx, (start_idx + 1) % k]

    alloc = [0] * k
    if target_count == 2 or len(chosen) == 2:
        f_heavy, f_light = chosen[0], chosen[1]
        # Asymmetric split (e.g. 60% / 40% when opponent plays 2 fields, or 52% / 48% when going all-in)
        if history and min(history[-1][2]) == 0 and spend < my_res:
            # Opponent is leaving a field at 0! Put ~64% on one contested field and ~36% on their 0-field.
            heavy = max(1, int(round(spend * 0.64)))
        else:
            heavy = (spend // 2) + 1 if spend >= 2 else spend
        light = max(0, spend - heavy)
        alloc[f_heavy] = heavy
        alloc[f_light] = light
        return alloc

    # General k > 3 with reserves
    per = spend // target_count
    rem = spend - per * target_count
    for idx, f in enumerate(chosen[:target_count]):
        alloc[f] = per + (1 if idx < rem else 0)
    return alloc


def _solve_blotto_fixed(
    *,
    k: int,
    values: list[int],
    budget: int,
    round_no: int,
    history: list[tuple[list[int], list[int], list[int]]],
    seat: int,
) -> list[int]:
    """Fixed-budget Colonel Blotto: exact sum == budget across k equal or weighted battlefields."""
    equal_values = len(set(values)) == 1

    # ---------------- CASE 1: Equal field values (e.g. 100 soldiers, 5 equal fields) ----------------
    if equal_values:
        win_fields = (k // 2) + 1  # e.g. 3 of 5
        if history:
            # Predict opponent's allocation per field from history
            opp_avg = [sum(h[2][i] for h in history) / len(history) for i in range(k)]
            last_opp = history[-1][2]
            pred = [0.65 * last_opp[i] + 0.35 * opp_avg[i] for i in range(k)]
            # Sort fields from cheapest to beat to most expensive
            order = sorted(range(k), key=lambda i: (pred[i], ((i + round_no + seat) % k)))
            targets = order[:win_fields]
            cost_needed = sum(int(pred[i]) + 2 for i in targets)
            if cost_needed <= budget:
                alloc = [0] * k
                surplus = budget - cost_needed
                # Put a small token on the (win_fields+1)-th field if cheap, rest on targets
                extra_field = order[win_fields] if win_fields < k else None
                if extra_field is not None and pred[extra_field] == 0 and surplus >= 2:
                    alloc[extra_field] = 2
                    surplus -= 2
                for idx, f in enumerate(targets):
                    alloc[f] = int(pred[f]) + 2 + (surplus // win_fields) + (1 if idx < (surplus % win_fields) else 0)
                return _fix_exact_sum(alloc, budget)

        # Round 1 or high-cost opponent: Asymmetric 3-field concentration + 1 token snipe
        # E.g. for budget=100, k=5: [36, 36, 26, 2, 0] permuted -> beats both (20,20,20,20,20) and (34,33,33,0,0)!
        if k == 5 and budget == 100:
            template = [36, 36, 26, 2, 0]
        else:
            main_budget = max(win_fields, budget - 2)
            base = main_budget // win_fields
            rem = main_budget - base * win_fields
            template = [base + (1 if i < rem else 0) for i in range(win_fields)] + [budget - main_budget] + [0] * max(0, k - win_fields - 1)
            template = template[:k]

        shift = (round_no * 2 + seat + 1) % k
        alloc = [template[(i - shift) % k] for i in range(k)]
        return _fix_exact_sum(alloc, budget)

    # ---------------- CASE 2: Weighted field values (e.g. 60 soldiers, 6 fields worth 1-6) ----------------
    total_val = sum(values)
    target_val = (total_val // 2) + 1  # e.g. 11 out of 21

    # Rank fields by value descending
    by_val_desc = sorted(range(k), key=lambda i: (-values[i], i))

    if history:
        # Map how many soldiers opponent historically placed on each VALUE rank (since values shuffle each round!)
        by_rank_opp: dict[int, list[int]] = {v: [] for v in set(values)}
        for h_vals, _, h_opp in history:
            for v, o in zip(h_vals, h_opp):
                by_rank_opp.setdefault(v, []).append(o)
        pred_for_field = [
            (sum(by_rank_opp.get(values[i], [])) / len(by_rank_opp[values[i]]))
            if by_rank_opp.get(values[i])
            else (budget * values[i] / total_val)
            for i in range(k)
        ]
        # Dynamic programming / subset search (k <= 10) for subset with value >= target_val + 1 and minimal soldier cost
        best_subset = None
        best_score = -1e9
        for mask in range(1, 1 << k):
            pts = sum(values[i] for i in range(k) if (mask & (1 << i)))
            if pts < target_val:
                continue
            req = sum(int(pred_for_field[i]) + 2 for i in range(k) if (mask & (1 << i)))
            if req <= budget:
                # Prefer subsets that achieve >= target_val + 1 with plenty of surplus soldiers per field
                count_f = bin(mask).count("1")
                surplus_per_field = (budget - req) / max(1, count_f)
                score = (10.0 if pts >= target_val + 1 else 0.0) + surplus_per_field * 2.0 + pts * 0.5
                if score > best_score:
                    best_score = score
                    best_subset = [i for i in range(k) if (mask & (1 << i))]
        if best_subset:
            alloc = [0] * k
            req_total = sum(int(pred_for_field[i]) + 2 for i in best_subset)
            surplus = max(0, budget - req_total)
            w_sum = sum(values[i] for i in best_subset) or len(best_subset)
            for i in best_subset:
                alloc[i] = int(pred_for_field[i]) + 2 + int(surplus * values[i] / w_sum)
            return _fix_exact_sum(alloc, budget)

    # Round 1 default for weighted fields (e.g. values 1..6, total 21, budget 60):
    # Skip the single most contested top field (worth 6) on even rounds or contest (5, 4, 3, 2, 1 = 15 pts!)
    # Putting [0 on v=6, 23 on v=5, 23 on v=4, 10 on v=3, 3 on v=2, 1 on v=1] beats BOTH proportional
    # splitters (wins 5+4+3=12 pts) AND top-3 concentrators [20,20,20 on 6,5,4] (wins 5+4+3+2+1=15 pts)!
    alloc = [0] * k
    if k == 6 and sorted(values) == [1, 2, 3, 4, 5, 6] and budget == 60:
        if (round_no + seat) % 2 == 1:
            rank_plan = {6: 0, 5: 23, 4: 23, 3: 10, 2: 3, 1: 1}
        else:
            rank_plan = {6: 23, 5: 22, 4: 0, 3: 11, 2: 3, 1: 1}
        for i, v in enumerate(values):
            alloc[i] = rank_plan.get(v, 0)
        return _fix_exact_sum(alloc, budget)

    # General weighted fallback: concentrate on highest-value fields until value >= target_val + 1
    chosen: list[int] = []
    acc_val = 0
    for idx in by_val_desc:
        chosen.append(idx)
        acc_val += values[idx]
        if acc_val >= target_val + 1:
            break
    c_sum = sum(values[i] for i in chosen) or 1
    for i in chosen:
        alloc[i] = int(round(budget * values[i] / c_sum))
    return _fix_exact_sum(alloc, budget)


def _fix_exact_sum(alloc: list[int], budget: int) -> list[int]:
    alloc = [max(0, int(x)) for x in alloc]
    diff = budget - sum(alloc)
    if diff > 0:
        # Add remainder to the field that already has the largest allocation
        best_i = max(range(len(alloc)), key=lambda i: alloc[i])
        alloc[best_i] += diff
    elif diff < 0:
        need = -diff
        for i in sorted(range(len(alloc)), key=lambda idx: -alloc[idx]):
            take = min(alloc[i], need)
            alloc[i] -= take
            need -= take
            if need == 0:
                break
    return alloc


class SimultaneousSolver:
    """Nash-mixed and resource-allocation solver for simultaneous-move games."""

    name = "simul"

    def __init__(self) -> None:
        self.rules_text = ""
        self.has_model = False

    def configure(self, spec: object, rules: str = "", has_model: bool = False) -> None:
        scoring = str(getattr(spec, "scoring", "") or "")
        notes = str(getattr(spec, "winning_idea", "") or "")
        self.rules_text = "\n".join(part for part in (rules, scoring, notes) if part)
        self.has_model = bool(has_model)

    def matches(self, game_id: str, spec: object) -> bool:
        gid = str(game_id or "").lower()
        if "blotto" in gid or "simul" in gid or "rps" in gid:
            return True
        family = str(getattr(spec, "family", "") or "").lower()
        if family == "read_opponent":
            return True
        return bool(getattr(spec, "simultaneous", False)) and family not in _NOT_SIMUL_FAMILY

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        """Propose an allocation move (Blotto) or a mixed-Nash discrete move (RPS/simultaneous)."""
        if not observation or _GAME_OVER.search(observation):
            return None

        # 1. Colonel Blotto / Resource Allocation across battlefields (typically legal is None)
        blotto_move = _propose_blotto(observation, self.rules_text, seat, turn)
        if blotto_move:
            return blotto_move

        # 2. Finite discrete simultaneous choices (RPS, etc.)
        if not legal:
            return None
        clean_legal = [str(item) for item in legal]
        if not (2 <= len(clean_legal) <= 12):
            return None
        if sum(1 for item in clean_legal if _NUMERIC_LIKE.match(item.strip())) > len(clean_legal) // 2:
            return None

        rps = _rps_map(clean_legal)
        if rps is not None:
            weights = _rps_distribution(f"{self.rules_text}\n{observation}", observation)
            chosen_key = _RNG.choices(list(_RPS), weights=weights, k=1)[0]
            return rps[chosen_key]

        if self.has_model:
            return None
        return _RNG.choice(clean_legal)
