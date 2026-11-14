"""Deterministic advisor for Bluff family games:
1. One-card numeric poker (`cards 1-N`, e.g. `tg_bluff`).
2. Mini Poker (`Kuhn poker` 3-card J/Q/K and `Leduc poker` 6-card 2xJ/2xQ/2xK with community card).
3. Liar's Dice / Perudo style quantity-face bidding games.

Reads only the arena's public observation text and our private card/dice. Never invents hidden opponent state.
Returns None when the observation does not match a known bluff format, falling back cleanly to the subagents/LLM.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

_DECK = re.compile(r"cards?\s*1\s*[-–]\s*(\d+)", re.IGNORECASE)
_CARD_NUM = re.compile(r"your card:\s*(\d+)", re.IGNORECASE)
_CARD_RANK = re.compile(r"your card:\s*([JQKA])\b", re.IGNORECASE)
_COMMUNITY_RANK = re.compile(r"community card:\s*([JQKA])\b", re.IGNORECASE)
_COMMUNITY_NONE = re.compile(r"community card:\s*not revealed", re.IGNORECASE)
_ROUND_NUM = re.compile(r"betting round\s*(\d+)", re.IGNORECASE)
_FIRST_ME = re.compile(r"you act first in this hand", re.IGNORECASE)
_FIRST_OPP = re.compile(r"your opponent acts first in this hand", re.IGNORECASE)
_HAND = re.compile(r"Hand\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
_LINES = re.compile(r"This hand so far:\s*(.*)", re.IGNORECASE)
_ROUND_LINE = re.compile(r"(?:Round\s*(\d+)\s*(?:\([^)]*\))?|R([12]))\s*:\s*(.*)", re.IGNORECASE)
_EARLIER = re.compile(r"(?:Earlier|Previous)\s+hands:\s*(.*?)(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)
_ENTRY = re.compile(r"Hand\s+(\d+):\s*(.*?)(?:\s*->\s*(.*))?$", re.IGNORECASE)
_LEGAL_LINE = re.compile(r"Legal actions:\s*([^\n]+)", re.IGNORECASE)
_ACTIONS = ("check", "bet", "call", "raise", "fold")
_RANK_VAL = {"j": 1, "q": 2, "k": 3, "a": 4}


class BluffSolver:
    name = "bluff"

    @classmethod
    def matches(cls, game_id: str, spec=None) -> bool:
        gid = str(game_id or "").lower()
        if any(k in gid for k in ("bluff", "poker", "kuhn", "leduc", "liar", "dudo", "perudo")):
            return True
        family = str(getattr(spec, "family", "") or "").lower()
        return family == "bluff"

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        text = str(observation or "")
        if not text:
            return None

        # 1. Mini Poker (Kuhn / Leduc with J/Q/K ranks)
        rank_match = _CARD_RANK.search(text)
        if rank_match:
            choice = self._propose_mini_poker(text, rank_match.group(1).lower(), legal, seat, turn)
            if choice:
                return choice

        # 2. Numeric 1..N one-card poker (`tg_bluff`)
        deck = _DECK.search(text)
        card = _CARD_NUM.search(text)
        if deck and card:
            n_cards = int(deck.group(1))
            my_card = int(card.group(1))
            if n_cards >= 2 and 1 <= my_card <= n_cards:
                line = _LINES.search(text)
                current = self._current_actions(line.group(1) if line else "")
                if current is not None:
                    tendencies = self._tendencies(text, seat)
                    choice = self._choose(my_card, n_cards, current, tendencies, seat, turn)
                    if choice:
                        return self._legal(choice, legal, text)

        # 3. Liar's Dice / Perudo (private dice + quantity/face bid or liar/challenge)
        liar_move = self._propose_liars_dice(text, legal)
        if liar_move:
            return liar_move

        return None

    # ------------------------------------------------------------------ Liar's Dice / Perudo
    @staticmethod
    def _propose_liars_dice(text: str, legal: list[str] | None) -> str | None:
        if not legal:
            return None
        dice_m = re.search(r"your dice[^:\n]*:\s*([1-6][1-6 ,\-]*)", text, re.IGNORECASE)
        if not dice_m:
            return None
        my_dice = [int(d) for d in re.findall(r"[1-6]", dice_m.group(1))]
        if not my_dice:
            return None
        opp_m = re.search(r"opponent(?:'s)? dice(?:\s+left|\s+count)?\s*[:=]\s*(\d+)", text, re.IGNORECASE)
        opp_count = int(opp_m.group(1)) if opp_m else len(my_dice)
        ones_wild = bool(re.search(r"\bones?\s+are\s+wild\b|\b1s?\s+count\s+as\s+any\b", text, re.IGNORECASE))

        legal_strs = [str(a).strip() for a in legal]
        challenge_act = next(
            (a for a in legal_strs if a.lower() in ("liar", "challenge", "dudo", "call", "bluff", "doubt")),
            None,
        )
        word_to_face = {
            "1": 1, "one": 1, "ones": 1,
            "2": 2, "two": 2, "twos": 2,
            "3": 3, "three": 3, "threes": 3,
            "4": 4, "four": 4, "fours": 4,
            "5": 5, "five": 5, "fives": 5,
            "6": 6, "six": 6, "sixes": 6,
        }
        face_pat = r"([1-6]|ones?|twos?|threes?|fours?|fives?|six(?:es)?)"
        bid_options: list[tuple[int, int, str]] = []
        for a in legal_strs:
            m = re.match(rf"^(?:bid\s+)?(\d+)\s*(?:x|\s+|of\s+){face_pat}s?$", a.strip(), re.IGNORECASE)
            if m:
                f_val = word_to_face.get(m.group(2).lower().rstrip("s"), word_to_face.get(m.group(2).lower(), 0))
                if f_val:
                    bid_options.append((int(m.group(1)), f_val, a))

        def expected_for(face: int) -> float:
            if ones_wild and face != 1:
                ours = sum(1 for d in my_dice if d in (face, 1))
                return ours + opp_count * (2.0 / 6.0)
            ours = sum(1 for d in my_dice if d == face)
            return ours + opp_count * (1.0 / 6.0)

        cur_m = re.search(
            rf"(?:current\s+bid|last\s+bid|opponent(?:'s)?\s+bids?)[^:\n]*:\s*(?:bid\s+)?(\d+)\s*(?:x|\s+|of\s+){face_pat}s?",
            text,
            re.IGNORECASE,
        )
        if cur_m and challenge_act:
            cur_q = int(cur_m.group(1))
            cur_f = word_to_face.get(cur_m.group(2).lower().rstrip("s"), word_to_face.get(cur_m.group(2).lower(), 1))
            if cur_q > expected_for(cur_f) + 0.65:
                return challenge_act

        if bid_options:
            # Pick the legal bid with the highest margin (expected_for(face) - quantity).
            best = max(bid_options, key=lambda item: (expected_for(item[1]) - item[0], -item[0], item[1]))
            if challenge_act and (expected_for(best[1]) - best[0] < -0.85):
                return challenge_act
            return best[2]
        return None

    # ------------------------------------------------------------------ Mini Poker (Kuhn & Leduc)
    def _propose_mini_poker(
        self,
        text: str,
        my_rank: str,
        legal: list[str] | None,
        seat: int,
        turn: int,
    ) -> str | None:
        allowed = self._extract_legal_words(legal, text)
        if not allowed:
            return None

        is_leduc = ("leduc" in text.lower()) or bool(_COMMUNITY_RANK.search(text) or _COMMUNITY_NONE.search(text))
        comm_match = _COMMUNITY_RANK.search(text)
        comm_rank = comm_match.group(1).lower() if comm_match else None
        round_match = _ROUND_NUM.search(text)
        bet_round = int(round_match.group(1)) if round_match else (2 if comm_rank else 1)

        hand_match = _HAND.search(text)
        hand_no = int(hand_match.group(1)) if hand_match else max(1, turn)
        act_first = True if _FIRST_ME.search(text) else (False if _FIRST_OPP.search(text) else ((hand_no - 1) % 2 == int(seat)))

        r1_acts, r2_acts = self._parse_mini_poker_rounds(text)
        cur_acts = r2_acts if bet_round == 2 else r1_acts
        stats = self._mini_poker_stats(text)

        facing_bet = "call" in allowed and ("fold" in allowed or "raise" in allowed)
        facing_raise = facing_bet and ("raise" not in allowed) and any(a == "raise" for a in cur_acts)

        if not is_leduc:
            choice = self._choose_kuhn(my_rank, act_first, facing_bet, stats, hand_no, seat)
        else:
            choice = self._choose_leduc(
                my_rank=my_rank,
                comm_rank=comm_rank,
                bet_round=bet_round,
                act_first=act_first,
                facing_bet=facing_bet,
                facing_raise=facing_raise,
                r1_acts=r1_acts,
                r2_acts=r2_acts,
                allowed=allowed,
                stats=stats,
                hand_no=hand_no,
                seat=seat,
            )
        return self._legal(choice, legal, text)

    @staticmethod
    def _parse_mini_poker_rounds(text: str) -> tuple[list[str], list[str]]:
        r1: list[str] = []
        r2: list[str] = []
        for match in _ROUND_LINE.finditer(text):
            r_idx = int(match.group(1) or match.group(2))
            body = match.group(3).lower()
            if "no actions yet" in body:
                acts: list[str] = []
            else:
                acts = [w for w in re.findall(r"\b(?:check|bet|call|raise|fold)\b", body)]
            if r_idx == 1:
                r1 = acts
            elif r_idx == 2:
                r2 = acts
        return r1, r2

    @staticmethod
    def _mini_poker_stats(text: str) -> dict:
        """Read previous hands in Kuhn/Leduc format to gauge opponent fold rate, aggression, and bluffs."""
        stats = {"hands": 0, "opp_folds": 0, "opp_bets": 0, "opp_raises": 0, "opp_bluffs_shown": 0, "showdowns": 0}
        block = _EARLIER.search(text)
        if not block:
            return stats
        for line in block.group(1).splitlines():
            row = line.strip().lower()
            if not row or not row.startswith("hand "):
                continue
            stats["hands"] += 1
            stats["opp_bets"] += len(re.findall(r"opponent\s+bet", row))
            stats["opp_raises"] += len(re.findall(r"opponent\s+raise", row))
            if "opponent fold" in row:
                stats["opp_folds"] += 1
            if "opponent had j" in row and ("opponent bet" in row or "opponent raise" in row):
                stats["opp_bluffs_shown"] += 1
            if "opponent had " in row and "not shown" not in row:
                stats["showdowns"] += 1
        return stats

    @staticmethod
    def _choose_kuhn(my_rank: str, act_first: bool, facing_bet: bool, stats: dict, hand_no: int, seat: int) -> str:
        """Game-theoretic Kuhn Poker (J < Q < K, 1 betting round, bet=1 into pot=2)."""
        hands = max(1, stats["hands"])
        opp_fold_rate = stats["opp_folds"] / hands
        opp_agg = (stats["opp_bets"] + stats["opp_raises"]) / hands

        if my_rank == "k":
            if facing_bet:
                return "call"
            # Acting first: mix value bet (75%) and trap check (25%, or more vs hyper-aggressive bettors).
            if act_first and opp_agg > 0.65 and (hand_no + seat) % 3 == 0:
                return "check"
            return "bet"

        if my_rank == "q":
            if not facing_bet:
                # Never bet Q yourself in Kuhn: K always calls and J always folds.
                return "check"
            # Facing a bet of 1 to win 3 (pot odds 25%; Nash call is 1/3).
            # Call more against loose/bluffy opponents, fold against passive opponents who only bet K.
            if stats["opp_bluffs_shown"] > 0 or opp_agg > 0.55:
                return "call"
            if stats["hands"] >= 3 and opp_agg < 0.25:
                return "fold"
            return "call" if (hand_no + seat) % 3 == 0 else "fold"

        # Holding J (worst card): never call a bet.
        if facing_bet:
            return "fold"
        # Bluff ~1/3 in Nash (more if opponent over-folds, never if opponent calls everything).
        if stats["hands"] >= 3 and stats["opp_folds"] == 0:
            return "check"
        if opp_fold_rate >= 0.45 and (hand_no + seat) % 2 == 0:
            return "bet"
        if not act_first and (hand_no + seat) % 3 == 0:
            return "bet"
        return "check"

    @staticmethod
    def _choose_leduc(
        *,
        my_rank: str,
        comm_rank: str | None,
        bet_round: int,
        act_first: bool,
        facing_bet: bool,
        facing_raise: bool,
        r1_acts: list[str],
        r2_acts: list[str],
        allowed: set[str],
        stats: dict,
        hand_no: int,
        seat: int,
    ) -> str:
        """Leduc Poker (2xJ, 2xQ, 2xK; Round 1 bet=2, Round 2 bet=4 with community card)."""
        hands = max(1, stats["hands"])
        opp_fold_rate = stats["opp_folds"] / hands
        paired = bool(comm_rank and my_rank == comm_rank)

        # ---------------- ROUND 1 (Pre-flop, no community card) ----------------
        if bet_round == 1:
            if my_rank == "k":
                if facing_bet:
                    # Value raise K pre-flop most of the time; call if raise not allowed.
                    if "raise" in allowed and (hand_no + seat) % 4 != 0:
                        return "raise"
                    return "call"
                return "bet"

            if my_rank == "q":
                if facing_raise:
                    # Opponent re-raised preflop: heavily weighted to K, but pot odds (2 into 10) allow calling unless tight.
                    return "call" if stats["opp_bluffs_shown"] > 0 else "fold"
                if facing_bet:
                    # Call 2 into pot 4 to see the flop and try to pair Q or outplay J. Never raise Q pre-flop.
                    return "call"
                # No bet yet: mostly check Q to control pot, occasional mix bet when acting second.
                if not act_first and opp_fold_rate > 0.5 and (hand_no % 3 == 0):
                    return "bet"
                return "check"

            # my_rank == "j" in Round 1
            if facing_bet:
                return "fold"
            if opp_fold_rate >= 0.55 and not act_first and (hand_no % 3 == 0):
                return "bet"
            return "check"

        # ---------------- ROUND 2 (Flop revealed: comm_rank is J, Q, or K) ----------------
        # 1. WE PAIRED THE BOARD (THE NUTS - unbeatable in Leduc!)
        if paired:
            if facing_bet:
                return "raise" if "raise" in allowed else "call"
            return "bet"

        # 2. UNPAIRED KING (strongest non-pair, but loses to pair of comm_rank!)
        if my_rank == "k":
            if facing_raise:
                # CRITICAL FIX (Hand 4 of T1): An opponent raising to 8 on the flop almost always holds the board pair
                # (especially on a Q board) or at best splits with the other K. Never pay off a Round 2 raise with unpaired K!
                return "fold"
            if facing_bet:
                # Opponent bet 4 on the flop.
                # If community card is Q and opponent called/bet in Round 1, Q is smack in the middle of their Round 1 range.
                if comm_rank == "q" and len(r1_acts) >= 2 and stats["opp_bluffs_shown"] == 0:
                    return "fold"
                return "call"
            # No bet in front of us in Round 2 with unpaired K:
            # If the board paired Q (which hits opponent's pre-flop call range), check for pot control!
            if comm_rank == "q" and any(a in ("bet", "call", "raise") for a in r1_acts):
                return "check"
            # On a J board (or checked-through pre-flop), bet thin value against Q.
            return "bet"

        # 3. UNPAIRED QUEEN (middle card: loses to pair AND loses to K-high!)
        if my_rank == "q":
            if facing_raise:
                return "fold"
            if facing_bet:
                # CRITICAL FIX (Hand 6 of T1): Calling both streets with Q-high on a J board loses to every K and every J-pair.
                # Only bluff-catch if opponent was passive in R1 (checked) AND has shown J bluffs before.
                r1_checked = all(a == "check" for a in r1_acts)
                if r1_checked and stats["opp_bluffs_shown"] > 0 and comm_rank == "k":
                    return "call"
                return "fold"
            return "check"

        # 4. UNPAIRED JACK (nut low: cannot win at showdown!)
        if facing_bet:
            return "fold"
        # Pure bluff spot only if everyone checked Round 1, board is K (scare card), and opponent folds often.
        if all(a == "check" for a in r1_acts) and comm_rank == "k" and opp_fold_rate >= 0.45 and (hand_no % 2 == 1):
            return "bet"
        return "check"

    # ------------------------------------------------------------------ 1..N numeric poker (`tg_bluff`)
    @staticmethod
    def _current_actions(raw: str) -> list[str] | None:
        """Our current hand's public actions, in order (unknown words are ignored)."""
        raw = raw.strip().rstrip(".")
        if not raw or "no action yet" in raw.lower():
            return []
        acts = [word for word in re.findall(r"[a-z]+", raw.lower()) if word in _ACTIONS]
        if not acts and raw not in ("", "(no action yet)"):
            return None
        return acts

    @staticmethod
    def _tendencies(text: str, seat: int) -> dict:
        """Count OUR opponent's public actions, from the hand summaries."""
        empty = {"bet": 0, "call": 0, "fold": 0, "check": 0, "showdowns": 0, "hands": 0}
        block = _EARLIER.search(text)
        if not block:
            return empty
        stats = dict(empty)
        for row in block.group(1).splitlines():
            row = row.strip()
            if not row:
                continue
            match = _ENTRY.search(row)
            if not match:
                continue
            hand_number = int(match.group(1))
            actions = [w for w in re.findall(r"[a-z]+", (match.group(2) or "").lower()) if w in _ACTIONS]
            outcome = (match.group(3) or "").lower()
            stats["hands"] += 1
            first_seat = (hand_number - 1) % 2
            for index, action in enumerate(actions):
                actor = (first_seat + index) % 2
                if actor != int(seat) and action in stats:
                    stats[action] += 1
            if "showdown" in outcome:
                stats["showdowns"] += 1
        return stats

    @staticmethod
    def _choose(my_card: int, n_cards: int, current: list[str], stats: dict, seat: int, turn: int) -> str:
        opponents = max(1, n_cards - 1)
        equity = (my_card - 1) / opponents
        bet_opportunities = max(1, stats["bet"] + stats["check"] + stats["fold"] + stats["call"])
        fold_rate = stats["fold"] / bet_opportunities
        aggressive = stats["bet"] / max(1, stats["bet"] + stats["check"])

        facing_bet = bool(current) and current[-1] == "bet"
        if facing_bet:
            threshold = max(0.25, 0.33 - 0.08 * aggressive)
            if equity >= threshold:
                return "call"
            return "fold"

        value_edge = 0.5 if n_cards <= 4 else 0.62
        if equity >= value_edge or (aggressive > 0.75 and equity >= 0.5):
            return "bet"
        bluff_spot = fold_rate >= 0.45 and equity < 0.4 and (turn + seat) % 3 == 0
        if bluff_spot:
            return "bet"
        return "check"

    @staticmethod
    def _extract_legal_words(legal: Iterable[str] | None, text: str = "") -> set[str]:
        words: set[str] = set()
        if legal:
            for item in legal:
                w = str(item).strip().lower().split()[0] if str(item).strip() else ""
                if w in _ACTIONS:
                    words.add(w)
        if not words and text:
            m = _LEGAL_LINE.search(text)
            if m:
                for w in re.findall(r"\b(check|bet|call|raise|fold)\b", m.group(1).lower()):
                    words.add(w)
        return words

    @classmethod
    def _legal(cls, choice: str | None, legal: Iterable[str] | None, text: str = "") -> str | None:
        if not choice:
            return None
        wanted = choice.strip().lower()
        if legal:
            for action in legal:
                cleaned = str(action).strip()
                if cleaned.lower() == wanted or cleaned.lower().startswith(wanted + " "):
                    return cleaned
            return None
        allowed = cls._extract_legal_words(None, text)
        if allowed and wanted not in allowed:
            for fallback in ("check", "fold", "call", "bet"):
                if fallback in allowed:
                    return fallback
            return None
        return wanted
