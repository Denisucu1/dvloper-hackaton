#!/usr/bin/env python3
"""Measured intelligence score for the rounds our bot actually played.

This stage does not trust the bench report: it loads the replay JSON files the arena wrote, rebuilds
every game with the real arena class ``(variant, seed)``, and walks the original timeline through
that fresh game. Every decision our seat made is checked against ``game.normalize`` and the arena's
legal list *before* it is applied, and every decision is also handed to our own expert stack to get
an independent reference move.

    python3 scripts/intelligence.py
    python3 scripts/intelligence.py --run bench/results/<run>/replays --limit 4 --json out.json

Formula (0-100, deterministic, the module docstring is the contract):

    legality   30 * legal_decisions / total_decisions
               forced to 0 when any replay error, our illegal move, forfeit or crash is present.
    tactics    35 * agree / judged          (agreement/quality with our OWN expert stack, not proven
               optimality; the neutral 0.5 is used only when judged == 0).
    outcome    20 * (wins + 0.5 * draws) / rounds, or 20 * report score_pct / 100 when available.
    efficiency 5 * clamp(1 - p95_ms / 20000) + 5 * clamp(1 - usd_per_match / 0.30)
               p95 is measured from our replay decisions; a missing cost scores the full 5 points.
    chat       5 * chat_reply_rate * (our chat lines <= 200 chars / all our chat lines)

Overall is the weighted sum rounded to one decimal. Components are printed on a 0-100 scale
(points / weight), so the component line adds up to the overall score.

Tactics matching policy (numeric actions are not strings):

    - discrete actions and board moves: exact canonical match (case/space-insensitive);
    - deals-family ``offer <n>``: agreement when our n is within +-5 of the reference OR not worse
      than it by more than 5 coins (offering more for ourselves counts as agreement); a strictly
      worse offer by more than 5 is a disagreement;
    - deals-family ``accept`` / ``walk``: exact match only;
    - ``bid <n>`` anywhere: exact matches count, non-matches are UNJUDGED (bids are not comparable
      without the private hand, so they must not lower the score);
    - think_ahead / Connect-K: no anytime search comparison. The observation is parsed with
      ``harness.solver_grid.parse_connect_k``; a position is judged only when we have an immediate
      winning drop (that column is forced) or, failing that, the opponent has an immediate win (any
      blocking column is forced). Positions with no immediate tactic are UNJUDGED. This is a
      deterministic blunder check (missed win / missed block), not a time-budget measurement.

The JSON breakdown records judged/agree decisions per reference family, so every tactics point is
auditable. Confidence is high when judged >= 10 and rounds >= 3, medium when judged >= 3, low
otherwise.

No network is used. A solver that raises, a replay that cannot be walked or an unknown game id is
counted and reported, never allowed to crash the stage. When no replay is found the stage prints the
SKIPPED line and exits 0.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import importlib
import inspect
import json
import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA = os.path.join(ROOT, "arena-lite")
for _path in (ROOT, ARENA):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Arena category -> harness spec family (the mapping scripts/solver_smoke.py uses).
CATEGORY_FAMILY = {
    "read": "read_opponent",
    "think": "think_ahead",
    "guess": "guessing",
    "guard": "guardrails",
    "practice": "unknown",
}

WEIGHTS = {"legality": 30.0, "tactics": 35.0, "outcome": 20.0, "efficiency": 10.0, "chat": 5.0}
MOVE_BUDGET_MS = 20_000.0
COST_BUDGET_USD = 0.30
TACTICS_MIN_JUDGED = 3
SOLVER_TIMEOUT = 0.1
LOW_CONFIDENCE = "low confidence"


# --------------------------------------------------------------------------- discovery
def _replay_files(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "*.json")))


def find_replays(run_dir: str | None = None, limit: int | None = None) -> tuple[str | None, list[str]]:
    """Return ``(replays_dir, files)`` for the explicit dir or the newest bench run with replays."""
    if run_dir:
        directory = os.path.abspath(os.path.expanduser(run_dir))
        files = _replay_files(directory)
        return (directory if files else None), (files[:limit] if limit else files)

    candidates: list[tuple[float, str]] = []
    for directory in glob.glob(os.path.join(ROOT, "bench", "results", "*", "replays")):
        files = _replay_files(directory)
        if not files:
            continue
        try:
            mtime = os.path.getmtime(directory)
        except OSError:
            mtime = 0.0
        candidates.append((mtime, directory))
    for _mtime, directory in sorted(candidates, reverse=True):
        files = _replay_files(directory)
        return directory, (files[:limit] if limit else files)
    return None, []


def load_report(replays_dir: str | None) -> dict:
    """The sibling ``report.json`` one level up from the replays directory, when present."""
    if not replays_dir:
        return {}
    path = os.path.join(os.path.dirname(replays_dir.rstrip(os.sep)), "report.json")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- arena access
def _game_classes() -> dict[str, type]:
    """Registry first (the arena's own order), falling back to a scan of the two game packages."""
    try:
        from arena.registry import all_games

        return dict(all_games())
    except Exception:  # noqa: BLE001 - fall back to the scanner solver_smoke.py uses
        pass
    found: dict[str, type] = {}
    try:
        from arena.core import Game
    except Exception:  # noqa: BLE001
        return found
    for package in ("arena.games", "arena.practice"):
        try:
            importlib.import_module(package)
        except ImportError:
            continue
        directory = os.path.join(ARENA, "arena", package.split(".")[1])
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if not name.endswith(".py") or name.startswith("__"):
                continue
            try:
                module = importlib.import_module(f"{package}.{name[:-3]}")
            except Exception:  # noqa: BLE001
                continue
            for obj in vars(module).values():
                if isinstance(obj, type) and issubclass(obj, Game) and obj is not Game and getattr(obj, "id", ""):
                    found[obj.id] = obj
    return found


def _canonical(action: object) -> str:
    return " ".join(str(action or "").strip().lower().split())


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _spec_for(game_id: str, category: str):
    from harness.spec import Spec

    return Spec(game_id=str(game_id or ""), family=CATEGORY_FAMILY.get(category, category or "unknown"))


# --------------------------------------------------------------------------- expert stack
def _build_experts() -> tuple[list, list[str]]:
    """``harness.engine._solvers()`` in engine order, plus the two engine-side specialists."""
    notes: list[str] = []
    try:
        from harness.engine import _solvers

        experts = list(_solvers())
    except Exception as exc:  # noqa: BLE001 - a broken engine must not fail the stage
        notes.append(f"experts unavailable: {type(exc).__name__}: {exc}")
        experts = []
    return experts, notes


def _specialists(experts: list, game_id: str, category: str, observation: str) -> list:
    """DealsSolver for deals-family games and ConnectKSolver for connect text, on demand."""
    selected = list(experts)
    family = CATEGORY_FAMILY.get(category, category or "unknown")
    lowered = f"{game_id} {observation}".lower()
    if family == "deals" or any(hint in lowered for hint in ("deal", "split the pot", "negotiat")):
        try:
            from harness.solver_deals import DealsSolver

            deals = DealsSolver()
            if not any(type(item).__name__ == "DealsSolver" for item in selected):
                selected.append(deals)
        except Exception as exc:  # noqa: BLE001
            pass
    if "connect" in lowered and not any(type(item).__name__ == "ConnectKSolver" for item in selected):
        try:
            from harness.solver_grid import ConnectKSolver

            selected.append(ConnectKSolver())
        except Exception as exc:  # noqa: BLE001
            pass
    return selected


def _configure(solver, spec, rules: str) -> None:
    configure = getattr(solver, "configure", None)
    if not callable(configure):
        return
    try:
        configure(spec, rules, False)
    except TypeError:
        try:
            configure(spec)
        except Exception:  # noqa: BLE001 - configuration is best effort only
            pass
    except Exception:  # noqa: BLE001
        pass


def _propose(solver, observation: str, legal, seat: int, turn: int) -> str | None:
    """Call ``propose`` with timeout 0.1, adapting to solvers with narrower signatures."""
    function = getattr(solver, "propose", None)
    if not callable(function):
        return None
    wanted = {"observation": observation, "legal": legal, "seat": seat, "turn": turn, "timeout": SOLVER_TIMEOUT}
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        parameters = {}
    has_var_kw = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    if parameters and "max_seconds" in parameters and "timeout" not in parameters:
        wanted.pop("timeout", None)
        wanted["max_seconds"] = SOLVER_TIMEOUT
    kwargs = dict(wanted) if (has_var_kw or not parameters) else {
        key: value for key, value in wanted.items() if key in parameters
    }
    try:
        return function(**kwargs)
    except TypeError:
        try:
            return function(observation=observation, legal=legal)
        except Exception:  # noqa: BLE001 - a solver bug is skipped, never fatal
            return None
    except Exception:  # noqa: BLE001
        return None


def _reference(experts: list, spec, rules: str, observation: str, legal, seat: int, turn: int,
               stats: dict) -> tuple[str | None, object]:
    """First non-None, legal proposal from our expert stack. Returns (action, solver)."""
    for solver in experts:
        name = str(getattr(solver, "name", type(solver).__name__))
        matches = getattr(solver, "matches", None)
        if callable(matches):
            try:
                if not matches(spec.game_id, spec):
                    continue
            except Exception:  # noqa: BLE001 - a matcher bug skips that solver
                stats["solver_errors"] = stats.get("solver_errors", 0) + 1
                continue
        _configure(solver, spec, rules)
        try:
            proposal = _propose(solver, observation, legal, seat, turn)
        except Exception:  # noqa: BLE001
            stats["solver_errors"] = stats.get("solver_errors", 0) + 1
            continue
        if proposal is None:
            continue
        text = str(proposal).strip()
        if not text:
            continue
        if legal:
            from harness.actions import normalize

            normalized = normalize(text, legal)
            if not normalized:
                stats["solver_illegal"] = stats.get("solver_illegal", 0) + 1
                continue
            text = normalized
        return text, solver
    return None, None


# --------------------------------------------------------------------------- chat helpers
def _chat_entries(doc: dict, game_doc: dict | None = None) -> list[dict]:
    entries: list[dict] = []
    seen: set[tuple] = set()
    sources = [doc.get("chat") or []]
    if game_doc is not None:
        sources.append(game_doc.get("chat") or [])
    for source in sources:
        for item in source:
            if not isinstance(item, dict):
                continue
            key = (item.get("n"), item.get("game"), item.get("seat"), item.get("team"), item.get("text"))
            if key in seen:
                continue
            seen.add(key)
            entries.append(item)
    return entries


def _add_chat(target: list, seen: set, entries) -> None:
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        key = (item.get("n"), item.get("game"), item.get("seat"), item.get("team"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        target.append(item)


def _authored_by_us(entry: dict, bot_name: str, our_seat: int | None,
                    seats_by_game: dict[int, int]) -> bool:
    team = entry.get("team")
    if team is not None:
        return str(team) == bot_name
    seat = entry.get("seat")
    if isinstance(seat, int):
        game_no = entry.get("game")
        expected = seats_by_game.get(int(game_no)) if isinstance(game_no, int) else our_seat
        return expected is not None and seat == expected
    return False


def _chat_score(entries: list[dict], bot_name: str, our_seat: int | None,
                seats_by_game: dict[int, int], report_rate: float | None) -> dict:
    """Reply rate * length compliance, counting only lines authored by us."""
    if not entries:
        return {"rate": 1.0, "compliance": 1.0, "lines": 0, "source": "no chat entries"}
    identifiable = any(entry.get("team") is not None or isinstance(entry.get("seat"), int) for entry in entries)
    if not identifiable:
        rate = 1.0 if report_rate is None else report_rate
        return {"rate": rate, "compliance": 1.0, "lines": 0,
                "source": "replay format unclear; report chat_reply_rate and full length compliance"}

    ours = [entry for entry in entries
            if _authored_by_us(entry, bot_name, our_seat, seats_by_game)]
    long_lines = sum(1 for entry in ours if len(str(entry.get("text") or "")) > 200)
    compliance = 1.0 if not ours else (len(ours) - long_lines) / len(ours)

    if report_rate is not None:
        rate = report_rate
        source = "report chat_reply_rate"
    else:
        ordered = sorted(entries, key=lambda item: (int(item.get("game") or 0), int(item.get("n") or 0)))
        opponent = replied = pending = 0
        last_game = None
        for entry in ordered:
            game_no = entry.get("game")
            if game_no != last_game:
                pending = 0
                last_game = game_no
            if _authored_by_us(entry, bot_name, our_seat, seats_by_game):
                if pending > 0:
                    replied += 1
                    pending -= 1
            else:
                opponent += 1
                pending += 1
        rate = 1.0 if not opponent else replied / opponent
        source = "replay chat sequence"
    return {"rate": float(rate), "compliance": compliance, "lines": len(ours), "source": source}


# --------------------------------------------------------------------------- scoring
def _points(legal: int, total: int, illegal: int, empty: int, errors: int, forfeits: int,
            crashes: int) -> tuple[float, float]:
    """Legality points (out of 30) and the legal fraction behind them."""
    fraction = 1.0 if not total else legal / total
    if errors or illegal or forfeits or crashes:
        return 0.0, 0.0
    return WEIGHTS["legality"] * fraction, fraction


def _connect_forced_actions(observation: str) -> list[str] | None:
    """Deterministic Connect-K blunder check: immediate wins for us, else the opponent's threats.

    Returns observation-spelled actions (``drop E``) that are forced, ``[]`` when the position has no
    immediate tactic, or ``None`` when the observation cannot be parsed. Never raises.
    """
    try:
        from harness.solver_grid import parse_connect_k

        state = parse_connect_k(observation)
    except Exception:  # noqa: BLE001 - a parser bug means this decision is unjudged
        return None
    if not state:
        return None
    try:
        rows = int(state["rows"])
        cols = int(state["cols"])
        k = int(state["k"])
        board = [list(row) for row in state["board"]]
        me = str(state.get("me") or "X")
        opp = str(state.get("opp") or ("O" if me == "X" else "X"))
        open_cols = [int(c) for c in (state.get("open_cols") or [])]
    except Exception:  # noqa: BLE001
        return None

    def landing(col: int) -> int:
        for row in range(rows - 1, -1, -1):
            if board[row][col] == ".":
                return row
        return -1

    def wins(col: int, piece: str) -> bool:
        row = landing(col)
        if row < 0:
            return False
        board[row][col] = piece
        won = False
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                rr, cc = row + sign * dr, col + sign * dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == piece:
                    count += 1
                    rr, cc = rr + sign * dr, cc + sign * dc
            if count >= k:
                won = True
                break
        board[row][col] = "."
        return won

    winning = [col for col in open_cols if wins(col, me)]
    forced = winning or [col for col in open_cols if wins(col, opp)]
    if not forced:
        return []
    letter_mode = bool(state.get("letter_mode"))
    tokens = state.get("col_tokens") or {}
    actions: list[str] = []
    for col in forced:
        if letter_mode:
            token = tokens.get(col) or chr(ord("A") + col)
            actions.append(f"drop {token}")
        else:
            actions.append(f"drop {col}")
    return actions


def _think_ahead_verdict(observation: str, action: str) -> tuple[bool, bool]:
    """(judged, agree) for a think_ahead/Connect-K decision: only forced win/block positions count."""
    forced = _connect_forced_actions(observation)
    if not forced:
        return False, False
    ours = _canonical(action)
    return True, any(_canonical(item) == ours for item in forced)


def _first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text or "")
    return int(match.group(0)) if match else None


def _tactic_match(family: str, ours: str, reference: str) -> tuple[bool, bool]:
    """Compare our action with the expert reference for the tactics score.

    Returns ``(judged, agree)``. Discrete actions are exact canonical matches. Deals-family offers
    are compared numerically with a +-5 coin tolerance (and a better offer for us always agrees).
    ``bid <n>`` non-matches are unjudged: without the private hand the numbers are not comparable.
    """
    ours_c = _canonical(ours)
    ref_c = _canonical(reference)
    exact = ours_c == ref_c
    if ours_c.startswith("bid ") or ref_c.startswith("bid "):
        return (True, True) if exact else (False, False)
    if family == "deals" and ours_c.startswith("offer ") and ref_c.startswith("offer "):
        ours_n = _first_int(ours_c)
        ref_n = _first_int(ref_c)
        if ours_n is None or ref_n is None:
            return True, exact
        within_tolerance = abs(ours_n - ref_n) <= 5
        not_worse = ours_n >= ref_n - 5
        return True, bool(within_tolerance or not_worse)
    return True, exact


def _tactics_points(agree: int, judged: int) -> tuple[float, float]:
    """Tactics points (out of 35) and the agree fraction (0.5 only when nothing was judged)."""
    if judged == 0:
        return WEIGHTS["tactics"] * 0.5, 0.5
    return WEIGHTS["tactics"] * (agree / judged), agree / judged


def _outcome_points(fraction: float) -> float:
    return WEIGHTS["outcome"] * _clamp(fraction)


def _efficiency_points(p95_ms: float, usd_per_match: float | None) -> tuple[float, float, float]:
    """(points out of 10, time fraction, cost fraction); a missing cost earns the full cost half."""
    time_fraction = _clamp(1.0 - (max(0.0, p95_ms) / MOVE_BUDGET_MS))
    cost_fraction = 1.0 if usd_per_match is None else _clamp(1.0 - (max(0.0, usd_per_match) / COST_BUDGET_USD))
    return 5.0 * time_fraction + 5.0 * cost_fraction, time_fraction, cost_fraction


def _outcome_fraction(game_doc: dict, our_seat: int, bot_name: str, game) -> tuple[str, float]:
    """Win/draw/loss for one game, preferring the replay's scores, then its winner, then the rebuilt game."""
    result = game_doc.get("result") or {}
    scores = result.get("scores")
    if isinstance(scores, list) and len(scores) == 2:
        try:
            ours, theirs = float(scores[our_seat]), float(scores[1 - our_seat])
        except (TypeError, ValueError):
            ours = theirs = None
        if ours is not None and theirs is not None:
            if abs(ours - theirs) < 1e-9:
                return "draw", 0.5
            return ("win", 1.0) if ours > theirs else ("loss", 0.0)

    winner = result.get("winner_seat", result.get("winner"))
    if winner is None and game is not None:
        try:
            rebuilt = game.result() or {}
            winner = rebuilt.get("winner_seat", rebuilt.get("winner"))
        except Exception:  # noqa: BLE001
            winner = None
    if winner is None:
        return "draw", 0.5
    if isinstance(winner, (int, float)) or (isinstance(winner, str) and winner.strip().lstrip("-").isdigit()):
        seat = int(winner)
        return ("win", 1.0) if seat == our_seat else ("loss", 0.0)
    name = str(winner)
    if name == bot_name:
        return "win", 1.0
    return ("draw", 0.5) if name in ("", "draw", "none", "None") else ("loss", 0.0)


def _check_action(game, seat: int, action: object) -> str | None:
    """Arena-canonical action, or None when the arena would reject it."""
    text = "" if action is None else str(action).strip()
    if not text:
        return None
    try:
        legal = game.legal_actions(seat)
    except Exception:  # noqa: BLE001
        legal = None
    if legal:
        from harness.actions import normalize

        return normalize(text, legal)
    try:
        return game.normalize(seat, text)
    except Exception:  # noqa: BLE001
        return None


def _seat_from_key(key: object, seats: list[str]) -> int | None:
    text = str(key)
    if text.isdigit():
        return int(text)
    if text in seats:
        return seats.index(text)
    return None


def _observe(game, seat: int) -> str:
    try:
        return str(game.observation(seat))
    except Exception:  # noqa: BLE001
        return ""


def analyse_game(doc: dict, game_doc: dict, game_index: int, experts: list,
                 seats_by_game: dict[int, int], report_rate: float | None) -> tuple[dict, dict]:
    """Rebuild and verify one game. Always returns a row; never raises."""
    bot_name = str(doc.get("a") or "adn")
    seats = [str(item) for item in (game_doc.get("seats") or [])]
    stats: dict = {"solver_errors": 0, "solver_illegal": 0, "opponent_illegal": 0,
                   "errors": 0, "empty": 0, "error_text": "", "families": {}}
    game_id = str(game_doc.get("game_id") or "")
    variant_label = str((game_doc.get("variant") or {}).get("label") or "")
    row = {
        "game": game_id, "variant": variant_label, "seat": None, "decisions": 0,
        "legal": 0, "illegal": 0, "empty": 0, "legal_pct": 0.0,
        "judged": 0, "agree": 0, "outcome": "error", "p95_ms": 0.0,
        "chat": 100.0, "score": 0.0, "error": "", "ms": [],
    }
    if bot_name not in seats:
        stats["errors"] = 1
        stats["error_text"] = f"our bot {bot_name!r} is not in seats {seats}"
        row.update({"legal_pct": 0.0, "error": stats["error_text"]})
        return row, stats
    our_seat = seats.index(bot_name)
    row["seat"] = our_seat
    seats_by_game[int(game_doc.get("index") or game_index)] = our_seat

    category = str(game_doc.get("category") or "")
    cls = _game_classes().get(game_id)
    if cls is None:
        stats["errors"] = 1
        stats["error_text"] = f"unknown arena game_id {game_id!r}"
        row["error"] = stats["error_text"]
        return row, stats

    try:
        game = cls(dict(game_doc.get("variant") or {}), int(game_doc.get("seed") or 0))
    except Exception as exc:  # noqa: BLE001 - a bad variant/seed is a replay error
        stats["errors"] = 1
        stats["error_text"] = f"cannot build {game_id}: {type(exc).__name__}: {exc}"[:160]
        row["error"] = stats["error_text"]
        return row, stats

    spec = _spec_for(game_id, category)
    row["family"] = spec.family
    game_experts = _specialists(experts, game_id, category, "")
    try:
        rules = str(game.rules())
    except Exception:  # noqa: BLE001
        rules = ""

    ms_values: list[float] = row["ms"]
    for step_index, step in enumerate(game_doc.get("timeline") or []):
        if not isinstance(step, dict):
            continue
        actions = step.get("actions") or {}
        if not isinstance(actions, dict):
            continue
        by_seat: dict[int, dict] = {}
        for key, entry in actions.items():
            seat = _seat_from_key(key, seats)
            if seat is None or not isinstance(entry, dict):
                continue
            by_seat[seat] = entry

        entry = by_seat.get(our_seat)
        applied: dict[int, str] = {}
        if entry is not None:
            row["decisions"] += 1
            raw = entry.get("action")
            observation = str(entry.get("observation") or "") or _observe(game, our_seat)
            if not str(raw or "").strip():
                row["empty"] += 1
                stats["empty"] += 1
            else:
                normalized = _check_action(game, our_seat, raw)
                if normalized:
                    row["legal"] += 1
                    applied[our_seat] = normalized
                else:
                    row["illegal"] += 1
                if isinstance(entry.get("ms"), (int, float)):
                    ms_values.append(float(entry["ms"]))
                judged = agreed = False
                if spec.family == "think_ahead":
                    # Anytime search moves depend on the time budget; judge the policy by a
                    # deterministic immediate win/block oracle instead (blunder check).
                    judged, agreed = _think_ahead_verdict(observation, normalized or str(raw))
                else:
                    reference, _solver = _reference(game_experts, spec, rules, observation,
                                                    _safe_legal(game, our_seat), our_seat,
                                                    step_index, stats)
                    if reference is not None:
                        judged, agreed = _tactic_match(spec.family, normalized or str(raw), reference)
                if judged:
                    bucket = stats["families"].setdefault(spec.family, {"judged": 0, "agree": 0})
                    bucket["judged"] += 1
                    row["judged"] += 1
                    if agreed:
                        bucket["agree"] += 1
                        row["agree"] += 1

        for seat, other in by_seat.items():
            if seat == our_seat or not isinstance(other, dict):
                continue
            normalized = _check_action(game, seat, other.get("action"))
            if normalized:
                applied[seat] = normalized
            elif str(other.get("action") or "").strip():
                # The arena warns on an opponent's illegal move and does NOT advance the state.
                stats["opponent_illegal"] += 1

        if not applied:
            continue
        try:
            game.apply({seat: applied[seat] for seat in sorted(applied)})
        except Exception as exc:  # noqa: BLE001 - divergence ends this game, not the stage
            stats["errors"] = 1
            stats["error_text"] = f"replay walk diverged at step {step_index}: {type(exc).__name__}: {exc}"[:180]
            break

    outcome, outcome_fraction = _outcome_fraction(game_doc, our_seat, bot_name, game)
    reason = str((game_doc.get("result") or {}).get("reason") or "").lower()
    if game_doc.get("forfeit") and bot_name.lower() in reason:
        outcome, outcome_fraction = "loss", 0.0
        stats["errors"] = 1
        stats["error_text"] = stats["error_text"] or "our forfeit"

    row.update({
        "outcome": outcome,
        "p95_ms": round(_p95(ms_values), 1),
        "error": stats["error_text"],
    })
    row["legal_pct"] = 0.0 if stats["errors"] else (100.0 if not row["decisions"]
                                                    else 100.0 * row["legal"] / row["decisions"])
    chat_entries = _chat_entries(doc, game_doc)
    chat = _chat_score(chat_entries, bot_name, our_seat, seats_by_game, report_rate)
    row["chat"] = round(100.0 * chat["rate"] * chat["compliance"], 1)

    row["score"] = round(_row_score(row, report_rate), 1)
    return row, stats


def _row_has_error(row: dict) -> bool:
    return bool(str(row.get("error") or ""))


def _safe_legal(game, seat: int):
    try:
        return game.legal_actions(seat)
    except Exception:  # noqa: BLE001
        return None


def _row_score(row: dict, report_rate: float | None) -> float:
    """The same weighted formula, per game, so the table's last column is self-explanatory."""
    legality, _ = _points(row["legal"], row["decisions"], row["illegal"], row["empty"],
                          1 if _row_has_error(row) else 0, 0, 0)
    tactics, _ = _tactics_points(row["agree"], row["judged"])
    outcome_frac = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(row["outcome"], 0.0)
    outcome = _outcome_points(outcome_frac)
    efficiency, _, _ = _efficiency_points(row["p95_ms"], None)
    chat = WEIGHTS["chat"] * (row["chat"] / 100.0)
    return legality + tactics + outcome + efficiency + chat


# --------------------------------------------------------------------------- analysis
def analyse(replays_dir: str | None = None, limit: int | None = None, report: dict | None = None) -> dict | None:
    """Verify every replay in the directory and return the full breakdown, or None when none exist."""
    directory, files = find_replays(replays_dir, limit)
    if not directory or not files:
        return None
    if report is None:
        report = load_report(directory)
    totals = report.get("totals") or {}
    report_rate = totals.get("chat_reply_rate")
    report_rate = float(report_rate) if isinstance(report_rate, (int, float)) else None
    usd_per_match = totals.get("usd_per_match")
    usd_per_match = float(usd_per_match) if isinstance(usd_per_match, (int, float)) else None

    experts, expert_notes = _build_experts()
    rows: list[dict] = []
    seats_by_game: dict[int, int] = {}
    all_chat: list[dict] = []
    seen_chat: set = set()
    judged_by_family: dict[str, dict] = {}
    all_ms: list[float] = []
    totals_counters = {"decisions": 0, "legal": 0, "illegal": 0, "empty": 0, "judged": 0,
                       "agree": 0, "errors": 0, "forfeits": 0, "crashes": 0, "wins": 0,
                       "draws": 0, "losses": 0, "solver_errors": 0, "solver_illegal": 0,
                       "opponent_illegal": 0, "unreadable": 0}
    notes: list[str] = list(expert_notes)
    inputs: list[dict] = []

    for path in files:
        name = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
            if not isinstance(doc, dict):
                raise ValueError("replay is not an object")
        except (OSError, ValueError) as exc:
            totals_counters["errors"] += 1
            totals_counters["unreadable"] += 1
            inputs.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        bot_name = str(doc.get("a") or "adn")
        games = doc.get("games") or []
        inputs.append({"file": name, "bot": bot_name, "opponent": str(doc.get("b") or ""),
                       "games": len(games), "winner": doc.get("winner")})
        _add_chat(all_chat, seen_chat, doc.get("chat"))
        if not games:
            rows.append({"game": "?", "variant": "", "decisions": 0, "legal_pct": 0.0, "judged": 0,
                         "agree": 0, "outcome": "error", "p95_ms": 0.0, "chat": 100.0, "score": 0.0,
                         "error": "replay has no games"})
            totals_counters["errors"] += 1
            continue
        for game_index, game_doc in enumerate(games, 1):
            if not isinstance(game_doc, dict):
                continue
            row, stats = analyse_game(doc, game_doc, game_index, experts, seats_by_game, report_rate)
            rows.append(row)
            all_ms.extend(row.get("ms") or [])
            for key in ("decisions", "legal", "illegal", "empty", "judged", "agree"):
                totals_counters[key] += int(row.get(key) or 0)
            for key in ("errors", "solver_errors", "solver_illegal", "opponent_illegal"):
                totals_counters[key] += int(stats.get(key) or 0)
            for family, counts in (stats.get("families") or {}).items():
                bucket = judged_by_family.setdefault(family, {"judged": 0, "agree": 0})
                bucket["judged"] += int(counts.get("judged") or 0)
                bucket["agree"] += int(counts.get("agree") or 0)
            reason = str((game_doc.get("result") or {}).get("reason") or "").lower()
            if game_doc.get("forfeit") and bot_name.lower() in reason:
                totals_counters["forfeits"] += 1
            if row["outcome"] == "win":
                totals_counters["wins"] += 1
            elif row["outcome"] == "draw":
                totals_counters["draws"] += 1
            elif row["outcome"] == "loss":
                totals_counters["losses"] += 1
            _add_chat(all_chat, seen_chat, game_doc.get("chat"))

    rounds = len(rows)
    if not rounds:
        return None

    totals_counters["crashes"] += int(totals.get("crashes") or 0)
    totals_counters["forfeits"] += int(totals.get("forfeits") or 0)

    legality_points, legality_fraction = _points(
        totals_counters["legal"], totals_counters["decisions"], totals_counters["illegal"],
        totals_counters["empty"], totals_counters["errors"], totals_counters["forfeits"],
        totals_counters["crashes"])
    tactics_points, agree_fraction = _tactics_points(totals_counters["agree"], totals_counters["judged"])
    if isinstance(totals.get("score_pct"), (int, float)):
        outcome_fraction = _clamp(float(totals["score_pct"]) / 100.0)
        outcome_source = "report score_pct"
    else:
        outcome_fraction = (totals_counters["wins"] + 0.5 * totals_counters["draws"]) / max(1, rounds)
        outcome_source = "replayed wins/draws"
    outcome_points = _outcome_points(outcome_fraction)
    replay_p95 = _p95(all_ms)
    report_p95 = totals.get("move_ms_p95")
    p95_ms = replay_p95 if all_ms else (float(report_p95) if isinstance(report_p95, (int, float)) else 0.0)
    efficiency_points, time_fraction, cost_fraction = _efficiency_points(p95_ms, usd_per_match)
    chat = _chat_score(all_chat, str(inputs[0].get("bot") if inputs else "adn"), None,
                       seats_by_game, report_rate)
    chat_fraction = _clamp(chat["rate"] * chat["compliance"])
    chat_points = WEIGHTS["chat"] * chat_fraction

    overall = (legality_points + tactics_points + outcome_points + efficiency_points + chat_points)
    judged = totals_counters["judged"]
    if judged >= 10 and rounds >= 3:
        confidence = "high"
    elif judged >= TACTICS_MIN_JUDGED:
        confidence = "medium"
    else:
        confidence = "low"

    tactics_note = ("tactics = agreement/quality vs our own expert stack, not proven optimality; "
                    "think_ahead judged by deterministic immediate win/block (blunder check)")
    if judged == 0:
        tactics_note += f"; {LOW_CONFIDENCE} (0 judged decisions, neutral 0.5 used)"
    elif judged < TACTICS_MIN_JUDGED:
        tactics_note += f"; {LOW_CONFIDENCE} ({judged} judged decision(s))"

    notes.append(tactics_note)
    if totals_counters["errors"]:
        notes.append(f"{totals_counters['errors']} replay error(s); legality forced to 0")
    if totals_counters["illegal"]:
        notes.append(f"{totals_counters['illegal']} illegal decision(s) of ours; legality forced to 0")
    if totals_counters["forfeits"]:
        notes.append(f"{totals_counters['forfeits']} forfeit(s) present; legality forced to 0")
    if totals_counters["crashes"]:
        notes.append(f"{totals_counters['crashes']} crash(es) reported; legality forced to 0")
    if totals_counters["solver_errors"]:
        notes.append(f"{totals_counters['solver_errors']} solver error(s) skipped")
    if totals_counters["solver_illegal"]:
        notes.append(f"{totals_counters['solver_illegal']} illegal solver proposal(s) skipped")
    if totals_counters["opponent_illegal"]:
        notes.append(f"{totals_counters['opponent_illegal']} opponent illegal move(s) skipped, not ours")
    notes.append(f"chat rate source: {chat['source']}")

    components = {
        "legality": {"points": round(legality_points, 2), "weight": WEIGHTS["legality"],
                     "pct": round(100.0 * legality_points / WEIGHTS["legality"], 1),
                     "fraction": round(legality_fraction, 4)},
        "tactics": {"points": round(tactics_points, 2), "weight": WEIGHTS["tactics"],
                    "pct": round(100.0 * tactics_points / WEIGHTS["tactics"], 1),
                    "fraction": round(agree_fraction, 4), "judged": judged,
                    "agree": totals_counters["agree"], "judged_by_family": judged_by_family,
                    "note": tactics_note},
        "outcome": {"points": round(outcome_points, 2), "weight": WEIGHTS["outcome"],
                    "pct": round(100.0 * outcome_points / WEIGHTS["outcome"], 1),
                    "fraction": round(outcome_fraction, 4), "source": outcome_source},
        "efficiency": {"points": round(efficiency_points, 2), "weight": WEIGHTS["efficiency"],
                       "pct": round(100.0 * efficiency_points / WEIGHTS["efficiency"], 1),
                       "p95_ms": round(p95_ms, 1), "usd_per_match": usd_per_match,
                       "time_fraction": round(time_fraction, 4),
                       "cost_fraction": round(cost_fraction, 4)},
        "chat": {"points": round(chat_points, 2), "weight": WEIGHTS["chat"],
                 "pct": round(100.0 * chat_fraction, 1), "rate": round(chat["rate"], 4),
                 "compliance": round(chat["compliance"], 4), "lines": chat["lines"],
                 "source": chat["source"]},
    }

    return {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {"replays_dir": directory, "files": inputs, "report_totals": totals,
                   "rounds": rounds, "decisions": totals_counters["decisions"],
                   "judged": judged, "agree": totals_counters["agree"],
                   "judged_by_family": judged_by_family},
        "counters": totals_counters,
        "components": components,
        "weights": dict(WEIGHTS),
        "rounds": rows,
        "overall": round(overall, 1),
        "confidence": confidence,
        "notes": notes,
    }


# --------------------------------------------------------------------------- rendering
def _fmt_row(row: dict) -> str:
    variant = str(row.get("variant") or "")[:28]
    agree = f"{row.get('agree', 0)}/{row.get('judged', 0)}"
    outcome = str(row.get("outcome") or "")[:7]
    return (f" {str(row.get('game') or '?'):<20} {variant:<28} {row.get('decisions', 0):>9} "
            f"{row.get('legal_pct', 0.0):>6.1f} {agree:>9} {outcome:>7} "
            f"{row.get('p95_ms', 0.0):>6.0f} {row.get('chat', 0.0):>5.0f} {row.get('score', 0.0):>5.1f}")


def render(result: dict) -> str:
    lines: list[str] = []
    directory = result["inputs"]["replays_dir"]
    lines.append("=" * 108)
    lines.append(f" INTELLIGENCE REPORT - {directory}")
    lines.append("=" * 108)
    lines.append(f" {'game':<20} {'variant':<28} {'decisions':>9} {'legal%':>6} {'ref-agree':>9} "
                 f"{'outcome':>7} {'p95ms':>6} {'chat':>5} {'score':>5}")
    lines.append("-" * 108)
    for row in result["rounds"]:
        lines.append(_fmt_row(row))
    lines.append("-" * 108)
    inputs = result["inputs"]
    lines.append(f" inputs: {len(result['rounds'])} rounds from {len(inputs['files'])} replay file(s); "
                 f"{inputs['decisions']} decisions, {inputs['judged']} judged")
    for note in result["notes"]:
        lines.append(f" note: {note}")
    components = result["components"]
    line = (f" INTELLIGENCE {result['overall']:.1f}/100  "
            f"legality {components['legality']['pct']:.1f}  "
            f"tactics {components['tactics']['pct']:.1f}  "
            f"outcome {components['outcome']['pct']:.1f}  "
            f"efficiency {components['efficiency']['pct']:.1f}  "
            f"chat {components['chat']['pct']:.1f}  "
            f"({len(result['rounds'])} rounds, {inputs['decisions']} decisions, "
            f"{inputs['judged']} judged; confidence: {result['confidence']})")
    lines.append(line)
    lines.append("=" * 108)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the bench rounds and print a measured intelligence score.")
    parser.add_argument("--run", default="", help="explicit replays directory (default: newest bench run)")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of replays (default: all)")
    parser.add_argument("--json", default="", help="write the full breakdown here")
    parser.add_argument("--no-color", action="store_true", help="disable color (no color is emitted anyway)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    with contextlib.redirect_stdout(sys.stderr):
        result = analyse(args.run or None, limit)
    if result is None:
        print("INTELLIGENCE SKIPPED (no bench replays; run scripts/bench.py or pipeline.py --bench first)")
        return 0

    print(render(result))
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=1)
        except OSError as exc:
            print(f"intelligence: cannot write {args.json}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
