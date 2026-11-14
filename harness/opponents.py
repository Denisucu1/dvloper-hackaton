"""Opponent profiles in /memory (rule 5.15: opponent modeling is allowed and encouraged).

What we keep is how a team plays the chat: how often it tries to inject instructions, fish for private
state or make claims, and how our games against it ended. One small file per match, so the containers of
one match (one per game, playing in sequence) never race with the containers of another match.
"""
from __future__ import annotations

from pathlib import Path

from . import config, memory
from .log import log

_DIR = "_opponents"
_COUNTERS = ("opponent_lines", "injections", "probes", "baits", "claims", "offers")


def _folder(opponent: object) -> Path:
    return Path(config.MEMORY_DIR) / _DIR / memory.safe_name(opponent)


def record(opponent: object, match_id: object, game_index: object, stats: dict, result: str | None = None) -> None:
    """Called at `end`. The chat is match-wide, so the latest stats replace the earlier ones for the match."""
    if not opponent:
        return
    try:
        path = _folder(opponent) / f"{memory.safe_name(match_id or 'match')}.json"
        data = memory.read_json(path, default=None) or {"opponent": str(opponent), "match": str(match_id), "results": {}}
        data["chat"] = {key: int(stats.get(key) or 0) for key in _COUNTERS}
        if result:
            data.setdefault("results", {})[str(game_index)] = str(result)
        memory.atomic_write_json(path, data)
    except Exception as exc:  # noqa: BLE001 - a profile is a bonus, never a reason to fail
        log("opponent profile write failed:", type(exc).__name__)


def load(opponent: object, limit: int = 40) -> dict:
    """Everything we remember about this team, summed over the matches on file."""
    profile = {"games": 0, "wins": 0, "losses": 0, "draws": 0, **{key: 0 for key in _COUNTERS}}
    if not opponent:
        return profile
    try:
        folder = _folder(opponent)
        if not folder.is_dir():
            return profile
        for path in sorted(folder.glob("*.json"))[-limit:]:
            data = memory.read_json(path, default=None) or {}
            for key in _COUNTERS:
                profile[key] += int((data.get("chat") or {}).get(key) or 0)
            for result in (data.get("results") or {}).values():
                profile["games"] += 1
                bucket = {"win": "wins", "loss": "losses", "draw": "draws"}.get(str(result))
                if bucket:
                    profile[bucket] += 1
    except Exception as exc:  # noqa: BLE001
        log("opponent profile read failed:", type(exc).__name__)
    return profile
