"""Concurrency-safe /memory storage for AI Harness Arena (Team adn).

Owner: adn-1 (Architecture & Infra).
All team containers share `/memory` simultaneously, so every game gets its own
rules-fingerprinted folder and files are written atomically via temp + os.replace.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

MEMORY_DIR = os.environ.get("MEMORY_DIR", "/memory")


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", str(value or "unknown").lower())


def rules_fingerprint(variants: list[dict[str, Any]]) -> str:
    """Stable short hash over all variant labels and rules texts."""
    payload = "\n".join(
        f"{v.get('label', '')}\n{v.get('rules', '')}"
        for v in sorted(variants, key=lambda x: str(x.get("label", "")))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def game_memory_dir(game_id: str, fp: str | None = None) -> str:
    slug = safe_slug(game_id)
    folder = f"{slug}_{fp}" if fp else slug
    return os.path.join(MEMORY_DIR, folder)


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically so concurrent containers never observe partial writes."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def save_specialist(game_id: str, fp: str, specialist: dict[str, Any]) -> None:
    path = os.path.join(game_memory_dir(game_id, fp), "specialist.json")
    atomic_write_json(path, specialist)
    # Also save latest pointer under game_id for cold containers matching by game_id.
    latest_path = os.path.join(game_memory_dir(game_id), "latest_specialist.json")
    atomic_write_json(latest_path, specialist)


def load_specialist(game_id: str, fp: str | None = None) -> dict[str, Any] | None:
    if fp:
        found = read_json(os.path.join(game_memory_dir(game_id, fp), "specialist.json"))
        if isinstance(found, dict):
            return found
    found = read_json(os.path.join(game_memory_dir(game_id), "latest_specialist.json"))
    return found if isinstance(found, dict) else None


def save_samples(game_id: str, fp: str, variants: list[dict[str, Any]]) -> None:
    samples = {
        v.get("label", "default"): v["sample_turn"]
        for v in variants
        if v.get("sample_turn")
    }
    if not samples:
        return
    path = os.path.join(game_memory_dir(game_id, fp), "samples.json")
    atomic_write_json(path, {"game_id": game_id, "fingerprint": fp, "sample_turns": samples})


def append_match_result(game_id: str, record: dict[str, Any]) -> None:
    folder = game_memory_dir(game_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "results.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_opponent_profile(opponent_id: str) -> dict[str, Any]:
    path = os.path.join(MEMORY_DIR, "_opponents", f"{safe_slug(opponent_id)}.json")
    data = read_json(path)
    return data if isinstance(data, dict) else {
        "opponent": opponent_id,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "injection_attempts": 0,
        "notes": [],
    }


def save_opponent_profile(opponent_id: str, profile: dict[str, Any]) -> None:
    path = os.path.join(MEMORY_DIR, "_opponents", f"{safe_slug(opponent_id)}.json")
    atomic_write_json(path, profile)
