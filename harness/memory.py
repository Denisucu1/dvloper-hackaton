"""Per-game /memory handling.

`/memory` is shared by every container of the team at the same time (rule 5.14), so:
- one folder per game_id, sanitized for the filesystem;
- artifact names derived from a hash of the rules text (a later prep of a known game is instant);
- writes are atomic (tmp file + os.replace) so a reader never sees half a file;
- append-only logs get a unique name where two containers could collide.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from . import config
from .log import log


def safe_name(value: Any) -> str:
    """Filesystem-safe, stable name for a game id (rule 5.6: use game_id to file artifacts)."""
    name = re.sub(r"[^a-z0-9_-]+", "_", str(value).strip().lower()).strip("_")
    return name or "unknown"


def game_dir(game_id: Any, root: str | os.PathLike[str] | None = None) -> Path:
    path = Path(root or config.MEMORY_DIR) / safe_name(game_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fingerprint(*texts: str) -> str:
    """Short stable hash of the rules text, used to key every artifact for one game."""
    h = hashlib.sha256()
    for text in texts:
        h.update(str(text).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def rules_fingerprint(rules: str) -> str:
    return fingerprint(rules or "")


def variants_rules(variants: list[dict] | None) -> str:
    """The canonical text every artifact is keyed on: labels + rules, in order."""
    parts = []
    for variant in variants or []:
        parts.append(str(variant.get("label", "")))
        parts.append(str(variant.get("rules", "")))
    return "\n".join(parts)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(25):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 24:
                    raise
                time.sleep(0.002 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=1))


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    for attempt in range(5):
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except PermissionError:
            if attempt < 4:
                time.sleep(0.001 * (attempt + 1))
                continue
            return default
        except (OSError, ValueError) as exc:
            if not isinstance(exc, FileNotFoundError):
                log("memory read failed:", path, type(exc).__name__)
            return default
    return default


def read_text(path: str | os.PathLike[str], default: str = "") -> str:
    for attempt in range(5):
        try:
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        except PermissionError:
            if attempt < 4:
                time.sleep(0.001 * (attempt + 1))
                continue
            return default
        except OSError:
            return default
    return default


def append_jsonl(path: str | os.PathLike[str], payload: Any) -> None:
    """Append one record. Used for results and notes; never for artifacts another container must parse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    except OSError as exc:
        log("memory append failed:", path, type(exc).__name__)


def artifact_path(game_id: Any, rules_hash: str, filename: str) -> Path:
    return game_dir(game_id) / f"{rules_hash}-{filename}"


def record_result(game_id: Any, payload: dict) -> None:
    """One line per finished game; keyed by day so it never grows unbounded within a tournament."""
    path = game_dir(game_id) / f"{config.ARTIFACT_RESULTS.split('.')[0]}-{time.strftime('%m%d')}.jsonl"
    append_jsonl(path, payload)
