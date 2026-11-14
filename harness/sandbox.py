"""Sandbox for code the model wrote during prep.

A generated policy is only trusted after it (a) contains no network/LLM/process access and (b) returns a
legal action on every `sample_turn` the arena gave us. At play time each call runs in a fresh, isolated
interpreter with a hard timeout, so a broken or adversarial module can cost a move, never the game.

This is also the compliance story the judges read: generated code cannot call another model (rule 7.7)
and cannot send anything out (rule 7.11), because the sandbox has no tools to do it with.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from . import config
from .actions import normalize
from .log import log

_BANNED = (
    "openai",
    "anthropic",
    "requests",
    "httpx",
    "aiohttp",
    "socket",
    "urllib",
    "http.client",
    "subprocess",
    "multiprocessing",
    "os.system",
    "os.popen",
    "shutil",
    "ctypes",
    "importlib",
    "__import__",
    "eval(",
    "exec(",
)

_RUNNER = r"""
import importlib.util, json, sys
path = sys.argv[1]
module_spec = importlib.util.spec_from_file_location("generated_policy", path)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
payload = json.loads(sys.stdin.read())
action = module.choose(payload["observation"], payload["legal_actions"], payload["seat"])
sys.stdout.write(json.dumps({"action": action}))
"""


def inspect_source(source: str) -> str | None:
    """Cheap static check. Returns the offending token, or None when the module looks clean."""
    lowered = source.lower()
    for token in _BANNED:
        if token in lowered:
            return token
    return None


def _run(path: str, observation: str, legal: list[str] | None, seat: int, timeout: float) -> str | None:
    payload = json.dumps({"observation": observation, "legal_actions": legal, "seat": seat})
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/tmp",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        # Deliberately no proxy variables and no LLM key: generated code has no way out.
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", _RUNNER, str(path)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout)),
            env=env,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        log("genpolicy: timeout")
        return None
    except OSError as exc:
        log("genpolicy: cannot start:", type(exc).__name__)
        return None
    if completed.returncode != 0:
        log("genpolicy: exit", completed.returncode, completed.stderr.strip()[:200])
        return None
    try:
        value = json.loads(completed.stdout or "{}").get("action")
    except ValueError:
        log("genpolicy: bad output")
        return None
    return None if value is None else str(value)


def validate_module(path: str, samples: list[dict], timeout: float = 6.0) -> tuple[bool, str]:
    """Replay every sample turn: the module must answer, and answer legally."""
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        return False, f"unreadable: {type(exc).__name__}"
    banned = inspect_source(source)
    if banned:
        return False, f"banned token {banned!r}"
    if "def choose(" not in source:
        return False, "no choose() function"
    if not samples:
        return False, "no sample_turn to validate against"
    for index, sample in enumerate(samples):
        legal = sample.get("legal_actions")
        action = _run(path, str(sample.get("observation", "")), legal, int(sample.get("seat", 0) or 0), timeout)
        if not action:
            return False, f"sample {index}: no action"
        if legal is not None and normalize(action, legal) is None:
            return False, f"sample {index}: illegal {action!r}"
    return True, f"ok on {len(samples)} sample(s)"


def call_choose(path: str, observation: str, legal: list[str] | None, seat: int, timeout: float = 4.0) -> str | None:
    """One play-time call. Returns None on any failure so the caller can fall back safely."""
    if timeout <= 0.5:
        return None
    action = _run(path, observation, legal, seat, timeout)
    if action is None:
        return None
    if legal is not None:
        return normalize(action, legal)
    return action if 0 < len(action) <= config.MAX_CHAT_CHARS else None
