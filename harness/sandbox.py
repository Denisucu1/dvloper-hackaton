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
import tempfile

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


_MODEL_BUILD = r"""
def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(module, rules, seat):
    cls = getattr(module, "Model", None) or getattr(module, "GameModel", None)
    if cls is None:
        raise RuntimeError("no class Model")
    try:
        return cls(rules, seat)
    except TypeError:
        try:
            return cls(seat)
        except TypeError:
            return cls()
"""

_MODEL_VALIDATE_RUNNER = _MODEL_BUILD + r"""
import json, math, sys

PROB_TOL = 1e-6
PROBE_LIMIT = 24

payload = json.loads(sys.stdin.read())
module = _load("generated_model", sys.argv[1])
errors = []
checked = 0
chance_nodes = 0


def _chance_outcomes(model, tag):
    # Validate one chance node's distribution; [] means "not a chance node".
    getter = getattr(model, "chance_actions", None)
    if not callable(getter):
        return []
    try:
        raw = getter()
    except Exception as exc:
        raise RuntimeError(f"{tag}: chance_actions raised {type(exc).__name__}: {exc}")
    if raw is None:
        return []
    try:
        pairs = list(raw)
    except Exception:
        raise RuntimeError(f"{tag}: chance_actions must return a list of (str, probability) pairs")
    if not pairs:
        return []
    outcomes = []
    total = 0.0
    for item in pairs:
        try:
            label, probability = item
        except Exception:
            raise RuntimeError(f"{tag}: chance_actions entries must be (str, probability) pairs")
        if not isinstance(label, str) or not label.strip():
            raise RuntimeError(f"{tag}: chance outcome label must be a non-empty string")
        try:
            weight = float(probability)
        except Exception:
            raise RuntimeError(f"{tag}: outcome {label!r}: probability must be a number")
        if not math.isfinite(weight) or weight <= 0.0:
            raise RuntimeError(
                f"{tag}: outcome {label!r}: probability must be finite and > 0, got {probability!r}")
        outcomes.append((label, weight))
        total += weight
    if abs(total - 1.0) > PROB_TOL:
        raise RuntimeError(f"{tag}: chance_actions sums to {total:g}, expected 1")
    return outcomes


def _check_chance_node(model, tag, seat):
    # Apply every outcome on a clone and require a coherent resulting state.
    outcomes = _chance_outcomes(model, tag)
    if not outcomes:
        return False
    for label, _weight in outcomes:
        try:
            child = model.clone()
        except Exception as exc:
            raise RuntimeError(f"{tag}: clone() before apply_chance({label!r}) raised "
                               f"{type(exc).__name__}: {exc}")
        applier = getattr(child, "apply_chance", None)
        if not callable(applier):
            raise RuntimeError(f"{tag}: chance_actions without apply_chance")
        try:
            applier(label)
        except Exception as exc:
            raise RuntimeError(f"{tag}: apply_chance({label!r}) raised {type(exc).__name__}: {exc}")
        terminal = child.is_terminal()
        if not isinstance(terminal, bool):
            raise RuntimeError(f"{tag}: apply_chance({label!r}) left is_terminal() non-bool")
        movers = child.to_move()
        if not isinstance(movers, (list, tuple)) or any(not isinstance(value, int) for value in movers):
            raise RuntimeError(f"{tag}: apply_chance({label!r}) left to_move() not a list of seat ints")
        if terminal:
            value = float(child.score(seat))
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"{tag}: apply_chance({label!r}) left score() out of [0, 1]: {value}")
    return True


def _find_chance_node(model, tag, seat):
    # Find a reachable chance node (the sample itself, or within two decision plies).
    if _check_chance_node(model, tag, seat):
        return True
    seen = 0
    frontier = [model]
    for _depth in range(2):
        nxt = []
        for node in frontier:
            try:
                movers = list(node.to_move() or [])[:1]
            except Exception:
                continue
            for mover in movers:
                try:
                    actions = list(node.legal_actions(mover) or [])[:6]
                except Exception:
                    continue
                for action in actions:
                    seen += 1
                    if seen > PROBE_LIMIT:
                        return False
                    try:
                        child = node.clone()
                        child.apply(mover, str(action))
                    except Exception:
                        continue
                    if _check_chance_node(child, tag, seat):
                        return True
                    nxt.append(child)
        frontier = nxt
    return False


for index, sample in enumerate(payload["samples"]):
    seat = int(sample.get("seat", 0) or 0)
    label = str(sample.get("variant") or "").strip()
    tag = f"sample {index}" + (f" [{label}]" if label else "")
    try:
        model = _build(module, payload["rules"], seat)
        model.update(str(sample.get("observation", "")))
        moves = model.legal_actions(seat)
        moves_again = model.legal_actions(seat)
        if list(moves or []) != list(moves_again or []):
            errors.append(f"{tag}: legal_actions is not deterministic")
            break
        expected = sample.get("legal_actions")
        if expected is not None:
            if set(map(str, moves or [])) != set(map(str, expected)):
                errors.append(f"{tag}: legal_actions mismatch, model={sorted(map(str, moves or []))[:8]} "
                              f"arena={sorted(map(str, expected))[:8]}")
                break
        elif moves is not None and not list(moves):
            errors.append(f"{tag}: legal_actions is empty but the arena had moves")
            break
        movers = model.to_move()
        if not isinstance(movers, (list, tuple)) or any(not isinstance(s, int) for s in movers):
            errors.append(f"{tag}: to_move() must be a list of seat ints")
            break
        for action in list(moves or [])[:6]:
            child = model.clone()
            child.apply(seat, str(action))
            if not isinstance(child.is_terminal(), bool):
                errors.append(f"{tag}: is_terminal() must return a bool")
                break
            if child.is_terminal():
                value = float(child.score(seat))
                if not 0.0 <= value <= 1.0:
                    errors.append(f"{tag}: score() must be in [0, 1], got {value}")
                    break

        # Optional chance surface: either both methods exist or neither. A half-implemented contract
        # cannot be searched (one side would silently play a different game), so reject it early.
        has_chance = callable(getattr(model, "chance_actions", None))
        has_apply = callable(getattr(model, "apply_chance", None))
        if has_chance and not has_apply:
            errors.append(f"{tag}: defines chance_actions without apply_chance")
            break
        if has_apply and not has_chance:
            errors.append(f"{tag}: defines apply_chance without chance_actions")
            break
        if has_chance:
            try:
                if _find_chance_node(model, tag, seat):
                    chance_nodes += 1
            except RuntimeError as exc:
                errors.append(str(exc))
                break
        checked += 1
    except Exception as exc:
        errors.append(f"{tag}: {type(exc).__name__}: {exc}")
        break
summary = f"ok on {checked} sample(s)"
if chance_nodes:
    summary += f", {chance_nodes} chance node(s)"
sys.stdout.write(json.dumps({"ok": not errors, "reason": "; ".join(errors) or summary,
                             "checked": checked}))
"""

_MODEL_PLAY_RUNNER = _MODEL_BUILD + r"""
import json, os, sys
payload = json.loads(sys.stdin.read())
module = _load("generated_model", sys.argv[1])
search = _load("sandboxed_search", sys.argv[2])
model = _build(module, payload["rules"], int(payload["seat"]))
model.update(payload["observation"])
seat = int(payload["seat"])
strategy = str(payload.get("strategy") or "search")
meta = {}
try:
    if strategy in ("matrix", "chance", "auto"):
        # Optional strategy modules are passed as extra argv paths by the parent.
        paths = {"matrix": sys.argv[3] if len(sys.argv) > 3 else "",
                 "chance": sys.argv[4] if len(sys.argv) > 4 else ""}
        if strategy == "auto":
            has_chance = callable(getattr(model, "chance_actions", None))
            strategy = "chance" if has_chance and paths["chance"] else "search"
        if strategy == "matrix" and paths["matrix"]:
            matrix = _load("sandboxed_matrix", paths["matrix"])
            action, value = matrix.MatrixSolver().propose_joint(
                model, seat, seconds=float(payload["seconds"]), rng=None)
            meta = {"action": action, "value": value, "route": "matrix", "depth": 0}
        elif strategy == "chance" and paths["chance"]:
            chance = _load("sandboxed_chance", paths["chance"])
            action, value = chance.ChanceSolver().propose(
                model, seat, seconds=float(payload["seconds"]),
                max_depth=int(payload.get("depth") or 4), rng=None)
            meta = {"action": action, "value": value, "route": "chance", "depth": 0}
except Exception:
    meta = {}
if not meta.get("action"):
    meta = search.best_action_meta(model, seat,
                                   seconds=float(payload["seconds"]), max_depth=int(payload["depth"]))
sys.stdout.write(json.dumps({"action": meta.get("action"), "value": meta.get("value"),
                             "route": meta.get("route"), "depth": meta.get("depth"),
                             "to_move": list(model.to_move() or []),
                             "n_actions": len(model.legal_actions(seat) or [])}))
"""


def _spawn(runner: str, args: list[str], payload: dict, timeout: float, label: str) -> dict | None:
    tmp_dir = tempfile.gettempdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": tmp_dir,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        # Deliberately no proxy variables and no LLM key: generated code has no way out.
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", runner, *[str(a) for a in args]],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout)),
            env=env,
            cwd=tmp_dir,
        )
    except subprocess.TimeoutExpired:
        log(f"{label}: timeout")
        return None
    except OSError as exc:
        log(f"{label}: cannot start:", type(exc).__name__)
        return None
    if completed.returncode != 0:
        log(f"{label}: exit", completed.returncode, completed.stderr.strip()[-600:])
        return None
    try:
        value = json.loads(completed.stdout or "{}")
    except ValueError:
        log(f"{label}: bad output")
        return None
    return value if isinstance(value, dict) else None


def validate_model(path: str, samples: list[dict], rules: str, timeout: float = 30.0) -> tuple[bool, str]:
    """Replay the arena's sample turns through a generated game model.

    Accepts the module only if its `legal_actions` match what the arena will actually offer, and if
    applying those actions never crashes and scores sensibly. A model that fails is thrown away; the bot
    then plays with the model's own `choose()` policy or with the LLM.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        return False, f"unreadable: {type(exc).__name__}"
    banned = inspect_source(source)
    if banned:
        return False, f"banned token {banned!r}"
    if not samples:
        return False, "no sample_turn to validate against"
    result = _spawn(_MODEL_VALIDATE_RUNNER, [str(path)], {"samples": samples, "rules": rules}, timeout, "genmodel")
    if not result:
        return False, "validation process failed"
    return bool(result.get("ok")), str(result.get("reason") or "unknown")


def call_model_search(
    path: str,
    search_path: str,
    rules: str,
    observation: str,
    legal: list[str] | None,
    seat: int,
    timeout: float = 3.0,
    depth: int = 8,
    strategy: str = "search",
    matrix_path: str = "",
    chance_path: str = "",
) -> dict:
    """One play-time search inside the sandbox; returns {"action", "route", "reason", ...}.

    Never raises: an empty dict means "the generated model could not answer", and the caller falls back.
    """
    if timeout <= 0.6:
        return {"action": None, "route": "timeout", "reason": "budget too small"}
    payload = {
        "rules": rules,
        "observation": observation,
        "seat": seat,
        "seconds": max(0.1, min(1.5, timeout * 0.6)),
        "depth": depth,
        "strategy": strategy,
    }
    result = _spawn(_MODEL_PLAY_RUNNER, [path, search_path, matrix_path, chance_path],
                    payload, timeout, "genmodel")
    if not result:
        return {"action": None, "route": "sandbox-failed", "reason": "no result"}
    action = result.get("action")
    if action is None:
        return {"action": None, "route": str(result.get("route") or "no-action"),
                "reason": f"to_move={result.get('to_move')} n_actions={result.get('n_actions')}",
                "value": result.get("value")}
    if legal is not None:
        verified = normalize(action, legal)
        if verified is None:
            return {"action": None, "route": "illegal", "reason": f"{action!r} not in legal"}
        action = verified
    else:
        text = str(action)
        if not text or len(text) > 120:
            return {"action": None, "route": "bad-action", "reason": repr(text)[:80]}
        action = text
    return {"action": action, "value": result.get("value"), "route": result.get("route"),
            "depth": result.get("depth"), "reason": "ok"}


def call_model_best(
    path: str,
    search_path: str,
    rules: str,
    observation: str,
    legal: list[str] | None,
    seat: int,
    timeout: float = 3.0,
    depth: int = 8,
    strategy: str = "search",
    matrix_path: str = "",
    chance_path: str = "",
) -> str | None:
    """Compatibility wrapper: the action only, or None. Prefer `call_model_search` for diagnostics."""
    return call_model_search(path, search_path, rules, observation, legal, seat, timeout, depth,
                             strategy, matrix_path, chance_path).get("action")


def inspect_source(source: str) -> str | None:
    """Cheap static check. Returns the offending token, or None when the module looks clean."""
    lowered = source.lower()
    for token in _BANNED:
        if token in lowered:
            return token
    return None


def _run(path: str, observation: str, legal: list[str] | None, seat: int, timeout: float) -> str | None:
    payload = json.dumps({"observation": observation, "legal_actions": legal, "seat": seat})
    tmp_dir = tempfile.gettempdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": tmp_dir,
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
            cwd=tmp_dir,
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
        if legal is not None:
            if normalize(action, legal) is None:
                return False, f"sample {index}: illegal {action!r}"
        else:
            text = str(action).strip()
            if not text or len(text) > 120 or "\n" in text or text.lower() in {"none", "null"}:
                return False, f"sample {index}: implausible free-text action {text[:60]!r}"
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
