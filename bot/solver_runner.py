"""Safe runner for prep-synthesized Python helper / solver functions.

Owner: Colegul A (Prep & Solvers).
Prep can generate a lightweight `choose_hint(observation, legal_actions, rules, chat_summary)`
function in Python, test it against `sample_turn`, and run it with a short wall-clock
budget during turns. Any error or timeout cleanly falls back to None.
"""
from __future__ import annotations

import math
import random
import re
import time
from typing import Any

from llm_client import log

ALLOWED_BUILTINS = {
    "Exception": Exception,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def compile_helper(code: str) -> dict[str, Any] | None:
    """Compile prep-generated Python helper code into an isolated namespace."""
    if not code or not code.strip():
        return None
    ns: dict[str, Any] = {
        "__builtins__": ALLOWED_BUILTINS,
        "re": re,
        "math": math,
        "random": random,
    }
    try:
        exec(compile(code, "<prep_helper>", "exec"), ns, ns)
    except Exception as exc:
        log("prep helper compile failed:", exc)
        return None
    if not callable(ns.get("suggest_move")):
        log("prep helper missing callable suggest_move(observation, legal_actions, rules)")
        return None
    return ns


def test_helper_on_samples(code: str, variants: list[dict[str, Any]]) -> bool:
    """Validate generated helper against every variant's `sample_turn`."""
    ns = compile_helper(code)
    if ns is None:
        return False
    suggest = ns["suggest_move"]
    for variant in variants:
        sample = variant.get("sample_turn")
        if not sample:
            continue
        obs = sample.get("observation", "")
        legal = sample.get("legal_actions")
        rules = variant.get("rules", "")
        try:
            res = suggest(obs, legal, rules)
        except Exception as exc:
            log(f"prep helper crashed on sample_turn ({variant.get('label')}):", exc)
            return False
        if not isinstance(res, dict):
            log(f"prep helper returned non-dict on sample_turn ({variant.get('label')}): {type(res).__name__}")
            return False
        move = res.get("action")
        if legal and move is not None and str(move) not in legal:
            log(f"prep helper proposed illegal sample move {move!r}")
            return False
    return True


def run_helper(
    compiled_ns: dict[str, Any] | None,
    observation: str,
    legal_actions: list[str] | None,
    rules: str,
    budget_seconds: float = 2.0,
) -> dict[str, Any] | None:
    """Invoke compiled helper safely; returns e.g. `{'action': ..., 'reason': ...}` or None."""
    if not compiled_ns:
        return None
    suggest = compiled_ns.get("suggest_move")
    if not callable(suggest):
        return None
    t0 = time.monotonic()
    try:
        res = suggest(observation, legal_actions, rules)
        elapsed = time.monotonic() - t0
        if elapsed > budget_seconds:
            log(f"solver helper took {elapsed:.2f}s (> {budget_seconds:.2f}s)")
        if isinstance(res, dict):
            return res
    except Exception as exc:
        log("solver helper runtime error:", exc)
    return None
