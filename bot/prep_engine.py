"""Prep engine (10-minute specialist builder) for AI Harness Arena (Team adn).

Owner: Colegul A (Prep & Solvers).
Turns a game-agnostic container into a specialist during `prep` (600s clock), and
also supports fast fallback study during a cold `start` (120s).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from llm_client import GatewayLLMClient, log
from solver_runner import test_helper_on_samples

GAME_FAMILIES = (
    "bluff",
    "deals",
    "read_opponent",
    "guessing",
    "think_ahead",
    "truth",
    "guardrails",
)

PREP_SYSTEM = """You are preparing an AI harness to win a two-player tournament game it has never seen before.
Analyze all variants and optional sample_turn layouts carefully.
Return ONLY valid JSON with these keys:
{
  "family": "one of: bluff | deals | read_opponent | guessing | think_ahead | truth | guardrails",
  "action_format": "concise specification of exact valid action string format",
  "action_regex": "Python regex matching one valid action string, or empty string if complex",
  "fallback_action": "single literal valid action token from the rules (e.g. 'B2' or '50' or 'pass') or empty string '' — NEVER a sentence or instruction",
  "forbidden_terms": ["list of forbidden words/topics if Guardrails game, else []"],
  "playbook": "single string with at most 20 sharp lines separated by newlines: winning strategy, opening, state tracking, traps, chat tactics",
  "chat_strategy": "how to use chat offensively (bluff/negotiate/pressure) and defensively in this game"
}"""

SYNTH_SYSTEM = """You write tiny, deterministic, bug-free Python helper code to assist a game-playing agent.
Output raw Python code ONLY: NO prose, NO explanation, NO markdown backticks, NO import statements (`re`, `math`, `random` are already pre-injected in globals).
Start line 1 directly with `def suggest_move(observation: str, legal_actions: list[str] | None, rules: str) -> dict:`:
- Parse `observation` accurately using the exact format shown in `sample_turn`.
- Return `{"action": "<best candidate or None>", "reason": "<short tactical analysis / odds / bounds>"}`.
- If `legal_actions` is provided and non-empty, `action` MUST be an exact element of `legal_actions` (or `None`).
- If unsure, return `{"action": None, "reason": "<parsed state summary>"}`.
- Never raise exceptions; execute in <50ms."""


def _extract_json_block(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        text = re.sub(r"^```(?:python|py)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text).strip()
        if not text.startswith(("def ", "#")):
            m = re.search(r"^(def\s+suggest_move\b.*)$", text, re.MULTILINE | re.DOTALL)
            if m:
                text = m.group(1).strip()
    # Strip top-level import lines since re, math, random are already injected without __import__
    lines = [
        ln
        for ln in text.splitlines()
        if not re.match(r"^\s*(import|from)\s+(re|math|random|typing|collections)\b", ln)
    ]
    return "\n".join(lines).strip()


def _format_variants_prompt(variants: list[dict[str, Any]]) -> str:
    parts = []
    for v in variants:
        chunk = f"=== VARIANT: {v.get('label', 'default')} ===\n{v.get('rules', '')}"
        sample = v.get("sample_turn")
        if sample:
            chunk += (
                f"\n--- Sample Turn (turn {sample.get('turn')}) ---\n"
                f"Observation:\n{sample.get('observation', '')}\n"
                f"Legal actions: {sample.get('legal_actions')}"
            )
        parts.append(chunk)
    return "\n\n".join(parts)


def prepare_specialist(
    game_id: str,
    game_name: str,
    variants: list[dict[str, Any]],
    total_seconds: float,
    llm: GatewayLLMClient,
    *,
    enable_code_synthesis: bool = True,
) -> dict[str, Any]:
    """Build a structured specialist package for `game_id` within `total_seconds`."""
    t0 = time.monotonic()
    deadline = t0 + max(10.0, float(total_seconds) - 30.0)
    prompt_body = f"Game: {game_name} ({game_id})\n\n" + _format_variants_prompt(variants)

    # Step 1: Deep rule analysis & classification (bounded thinking when budget allows).
    remaining = deadline - time.monotonic()
    thinking_budget = 1500 if remaining >= 60 else 0
    raw = llm.ask(
        PREP_SYSTEM,
        prompt_body,
        timeout_seconds=min(60.0, max(5.0, remaining * 0.5)),
        max_tokens=2000,
        thinking_budget=thinking_budget,
    )
    parsed = _extract_json_block(raw) or {}

    family = str(parsed.get("family", "think_ahead")).lower().strip()
    if family not in GAME_FAMILIES:
        family = "think_ahead"

    raw_playbook = parsed.get("playbook", "")
    if isinstance(raw_playbook, list):
        playbook = "\n".join(str(x).strip() for x in raw_playbook if str(x).strip())
    else:
        playbook = str(raw_playbook).strip()
    if not playbook:
        playbook = raw.strip()[:2500]

    fallback_action = str(parsed.get("fallback_action", "")).strip()
    if (
        len(fallback_action) > 40
        or "\n" in fallback_action
        or "legal action" in fallback_action.lower()
        or "first entry" in fallback_action.lower()
    ):
        fallback_action = ""

    action_regex = str(parsed.get("action_regex", "")).strip()
    if "(?i)" in action_regex and not action_regex.startswith("(?i)"):
        action_regex = "(?i)" + action_regex.replace("(?i)", "")
    if action_regex:
        try:
            re.compile(action_regex)
        except re.error:
            action_regex = ""

    specialist: dict[str, Any] = {
        "game_id": game_id,
        "game_name": game_name,
        "family": family,
        "action_format": str(parsed.get("action_format", "")).strip(),
        "action_regex": action_regex,
        "fallback_action": fallback_action,
        "forbidden_terms": [
            str(x).strip() for x in (parsed.get("forbidden_terms") or []) if str(x).strip()
        ],
        "playbook": playbook,
        "chat_strategy": str(parsed.get("chat_strategy", "")).strip(),
        "helper_code": "",
        "helper_verified": False,
        "prepared_at": time.time(),
    }

    # Step 2: Optional deterministic helper synthesis + self-verification on sample_turn.
    remaining = deadline - time.monotonic()
    has_samples = any(v.get("sample_turn") for v in variants)
    if enable_code_synthesis and has_samples and remaining >= 25.0:
        synth_user = (
            f"Game family: {family}\n"
            f"Action format: {specialist['action_format']}\n"
            f"Strategy summary:\n{specialist['playbook']}\n\n"
            + prompt_body
        )
        for attempt in range(2):
            rem_now = deadline - time.monotonic()
            if rem_now < 15.0:
                break
            raw_code = llm.ask(
                SYNTH_SYSTEM,
                synth_user
                if attempt == 0
                else synth_user + "\n\nCRITICAL: Start line 1 with `def suggest_move(observation, legal_actions, rules):` and NO markdown or prose.",
                timeout_seconds=min(40.0, max(5.0, rem_now - 5.0)),
                max_tokens=1600,
                thinking_budget=0,
            )
            code = _strip_code_fences(raw_code)
            if code and test_helper_on_samples(code, variants):
                specialist["helper_code"] = code
                specialist["helper_verified"] = True
                log(f"prep {game_id}: helper verified on sample_turn ({len(code)} chars, attempt {attempt + 1})")
                break
            log(f"prep {game_id}: attempt {attempt + 1} unverified; head={code[:80]!r}")
        if not specialist["helper_verified"]:
            log(f"prep {game_id}: skipped unverified helper; keeping playbook + LLM")

    log(
        f"prep {game_id}: family={specialist['family']} "
        f"verified_helper={specialist['helper_verified']} "
        f"forbidden={len(specialist['forbidden_terms'])} in {time.monotonic() - t0:.1f}s"
    )
    return specialist
