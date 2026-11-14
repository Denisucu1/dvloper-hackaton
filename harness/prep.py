"""The 10 minutes of prep: turn one game's rules into something code can use.

Output (all in /memory/<game_id>/, keyed by a hash of the rules text):
- spec.json      structured understanding: family, action format, hidden info, speech rules, traps
- playbook.txt   a short, human-readable plan (also what the judges can read)
- samples.json   the `sample_turn` messages, kept to test parsers and validate generated code
- rules.txt      the exact rules we prepared from (traceability for the judges)
- policy.py      optional, only with HARNESS_GENPOLICY=1: a module the model wrote, sandboxed and validated

Budget: one structured call for the whole game (all variants at once), so prep cost is a fraction of a cent.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, memory, sandbox
from .llm import Gateway
from .log import log, kv
from .spec import FAMILIES, Spec, fallback_spec

_SPEC_SYSTEM = """You are the preparation engine of a game-playing harness. You get the rules of one
two-player game (every variant) and real sample turns. You have never seen this game before.

Return exactly ONE JSON object, no markdown, with these keys:
{
  "family": one of __FAMILIES__,
  "action_format": "the exact textual format of a move, as a template",
  "action_examples": ["3-6 concrete legal action strings (e.g. 'B2', 'guess 50', 'pass') — no placeholders"],
  "fallback_action": "single literal valid action token from the rules/samples (e.g. 'B2' or '50') or ''",
  "winning_idea": "one sentence: how this game is won",
  "hidden_information": true|false,
  "simultaneous": true|false,
  "scoring": "how a game/round is scored, including anything unusual",
  "speech_rules": "every restriction on what a player may SAY in chat, or empty string",
  "forbidden_subjects": ["every word/topic that must never be mentioned in speech or actions"],
  "traps": ["up to 5 tricks, illegal-move traps or common losing mistakes in this game"],
  "notes": "variant differences, deal setup, and anything the observation text will contain",
  "playbook": "at most 25 short lines: opening plan, per-turn checklist, endgame, how to use or resist chat"
}

Be precise about the action format: the bot must never send an illegal move. If a variant restricts speech,
fill speech_rules and forbidden_subjects with the literal words from the rules.
""".replace("__FAMILIES__", ", ".join(FAMILIES))

_SPEC_USER = """GAME ID: {game_id}
GAME NAME: {game_name}

{variants}

Return the JSON object now."""

_GEN_SYSTEM = """Write ONE standalone Python 3 module that plays the given game.
Output raw Python code ONLY — no prose, no explanations, no markdown backticks.
Start with optional allowed imports (`math`, `random`, `re`, `itertools`, `collections`, `heapq`) and `def choose(observation: str, legal_actions: list[str] | None, seat: int) -> str:`.
- `observation` is the exact turn text shown in the sample turns below.
- Return ONE valid action string in the exact action format; if `legal_actions` is provided and non-empty, return an element of `legal_actions` verbatim.
- Handle ALL variants shown cleanly. Never print, never raise, never use files/network/subprocesses, finish in <50ms."""

_GEN_USER = """RULES (all variants):
{rules}

SAMPLE TURNS TO PASS VALIDATION:
{samples_blob}"""

_GEN_FIX = """Your previous module failed validation with: {error}
Fix it so it handles EVERY sample turn shown above cleanly. Output the full corrected Python module only."""


@dataclass
class PrepResult:
    spec: Spec
    playbook: str = ""
    rules: str = ""
    genpolicy: str | None = None
    cached: bool = False
    elapsed: float = 0.0
    calls: int = 0
    notes: list[str] = field(default_factory=list)


def _variant_blob(variants: list[dict]) -> str:
    chunks = []
    for variant in variants:
        part = [f"=== VARIANT: {variant.get('label', '?')} ===", str(variant.get("rules", ""))[:3500]]
        sample = variant.get("sample_turn") or {}
        if sample:
            part.append(f"--- sample turn {sample.get('turn', '?')} ---")
            part.append(str(sample.get("observation", ""))[:1200])
            part.append(f"legal_actions = {sample.get('legal_actions')}")
        chunks.append("\n".join(part))
    return "\n\n".join(chunks)


def run_prep(msg: dict, gateway: Gateway) -> PrepResult:
    started = time.time()
    seconds = float(msg.get("seconds") or 600)
    game_id = str(msg.get("game_id") or "unknown")
    variants = msg.get("variants") or []
    rules_all = memory.variants_rules(variants)
    fp = memory.fingerprint(rules_all)
    directory = memory.game_dir(game_id)
    notes: list[str] = []

    # Samples first: they are also the validator for generated code and cost nothing.
    samples = [
        {**dict(v.get("sample_turn") or {}), "seat": v.get("seat", 0)}
        for v in variants
        if v.get("sample_turn")
    ]
    memory.atomic_write_json(directory / config.ARTIFACT_SAMPLES, {"game_id": game_id, "fingerprint": fp, "turns": samples})
    memory.atomic_write_text(directory / config.ARTIFACT_RULES, rules_all)

    cached = memory.read_json(memory.artifact_path(game_id, fp, config.ARTIFACT_SPEC))
    if not (cached and cached.get("fingerprint") == fp):
        latest = memory.read_json(directory / config.ARTIFACT_SPEC)
        cached = latest if latest and latest.get("fingerprint") == fp else None
    if cached and cached.get("fingerprint") == fp:
        spec = Spec.from_dict(cached, fp)
        playbook = memory.read_text(memory.artifact_path(game_id, fp, config.ARTIFACT_PLAYBOOK)) or memory.read_text(directory / config.ARTIFACT_PLAYBOOK)
        manifest = memory.read_json(memory.artifact_path(game_id, fp, "manifest.json")) or memory.read_json(directory / "manifest.json") or {}
        policy_name = str(manifest.get("genpolicy") or "")
        policy_path = directory / policy_name if policy_name else None
        genpolicy = str(policy_path) if policy_path and policy_path.name == policy_name and policy_path.exists() else None
        log(kv(prep="cached", game=game_id, family=spec.family, genpolicy=bool(genpolicy)))
        return PrepResult(spec=spec, playbook=playbook, rules=rules_all, genpolicy=genpolicy,
                          cached=True, elapsed=time.time() - started, notes=["cached artifact reused"])

    budget_seconds = max(20.0, seconds * config.PREP_SHARE)
    call_timeout = min(180.0, max(20.0, budget_seconds * 0.55))

    data = gateway.ask_json(
        _SPEC_SYSTEM,
        _SPEC_USER.format(game_id=game_id, game_name=msg.get("game_name", game_id), variants=_variant_blob(variants)),
        max_tokens=1600,
        timeout=call_timeout,
        temperature=0.2,
    )
    if data:
        spec = Spec.from_dict({**data, "game_id": game_id}, fp)
        spec.source = "prep"
        raw_plan = data.get("playbook")
        if isinstance(raw_plan, list):
            playbook = "\n".join(f"- {str(item).strip()}" for item in raw_plan if str(item).strip())
        else:
            playbook = str(raw_plan or "").strip()
        notes.append("spec from model")
    else:
        spec = fallback_spec(game_id, rules_all, samples, fp)
        playbook = ""
        notes.append("model unavailable: regex fallback spec")
        log("prep: model returned nothing usable, using fallback spec")

    if not spec.action_examples:
        spec.action_examples = fallback_spec(game_id, rules_all, samples, fp).action_examples
    if not spec.family or spec.family == "unknown":
        spec.family = fallback_spec(game_id, rules_all, samples, fp).family

    memory.atomic_write_json(memory.artifact_path(game_id, fp, config.ARTIFACT_SPEC), spec.to_dict())
    memory.atomic_write_text(memory.artifact_path(game_id, fp, config.ARTIFACT_PLAYBOOK), playbook or "(no playbook)")
    memory.atomic_write_json(directory / config.ARTIFACT_SPEC, spec.to_dict())
    memory.atomic_write_text(directory / config.ARTIFACT_PLAYBOOK, playbook or "(no playbook)")

    genpolicy = None
    if config.GENPOLICY and samples:
        genpolicy = _build_genpolicy(
            gateway, directory, fp, rules_all, samples, spec, notes,
            timeout=min(150.0, max(30.0, seconds - (time.time() - started) - 60.0)),
        )
    elif config.GENPOLICY:
        notes.append("genpolicy skipped: no sample_turn")

    manifest_doc = {
        "game_id": game_id,
        "fingerprint": fp,
        "genpolicy": (Path(genpolicy).name if genpolicy else ""),
        "family": spec.family,
        "seconds": seconds,
        "made": time.time(),
        "notes": notes,
    }
    memory.atomic_write_json(memory.artifact_path(game_id, fp, "manifest.json"), manifest_doc)
    memory.atomic_write_json(directory / "manifest.json", manifest_doc)

    result = PrepResult(spec=spec, playbook=playbook, rules=rules_all, genpolicy=genpolicy,
                        cached=False, elapsed=time.time() - started, calls=gateway.calls, notes=notes)
    log(kv(prep="done", game=game_id, family=spec.family, variants=len(variants),
           plan_chars=len(playbook), genpolicy=bool(genpolicy), secs=round(result.elapsed, 1),
           spend=gateway.summary()["spend_usd"]))
    return result


def load_artifacts(game_id: str) -> PrepResult | None:
    """Load what a previous prep (possibly in another container) left in /memory. Rule 4.5."""
    directory = memory.game_dir(game_id)
    data = memory.read_json(directory / config.ARTIFACT_SPEC)
    if not isinstance(data, dict) or not data:
        return None
    spec = Spec.from_dict(data, str(data.get("fingerprint") or ""))
    playbook = memory.read_text(directory / config.ARTIFACT_PLAYBOOK)
    manifest = memory.read_json(directory / "manifest.json") or {}
    name = str(manifest.get("genpolicy") or "")
    policy = directory / name
    genpolicy = str(policy) if name and policy.name == name and policy.exists() else None
    rules = memory.read_text(directory / config.ARTIFACT_RULES)
    log(kv(memory="loaded", game=game_id, family=spec.family, genpolicy=bool(genpolicy)))
    return PrepResult(spec=spec, playbook=playbook, rules=rules, genpolicy=genpolicy, cached=True)


_QUICK_SYSTEM = """You are given the rules of one two-player game. Return ONE JSON object (no markdown) with:
{"family": "bluff|deals|read_opponent|guessing|think_ahead|truth|guardrails|unknown",
 "action_format": "exact move format", "action_examples": ["2-4 concrete actions"],
 "winning_idea": "one sentence", "speech_rules": "restrictions on what a player may say, or empty",
 "forbidden_subjects": ["words/topics that must never be mentioned"]}"""


def quick_spec(gateway: Gateway, game_id: str, rules: str, timeout: float = 20.0) -> Spec | None:
    """A cold container's mini-prep: one small call inside the 120 s study clock, then play."""
    data = gateway.ask_json(
        _QUICK_SYSTEM,
        f"GAME ID: {game_id}\n\nRULES:\n{str(rules)[:6000]}\n\nJSON now:",
        max_tokens=500,
        timeout=timeout,
        temperature=0.2,
    )
    if not data:
        return None
    spec = Spec.from_dict({**data, "game_id": game_id}, memory.rules_fingerprint(rules))
    spec.source = "prep"
    if not spec.action_examples:
        spec.action_examples = fallback_spec(game_id, rules).action_examples
    if spec.family == "unknown":
        spec.family = fallback_spec(game_id, rules).family
    log(kv(cold="mini-prep", game=game_id, family=spec.family))
    return spec


def _build_genpolicy(
    gateway: Gateway,
    directory: Path,
    fp: str,
    rules_all: str,
    samples: list[dict],
    spec: Spec,
    notes: list[str],
    timeout: float,
) -> str | None:
    """Ask the model for a `choose()` policy and accept it only if it answers every sample legally."""
    path = directory / f"{fp}-{config.ARTIFACT_GENPOLICY}"
    samples_chunks = []
    for idx, sample in enumerate(samples[:4]):
        samples_chunks.append(
            f"--- Sample {idx + 1} (seat={sample.get('seat', 0)}, turn={sample.get('turn', '?')}) ---\n"
            f"{str(sample.get('observation', ''))[:1500]}\n"
            f"legal_actions = {sample.get('legal_actions')}"
        )
    user = _GEN_USER.format(
        rules=rules_all[:6000],
        samples_blob="\n\n".join(samples_chunks),
    )
    error = ""
    for attempt in (1, 2):
        prompt = user if attempt == 1 else user + "\n\n" + _GEN_FIX.format(error=error)
        text = gateway.ask(
            _GEN_SYSTEM if attempt == 1 else "You fix standalone Python 3 game modules.",
            prompt,
            max_tokens=1800,
            timeout=timeout / 2,
            temperature=0.2,
        )
        source = _strip_code(text)
        if not source:
            error = "no valid Python `def choose(...)` function returned"
            continue
        memory.atomic_write_text(path, source)
        ok, reason = sandbox.validate_module(str(path), samples)
        notes.append(f"genpolicy attempt {attempt}: {reason}")
        log(f"genpolicy attempt {attempt}: {reason}")
        if ok:
            return str(path)
        error = reason
        if "banned token" in reason:
            break
    try:
        path.unlink()
    except OSError:
        pass
    return None


def _strip_code(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:python|py|python3)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        text = re.sub(r"^```(?:python|py|python3)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text).strip()
        if not text.startswith(("import ", "from ", "def ", "#")):
            m = re.search(r"^((?:import\s+\w+|from\s+\w+\s+import|def\s+choose\b).*)$", text, re.MULTILINE | re.DOTALL)
            if m:
                text = m.group(1).strip()
    return text.strip() if "def choose(" in text else ""
