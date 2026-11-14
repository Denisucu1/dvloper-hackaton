"""The 10 minutes of prep: turn one game's rules into something code can use.

Output (all in /memory/<game_id>/, keyed by a hash of the rules text):
- spec.json      structured understanding: family, action format, hidden info, speech rules, traps
- playbook.txt   a short, human-readable plan (also what the judges can read)
- samples.json   the `sample_turn` messages, kept to test parsers and validate generated code
- rules.txt      the exact rules we prepared from (traceability for the judges)
- policy.py      optional, only with HARNESS_GENPOLICY=1: a choose() policy the model wrote, sandboxed
- model.py       the simulator the model wrote (harness/simgen.py): our search.py plays it, after the
                 arena's sample turns prove its legal_actions and apply() are correct
- manifest.json  what was built for this rules hash; a new ARTIFACT_VERSION invalidates stale artifacts

Budget: one structured call for the whole game (all variants at once), so prep cost is a fraction of a cent.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, memory, sandbox
from .game_model import ARTIFACT_VERSION, MODEL_SUFFIX
from .llm import Gateway
from .log import log, kv
from .spec import FAMILIES, Spec, fallback_spec

# Prep spends its budget in this order: understanding (always), policy (cheap), simulator (expensive).
GENMODEL = os.environ.get("HARNESS_GENMODEL", "1") == "1"

# Code generation is not free: prep has one clock and one key. These families cannot use a generated
# simulator, so we skip it instead of spending two attempts on a model that would be wrong or useless.
_SKIP_SIM_FAMILIES = {
    "truth": "QA game: the answer comes from the document, there is no game tree",
    "guardrails": "speech game: the move is a safe answer, not a search",
    "bluff": "private-card family: minimax cannot see the opponent's card, a simulator would invent it",
    "deals": "negotiation: the move depends on private values and chat, not a search tree",
}


def simgen_worthwhile(spec: Spec) -> tuple[bool, str]:
    """Whether a generated simulator can help this game. Cheap, deterministic, no model call."""
    reason = _SKIP_SIM_FAMILIES.get(spec.family)
    if reason:
        return False, f"{spec.family}: {reason}"
    if spec.hidden_information and not spec.simultaneous:
        return False, "hidden information: a deterministic model cannot see the opponent's state"
    return True, ""


def genpolicy_worthwhile(spec: Spec) -> tuple[bool, str]:
    """A generated choose() is a shallow imitation; for QA/speech games it only adds noise."""
    if spec.family in ("truth", "guardrails"):
        return False, f"{spec.family}: a learned move adds nothing over the LLM and the document"
    return True, ""

_SPEC_SYSTEM = """You are the preparation engine of a game-playing harness. You get the rules of one
two-player game (every variant) and real sample turns. You have never seen this game before.

Return exactly ONE JSON object, no markdown, with these keys:
{
  "family": one of __FAMILIES__,
  "action_format": "the exact textual format of a move, as a template",
  "action_examples": ["3-6 concrete legal action strings (e.g. 'B2', 'guess 50', 'pass') — no placeholders"],
  "fallback_action": "single literal valid action token from the rules/samples (e.g. 'B2' or '50') or ''",
  "winning_idea": "one sentence: how this game is won",
  "hidden_information": true|false,  // true ONLY when part of the current state is private (cards, secret
                                    // word/values). Simultaneous moves are NOT hidden information.
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

_GEN_SYSTEM = """Write ONE standalone Python 3 module that plays the given game, including games that are
not boards: negotiations, auctions, word guesses, simultaneous choices, free-text answers.
Output raw Python code ONLY — no prose, no explanations, no markdown backticks.
Start with optional allowed imports (`math`, `random`, `re`, `itertools`, `collections`, `heapq`) and
`def choose(observation: str, legal_actions: list[str] | None, seat: int) -> str:`.

How to return the action, by regime:
- `legal_actions` is a non-empty list: return one element of it verbatim (copy the exact spelling).
- `legal_actions` is None and the action is numeric/structured: build the string from the observation and
  the rules, exactly in the action format (e.g. "bid 7", "guess crane", "offer 45 50", "vote b").
- `legal_actions` is None and the action is free text: return ONE short concrete answer, at most 50
  characters, no explanation and no quotes. Never return an empty string, "none", "null" or a sentence.
- Hidden information (an unseen card, a secret word, a private value): never invent it. Use only facts the
  observation states; for unseen things play a robust mixed or expected-value choice.
- Simultaneous moves: assume the opponent also moves now and cannot be seen; do not reason as if their
  move were known. Prefer a mixed or robust choice, and randomize with `random` when the rules allow it.
- If the rules say a safe abstention exists (for example "unknown"), prefer it over inventing an answer.

`observation` is the exact turn text shown in the sample turns below. Handle ALL variants and ALL sample
turns cleanly. Never print, never raise, never use files/network/subprocesses; finish in <50 ms."""

_GEN_USER = """RULES (all variants):
{rules}

WHAT WE ALREADY KNOW ABOUT THE GAME:
{context}

SAMPLE TURNS TO PASS VALIDATION:
{samples_blob}"""

_GEN_FIX = """Your previous module failed validation with:
{error}

PREVIOUS MODULE (repair this one; do not start over):
```python
{source}
```

Fix it so it handles EVERY sample turn shown above cleanly. Output the full corrected Python module only."""


@dataclass
class PrepResult:
    spec: Spec
    playbook: str = ""
    rules: str = ""
    genpolicy: str | None = None
    genmodel: str | None = None
    strategy: str = ""
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
    # The variant label travels with the sample: one game_id may have several variants, and a generated
    # simulator that hardcodes the first variant's geometry must be rejected (and told which one it broke).
    samples = [
        {
            **dict(v.get("sample_turn") or {}),
            "seat": v.get("seat", 0),
            "variant": str(v.get("label", "")),
            "settings": {k: val for k, val in v.items() if k not in ("rules", "sample_turn", "label")},
        }
        for v in variants
        if v.get("sample_turn")
    ]
    memory.atomic_write_json(directory / config.ARTIFACT_SAMPLES, {"game_id": game_id, "fingerprint": fp, "turns": samples})
    memory.atomic_write_text(directory / config.ARTIFACT_RULES, rules_all)

    cached = memory.read_json(memory.artifact_path(game_id, fp, config.ARTIFACT_SPEC))
    if not (cached and cached.get("fingerprint") == fp
                and cached.get("artifact_version") == ARTIFACT_VERSION):
        latest = memory.read_json(directory / config.ARTIFACT_SPEC)
        cached = (latest if latest and latest.get("fingerprint") == fp
                  and latest.get("artifact_version") == ARTIFACT_VERSION else None)
    if cached and cached.get("fingerprint") == fp:
        spec = Spec.from_dict(cached, fp)
        playbook = memory.read_text(memory.artifact_path(game_id, fp, config.ARTIFACT_PLAYBOOK)) or memory.read_text(directory / config.ARTIFACT_PLAYBOOK)
        manifest = memory.read_json(memory.artifact_path(game_id, fp, "manifest.json")) or memory.read_json(directory / "manifest.json") or {}
        policy_name = str(manifest.get("genpolicy") or "")
        policy_path = directory / policy_name if policy_name else None
        genpolicy = str(policy_path) if policy_path and policy_path.name == policy_name and policy_path.exists() else None
        model_name = str(manifest.get("genmodel") or "")
        model_path = directory / model_name if model_name else None
        genmodel = str(model_path) if model_path and model_path.name == model_name and model_path.exists() else None
        strategy = str(manifest.get("strategy") or "")
        log(kv(prep="cached", game=game_id, family=spec.family, genpolicy=bool(genpolicy),
               genmodel=bool(genmodel), strategy=strategy or "-"))
        return PrepResult(spec=spec, playbook=playbook, rules=rules_all, genpolicy=genpolicy,
                          genmodel=genmodel, strategy=strategy, cached=True,
                          elapsed=time.time() - started, notes=["cached artifact reused"])

    budget_seconds = max(20.0, seconds * config.PREP_SHARE)
    call_timeout = min(180.0, max(20.0, budget_seconds * 0.55))

    spec_user = _SPEC_USER.format(game_id=game_id, game_name=msg.get("game_name", game_id),
                                  variants=_variant_blob(variants))
    data = gateway.ask_json(
        _SPEC_SYSTEM,
        spec_user,
        max_tokens=1600,
        timeout=call_timeout,
        temperature=0.2,
    )
    if not data:
        # One retry is cheap insurance on an unseen game: a failed parse leaves the specialist without a
        # family, playbook or speech rules, which changes chat and turn policy for the whole game.
        left = seconds - (time.time() - started) - 120.0
        if gateway.affordable and left > 20.0:
            notes.append("spec retry after an unparseable answer")
            data = gateway.ask_json(
                _SPEC_SYSTEM + "\nReturn ONLY the JSON object, with no prose before or after it.",
                spec_user,
                max_tokens=2000,
                timeout=min(90.0, max(20.0, left / 2.0)),
                temperature=0.0,
            )
            if data:
                notes.append("spec from model (retry)")
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

    spec_doc = {**spec.to_dict(), "artifact_version": ARTIFACT_VERSION}
    memory.atomic_write_json(memory.artifact_path(game_id, fp, config.ARTIFACT_SPEC), spec_doc)
    memory.atomic_write_text(memory.artifact_path(game_id, fp, config.ARTIFACT_PLAYBOOK), playbook or "(no playbook)")
    memory.atomic_write_json(directory / config.ARTIFACT_SPEC, spec_doc)
    memory.atomic_write_text(directory / config.ARTIFACT_PLAYBOOK, playbook or "(no playbook)")

    genpolicy = None
    policy_worth, policy_why = genpolicy_worthwhile(spec)
    if config.GENPOLICY and samples and policy_worth:
        genpolicy = _build_genpolicy(
            gateway, directory, fp, rules_all, samples, spec, notes,
            timeout=min(150.0, max(30.0, seconds - (time.time() - started) - 60.0)),
        )
    elif config.GENPOLICY and samples:
        notes.append(f"genpolicy skipped: {policy_why}")
        log(f"prep: genpolicy skipped ({policy_why})")
    elif config.GENPOLICY:
        notes.append("genpolicy skipped: no sample_turn")

    genmodel = None
    sim_worth, sim_why = simgen_worthwhile(spec)
    if GENMODEL and samples and sim_worth:
        genmodel = _build_genmodel(
            gateway, directory, fp, game_id, rules_all, samples, spec, notes,
            timeout=min(240.0, max(0.0, seconds - (time.time() - started) - 45.0)),
        )
    elif GENMODEL and samples:
        notes.append(f"genmodel skipped: {sim_why}")
        log(f"prep: simgen skipped ({sim_why})")
    elif GENMODEL:
        notes.append("genmodel skipped: no sample_turn")

    best_strategy = ""
    if genmodel:
        best_strategy = _maybe_tournament(
            genmodel, rules_all, spec, notes,
            budget=seconds - (time.time() - started) - 60.0,
            seed=int(fp[:8], 16) if len(fp) >= 8 else 0,
        )

    manifest_doc = {
        "game_id": game_id,
        "fingerprint": fp,
        "artifact_version": ARTIFACT_VERSION,
        "strategy": best_strategy,
        "genpolicy": (Path(genpolicy).name if genpolicy else ""),
        "genmodel": (Path(genmodel).name if genmodel else ""),
        "family": spec.family,
        "seconds": seconds,
        "made": time.time(),
        "notes": notes,
    }
    memory.atomic_write_json(memory.artifact_path(game_id, fp, "manifest.json"), manifest_doc)
    memory.atomic_write_json(directory / "manifest.json", manifest_doc)

    result = PrepResult(spec=spec, playbook=playbook, rules=rules_all, genpolicy=genpolicy,
                        genmodel=genmodel, strategy=best_strategy, cached=False,
                        elapsed=time.time() - started,
                        calls=gateway.calls, notes=notes)
    log(kv(prep="done", game=game_id, family=spec.family, variants=len(variants),
           plan_chars=len(playbook), genpolicy=bool(genpolicy), genmodel=bool(genmodel),
           secs=round(result.elapsed, 1), spend=gateway.summary()["spend_usd"]))
    return result


def load_artifacts(game_id: str) -> PrepResult | None:
    """Load what a previous prep (possibly in another container) left in /memory. Rule 4.5."""
    directory = memory.game_dir(game_id)
    data = memory.read_json(directory / config.ARTIFACT_SPEC)
    if not isinstance(data, dict) or not data:
        return None
    spec = Spec.from_dict(data, str(data.get("fingerprint") or ""))
    if data.get("artifact_version") != ARTIFACT_VERSION:
        log(kv(memory="stale artifacts ignored", game=game_id,
               have=data.get("artifact_version"), want=ARTIFACT_VERSION))
        return None
    playbook = memory.read_text(directory / config.ARTIFACT_PLAYBOOK)
    manifest = memory.read_json(directory / "manifest.json") or {}
    name = str(manifest.get("genpolicy") or "")
    policy = directory / name
    genpolicy = str(policy) if name and policy.name == name and policy.exists() else None
    model_name = str(manifest.get("genmodel") or "")
    model = directory / model_name
    genmodel = str(model) if model_name and model.name == model_name and model.exists() else None
    rules = memory.read_text(directory / config.ARTIFACT_RULES)
    log(kv(memory="loaded", game=game_id, family=spec.family,
           genpolicy=bool(genpolicy), genmodel=bool(genmodel)))
    return PrepResult(spec=spec, playbook=playbook, rules=rules, genpolicy=genpolicy,
                      genmodel=genmodel, cached=True)


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
    context = ""
    try:
        context = str(spec.prompt_block(800) or "").strip()
    except Exception:  # noqa: BLE001 - a spec is a hint, never a reason to fail prep
        context = ""
    user = _GEN_USER.format(
        rules=rules_all[:6000],
        context=context or "(no structured spec)",
        samples_blob="\n\n".join(samples_chunks),
    )
    error = "no valid Python `def choose(...)` function returned"
    previous = ""
    for attempt in (1, 2):
        if attempt == 1:
            prompt = user
        elif "banned token" in error:
            # Do not resend the banned module: start fresh with an explicit allow-list. This is a cheap
            # retry, not a security decision - the sandbox re-checks the next answer anyway.
            prompt = (f"{user}\n\nYour previous answer used a banned construct ({error[:200]}). "
                      "Write a fresh module. Allowed imports: math, random, re, itertools, collections, "
                      "heapq. Never use __import__, eval, exec, open, subprocess, sockets, requests or urllib.")
        else:
            prompt = f"{user}\n\n{_GEN_FIX.format(error=error[:900], source=previous[:5000])}"
        text = gateway.ask(
            _GEN_SYSTEM if attempt == 1 else "You fix standalone Python 3 game modules.",
            prompt,
            max_tokens=2000,
            timeout=timeout / 2,
            temperature=0.2,
        )
        source = _strip_code(text)
        if not source:
            error = "no valid Python `def choose(...)` function returned"
            continue
        previous = source
        memory.atomic_write_text(path, source)
        ok, reason = sandbox.validate_module(str(path), samples)
        detail = " ".join(str(reason).split())
        notes.append(f"genpolicy attempt {attempt}: {detail[:160]}")
        log(f"genpolicy attempt {attempt}: {detail[:160]}")
        if ok:
            return str(path)
        error = detail
        if "banned token" in detail:
            previous = ""
    try:
        path.unlink()
    except OSError:
        pass
    return None


def _build_genmodel(
    gateway: Gateway,
    directory,
    fingerprint: str,
    game_id: str,
    rules: str,
    samples: list[dict],
    spec: Spec,
    notes: list[str],
    timeout: float,
) -> str | None:
    """Ask the model for a simulator, then let the sandbox decide if it is real (see harness.simgen)."""
    if timeout < 25.0:
        notes.append("genmodel skipped: no budget left")
        return None
    try:
        from . import simgen
    except ImportError as exc:
        notes.append(f"genmodel skipped: simgen missing ({type(exc).__name__})")
        return None
    path = simgen.build_model(
        gateway=gateway,
        directory=directory,
        fingerprint=fingerprint,
        game_id=game_id,
        rules=rules,
        samples=samples,
        spec=spec,
        notes=notes,
        timeout=timeout,
    )
    if path:
        log(kv(genmodel="ready", file=Path(path).name, chars=len(memory.read_text(path))))
    else:
        notes.append("genmodel rejected: falling back to policy/LLM")
    return path


def _maybe_tournament(model_path: str, rules: str, spec: Spec, notes: list[str],
                      budget: float, seed: int) -> str:
    """Rank decision strategies on the validated model by self-play; return the winner's name.

    Only for perfect-information games: a deterministic model cannot referee a hidden-information game.
    Everything is best-effort: missing modules, a timeout or an exception simply keep the default search.
    """
    tournament_on = (os.environ.get("HARNESS_TOURNAMENT", "0") == "1"
                     or os.environ.get("BOT_HARNESS_TOURNAMENT", "0") == "1")
    if not tournament_on or budget < 60.0:
        return ""
    if spec.hidden_information or spec.family in ("bluff", "deals", "truth", "guardrails"):
        return ""
    try:
        from . import candidates as candidate_module
        from . import selfplay, tournament
    except ImportError:
        notes.append("tournament skipped: modules not available")
        return ""

    here = os.path.dirname(os.path.abspath(__file__))
    strategies = ["search", "greedy"]
    if (spec.simultaneous or spec.family == "read_opponent") and os.path.exists(
            os.path.join(here, "solver_matrix.py")):
        strategies.insert(1, "matrix")
    if os.path.exists(os.path.join(here, "solver_chance.py")):
        strategies.append("chance")

    def candidate(name: str) -> object:
        return candidate_module.Candidate(kind="strategy", name=name, path=None,
                                          meta={"strategy": name})

    errors = [0]

    def play(a, b, game_seed: int) -> float:
        result = selfplay.play_game(model_path, rules,
                                    (a.meta["strategy"], b.meta["strategy"]),
                                    seed=game_seed, max_plies=80, seconds_per_move=0.03)
        if result.error:
            errors[0] += 1
            return 0.5
        if result.winner is None:
            return 0.5
        return 1.0 if result.winner == 0 else 0.0

    try:
        best, stats = tournament.run(
            candidates=[candidate(name) for name in strategies],
            baselines=[candidate("random")],
            play=play,
            budget_s=min(150.0, budget),
            seed=seed,
            games_per_pair=3,
        )
    except Exception as exc:  # noqa: BLE001 - prep must never die for an experiment
        notes.append(f"tournament skipped: {type(exc).__name__}")
        return ""
    total = max(1, int(stats.get("games") or 0))
    if errors[0] * 2 >= total:
        notes.append(f"tournament skipped: model does not self-play ({errors[0]}/{total} errors)")
        return ""
    winner = str(best.meta.get("strategy") or best.name)
    notes.append(f"tournament: {winner} ({stats.get('games', '?')} games, "
                 f"score {best.score:.2f}, timed_out={stats.get('timed_out')})")
    log(kv(tournament=winner, games=stats.get("games"), score=round(best.score, 3),
           timed_out=stats.get("timed_out")))
    return winner


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
