"""Prep step where the model writes a *simulator* of the game instead of a policy.

The design we want the judges to read is split in two: the model supplies the rules engine during prep,
our own `search.py` supplies the move during play. A generated `choose()` policy can only imitate the
one line of play the model imagined; a generated `class Model` lets negamax/alpha-beta try lines the
model never wrote down, and every one of them is checked against real sample turns.

So this module asks exactly one thing: implement the GameModel contract for this game. It reuses the
shape of `prep._build_genpolicy` on purpose - strict system prompt, strip code, validate in the
sandbox, send the concrete validation error back once, then give up cleanly. A broken simulator must
cost us nothing: the caller keeps its deterministic solvers and the LLM.

Budget discipline is part of the contract too: prep shares one clock, so no single model call may use
more than half of `timeout`, and a retry is pointless once ~20 s of the budget is left.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from . import memory, sandbox
from .game_model import MODEL_SUFFIX
from .llm import Gateway
from .log import kv, log
from .spec import Spec

# A second attempt needs enough room to both answer and validate; below this the budget is spent.
_RETRY_FLOOR = 20.0

_CONTRACT = '''class Model:
    def __init__(self, rules: str, seat: int): ...   # seat is MY seat; store it
    def update(self, observation: str) -> None: ...  # parse the exact arena turn text into the state
    def legal_actions(self, seat: int) -> list[str] | None: ...  # exact action strings; None only if >200
    def apply(self, seat: int, action: str) -> None: ...
    def is_terminal(self) -> bool: ...
    def to_move(self) -> list[int]: ...              # seats that must move now
    def score(self, seat: int) -> float: ...         # terminal: 1 win, 0.5 draw, 0 loss
    def evaluate(self, seat: int) -> float: ...      # non-terminal heuristic, 0.5 neutral
    def move_score(self, action: str) -> float: ...  # optional: higher = try first
    def clone(self) -> "Model": ...                  # deep enough copy; search mutates clones
    def key(self) -> str: ...                        # optional transposition key'''

_SYSTEM = """Write ONE standalone Python 3 module that SIMULATES the given game, so a separate search
engine can play it. You write the rules engine; our code picks the move.

The module must define exactly this interface (keep the names and signatures):
""" + _CONTRACT + """

ONE GAME, SEVERAL VARIANTS - this is where most simulators fail:
The same game_id can arrive in ONE prep with several variants: different board sizes, limits, target
scores or scoring twists. The samples below can come from DIFFERENT variants and are shown with their
variant label and settings. Therefore:
- update(observation) must derive the board size, limits and targets DYNAMICALLY from the observation
  text itself (count the grid's rows and columns, read "1-1000", "first to 60", the target score, the
  number of dice, ...), using `rules` only as a fallback when the text is silent.
- NEVER hardcode a size or limit taken from one sample. A model that assumes a 3x3 board will be rejected
  by the 5x5 sample, and vice versa.
- legal_actions must be correct for EVERY variant, and must exclude occupied/illegal cells by parsing
  the observation, not by replaying a fixed opening.

The engine constructs Model(rules, MY_SEAT), calls update(observation) once per turn, then repeatedly
clones the model and applies actions while searching. Therefore:
- update() must rebuild the whole state from the observation text alone; it is the only input.
- clone() must copy every mutable field; the search mutates clones and never the live object.
- legal_actions(seat) must return the arena's exact action strings (any order), or [] when it is not
  that seat's turn or the game is over. The search needs the FULL list (up to ~200 actions); return None
  only when the game is genuinely wider than that. When a sample lists legal_actions, reproduce that
  list exactly, even if it is long: validation compares the sets.
- apply(seat, action) must accept exactly those strings and advance one ply.
- is_terminal() returns a bool. to_move() returns a list of seat ints (one seat for alternating games,
  two for simultaneous games, [] when the game is over).
- score(seat) is only called at terminal states: 1.0 win, 0.5 draw, 0.0 loss from that seat's view.
- evaluate(seat) is a real heuristic in [0, 1] (0.5 neutral), NOT a constant: count how close each side
  is to completing a line (e.g. open lines, stones in a row, threats). The searcher calls it at the
  rollout horizon and as a root prior, so a constant value makes the search blind.
- move_score(action) is optional and recommended: a higher number for moves that complete a line, block
  the opponent's line, or build a double threat. The searcher tries those moves first.
- key() is optional: a string that uniquely identifies the position.

Rules for the code:
- stdlib imports only: math, random, re, itertools, collections, heapq.
- No prints, no file/network/process access, no other modules. The source is checked statically and
  rejected if it mentions banned tools.
- Handle EVERY sample turn shown below; parsing must rely on the observation text.
- No randomness: the search needs firm values. For dice or hidden numbers use expected values and only
  facts the observation actually states; never guess an unknown secret.
- Keep it under ~200 lines. apply() and clone() must stay under 1 ms.

Output raw Python code ONLY, starting with the imports and `class Model:` - no prose, no markdown
fences."""

_USER = """GAME ID: {game_id}

RULES (all variants):
{rules}

WHAT WE ALREADY KNOW ABOUT THE GAME:
{context}

SAMPLE TURNS (the model must handle every one):
{samples}

Write the module now."""

_FIX = """Your previous module failed validation with: {error}
Fix that error and output the full corrected Python module only. It must still handle every sample turn
shown above and reproduce each sample's legal_actions exactly."""

_FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _sample_blob(samples: list[dict]) -> str:
    """The arena's own turn texts: the only ground truth the simulated parser is scored against."""
    chunks = []
    for index, sample in enumerate(samples):
        legal = sample.get("legal_actions")
        legal_text = "None" if legal is None else repr([str(action) for action in legal])
        variant = str(sample.get("variant") or "").strip()
        settings = sample.get("settings") or {}
        header = f"--- Sample {index + 1}"
        if variant:
            header += f" | VARIANT: {variant}"
        header += f" | seat={sample.get('seat', 0)}, turn={sample.get('turn', '?')}"
        if settings:
            header += f" | settings={settings}"
        header += " ---"
        chunks.append(
            f"{header}\n{str(sample.get('observation', ''))[:1800]}\nlegal_actions = {legal_text}"
        )
    return "\n\n".join(chunks)


def _context(spec: Spec | None) -> str:
    """What prep already understood, in the compact form spec.py already uses for prompts."""
    block = getattr(spec, "prompt_block", None)
    text = ""
    if callable(block):
        try:
            text = str(block(1200) or "").strip()
        except Exception:  # noqa: BLE001 - a spec is a hint, never a reason to fail prep
            text = ""
    if not text:
        fields = []
        for name in ("family", "action_format", "winning_idea", "scoring", "notes"):
            value = str(getattr(spec, name, "") or "").strip()
            if value:
                fields.append(f"{name}: {value}")
        text = "\n".join(fields)
    return text or "(no structured spec)"


def _strip_code(text: str) -> str:
    """Keep just the module, tolerating fences and a prose preamble; require `class Model`."""
    text = (text or "").strip()
    if not text:
        return ""
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    else:
        text = re.sub(r"^```(?:python|py|python3)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    if "class Model" not in text:
        return ""
    # Models sometimes narrate before the code. Drop the prose, but keep everything that is code:
    # module-level helpers are common (geometry tables, parsers) and must never be cut away.
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "#", "@", '"""', "'''", "class Model")):
            start = index
            break
    return "\n".join(lines[start:]).strip()


def _left(started: float, timeout: float) -> float:
    return float(timeout) - (time.time() - started)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def build_model(
    *,
    gateway: Gateway,
    directory: Path,
    fingerprint: str,
    game_id: str,
    rules: str,
    samples: list[dict],
    spec: Spec | None,
    notes: list[str],
    timeout: float = 120.0,
) -> str | None:
    """Ask the model for a simulator, validate it, retry once on a concrete error, return its path.

    `None` means "no usable simulator"; that is a normal outcome, not an error. The caller then plays
    with a deterministic solver or the LLM, so a refused model costs correctness, never a move.
    """
    if not samples:
        notes.append("simgen skipped: no sample_turn to validate against")
        return None
    if timeout <= 0:
        notes.append("simgen skipped: no time budget")
        return None

    started = time.time()
    directory = Path(directory)
    path = directory / f"{fingerprint}{MODEL_SUFFIX}"
    user = _USER.format(
        game_id=game_id,
        rules=str(rules or "")[:6000],
        context=_context(spec),
        samples=_sample_blob(samples),
    )
    error = "no standalone Python module with class Model"
    last_source = ""

    for attempt in (1, 2):
        left = _left(started, timeout)
        if attempt == 2 and left < _RETRY_FLOOR:
            notes.append(f"simgen attempt 2 skipped: {max(0, int(left))}s left")
            break
        if left <= 2.0:
            notes.append(f"simgen attempt {attempt} skipped: no time left")
            break

        # Hard ceiling: half the total budget, so a slow answer cannot eat the validation window.
        ask_timeout = min(float(timeout) / 2.0, max(0.5, left - 5.0))
        if attempt == 1:
            prompt = user
        else:
            prev_block = f"\n\nYOUR PREVIOUS MODULE THAT FAILED:\n```python\n{last_source[:4500]}\n```" if last_source else ""
            prompt = f"{user}{prev_block}\n\n{_FIX.format(error=error[:600])}"
        try:
            text = gateway.ask(_SYSTEM, prompt, max_tokens=4000, timeout=ask_timeout, temperature=0.2)
        except Exception as exc:  # noqa: BLE001 - prep must survive a gateway hiccup and try once more
            error = f"gateway error {type(exc).__name__}"
            notes.append(f"simgen attempt {attempt}: {error}")
            log(kv(simgen="attempt", n=attempt, error=type(exc).__name__))
            continue

        source = _strip_code(str(text or ""))
        if not source:
            error = "no standalone Python module with class Model"
            notes.append(f"simgen attempt {attempt}: {error}")
            log(kv(simgen="attempt", n=attempt, error="no code"))
            continue

        last_source = source
        memory.atomic_write_text(path, source)
        validate_timeout = max(2.0, min(30.0, _left(started, timeout) - 1.0))
        ok, reason = sandbox.validate_model(str(path), samples, rules, timeout=validate_timeout)
        reason = " ".join(str(reason).split())[:160]
        notes.append(f"simgen attempt {attempt}: {reason}")
        log(kv(simgen="attempt", n=attempt, ok=ok, reason=reason))
        if ok:
            return str(path)
        error = reason

    _unlink(path)
    return None
