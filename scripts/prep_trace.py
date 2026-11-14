#!/usr/bin/env python3
"""Visual trace of the 10-minute prep: stages, model calls, validation, artifacts, and the first move.

The arena gives a container 600 s to turn one game's rules into a specialist. This script runs the real
`harness.prep` pipeline and prints what happened, in order, with timings:

    python3 scripts/prep_trace.py                      # offline, scripted model, no key, ~1 s
    python3 scripts/prep_trace.py --real --game pig    # the real gateway, real arena prep message
    python3 scripts/prep_trace.py --keep               # keep the memory dir and print its path

Offline mode is deterministic on purpose: a fake gateway returns a spec, a `choose()` policy and a
`class Model` simulator for a synthetic 2-variant game, so the trace also proves the variant handling
(3 numbers vs 5 numbers) and the validation gate without spending a token.

What it shows, in this order:
1. the prep input (variants, samples, clock);
2. every gateway call (prompt size, reply size, duration, USD);
3. the validation verdicts recorded in the manifest (policy and simulator, per variant);
4. the artifacts written to /memory and what each one is for;
5. one real `Engine.propose()` on a sample turn, i.e. prep -> play: the generated simulator is searched
   by our code and the move comes back.

Exit code is non-zero if the pipeline produced no usable specialist (no spec or no generated code).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ARENA = os.path.join(ROOT, "arena-lite")


# --------------------------------------------------------------------------- scripted (offline) world
SPEC_JSON = {
    "family": "think_ahead",
    "action_format": "pick <n>",
    "action_examples": ["pick 1", "pick 2", "pick 5"],
    "winning_idea": "Take the highest remaining number before the opponent can.",
    "hidden_information": False,
    "simultaneous": False,
    "scoring": "the higher number wins; equal numbers are a draw",
    "speech_rules": "",
    "forbidden_subjects": [],
    "traps": ["picking a number that is already gone is an illegal move"],
    "notes": "Two variants: 3 numbers and 5 numbers. The open numbers are listed in the observation.",
    "playbook": [
        "Read the open numbers from the observation; never trust a remembered board.",
        "Take the highest open number: it can never lose.",
        "Keep the chat consistent with the move you actually play.",
    ],
}

POLICY_CODE = '''def choose(observation, legal_actions, seat):
    """Deliberately naive: the sandbox only checks that it is legal on every sample."""
    if legal_actions:
        return legal_actions[-1]
    return "pick 1"
'''

MODEL_CODE = r'''import re

OPEN = re.compile(r"Open numbers:\s*(.*)", re.I)


class Model:
    """Simulator for the synthetic NUMBER PICK game: parse the open numbers, one pick each, higher wins."""

    def __init__(self, rules, seat):
        self.seat = int(seat)
        self.open = []
        self.picks = []

    def update(self, observation):
        match = OPEN.search(observation or "")
        self.open = [int(n) for n in re.findall(r"\d+", match.group(1))] if match else []
        self.picks = []

    def legal_actions(self, seat):
        return [f"pick {n}" for n in self.open]

    def apply(self, seat, action):
        number = int(str(action).split()[-1])
        if number not in self.open:
            raise ValueError("illegal pick")
        self.open.remove(number)
        self.picks.append((int(seat), number))

    def is_terminal(self):
        return len(self.picks) >= 2

    def to_move(self):
        if self.is_terminal():
            return []
        return [0 if not self.picks else 1]

    def score(self, seat):
        chosen = dict(self.picks)
        if len(chosen) < 2:
            return 0.5
        mine, theirs = chosen[seat], chosen[1 - seat]
        return 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        other = Model("", self.seat)
        other.open = list(self.open)
        other.picks = list(self.picks)
        return other

    def key(self):
        return ",".join(map(str, self.open)) + "/" + ",".join(f"{s}:{n}" for s, n in self.picks)
'''


def synthetic_message(numbers: tuple[int, ...] = (3, 5)) -> dict:
    variants = []
    for count in numbers:
        label = f"{count} numbers"
        open_numbers = ", ".join(str(i) for i in range(1, count + 1))
        observation = (
            f"NUMBER PICK - {label}. You are seat 0.\n"
            f"Open numbers: {open_numbers}\n"
            f'It is YOUR turn: "pick <n>" with n one of the open numbers.'
        )
        variants.append({
            "label": label,
            "rules": (
                f"NUMBER PICK - {label}. Two players alternate picking one number from 1 to {count}.\n"
                'A move is "pick <n>" for an open number. The higher number wins; equal is a draw.'
            ),
            "sample_turn": {
                "type": "turn", "game_index": 1, "turn": 1,
                "observation": observation,
                "legal_actions": [f"pick {i}" for i in range(1, count + 1)],
                "seconds": 20, "chat": [], "seat": 0,
            },
            "size": count,
        })
    return {
        "type": "prep", "game_id": "number_pick", "game_name": "Number Pick",
        "variants": variants, "seconds": 600, "you": "adn", "study_seconds": 30, "move_seconds": 20,
        "llm": {"base_url": "scripted://offline", "api_key": "sk-offline", "model": "haiku"},
    }


class ScriptedGateway:
    """Implements just enough of Gateway for prep, with zero network access."""

    def __init__(self) -> None:
        self.calls = 0
        self.spend = 0.0
        self.cap = 0.75
        self.input_tokens = 0
        self.output_tokens = 0
        self.failures = 0
        self.base_url = "scripted://offline"
        self.api_key = "sk-offline"
        self.model = "haiku"

    def configure(self, info: dict | None, cap: float | None = None) -> None:
        if cap is not None:
            self.cap = float(cap)

    @property
    def available(self) -> bool:
        return True

    @property
    def affordable(self) -> bool:
        return True

    def ask_json(self, system: str, user: str, **kwargs):
        self.calls += 1
        self.spend += 0.0001
        return dict(SPEC_JSON)

    def ask(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        self.spend += 0.0001
        return MODEL_CODE if "SIMULATES" in system else POLICY_CODE

    def key_info(self) -> dict:
        return {}

    def summary(self) -> dict:
        return {"calls": self.calls, "spend_usd": round(self.spend, 5), "cap_usd": self.cap,
                "input_tokens": 0, "output_tokens": 0, "failures": 0}


# --------------------------------------------------------------------------- trace plumbing
class Trace:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.t0 = time.perf_counter()

    def mark(self, stage: str, detail: str, ms: float = 0.0) -> None:
        self.events.append({"t": time.perf_counter() - self.t0, "stage": stage,
                            "detail": detail, "ms": float(ms)})


class TracingGateway:
    """Delegates to the real (or scripted) gateway and records every call with its cost and duration."""

    def __init__(self, inner, trace: Trace) -> None:
        self._inner = inner
        self._trace = trace

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def configure(self, info: dict | None, cap: float | None = None) -> None:
        self._inner.configure(info, cap)
        key = str((info or {}).get("api_key", "?"))
        self._trace.mark("gateway", f"key configured (…{key[-4:]}, cap ${cap if cap is not None else '?'})")

    def ask_json(self, system: str, user: str, **kwargs):
        label = "spec extraction"
        return self._call(label, self._inner.ask_json, system, user, **kwargs)

    def ask(self, system: str, user: str, **kwargs):
        if "SIMULATES" in system:
            label = "simulator generation"
        elif "choose(observation" in system or "module that plays" in system:
            label = "policy generation"
        else:
            label = "model call"
        return self._call(label, self._inner.ask, system, user, **kwargs)

    def _call(self, label: str, fn, system: str, user: str, **kwargs):
        before = float(getattr(self._inner, "spend", 0.0))
        started = time.perf_counter()
        try:
            out = fn(system, user, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the trace must survive a gateway failure
            self._trace.mark("llm", f"{label}: EXCEPTION {type(exc).__name__}",
                             (time.perf_counter() - started) * 1000)
            return ""
        elapsed = (time.perf_counter() - started) * 1000
        after = float(getattr(self._inner, "spend", 0.0))
        size = len(out) if isinstance(out, str) else len(json.dumps(out or {}))
        verdict = "EMPTY" if not out else f"{size:,} B out"
        self._trace.mark("llm", f"{label}: {len(system) + len(user):,} B in -> {verdict}, "
                                f"${after - before:.5f}", elapsed)
        return out


# --------------------------------------------------------------------------- report
STAGE_STYLE = {
    "input": "INPUT",
    "gateway": "GATEWAY",
    "llm": "MODEL",
    "artifacts": "DISK",
    "validation": "CHECK",
    "engine": "PLAY",
    "ready": "READY",
}


def _line(char: str = "-", width: int = 96) -> str:
    return char * width


def _print_timeline(events: list[dict]) -> None:
    print(_line("="))
    print("PREP TIMELINE")
    print(_line("="))
    for event in events:
        stage = STAGE_STYLE.get(event["stage"], event["stage"].upper())
        took = f"{event['ms']:8.1f} ms" if event["ms"] else " " * 11
        print(f"  +{event['t']:6.3f}s  {stage:<7} {took}  {event['detail']}")


def _artifact_rows(directory: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    for path in sorted(directory.glob("*")):
        if path.name == "__pycache__" or path.is_dir():
            continue
        text = ""
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
        size = len(text.encode("utf-8", "replace"))
        summary = ""
        if path.suffix == ".json":
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    keys = [k for k in ("family", "genpolicy", "genmodel", "artifact_version",
                                        "fingerprint", "shots", "turns") if k in data]
                    summary = ", ".join(f"{k}={str(data[k])[:24]}" for k in keys)
                    if path.name == "manifest.json":
                        summary = f"family={data.get('family')}; genpolicy={bool(data.get('genpolicy'))}; " \
                                  f"genmodel={bool(data.get('genmodel'))}"
            except ValueError:
                summary = "invalid json"
        elif path.suffix == ".py":
            summary = f"{len(text.splitlines())} lines, " + \
                      ("class Model" if "class Model" in text else "def choose" if "def choose" in text else "code")
        elif path.name.endswith("playbook.txt"):
            summary = f"{len(text.splitlines())} lines: {text.splitlines()[0][:60] if text else 'empty'}"
        elif path.name.endswith("rules.txt"):
            summary = f"{len(text.splitlines())} lines of rules"
        elif path.name == "samples.json":
            summary = "sample turns used as validation ground truth"
        rows.append((path.name, f"{size:,} B", summary, "per game, keyed by rules hash"))
    return rows


def _print_artifacts(directory: Path, trace: Trace) -> list[str]:
    rows = _artifact_rows(directory)
    print()
    print(_line("="))
    print(f"ARTIFACTS  {directory}")
    print(_line("="))
    print(f"  {'file':<40} {'size':>9}  what it is")
    for name, size, summary, _ in rows:
        print(f"  {name:<40} {size:>9}  {summary}")
    trace.mark("artifacts", f"{len(rows)} files written under /memory/<game_id>/")
    return [name for name, _, _, _ in rows]


def _print_validation(manifest: dict, samples: list[dict], trace: Trace) -> dict:
    notes = manifest.get("notes") or []
    print()
    print(_line("="))
    print("VALIDATION (what the sandbox accepted or rejected)")
    print(_line("="))
    for note in notes:
        print(f"  - {note}")
        if "ok on" in note:
            trace.mark("validation", note)
    variants = [str(sample.get("variant") or "?") for sample in samples]
    print()
    print(f"  samples         : {len(samples)} ({', '.join(variants)}) - validated one by one")
    print(f"  spec            : {'yes' if manifest.get('family') else 'NO'}")
    print(f"  generated policy: {'yes' if manifest.get('genpolicy') else 'no'}")
    print(f"  generated model : {'yes' if manifest.get('genmodel') else 'NO'}")
    return {"genpolicy": bool(manifest.get("genpolicy")), "genmodel": bool(manifest.get("genmodel"))}


def _print_engine(spec, samples: list[dict], trace: Trace) -> tuple[str | None, str]:
    from harness import engine as engine_module

    if not samples:
        return None, "none"
    sample = samples[-1]
    legal = sample.get("legal_actions")
    print()
    print(_line("="))
    print("PREP -> PLAY (the generated simulator, searched by our own code)")
    print(_line("="))
    print(f"  variant: {sample.get('variant') or '?'}   observation: "
          f"{str(sample.get('observation', '')).splitlines()[0][:70]}")
    started = time.perf_counter()
    proposal, intent = engine_module.Engine(spec).propose(
        observation=str(sample.get("observation", "")),
        legal=legal,
        seat=int(sample.get("seat", 0) or 0),
        turn=int(sample.get("turn", 0) or 0),
        timeout=2.0,
    )
    elapsed = (time.perf_counter() - started) * 1000
    print(f"  engine proposal : {proposal!r}")
    print(f"  intent          : {intent[:100]}")
    print(f"  legal?          : {proposal in legal if legal else 'n/a'}")
    source = _proposal_source(intent)
    print(f"  source          : {source}")
    trace.mark("engine", f"Engine.propose -> {proposal!r} in {elapsed:.0f} ms ({source})", elapsed)
    return proposal, source


def _proposal_source(intent: str) -> str:
    if intent.startswith("generated model search"):
        return "generated model (our search)"
    if intent.startswith("generated policy"):
        return "generated policy"
    if " solver:" in intent:
        return "family solver"
    return "model / fallback"


# --------------------------------------------------------------------------- real arena input
def read_env(path: str) -> dict:
    values = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    except OSError:
        pass
    return values


def arena_prep_message(game: str, env_file: str) -> dict:
    """Ask arena-lite for the exact `prep` message it would send, without starting a bot."""
    completed = __import__("subprocess").run(
        [sys.executable, "-m", "arena", "prep", "--bot", "cmd:true", "--game", game,
         "--env-file", env_file, "--show"],
        cwd=ARENA, capture_output=True, text=True, timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"arena prep --show failed: {completed.stderr.strip()[:300]}")
    return json.loads(completed.stdout)


# --------------------------------------------------------------------------- main flow
def run_trace(*, real: bool = False, game: str = "tic_tac_toe", env_file: str | None = None,
              keep: bool = False, verbose: bool = False) -> dict:
    directory_root = Path(tempfile.mkdtemp(prefix="adn-prep-trace-"))
    previous_memory = os.environ.get("MEMORY_DIR")
    os.environ["MEMORY_DIR"] = str(directory_root)

    from harness import config, prep

    previous_config = config.MEMORY_DIR
    config.MEMORY_DIR = str(directory_root)

    trace = Trace()
    env_file = env_file or os.path.join(ROOT, "dev.env")

    if real:
        msg = arena_prep_message(game, env_file)
        # `prep --show` redacts the key on purpose, so a local trace uses the dev key from dev.env.
        credentials = read_env(env_file)
        llm_info = {
            "base_url": credentials.get("LLM_BASE_URL", ""),
            "api_key": credentials.get("LLM_API_KEY", ""),
            "model": credentials.get("LLM_MODEL", "haiku"),
        }
        if not llm_info["base_url"] or not llm_info["api_key"]:
            raise RuntimeError(f"no LLM_BASE_URL / LLM_API_KEY in {env_file}")
        trace.mark("input", f"REAL arena prep · {msg['game_id']} · {len(msg['variants'])} variant(s) · "
                            f"{msg.get('seconds', 600)} s clock · {llm_info['base_url']} · {llm_info['model']}")
        from harness.llm import Gateway

        inner = Gateway()
        inner.configure(llm_info)
    else:
        msg = synthetic_message()
        llm_info = msg["llm"]
        trace.mark("input", f"scripted prep · {msg['game_id']} · {len(msg['variants'])} variants · "
                            f"{msg.get('seconds', 600)} s clock · no network")
        inner = ScriptedGateway()

    gateway = TracingGateway(inner, trace)
    gateway.configure(llm_info, cap=config.PREP_BUDGET)

    printed = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stderr(printed):
        result = prep.run_prep(msg, gateway)
    elapsed = time.perf_counter() - started
    trace.mark("ready", f"prep finished in {elapsed:.1f}s and wrote manifest.json "
                        f"({'cached' if result.cached else 'fresh'})", elapsed * 1000)

    from harness import memory

    directory = memory.game_dir(msg["game_id"])
    artifacts = _print_artifacts(directory, trace)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8")) \
        if (directory / "manifest.json").exists() else {}
    samples = json.loads((directory / "samples.json").read_text(encoding="utf-8")).get("turns", []) \
        if (directory / "samples.json").exists() else []
    validation = _print_validation(manifest, samples, trace)
    proposal, covered_by = _print_engine(result.spec, samples, trace)

    print()
    _print_timeline(trace.events)
    print()
    summary = gateway.summary()
    print(_line("="))
    print("SUMMARY")
    print(_line("="))
    print(f"  total wall time : {elapsed:.1f} s of the {msg.get('seconds', 600)} s clock")
    print(f"  model calls     : {summary.get('calls', 0)}")
    print(f"  spend (prep key): ${summary.get('spend_usd', 0.0):.5f} of ${summary.get('cap_usd', 0.75):.2f}")
    print(f"  specialist      : spec={'yes' if result.spec.family else 'no'}, "
          f"policy={'yes' if validation['genpolicy'] else 'no'}, "
          f"model={'yes' if validation['genmodel'] else 'no'}")
    print(f"  first move      : {proposal!r} via {covered_by}")
    print(f"  memory dir      : {directory_root}")
    ok = bool(result.spec.family) and (
        validation["genmodel"] or validation["genpolicy"] or covered_by == "family solver"
    )

    if verbose and printed.getvalue().strip():
        print()
        print(_line("="))
        print("HARNESS LOG (stderr, captured)")
        print(_line("="))
        for line in printed.getvalue().strip().splitlines():
            print(f"  {line}")

    if not keep:
        shutil.rmtree(directory_root, ignore_errors=True)
        print("  (memory dir removed; use --keep to inspect it)")
    else:
        print("  (kept for inspection)")

    # Never leak the trace's temporary memory into the caller (tests run this in-process).
    config.MEMORY_DIR = previous_config
    if previous_memory is None:
        os.environ.pop("MEMORY_DIR", None)
    else:
        os.environ["MEMORY_DIR"] = previous_memory

    return {
        "ok": ok,
        "events": trace.events,
        "artifacts": artifacts,
        "manifest": manifest,
        "validation": validation,
        "engine_action": proposal,
        "total_s": round(elapsed, 2),
        "calls": summary.get("calls", 0),
        "spend_usd": summary.get("spend_usd", 0.0),
        "memory_dir": str(directory_root) if keep else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualise the 10-minute prep pipeline.")
    parser.add_argument("--real", action="store_true", help="use the real gateway (costs a fraction of a cent)")
    parser.add_argument("--game", default="tic_tac_toe", help="arena-lite game for --real")
    parser.add_argument("--env-file", default=os.path.join(ROOT, "dev.env"))
    parser.add_argument("--keep", action="store_true", help="keep the /memory folder")
    parser.add_argument("--verbose", action="store_true", help="also print the harness stderr log")
    args = parser.parse_args()

    try:
        result = run_trace(real=args.real, game=args.game, env_file=args.env_file,
                           keep=args.keep, verbose=args.verbose)
    except Exception as exc:  # noqa: BLE001 - a trace tool should explain, not dump a traceback
        print(f"prep trace failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not result["ok"]:
        print("\nPREP INCOMPLETE: the container would play with a degraded specialist.", file=sys.stderr)
        return 1
    print("\nPREP OK: a specialist was built and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
