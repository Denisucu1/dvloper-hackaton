# T2 interfaces — frozen before parallel work

Repo: `/home/enica/Desktop/hackaton/adn-t2`, branch `dev/om2-t2`. No pushes to main, no Harbor.
All new behaviour is additive and behind an env flag. Defaults stay off until the tournament/gauntlet
proves a gain.

## Rules for every workstream

- One owner per file. Shared files (`prep.py`, `engine.py`, `search.py`) are integration-only.
- No git commands. The integrator commits.
- stdlib only in the runtime path; no second hosted model; no GPU; tests must be offline.
- Run your own tests (`python3 -m unittest tests.test_X -v`), not the full suite, while others work.
- Report: files, test output tail, measured numbers, and anything the integrator must wire.

## Flags

| Flag | Meaning | Default |
|---|---|---|
| `HARNESS_MATRIX` | matrix-game solver for simultaneous moves | off |
| `HARNESS_CFR` | tabular CFR solver | off |
| `HARNESS_CHANCE` | expectimax over chance nodes | off |
| `HARNESS_TOURNAMENT` | prep candidate tournament | off |
| `HARNESS_TOURNAMENT_K` | candidates kept per prep | 3 |
| `HARNESS_TOURNAMENT_SECONDS` | wall-clock cap for the tournament | 180 |

Read flags with `os.environ.get(...)` inside the module; do not edit `config.py`.

## Candidate

```python
# harness/candidates.py (frozen)
@dataclass
class Candidate:
    kind: str          # "model" | "policy" | "strategy"
    name: str
    path: str | None
    meta: dict
    score: float
```

## GameModel extension (optional, backwards compatible)

`game_model.py` may declare, but **must not require**, these methods:

```python
def chance_actions(self) -> list[tuple[str, float]] | None: ...
def apply_chance(self, outcome: str) -> None: ...
def sample_state(self, rng) -> "GameModel" | None: ...
def info_state(self) -> str | None: ...
```

`_REQUIRED` in `game_model.py` is unchanged: a model without chance support stays valid for
deterministic games.

## Tournament (prep)

```python
# harness/tournament.py
def run(*, candidates, baselines, play, budget_s, seed, mutate=None) -> tuple[Candidate, dict]:
    """Round-robin/gauntlet with paired seeds, timeboxed. Returns (best, stats).
    `play(a, b, seed) -> float` is 1/0.5/0 for a from a's perspective.
    `mutate(best, stats, log) -> Candidate | None` is the optional improvement step.
    Never raises; always returns best-so-far."""
```

## Probes (testing interface)

```python
# scripts/probes/base.py
class Probe:
    name: str; component: str
    def run(self) -> dict: ...   # score, n, p50_ms, p95_ms, notes, details
```

`scripts/report.py` aggregates probes + unittest + bench reports into one ANSI table and a JSON history.

## Integration points (integrator only)

- `prep.py`: after spec/policy/simgen, build candidates, call `tournament.run(...)` when the flag is on,
  persist the winner in `manifest.json`, and always answer `ready` with the best candidate so far.
- `engine.py`: solver order becomes family solver → matrix/chance/CFR (flag-gated) → model search →
  policy → LLM. Every path keeps the legal-action verifier.
- `sandbox.py`: unchanged contract; chance methods are optional and validated only if present.

## Acceptance gates for T2

- 253 existing tests + each workstream's new tests green.
- `bench.py` gate passes on the probe set (`tic_tac_toe`, `tg_connect4` incl. 7x9, `tg_blotto_probe`,
  `tg_chess`, `pig`): zero warnings/timeouts/illegal, chat 1.00, p95 < 7 s.
- Prep budget: worst case < 420 s per game with the tournament on (leaves 180 s margin).
- Flags off by default in the image unless the measured delta is positive.
