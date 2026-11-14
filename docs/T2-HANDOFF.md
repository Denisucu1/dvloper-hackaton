# T2 lab — what landed, how to enable it, what is measured

Branch `dev/om2-t2` in `/home/enica/Desktop/hackaton/adn-t2` (clone; main untouched, no Harbor pushes).
Baseline before this work: 253 tests. After: **343 tests** + a 14-component probe report at 100/100.

## What exists now

| Area | Module | State |
|---|---|---|
| Testing interface | `scripts/report.py`, `scripts/probes/` | 14 probes, ANSI scorecard, history + deltas, `--fast/--watch/--json`; 0.44 s |
| Candidate model | `harness/candidates.py` | frozen Candidate dataclass |
| Tournament | `harness/tournament.py` | round-robin + baseline gauntlet, paired seeds, `games_per_pair`, Wilson CI, deadline, one mutation |
| Self-play referee | `harness/selfplay.py` | strategies on one validated model; search/matrix/chance/greedy/random; deadlines; safe draws |
| Matrix solver | `harness/solver_matrix.py` | k×k payoff matrix + regret matching; RPS 0.32/0.335/0.345; dominated 0.0003; k=48 in 19 ms |
| CFR | `harness/cfr.py` | generic tree + CFR+; Kuhn exploitability 2.4e-4 at 500 iters, 1.1e-5 at 20k |
| Chance | `harness/solver_chance.py` + optional `game_model` methods | expectimax; 147/147 DP-optimal Pig decisions; 0.7 ms/call |
| Chat gate | `harness/chatpolicy.py` | homoglyph/accent/small-cap/Hangul-fold skeleton + sentence redaction; leet injection tagged |
| Truth | `harness/decide.py`, `actions.py` | answer must be grounded in the document; abstain with a legal `unknown` |
| Prep tournament | `harness/prep.py` | hook after simgen; writes the winning strategy into `manifest.json` |
| Sandbox routing | `harness/sandbox.py` | child runs model with `strategy=search|matrix|chance|auto`; any failure falls back to alpha-beta |
| Engine routing | `harness/engine.py` | honors the manifest strategy; flags; CFR expert appended when enabled |

## Flags

| Flag | Effect | Default |
|---|---|---|
| `HARNESS_TOURNAMENT` | run the prep tournament and play the winning strategy | off |
| `HARNESS_MATRIX` | force matrix strategy for simultaneous/read-opponent | off |
| `HARNESS_CHANCE` | auto-detect chance methods and use expectimax | off |
| `HARNESS_CFR` | append the CFR solver (Kuhn-only today) | off |

`BOT_HARNESS_*` works in local arena-lite practice (`BOT_*` is the only custom namespace forwarded).

Auto mode (no env needed): the prep tournament runs by itself for `read_opponent`, `guessing` and
`unknown` families, and is skipped for `think_ahead` (alpha-beta is already proven there), hidden
information, bluff/deals/truth/guardrails. A non-search winner must beat search by >= 0.55 over 20 paired
games, otherwise search is kept. Forcing every family is still possible with `HARNESS_TOURNAMENT=1`.

## Measured on the probes (real generated models)

- `tg_blotto_probe` (simultaneous, 28 allocations): prep 18 s; simulator validated; tournament 60 games,
  winner **matrix** 0.77 in 11.7 s; real match vs a uniform allocator won 1.5–0.5.
- Matrix vs search head-to-head on the same model: **0.512** (a tie within noise); matrix vs random
  0.575, search vs random 0.713. The tournament is the arbiter, not a hand-written preference.
- Connect5 7x9: prep 19.5 s, simulator validated, search picks the centre at depth 7 and blocks at depth 8.
- Chance: one-die Pig decisions match exact DP 147/147; two-dice smoke passes; no-chance models fall back.
- Chat: all six unicode evasions blocked, leet injection tagged, benign numbers untouched, zero false
  positives; Truth abstains on invented numbers and on answers copied from the opponent's chat.
- Probe report: 14 components, 180 observations, 100.0/100, 0.44 s.

## Known limits / next

- Tournament self-play needs the generated `Model` to be usable from `__init__` (no observation text).
  Models that only work after `update(observation)` are detected (too many self-play errors) and skipped.
- Hidden-information families are skipped by the tournament by design (no sound referee).
- CFR is Kuhn-only; Leduc (mini_poker) needs a game-specific tree/parser. Template exists.
- Matrix and joint MCTS are close on Blotto; keep the tournament selection but add a margin guard before
  switching away from search (planned: winner must beat search by ≥0.55 over paired seeds).
- PIMC/ISMCTS for hidden information remains post-T2.

## Integration order for the main line

1. Merge the branch (no conflicts expected: new files + small edits in prep/engine/chatpolicy/decide).
2. `python3 -m unittest discover -s tests` (343) and `python3 scripts/report.py` (100/100).
3. `scripts/bench.py --label t2-merged --games tic_tac_toe,tg_connect4,tg_blotto_probe,tg_chess,pig`.
4. Enable `HARNESS_TOURNAMENT=1` in the Dockerfile only if the bench gate stays green and prep worst case
   stays under ~420 s.
5. Do not enable `HARNESS_MATRIX`/`HARNESS_CHANCE`/`HARNESS_CFR` globally: the tournament selects the
   winner per game; forcing a strategy is for experiments.
