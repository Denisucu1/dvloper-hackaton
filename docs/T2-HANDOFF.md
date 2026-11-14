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


## Testing pipeline (new)

One command runs the whole local acceptance suite:

```bash
python3 scripts/pipeline.py                 # offline: compile, tests, probes, protocol, games, prep trace
python3 scripts/pipeline.py --fast          # skip the slow protocol probe inside the report
python3 scripts/pipeline.py --bench         # + real arena matches and the bench gate (uses the gateway)
python3 scripts/pipeline.py --json out.json # machine-readable stage results
```

Measured: **6/6 stages, 8.6 s** offline. Each stage runs even if an earlier one fails; exit 1 on any
failure. `scripts/game_smoke.py` checks variants/rules/observation/frame/legal→normalize/random
playout/result for every game: **31/31 variant checks pass**.

## Games added in this lab

| Game | Family | What it exercises |
|---|---|---|
| `tg_kuhn_probe` | bluff | 3-card Kuhn, 1 or 3 hands; J/Q/K ranks so the CFR parser understands it; live match shows `solver cfr` decisions |
| `tg_dice_probe` | chance | one-die/two-die push-your-luck under a non-practice id |
| `tg_wordle_probe` | guessing | free-text five-letter guesses (`legal_actions` null) for the non-board policy path |
| `tg_auction_probe` | deals | sealed-bid auction, private values, simultaneous bids |

Plus the existing `tg_connect4` (now with the 7x9 connect-5 shape), `tg_blotto_probe`, `tg_chess`,
`tg_bluff`, `tg_deals`, `tg_guardrails`, `tg_simul`, `tg_truth`.


## Performance lab (battle royale)

`scripts/lab.py` measures the decision stack on three fixture simulators (wide 9x7 connect-5, simultaneous
Blotto, chance/dice) and repeats the prep tournament as a progressive battle royale:

```
iter 1: incumbent (greedy)
iter 2: + alpha-beta
iter 3: + family specialist (matrix / expectimax)
```

Every strategy is scored in the same gauntlet with the same seeds, so a new candidate can only raise the
best score. The performance index (mean best score) is stored in `scripts/lab/.history.jsonl`, and the next
run shows the delta. Current values:

| model | winner | score | improvement |
|---|---|---|---|
| connect5 | search | 0.825 | +0.350 |
| blotto | matrix | 1.000 | +0.350 |
| chance_pig | chance | 0.850 | +0.350 |
| **index** | | **0.892** | deterministic across runs |

The pipeline runs it as the `performance` stage (`--verbose` prints every iteration, including the
per-strategy scores). Search/UCT and the solvers are seeded and iteration-capped inside self-play, so the
index is reproducible; a changed index means changed code, not CPU timing.

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
