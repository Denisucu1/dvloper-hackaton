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

Measured: **9/9 stages, ~25 s** offline (slowest: performance, 12 s; the summary prints it). Stages:
compile, unit tests, probes (17), solvers (solver smoke over every arena observation), protocol, game
API, performance lab, prep trace, intelligence (replays -> decision quality). Each stage runs even if an earlier one fails; exit 1 on any failure.
`scripts/game_smoke.py` checks variants/rules/observation/frame/legal→normalize/random playout/result
for every game: **48/48 variant checks pass**. `--json` now carries per-stage seconds, `total_seconds`
and `slowest`.

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
| connect5 | search | 1.000 | +0.700 |
| blotto | matrix | 1.000 | +1.000 |
| chance_pig | chance | 0.850 | +0.350 |
| **index** | | **0.950** | deterministic across runs (0.892 before the wide-root fix) |

The pipeline runs it as the `performance` stage (`--verbose` prints every iteration, including the
per-strategy scores). Search/UCT and the solvers are seeded and iteration-capped inside self-play, so the
index is reproducible; a changed index means changed code, not CPU timing.


## Luck-based games — state after the T2 lab

The chain is now closed end-to-end:

1. **Generation**: simgen asks for `chance_actions`/`apply_chance` when the game rolls, deals or draws;
   `apply()` only enters a pending-chance state and never samples. `ARTIFACT_VERSION=7`.
2. **Validation**: the sandbox rejects half contracts, distributions that do not sum to 1, and
   `apply_chance` that raises; deterministic models are unchanged.
3. **Play**: `solver_chance.py` runs expectimax over chance nodes (147/147 DP-optimal Pig decisions,
   0.7 ms/call); `solver_luck.py` is a text-only fallback for unseen dice games (exact EV, one-die
   boundary 19→roll / 20→hold, 0.06 ms/call, 31 tests).
4. **Selection**: the prep tournament includes the chance strategy; on the chance fixture it wins and
   adds +0.35 to the performance index (0.892, deterministic).
5. **Coverage**: probes include `chance` and `luck`; game smoke includes `tg_dice_probe`.

Leduc (mini_poker in the round robin): `tg_leduc_probe` + `LeducSolver` parse private card, community
card, pot and the betting line, and use pot odds/hand strength. It runs before the generic BluffSolver;
872 fuzzed decisions, 0 declines, 0.1 ms. A real match shows `solver leduc: check/bet`. The parser is
conservative: Kuhn and one-card text are declined, so it cannot misplay the wrong shape.


## T1-shape reconstructions (not official engines)

The official T1 game files are not in the repo or any remote branch. To still exercise the T1 ids and
shapes, four arena games were added as **reconstructions from the brief + our probes**; each module
docstring says so explicitly:

| id | shape | notes |
|---|---|---|
| `connect_four` | 7x9 connect-5 (T1 shape) and 7x6 connect-4 | inherits the shared Connect4 logic |
| `mini_poker` | simplified Leduc, 1 or 3 hands | observation keeps the layout `solver_leduc` parses |
| `colonel_blotto` | 5x100 with reserves, 3x20, 3x4 | unspent troops carry over; legal_actions capped at 50 |
| `wordle_duel` | Wordle duel, 6/4 guesses | free-text actions, hidden secret |

Run them like any game:

```bash
python3 scripts/pipeline.py --fast
python3 scripts/bench.py --label t1-shape --games connect_four,mini_poker,colonel_blotto,wordle_duel   --seeds 1 --best-of 1
```

These results are **indicative**: they test our harness on the right shapes and ids, not against the
organizers' engines. If the official files are released, drop them in `arena-lite/arena/games/` (same ids)
and rerun the same commands.

## Post-audit fixes (T2 freeze)

A read-only audit ranked bottlenecks beyond "the game has luck"; the top ones were fixed and measured:

- **Leduc round-1 equity** (`f71facd`): enumerate the opponent card *and* the community draw. J was
  0.10 (fold) vs a 0.25 pot-odds threshold; now 0.40 (call), K 0.70. Dominated round-2 hands still fold.
  mini_poker over 10 seeds: 50% win / 64.3% score.
- **Null `legal_actions` last resort** (`8671e8d`): format/family-aware action (`unknown`, `offer n`,
  a five-letter word, a concrete example) in both `decide_turn` and `main.turn_fallback`; never empty.
- **Deals offer audit** (`8671e8d`): no longer skipped when the arena enumerates offers; corrected
  offers are normalized or snapped to the nearest legal one.
- **Truth numeric grounding** (`8671e8d`): `2,300` == `2300`; absent number-only answers rejected.
- **Drawn-variant rules** (`8671e8d`): `start.rules` survives `adopt`.
- **Wide alternating boards** (`b6cafe8`): one-ply root ordering + cap 24 (`WIDE_ROOT_ACTIONS`); a
  63-action 9x7 board reaches depth 3 instead of 2. The lab model still reaches only depth 3 at 8 s
  because its `evaluate` costs ~0.29 ms. Real `connect_four` play after the cap: 3/3 wins, depth 6-8,
  p95 4.27 s, gate passed.

Still open after T2: the simgen skip is keyed on family rather than public randomness (blackjack is
classified `deals` and loses its simulator), the prep tournament still self-plays under a different
clock, every turn pays a full LLM call when a solver move is locked, and PIMC/CFR for Leduc.

## Ported from the lab branch (after the audit)

The T2 lab branch added a solver stack and four robustness fixes; they are now part of this branch as
additive files, not a merge. Our own experts stay first (CFR on Kuhn, the equity Leduc solver, chance
expectimax); the ported heuristics only fill gaps and official-text formats:

- `solver_bluff` (Liar's Dice + official Mini Poker/Kuhn text), `solver_guess` (Wordle G/Y/X feedback,
  candidate filtering), `solver_simul` (reserve-aware Blotto), `words_en`, `solver_deals` (bargaining,
  Prisoner's Dilemma, Vickrey/first-price) and `solver_grid` (Connect-K tactical + alpha-beta fallback).
- Deals fixes before adopting: decay inferred from the parsed initial pot (never a hardcoded 100), the
  corrected offer snaps to the nearest legal one, one-shot Prisoner's Dilemma defects, "second-highest"
  auctions are recognised.
- `solver_grid` runs only after the generated model fails. It now understands both the lab's numeric
  board text and the arena's own `CONNECT FOUR - 7x9, connect 5` letter-column form (`drop E` on the
  real 7x9 start), so it is a true fallback when simgen is missing.
- Gateway retry on 429/5xx inside the caller's timeout; prep rejects an incomplete cache; spec retries
  with growing waits while the clock allows; `ready` subtracts the study time already spent.
- Deals audit recomputes the split from the opponent's offer number and never leaves a rejected
  acceptance standing; when the model produces nothing, `DealsSolver` is the last resort before a
  format fallback.
- Pipeline: a new `solvers` stage plays one seeded playout per arena variant and feeds every observation
  to the solver layer (509 proposals, 51 declines, 0 illegal, 92 ms); a `port` probe checks Deals,
  Wordle, Liar's Dice and Connect-K; the summary prints the slowest stage and `--json` carries totals.

Measured after the port: **8/8 pipeline stages, 467 tests, 17 probes at 100/100, performance index
0.950**, solver smoke 0 illegal. Bench `tg_deals` + `tg_liars_dice_probe`, 2 seeds each: **4/4 wins**,
0 warnings/timeouts, p95 2.38 s, **$0.00034/match** (the solver path is also the cheap path); Liar's Dice
shows `solver bluff: 1 4` / `liar` in the warm log.

## Intelligence score (rounds produced by the bot)

`scripts/intelligence.py` reads the newest `bench/results/*/replays/`, rebuilds every game from
`(game_id, variant, seed)` in the arena and walks the timeline again, so every decision is verified
against the real rules rather than self-reported. It prints a per-round table and a composite:

| Component | Weight | What it measures |
|---|---|---|
| legality | 30 | our decisions that normalize onto the legal list; zeroed by an illegal/forfeit/replay error |
| tactics | 35 | agreement/quality against our own expert stack (CFR/Leduc/chance/Deals/Connect-K); Deals offers count when within +-5 of the reference or better; bids that differ are unjudged |
| outcome | 20 | wins + 0.5*draws over the analysed rounds (or the bench score) |
| efficiency | 10 | half p95 latency against the 20 s move clock, half USD/match against the $0.30 cap |
| chat | 5 | reply rate and the share of our lines within 200 characters |

Current run (`bench/results/20261114-222346-port-verify`, 4 rounds, 6 decisions):
**INTELLIGENCE 93.6/100** - legality 100, tactics 83.3, outcome 100, efficiency 94, chat 100
(confidence: medium). It is a transparent proxy, not an IQ test: tactics is agreement with our own
experts and the sample is small, so `--bench` should be run to produce fresh rounds before quoting it.

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
2. `python3 -m unittest discover -s tests` (467) and `python3 scripts/report.py` (100/100, 17 probes).
3. `scripts/bench.py --label t2-merged --games tic_tac_toe,tg_connect4,tg_blotto_probe,tg_chess,pig`.
4. Enable `HARNESS_TOURNAMENT=1` in the Dockerfile only if the bench gate stays green and prep worst case
   stays under ~420 s.
5. Do not enable `HARNESS_MATRIX`/`HARNESS_CHANCE`/`HARNESS_CFR` globally: the tournament selects the
   winner per game; forcing a strategy is for experiments.
