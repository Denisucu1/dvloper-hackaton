# Bottleneck audit — beyond luck games

Read-only audit of the T2 lab (`dev/om2-t2`), based on code inspection, warm logs and the five bench
reports. Ordered by expected points at risk, not by how interesting the fix is. Status is updated as the
T2 fixes land.

| # | Bottleneck | Evidence | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| 1 | Hidden-info play is one-ply; Leduc round-1 equity ignores the community card | `solver_leduc._equity`; mini_poker J vs bet: reported 0.10, true ~0.40, threshold 0.25; bench mini_poker 0-1 | Bluff/Leduc match points | enumerate opponent card + community draw -> J=0.40 (call), K=0.70; dominated round-2 hands still fold; 10 seeds mini_poker 50% win / 64.3% score; PIMC/CFR still post-T2 | **fixed `f71facd`** |
| 2 | No last-resort action when `legal_actions` is null | `actions.choose_fallback` -> None; `main.turn_fallback` -> `action:""`; `decide` -> "pass" | −1.25 robustness per game + match loss | family/format-aware fallback (`unknown`, `offer n`, five-letter word, concrete example); never empty; 14 new tests | **fixed `8671e8d`** |
| 3 | Missing `sample_turn` silently removes generated code | prep only samples variants with sample_turn; wordle_duel 6 s prep, genpolicy/genmodel false; blotto 2/3 variants no sample | strength on free-text/wide games | synthesize samples from rules/action_examples, or built-in family policy | after T2 |
| 4 | Wide alternating boards stay at depth 2 | `route_for` sends only simultaneous nodes to MCTS; 63-action board depth 2 at 1.5-8 s | Think-ahead strength | root one-ply ordering + cap 24 (`WIDE_ROOT_ACTIONS`); 63-action 9x7 depth 2 -> 3 at 0.6 s; lab model depth 2 -> 3 at 8 s; ties deterministic | **fixed `b6cafe8`** |
| 5 | Prep tournament measures a different clock than play | selfplay search 30 s / depth 2, real play 1.5 s / depth 8; forced chess prep 150 s vs 23 s | wrong strategy choice + prep time | shared PLAY_PROFILE; cap tournament 60-90 s | after T2 |
| 6 | Artifact cache checks version only; drawn-variant rules overwritten by prep text | `prep.load_artifacts`, `state.adopt`, `main.handle_start` | wrong format/board size if a variant-specific rule is missed | drawn-variant rules restored after adopt; fingerprint/version check still open (low risk) | **fixed `8671e8d`** |
| 7 | Full LLM call per turn even when a solver move is locked | pig: 27 calls / 106k input tokens / $0.0141 vs $0.0033 on connect4 | cost points | chat-only compact prompt on solver turns | after T2 |
| 8 | Deals offer audit skipped when offers are enumerated | `decide.audit_offer` gated on `legal_list is None` | Deals match points | gate removed; corrected offer normalized against the list, else nearest legal offer; test with enumerated list | **fixed `8671e8d`** |
| 9 | Truth numeric grounding mis-handles `2,300` and number-only answers | `unsupported_answer` raw `\d+`; offline checks | Truth points | thousands separators stripped; absent number-only answers rejected; abstention unchanged | **fixed `8671e8d`** |
| 10 | `inject` chat tactic is self-defeating on board games | `other_moves_named` trims the line to empty; 2/2 inject turns -> canned 38 chars | chat quality / judges | skip inject on board families | after T2 |

## Not bottlenecks (checked)

- Luck/dice: chance DP 147/147, luck boundary exact, Pig won.
- LLM JSON failures: zero in the latest five reports; parser repair tested.
- Timeouts/crashes/illegal answers: zero across all reports; watchdog works.
- Cost spikes: max $0.0141/match, far below the $0.30 container cap (#7 is steady spend).
- Memory concurrency: 48 atomic writes / 21 concurrent reads, no partial JSON.
- Non-binary scores, simultaneous routing, solver over-claim, chat compliance: no failing evidence.

## Added during the luck-game pass

| # | Bottleneck | Evidence | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| 11 | Simgen skip is keyed on family, not on "public randomness" | `tg_blackjack_probe` classified `deals`; prep skipped genmodel with "negotiation: private values" although only the deck order is hidden; bench still won on policy/LLM | chance games lose the exact simulator they need | add `has_randomness` to the spec; skip only when `hidden_information && randomness`, else allow `chance_actions` | after T2 (needs ARTIFACT_VERSION bump) |
| 12 | Three new chance games prove the paths, not the strength | `tg_liars_dice_probe` (hidden dice, bids <=36), `tg_blackjack_probe` (public deck), `tg_roulette_probe` (39 bets + pass): 3/3 vs random, 0 warnings, p95 2.2 s, $0.0029 | coverage only | keep as regression probes; add stronger sparring bots | done (`1f39906`, `087bd0a`) |

## Measured after the fixes

- Pipeline `--fast`: **8/8 stages, 457 tests, 48/48 game variants, 17 probes, performance index 0.950** (was 0.892); new `solvers` stage: 509 proposals, 0 illegal.
- `scripts/report.py`: **100.0/100**, 17 probes (including the new `port` checks).
- mini_poker 10 seeds: 5 wins / 3 draws / 2 losses -> **50% win, 64.3% score**, gate passed.
- Luck games: liar's dice, blackjack, roulette -> 3/3, p95 2229 ms, $0.00289/match, chat 1.00.
- 63-action wide board: depth **3** at 0.6-1.5 s (lean evaluate) vs depth 2 before; lab model depth 3 at 8 s.
- Port verification `tg_deals` + `tg_liars_dice_probe`, 2 seeds: **4/4**, p95 2.38 s, $0.00034/match, gate passed.
- Connect-K fallback parses the real arena letter-column text and returns `drop E` on the 7x9 start.
- `connect_four` 3 seeds after the cap: **3/3 wins**, 0 warnings, depth **6-8** in real play
  (`route=alphabeta`), p95 4.27 s, $0.0037/match, gate passed.

## Ported after the audit

The lab branch's solver stack (`solver_bluff`/`guess`/`simul`, `words_en`, `solver_deals`, `solver_grid`)
and four robustness fixes (gateway retry, incomplete-cache guard, spec waits, ready clock) are now
additive files here. Our CFR/Leduc/chance experts stay ahead; the ported heuristics fill gaps and the
official T1 text formats. Deals fixes landed with it: parsed initial pot instead of 100, nearest-legal
offer snapping, one-shot PD defection, hyphenated auction detection.

## Fix-first shortlist

1. Null-legal last resort (#2) — deterministic protection against a lost game.
2. Leduc round-1 equity (#1) — the only measured game loss.
3. Wide-board root ordering/cap (#4) — depth 2 to real search on wide boards.

The best evidence gap remains that every bench opponent is `random_bot.py`: legality and robustness are
proven, strength is not. Fixing that needs stronger sparring partners (wordle bot, Leduc bot, Blotto bot)
or the official released games.
