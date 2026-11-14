# Bottleneck audit — beyond luck games

Read-only audit of the T2 lab (`dev/om2-t2`), based on code inspection, warm logs and the five bench
reports. Ordered by expected points at risk, not by how interesting the fix is. Status is updated as the
T2 fixes land.

| # | Bottleneck | Evidence | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| 1 | Hidden-info play is one-ply; Leduc round-1 equity ignores the community card | `solver_leduc._equity`; mini_poker J vs bet: reported 0.10, true ~0.40, threshold 0.25; bench mini_poker 0-1 | Bluff/Leduc match points | enumerate opponent card + community draw; PIMC later | **fix in progress** |
| 2 | No last-resort action when `legal_actions` is null | `actions.choose_fallback` -> None; `main.turn_fallback` -> `action:""`; `decide` -> "pass" | −1.25 robustness per game + match loss | family/format-aware fallback (`unknown`, `offer n`, a word, concrete example) | **fix in progress** |
| 3 | Missing `sample_turn` silently removes generated code | prep only samples variants with sample_turn; wordle_duel 6 s prep, genpolicy/genmodel false; blotto 2/3 variants no sample | strength on free-text/wide games | synthesize samples from rules/action_examples, or built-in family policy | after T2 |
| 4 | Wide alternating boards stay at depth 2 | `route_for` sends only simultaneous nodes to MCTS; 63-action board depth 2 at 1.5-8 s | Think-ahead strength | root one-ply ordering + branch cap to 24, or MCTS | **fix in progress** |
| 5 | Prep tournament measures a different clock than play | selfplay search 30 s / depth 2, real play 1.5 s / depth 8; forced chess prep 150 s vs 23 s | wrong strategy choice + prep time | shared PLAY_PROFILE; cap tournament 60-90 s | after T2 |
| 6 | Artifact cache checks version only; drawn-variant rules overwritten by prep text | `prep.load_artifacts`, `state.adopt`, `main.handle_start` | wrong format/board size if a variant-specific rule is missed | keep `start.rules`; fingerprint check | **fix in progress** |
| 7 | Full LLM call per turn even when a solver move is locked | pig: 27 calls / 106k input tokens / $0.0141 vs $0.0033 on connect4 | cost points | chat-only compact prompt on solver turns | after T2 |
| 8 | Deals offer audit skipped when offers are enumerated | `decide.audit_offer` gated on `legal_list is None` | Deals match points | remove gate; normalize corrected offer | **fix in progress** |
| 9 | Truth numeric grounding mis-handles `2,300` and number-only answers | `unsupported_answer` raw `\d+`; offline checks | Truth points | strip separators; number-only alignment | **fix in progress** |
| 10 | `inject` chat tactic is self-defeating on board games | `other_moves_named` trims the line to empty; 2/2 inject turns -> canned 38 chars | chat quality / judges | skip inject on board families | after T2 |

## Not bottlenecks (checked)

- Luck/dice: chance DP 147/147, luck boundary exact, Pig won.
- LLM JSON failures: zero in the latest five reports; parser repair tested.
- Timeouts/crashes/illegal answers: zero across all reports; watchdog works.
- Cost spikes: max $0.0141/match, far below the $0.30 container cap (#7 is steady spend).
- Memory concurrency: 48 atomic writes / 21 concurrent reads, no partial JSON.
- Non-binary scores, simultaneous routing, solver over-claim, chat compliance: no failing evidence.

## Fix-first shortlist

1. Null-legal last resort (#2) — deterministic protection against a lost game.
2. Leduc round-1 equity (#1) — the only measured game loss.
3. Wide-board root ordering/cap (#4) — depth 2 to real search on wide boards.

The best evidence gap remains that every bench opponent is `random_bot.py`: legality and robustness are
proven, strength is not. Fixing that needs stronger sparring partners (wordle bot, Leduc bot, Blotto bot)
or the official released games.
