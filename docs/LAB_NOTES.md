# Lab notes - what `bot-lab` changes on top of `main`

Base: `main` at `4bf1a0f` + `dev/om3-t2` (`40210bd`). Local only; nothing here was pushed.
Every item has an environment switch, so each can be taken or left at the merge without code changes.

| Change | File | Switch (default) | Clock it uses | Cost |
|---|---|---|---|---|
| Strategy brief written in prep with thinking | `harness/strategy.py`, `harness/prep.py` | `HARNESS_STRATEGY=1` | prep (about 90 s of 600) | about $0.009 per game |
| Strategy brief at a cold start, cached by rules hash | `harness/strategy.py`, `harness/decide.py` | same | study clock, only when 40 s are spare after a 35 s reserve | same, once |
| Chat plan for every variant in prep | `harness/prep.py` | `HARNESS_CHATPLAN_*` | prep (about 8 s per variant) | under $0.001 each |
| Voting: 3 answers in parallel when no code move exists | `harness/decide.py` | `HARNESS_VOTERS=3` (1 = off) | none extra (parallel) | 3x the turn call on those turns |
| Thinking on the move for deals, truth, guardrails, unknown | `harness/decide.py` | `HARNESS_THINK_FAMILIES` | 3-6 s on those turns | about $0.0006 a turn |
| Post-game lessons | `harness/lessons.py` | `HARNESS_LESSONS=1` | none (background thread at `end`) | about $0.0005 a game |
| Acceptance audit recomputes the split from the offer itself | `harness/decide.py` | - | - | - |
| `ready` sized on the study clock that is left | `main.py` | - | - | - |
| Playbook budget in the turn prompt 1800 -> 4200 chars | `harness/decide.py` | `HARNESS_MAX_PLAYBOOK` | - | input tokens only |
| smoke.py: bot stderr to a file | `scripts/smoke.py` | - | - | - |

## Found while measuring

- Prep used 13-25 s of 600 and $0.002-0.003 of $0.75 on four games before these changes.
- A thinking budget of 5000 or 8000 tokens returned an EMPTY answer for the strategy brief even with as
  much headroom again (rule 7.3). 2000 works. The brief steps down (2000 -> 1024 -> off) instead of failing.
- The acceptance audit read "opponent offered to keep 80" as mine=80 and a 20-of-100 deal was accepted.
  The audit now quotes the offer and says whose share its number is; code does the rest.
- `scripts/smoke.py` kept the bot's stderr in a pipe nobody read; with more log lines the bot blocked on
  its own log and never exited. Not a bot bug (the arena drains stderr), but the test was failing.
- `PrepGuardrailsSubagent` is run on "subjects" such as `walk-away value`, `private value`, `W` that the
  analyst lists as `extra_forbidden`: five alias calls in a negotiation game that has no speech rule, and
  those words then sit in `spec.forbidden_subjects`.
- Locally, voting can hit the dev key's 60 requests a minute (one `llm http 429` seen in a four-game match).
  Arena keys have no per-minute limit (rule 7.2).

## Measured in tournament mode (`--prep`, warm specialists), final lab code

| Game | Opponent | Result | Longest move |
|---|---|---|---|
| tg_deals x2 matches (8 games) | trickster (demands 80%) | 2-2 and 2-2: eight draws, no deal | 13.2 s |
| tg_deals | kit starter | 1-0 (kept 70 of 100, surplus +50 vs +8) | 2.9 s |
| tg_truth | trickster | 1-0 (6 to -12) | 3.2 s |
| tg_guardrails | trickster | 1-0 (6-2) | 2.9 s |
| tg_connect4 | kit starter | 1-0 | 3.7 s |
| tg_bluff | kit starter | 1-0 (+2 vs -2) | 2.5 s |
| tg_simul | kit starter | 0.5-1.5 (4-4, then 2-6 on "rock wins 3") | 2.3 s |

No warning, no illegal move and no missed clock in any of them. Prep with the brief and the chat plans
took 47-63 s of 600 and $0.005-0.006.

Earlier lab code lost two negotiation games; both causes are fixed and under test:
the audit reading the opponent's offer backwards, and a rejected acceptance being played because the
audit gave no alternative (now our last demand is repeated).

Two validation matches are not counted: the shared dev key hit its 60 requests a minute, every call
failed with 429 and the bot played its fallback (`walk`, a draw).

Open point: with thinking, three voters and the audit, a negotiation move took up to 13.2 s of 20. The
deadline guard still bounds it at 17 s, but total thinking time is the second knockout tie-break.

## Not verified here

- The Docker image: Docker Desktop was not running on this machine. Nothing in the lab adds a dependency
  (stdlib only), but the image was not built or run with these changes.
- No official T1 or T2 game engine is in arena-lite; everything was measured on the practice games and on
  the team's own `tg_*` test games.
