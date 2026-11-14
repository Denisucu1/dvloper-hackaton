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

## Not verified here

- The Docker image: Docker Desktop was not running on this machine. Nothing in the lab adds a dependency
  (stdlib only), but the image was not built or run with these changes.
- No official T1 or T2 game engine is in arena-lite; everything was measured on the practice games and on
  the team's own `tg_*` test games.
