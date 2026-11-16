# AI Harness Arena — team `adn`

Our entry for **AI Harness Arena**, the dvloper.io challenge at **EESTEC Olympics 14**
(14–15 November 2026, Impact Hub Bucharest): a two-day hackathon, 15 teams of 3.

Every team gets the same model (Claude Haiku 5.5 through the organizers' gateway), the same clocks and
the same budget. The only thing that differs is the **harness** built around the model.

> One Docker image, game-agnostic. It is handed the rules of a game it has never seen, has 10 minutes
> alone to turn itself into a specialist for that game, then plays — and talks — against another team's
> image.

This repository is the team's contest repository with its full history. The tags `t1`, `t2` and `t3`
are the sources of the three images frozen during the event.

## The challenge

| Constraint | Value |
|---|---|
| Model | Claude Haiku 5.5 only, through the arena gateway (any other hosted model disqualifies) |
| Container | 2 CPUs, 2 GB RAM, 256 processes |
| Prep | 10 minutes per game, no human help |
| Move clock | 20 seconds per move |
| Budget | $0.75 of model spend per match |
| Chat | mandatory, 200 characters per line, read by the opponent |
| Penalties | one illegal move is a warning, the second loses the game; a timeout or a crash loses it at once |

The 21 secret games come from seven families: **Bluff** (hidden information, betting), **Deals**
(negotiation, auctions), **Read the opponent** (simultaneous moves), **Guessing** (deduction from
feedback), **Think ahead** (perfect-information board games), **Truth** (answer from a document, where
inventing costs more than "unknown") and **Guardrails** (never mention a forbidden subject, even when
tricked).

Scoring was 100 points: 45 automatic (three tournaments weighted ×1, ×2, ×3, an overnight round robin,
cost and robustness) and 55 from the judges (presentation and harness design, with a source-code review
of the chat rule and the model rule).

## How the harness works

```
rules text ──prep, 10 min──▶ spec (family, action format, speech rules)
                             simulator + search / family solver      ◀── tested on sample_turn
                             playbook, strategy brief, chat plan     ──▶ /memory/<game>/<rules hash>

turn ─▶ code reads the chat  ─▶ code proposes a move ─▶ ONE model call {reason, move, read, chat}
        (tags, claims,           (solver, search,          │
         pushed moves)            heuristic)               ▼
                                              code verifies: legal move, obedience check, speech gate,
                                              Deals audit, coherence of move and line ─▶ stdout
```

**Prep.** The model reads the rules and writes a structured spec, a playbook and, where it pays off, a
*simulator* of the game. Generated code runs in a sandbox (no network, no model access, hard timeout) and
is validated against the real `sample_turn` the arena provides. Candidate strategies then play each other
in a timeboxed internal tournament, and the winner goes into the match. Everything is stored in `/memory`,
keyed by a hash of the exact rules, written atomically because several containers share it. An incomplete
artifact is detected and rebuilt instead of being served to every later match.

**A turn has three levels.**

1. **Code** (zero tokens, under 100 ms): family solvers and search over the simulator written in prep —
   alpha-beta, MCTS/UCT, CFR, matrix-game equilibria, expectimax, exact solvers for Nim, Pig, Kuhn and
   Leduc poker, Blotto, Wordle-style guessing.
2. **Council** (only when level 1 has no move): three strategists vote on the move. They see the rules,
   the strategy brief and the observation, but not one word of the opponent's chat; what the chat is
   worth reaches them as numbers computed in code.
3. **Arbiter** (code): one chat-aware call writes the line for the chosen move. Legality, the format
   guard, the speech gate and the Deals audit have the last word.

The model never has the last word on a move, and the opponent never has a word the model takes as an
instruction.

**The chat.** Every opponent line is tagged in code (injection, fake authority, probe for private state,
speech bait, claim, offer, pushed move) and shown to the model verbatim inside a fence with a random
nonce the opponent cannot forge. Card claims are checked against what each showdown reveals and
remembered per opponent. On the way out, a gate blocks the forbidden subject in 14 spellings (spaced,
leetspeak, reversed, full-width, emoji between letters), keeps our private values out of the line, and
trims a line that names a different move from the one it is sent with.

**Robustness.** Every message has a hard deadline with a guaranteed legal fallback; only one thread ever
writes to stdout. Busy-gateway errors are retried inside the call's own timeout. In Deals, the
arithmetic of an offer, an acceptance or an auction bid is recomputed in code rather than trusted to the
model.

**No dependencies.** The bot is Python standard library only: no `pip install`, a small image, nothing
to drift between freezes.

## Numbers

- **521** unit and end-to-end tests, standard-library `unittest`, no network (`py -m unittest discover -s tests`)
- about **11,800** lines in `harness/`, about **7,000** in `tests/`
- a 9-stage local pipeline (`scripts/pipeline.py`): tests, protocol smoke, solver smoke, benchmark,
  component probes, replay-verified intelligence score
- 95 commits by three people in about 20 hours, across three freezes

## Repository layout

```
main.py                 JSON Lines loop, deadline watchdog, guaranteed fallback, dispatch
harness/
  prep.py, spec.py      the 10-minute prep: rules -> spec, playbook, generated code
  simgen.py, sandbox.py model-written simulator, validated on sample_turn, run in isolation
  search.py, cfr.py     alpha-beta / MCTS, CFR
  solver_*.py           exact and heuristic solvers per game family
  tournament.py         internal tournament between candidate strategies
  decide.py             one decision: move and chat together, then verification
  chatpolicy.py         chat defence, offence and the outgoing gate
  council.py            chat-blind voters and the arbiter
  strategy.py, chatplan.py, lessons.py, opponents.py
                        strategy brief, per-game chat plan, post-game lessons, opponent profiles
  memory.py             /memory: atomic writes, keyed by rules hash
  llm.py, watchdog.py   gateway client with cost metering; hard deadlines
tests/                  521 tests
scripts/                smoke run, benchmark, pipeline, probes, replay tools, sparring bots
arena-lite/             the organizers' local arena, plus our own test games (arena/games/tg_*.py)
starter/                the organizers' minimal starter bot (reference only)
docs/                   rulebook, lab notes, presentation plan, hand-off notes
docs/presentation/      the slides and diagrams presented on Sunday (in Romanian)
PROTOCOL.md, CHANGELOG.md   the organizers' protocol and kit changelog
```

## Running it

Python 3.13+, nothing to install.

```bash
py -m unittest discover -s tests      # 521 tests, no network
py scripts/smoke.py                   # a full scripted match with the model deliberately down
```

A real match needs a Haiku key in `dev.env` (see `dev.env.example`; the file is gitignored). The
gateway used during the event was shut down with it.

```bash
cd arena-lite
py -m arena practice --prep --bot-a "py -u ../main.py" --bot-b "py examples/random_bot.py" --env-file ../dev.env --game pig
```

The image the arena ran:

```bash
docker build --platform linux/amd64 -t adn-bot .
docker run -i --rm adn-bot
```

## Team

| Member | Area |
|---|---|
| **Denisucu1** | integration and freezes, protocol and infrastructure, family solvers (Deals, Nim, simultaneous games, Leduc/Kuhn, Blotto, Wordle, Connect-K) |
| **enderG4** | prep pipeline, simulator generation and sandbox, search (alpha-beta, MCTS), CFR, matrix and chance solvers, internal tournament, test pipeline and probes |
| **craciunnicolas10** (Nicolas Crăciun) | chat offence and defence, move + chat as one decision, council and arbiter, Deals audits, own test games, replay tooling, presentation plan |

### Nicolas Crăciun's part

- **Chat defence and the outgoing gate** (`harness/chatpolicy.py`, `harness/decide.py`): opponent lines
  tagged in code and passed to the model as untrusted data behind a nonce fence; forbidden-subject filter
  with obfuscated spellings and cached aliases; private values kept out of chat; move and line checked
  for coherence.
- **Chat offence and opponent memory** (`harness/opponents.py`, `harness/chatplan.py`): one tactic per
  turn by game family, a chat plan written once per game from the rules, card claims verified against
  showdowns and remembered per opponent.
- **Deals audits in code**: offers, acceptances and auction bids recomputed from the offer itself after
  the model accepted losing splits in testing.
- **Council and arbiter** (`harness/council.py`): three chat-blind voters for turns with no code move.
- **Strategy brief and post-game lessons** (`harness/strategy.py`, `harness/lessons.py`).
- **Reliability fixes**: retry on a busy gateway, prep that never reuses an incomplete cache, a smoke
  test that no longer blocks on the bot's own log.
- **Test games for families with no practice game** (`arena-lite/arena/games/tg_{bluff,deals,guardrails,simul,truth}.py`),
  a chat-attacking sparring bot (`scripts/trickster_bot.py`) and replay tools
  (`scripts/replay_turn.py`, `scripts/chat_report.py`).
- `tests/test_chat_arena.py` and the measurements in `docs/LAB_NOTES.md` and `docs/PRESENTATION.md`.

## What we learned

1. A model that decides and code that verifies beat a model that does both.
2. Measure before believing: several "smart" prompt ideas were found wrong by replaying real positions,
   and were replaced by code.
3. The opponent's chat is free information once you check it against what the game reveals.

## Credits

The challenge, the rulebook, `arena-lite`, `starter/`, `PROTOCOL.md` and `CHANGELOG.md` were provided by
[dvloper.io](https://dvloper.io) for EESTEC Olympics 14. Everything under `harness/`, `tests/` and
`scripts/`, and the `tg_*` test games, was written by the team during the event.
