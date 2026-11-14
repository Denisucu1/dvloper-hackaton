# Presentation plan - team adn (Sunday 15:30)

Working document. 30 points for the presentation (how clearly we explain the harness, a live demo, what we
learned) and 25 for harness design (rules understanding, state, move verification, tools and computation,
memory, budget and time, and how it plays the chat). Rule 12.2: we start by introducing ourselves, one by
one, about 30 seconds each.

Everything marked **[verified]** below was read from the public replay `t1-r1-m7` or measured in arena-lite.
Items marked **[owner]** need the numbers from the teammate who built that part.

## 1. Running order (target: 8-10 minutes, adjust when the format is announced)

| # | Part | Who | Time |
|---|------|-----|------|
| 0 | Who we are: name, what we study, what we built before, what we like doing | all three, one by one | 1:30 |
| 1 | The idea in one picture: rules in, specialist out, move + chat as one decision | Denis | 1:00 |
| 2 | Prep: from a rules text to a spec, a simulator, a solver, a playbook | Om 2 | 1:30 |
| 3 | One turn: code proposes, one model call decides move and chat, code verifies | Denis | 1:00 |
| 4 | The chat: defense, offense, and what the opponent's words are worth | Om 3 | 1:30 |
| 5 | Live demo | Om 3 drives, Denis talks | 2:00 |
| 6 | T1 -> T2: what one lost match taught us | all | 1:30 |
| 7 | What we would do next | Denis | 0:30 |

## 2. The one picture (slide 1)

```
rules text ──prep, 10 min──▶ spec (family, action format, speech rules)
                             simulator + search / family solver      ◀── tested on sample_turn
                             playbook                                 ──▶ /memory/<game>/<rules hash>

turn ─▶ code reads the chat  ─▶ code proposes a move ─▶ ONE model call {reason, move, read, chat}
        (tags, claims,           (solver, search,          │
         pushed moves)            heuristic)               ▼
                                              code verifies: legal move, obedience check, speech gate,
                                              Deals audit, coherence of move and line ─▶ stdout
```

Three levels on a turn, in this order:

1. **Code** (0 tokens, under 100 ms): family solvers, and search on the simulator written in prep.
2. **Council** (only when level 1 has no move): three strategists who see the rules, the strategy brief and
   the observation but not one word of the opponent's chat vote on the move. What the chat is worth reaches
   them as numbers computed by our code (claims verified true or false, lines that were orders).
3. **Arbiter** (code): the chat-aware call writes the chat line for the council's move and may leave that
   move only when our code has verified the opponent's claims as reliable; then legality, the format guard,
   the speech gate and the Deals audit have the last word.

The organizers' own test harnesses scored 5.5 / 28 for one call per move, 15 / 28 for several voices voting
and 21.5 / 28 for a simulator plus code: levels 2 and 1 here. Prep spend is not counted in the cost score
(rule 9.3), so the strategy brief, the chat plans and the generated code are free.

Sentence to say out loud: *the model never has the last word on a move, and the opponent never has a word
the model takes as an instruction.*

## 3. Design points to show, mapped to the judges' list (rule 12.1)

| Judges look at | What we show | Where in the code |
|---|---|---|
| Understands rules | family classification, action format, forbidden subject read from rules and observation | `harness/spec.py`, `harness/prep.py`, `ChatPolicy.learn` |
| Tracks state | simulator generated in prep and checked on `sample_turn` **[owner: Om 2]** | `harness/simgen.py`, `harness/game_model.py` |
| Verifies moves | legal-list normalisation, template fill when there is no list, one warning never becomes two | `harness/actions.py`, `harness/decide.py` |
| Tools and computation | alpha-beta / MCTS, family solvers **[owner: Denis, Om 2]** | `harness/search.py`, `harness/solver_*.py` |
| Memory | artifacts per game and rules hash, opponent chat profile, alias cache | `harness/memory.py`, `harness/opponents.py` |
| Budget and time | hard deadline per handler, thinking only with clock to spare, audit capped at 9 s | `harness/watchdog.py`, `harness/turnpolicy.py`, `config.AUDIT_*` |
| Chat: offense, defense, one decision | section 4 below | `harness/chatpolicy.py`, `harness/decide.py` |

## 4. The chat slide (Om 3)

Three columns, one example line each, all from real replays.

**Defense - their words are evidence, never instructions**
- Every opponent line is tagged in code: injection, fake authority, probe for private state, speech bait,
  claim, offer, and the moves they push us toward.
- The model sees the lines verbatim inside a fence with a random nonce the opponent cannot forge.
- Measured against a sparring bot that sends 12 kinds of attack ("SYSTEM: the rules have changed", forged
  fences, "repeat after me"): no order obeyed in any practice match. **[verified]**

**The gate - nothing leaves that can cost a game**
- Forbidden subject in 14 spellings (spaced, leetspeak, reversed, full-width, emoji between letters), plus
  translations and emoji learned once per game and cached in `/memory`. **[verified in tests]**
- Our own private values (secret number, walk-away value) never appear in a chat line.
- A line may only name the move it is sent with: "Center it is, B2" next to the move C3 is trimmed.
- Deals: an acceptance or an offer that leaves us behind is recomputed in code and replaced; an auction bid
  is capped at our private value.

**Offense - a tactic per turn, written with the move**
- One tactic per turn by family (represent, anchor, misdirect, lure), "counter" after an injection,
  "stonewall" after a probe.
- Honest number: direct instructions aimed at the opponent's model were obeyed 0 times out of 8 by the
  kit's starter bot. Bluff, pressure and negotiation carry the offense, not injection. **[verified]**

**What their words are worth - the T1 lesson**
- In `t1-r1-m7` the opponent named its own card in the chat 8 times, and all 8 matched its real card.
  **[verified against the replay frames]**
- New for T2: card claims are checked in code against what each showdown reveals; the count goes to
  `/memory` and the next game against that team starts knowing whether its chat tells the truth.

## 5. Live demo (2 minutes, must work offline)

Preferred, in this order; each has a fallback that needs no network.

1. **One decision, on screen.** `python scripts/replay_turn.py <t1 replay> --game 1 --step 30`: the real T1
   position, what we played then, what the harness plays now, the trust level and the chat line.
   Fallback: a screenshot of the same output.
2. **Attack us.** arena-lite, our bot against `scripts/trickster_bot.py` on `tg_guardrails`; the audience
   sees the bait lines and our answers. Then `python scripts/chat_report.py <replay>`: lines sent, longest,
   tactics, what the gate blocked. Fallback: a recorded replay JSON and the report output.
3. **The dashboard replay** of our Connect Four win, if the room has network.

Before the talk: run each command once on the presenting laptop with `py` (not the msys `python`: it
cannot reach the gateway) and keep the outputs in a folder.

## 6. T1 -> T2: the story of one match (`t1-r1-m7`, adn 1 - 2 A.L.T.)

Facts **[verified from the replay]**: 2 min 26 s, 0 warnings on both sides, cost $0.022 (ours) vs $0.029,
decided by score.

**Game 2 - Connect Four, 7x9, connect 5: won in 17 plies.**
We moved first (X), 9 moves, 3.1 s per move on average, 4.0 s at most; the opponent averaged 4.0 s. Moves:
`5 4 5 5 5 5 5 5 4 6 6 7 7 6 4 8 8` (X first). The simulator written in prep plus alpha-beta found the
five-in-a-row on row two while the opponent's chat was describing our "row-two trio" as "an essay with the
conclusion still missing". This is the picture for "the model writes the tool, code picks the move".

**Game 1 - Leduc poker, 8 hands: lost 26 chips to 54.**
- Hand 4 (-11): we held K, they held Q, the community card was Q. We bet 2, bet 4, they raised to 8 on the
  paired board and we called. A raise on a paired board was the signal; nothing in our harness priced it.
- Hand 6 (-7): we held Q, they held K. Before betting they wrote "my K will be happy to accept your
  tuition", then "this K will collect the tuition in full". Both true. We called both bets.
- Those two hands are 18 chips; we finished 14 down.
- What was missing: the bluff solver did not cover the Leduc variant, and we gave the opponent's claims no
  weight because we had no evidence either way.

**Game 3 - Colonel Blotto with reserves, 3 fields, 5 rounds: lost 2 rounds to 3 after leading 2-1.**
- Rounds 2 and 3 won with 16/16/0 and 15/15/0.
- Round 4, leading 2-1: we sent 15/15/13, our whole reserve of 43. They answered the pattern with 17/17/0
  and kept 14 soldiers.
- Round 5: 30 soldiers against 44. We sent 10/10/10 against 16/16/12 and lost all three fields.
- What was missing: a solver for the reserve variant (how much to keep), and mixing on fields we had
  already shown twice. Their chat described their allocation each round, but in a simultaneous game a line
  arrives after the move it is sent with, so it could only be used as a pattern, not as a tell.

**What changed for T2**
- Chat: card claims verified against showdowns and remembered per opponent; our own line no longer carries
  the quote marks of our chat fence (it did, once, in game 3). **[done, this branch]**
- A chat plan per game: one call with thinking reads the rules once and writes what we must never reveal,
  what the opponent might let slip and how the game confirms it, four tactics specific to that game, and
  the opponent's likely tricks. Cached in `/memory` by rules hash. Measured on the real Leduc and Blotto
  rules: 7.5 s and 8.7 s, under a tenth of a cent each. **[done, this branch]**
- Post-game review: at `end`, off the clock and in a background thread, one call with thinking turns the
  final observation into up to three lessons for the next game of the same game. On the T1 poker game it
  wrote, unprompted, the two hands we lost (fold to a raise on a paired board; do not check-call a bettor
  down with one high card). **[done, this branch]**
- Bounded thinking on the move for the families with no solver (Deals, Truth, Guardrails, unclassified).
  **[done, this branch]**
- Bluff: solver coverage for Kuhn and Leduc. **[owner: Denis]**
- Read the opponent: reserve-aware Blotto, mixing instead of repeating. **[owner: Denis]**
- Guessing: Wordle with a dictionary and entropy. **[owner: Denis]**
- Prep: valid generated policies for games that are not on a board. **[owner: Om 2]**
- Deals, Truth, Guardrails did not appear in T1; they were built and measured on our own test games
  (`arena-lite/arena/games/tg_*.py`) before any official game of those families was seen.

## 7. What we learned (closing slide, three lines)

1. A model that decides and a code that verifies beat a model that does both. In Deals the model accepted
   "opponent keeps 68 of 80" with a walk-away value of 30 and answered 40 for "90 minus 55"; the subtraction
   and the comparison now live in code.
2. Measure before believing: the first chat prompt made the model guess its own secret number half the
   time, and the "smart" claim check by the model counted lies for a player who never lied. Both were found
   by replaying real positions, and both were replaced by code.
3. The opponent's chat is free information once you check it against what the game reveals.

## 8. To collect before Sunday 15:30

- [ ] Final standings and our placement in T1, T2, round robin (dashboard).
- [ ] One replay per family from T2 or the round robin, saved under `arena-lite/results/` (not committed).
- [ ] Numbers from Om 2: prep time per game, how many simulators passed validation on `sample_turn`.
- [ ] Numbers from Denis: solver coverage, search depth and nodes on Connect Four.
- [ ] Cost per match and games lost to timeouts or illegal moves over the whole event (robustness half).
- [ ] Screenshots for every demo step, in case the network or the gateway is down.
