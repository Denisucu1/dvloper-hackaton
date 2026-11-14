# AI Harness Arena — Detailed Reference

## 1. Seven Game Families (21 Secret Tournament Games)
1. **Bluff:** hidden information, betting, bluffing, odds.
2. **Deals:** negotiation, trust and betrayal, auctions with private values.
3. **Read the opponent:** simultaneous moves, patterns, mixed strategies.
4. **Guessing:** deduction from feedback.
5. **Think ahead:** perfect-information board games, search, tactics.
6. **Truth:** answering from a document; inventing costs more than `"unknown"`.
7. **Guardrails:** rules name a forbidden subject; never mention it even when tricked, while answering everything else.

## 2. Message Schemas & Clocks
- **`prep`** (`600s` clock):
  - Fields: `type`, `game_id`, `game_name`, `variants` (`[{label, rules, sample_turn?}]`), `seconds`, `you`, `study_seconds`, `move_seconds`, `llm`.
  - Reply: `{"type": "ready"}` (invalid JSON during prep is ignored and retried; missing clock falls back to cold container at match time).
- **`start`** (`30s` warm / `120s` cold clock):
  - Fields: `type`, `match_id`, `game_index`, `best_of`, `game_id`, `game_name`, `variant`, `rules`, `seat` (`0` or `1`), `you`, `opponent`, `match_score`, `study_seconds`, `move_seconds`, `chat`, `llm`.
  - Reply: `{"type": "ready", "chat": "<string <= 200 chars>"}`. **No warning on `start`**: malformed, missing, or late `ready` loses the game immediately!
- **`turn`** (`20s` clock):
  - Fields: `type`, `game_index`, `turn`, `observation`, `legal_actions` (list of <=50 strings or `null`), `seconds`, `chat` (full match history: `{from, game, game_name, phase, turn, kind, text}`), optional `warning` (`{count, max, reason, rejected}`).
  - Reply: `{"type": "action", "action": "<move>", "chat": "<string <= 200 chars>"}`.
  - Warning rule: 1st bad turn answer -> `warning` + turn resent with fresh 20s clock (chat dropped). 2nd bad turn answer in same game -> **game lost**. Timeout or crash -> **instant loss, no warning**.
- **`end`** (no reply):
  - Fields: `type`, `game_index`, `result`, `reason`, `scores`, `match_score`, `final_observation`, `match_over`, `chat`.
- **`shutdown`** (`15s` clock, no reply):
  - Save state to `/memory` and exit cleanly.

## 3. Environment Variables Inside Arena Containers
- `LLM_BASE_URL`, `LLM_API_KEY` (initial/expired—always prefer `msg["llm"]["api_key"]`), `LLM_MODEL` (`haiku`)
- `TEAM_ID`, `MEMORY_DIR` (`/memory`)
- `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` (`gateway,localhost,127.0.0.1`), `NODE_USE_ENV_PROXY=1`

## 4. Haiku 5.5 Budget & Pricing
- Budget: **$0.75 per match** (shared across containers playing that match) + **$0.75 per prep**.
- Pricing: **$0.10 / 1M input tokens**, **$0.50 / 1M output tokens** (cached reads **$0.01 / 1M**).
- Thinking control in Python OpenAI client:
  - Fast: `extra_body={"thinking": {"type": "disabled"}}`
  - Bounded: `extra_body={"thinking": {"type": "enabled", "budget_tokens": 2000}}` with `max_tokens >= budget + 1500`

## 5. Scoring (100 Points)
- **45 Automated (Live Leaderboard):**
  - **Tournaments (30 pts):** T1 (4 games, weight ×1), T2 (8 other games, weight ×2), T3 Final top 8 (all 21 games, weight ×3). Placement: 1st=10, 2nd=7, SF=5, QF=3, R16=1. Score = `(sum(placement * weight) / 60) * 30`.
  - **Round Robin (10 pts):** All 15 teams play each other overnight (105 matches, 2 boards × 2 swapped seats = 4 games per match on T1+T2's 12 games). Win=1, Draw=0.5. Top 8 qualify for T3.
  - **Cost & Robustness (5 pts):** `2.5` for low match spend (`2.5 * min(1, c_min' / c')`) + `2.5` robustness (`2.5 - 1.25` per game lost to timeout/invalid JSON/wrong type/illegal or missing action/crash/forfeit, floored at 0).
- **55 Judges (Sunday Afternoon):**
  - **Presentation (30 pts):** clarity explaining harness, live demo, lessons learned (starts with ~30s individual intros per teammate).
  - **Harness Design (25 pts):** rule understanding, state tracking, move verification, tools/simulators/solvers, memory, budget & clock management, offensive + defensive chat.

## 6. Schedule (14–15 Nov 2026)
- **Sat 12:00** Hacking starts
- **Sat 18:00** **T1 freeze** -> prep (~40 min) -> T1 live
- **Sat 23:59** **T2 freeze** -> prep (00:00–02:00) -> Round Robin (02:00–06:30)
- **Sun 09:00–10:30** T2 live
- **Sun 10:30–11:00** **T3 freeze** & draw (top 8) -> warmup (11:00–12:00) -> T3 stage matches (13:00–15:00)
- **Sun 15:30** Presentations & **17:00** Awards
