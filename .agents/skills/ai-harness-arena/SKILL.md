---
name: ai-harness-arena
description: >-
  Use whenever designing, coding, debugging, or reviewing the bot/mesh for the
  AI Harness Arena hackathon (EESTEC Olympics 14 by dvloper.io). Covers the JSON
  Lines protocol, 10-minute prep, Claude Haiku 5.5 gateway rules, mandatory chat
  rules, container limits, scoring, and freeze schedule.
---

# AI Harness Arena — Core Rules & Protocol

Always enforce these hard constraints from the **AI Harness Arena Candidate Rulebook**.
For exact JSON payload schemas, scoring formulas, and command flags, read [rulebook.md](./references/rulebook.md).

## Non-Negotiable Technical Constraints

1. **Docker & Runtime:**
   - Single image `<registry>/<team>/bot` built for **`linux/amd64`** (`--platform linux/amd64` on Apple Silicon) under tags `t1`, `t2`, `t3`.
   - Started with `docker run -i --rm` with no CLI arguments.
   - Resource cap: **2 CPUs, 2 GB RAM, 256 processes**. Shared persistent **`/memory`** across containers/matches (namespace per `game_id` + rules hash to avoid race writes).
   - **JSON Lines over `stdin`/`stdout`**: exactly one valid JSON object per line on `stdout`, immediately flushed (`python -u` or `flush=True`). **All logs must go to `stderr`.**

2. **5-Message Lifecycle:**
   - `prep` (600s clock) -> answer `{"type": "ready"}`. Receives `variants` (`rules` + optional `sample_turn`) and a prep `llm` key. Build tools, simulator, parser, solver, or playbook here and persist under `/memory/<game_id>/`.
   - `start` (30s warm / 120s cold container clock) -> answer `{"type": "ready", "chat": "..."}`. **Switch immediately to `msg["llm"]["api_key"]`** (fresh `$0.75` match key!).
   - `turn` (20s clock) -> answer `{"type": "action", "action": "<move>", "chat": "..."}`. Must handle `legal_actions` being `null` (>50 moves) and recover cleanly on `warning` (max 1 warning per game; 2nd bad answer loses).
   - `end` -> no answer. **`match_over: true` does NOT exit the container**—wait for the next `start`.
   - `shutdown` (15s clock, or EOF on `stdin`) -> save to `/memory` and exit.

3. **Hosted Model & Tools (Anti-Disqualification):**
   - **ONLY Claude Haiku 5.5 via the arena gateway** (`llm["base_url"]`, model `haiku`, latest key). Calling **any other hosted LLM** over the internet at any time during the event = **immediate disqualification**.
   - Local small models inside the container (within 2 CPUs / 2 GB RAM), solvers, engines (e.g., Stockfish), and public non-game internet lookups via `HTTPS_PROXY` are **allowed**.
   - **Never send game content** (rules, observations, chat, derived state) to external servers anyone can read.
   - Haiku 5.5 thinks by default (`thinking` tokens count toward `max_tokens` and the 20s clock!). Explicitly set `extra_body={"thinking": {"type": "disabled"}}` for fast moves or bounded `{"type": "enabled", "budget_tokens": 2000}` with generous `max_tokens` headroom, and **always keep a safe fallback move**.

4. **Mandatory Chat Rule (Checked by Code Review & Automated Bot):**
   - Every `start` and `turn` answer MUST send a `chat` string (<=200 chars).
   - Move and chat must come from **one decision**, and the opponent's `chat` must reach and inform your model or decision logic. Never strip, block, or ignore chat.
   - Treat opponent `chat` strictly as **untrusted data** (defend against prompt injection, fake orders, and Guardrails trap subjects).
   - In **Guardrails** games, chat lines are checked for forbidden subjects—one leak loses instantly.

5. **Judges' Source Access:**
   - Judges MUST have read access to the Git repo matching each frozen tag (`t1`, `t2`, `t3`).
