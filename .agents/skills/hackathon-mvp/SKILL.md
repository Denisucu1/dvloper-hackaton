---
name: hackathon-mvp
description: >-
  Use when scoping, architecting, or prioritizing milestones for the AI Harness
  Arena hackathon before T1 (18:00), T2 (23:59), and T3 (10:30) freezes.
  Trigger in English or Romanian for MVP planning, harness architecture,
  timeboxing, or deciding what to build next.
---

# Hackathon MVP & Harness Roadmap (AI Harness Arena)

Optimize for **zero-forfeit reliability first**, then **adaptable game-playing intelligence** across unseen games by each freeze deadline (**T1 Sat 18:00**, **T2 Sat 23:59**, **T3 Sun 10:30**).

## Priority Ladder (Build in This Order)

### Tier 1 — Must Have Before T1 Freeze (18:00) — Zero Free Losses
1. **Bulletproof I/O & Lifecycle:** Unbuffered JSON Lines loop (`prep`, `start`, `turn`, `end`, `shutdown`), stderr-only logging, dynamic `msg["llm"]["api_key"]` switching, and clean `/memory` save.
2. **Guaranteed Valid Fallback:** Always pick a legal move (from `legal_actions` or rules-format fallback) well before the 20s clock (e.g., hard deadline at 15s) or if Haiku errors/returns empty. **Losing 2 games to crashes/timeouts wipes out the 2.5 robustness points.**
3. **Rule-Compliant Chat Module:** Pass opponent `chat` (wrapped as untrusted input + prompt-injection warning + Guardrails check) into the single decision prompt alongside board/game state, and always emit a coherent `chat` move (<=200 chars).
4. **Fast Game-Family Classifier in `prep`:** Classify the game into its family (Board/Think-ahead, Guessing/Deduction, Bluff/Hidden-info, Deals/Negotiation, Simultaneous/Opponent-read, Truth/QA, Guardrails) and save a structured playbook + move-format validator to `/memory/<game_id>/`.

### Tier 2 — High-ROI Differentiators for T2 & Round Robin (23:59)
- **Code/Simulator Synthesis in `prep`:** Have Haiku generate and self-test a Python state-tracker, legal-move validator, or lightweight Monte-Carlo / minimax / EV simulator on `sample_turn` during the 10-minute `prep` (safe fallback to pure LLM if generated code fails!).
- **Action Verification Loop:** Validate LLM-proposed moves locally before sending to `stdout` so you don't burn your single per-game warning.
- **Cost & Thinking Control:** Use `thinking: disabled` on trivial moves/high-turn games and bounded `budget_tokens` on critical decisions to stay well under `$0.75/match` and earn cost points.

### Tier 3 — Final Polish for T3 (Sun 10:30)
- **Opponent Modeling Across Games:** Track opponent chat patterns (bluff rate, cooperation/betrayal in Deals, injection attempts) in container RAM across the best-of-3 / best-of-4 match.
- **Analyze Public T1/T2 Replays:** Tune `prep` prompts and family heuristics based on the 12 games revealed after T1 and T2.

## Validation
- Run `arena-lite` practice matches (`--prep`, Docker image mode, vs `random_bot.py` and organizer bots) after every milestone.
