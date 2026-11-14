---
name: hackathon-pitch
description: >-
  Use when preparing the Sunday afternoon presentation (30 pts Presentation +
  25 pts Harness Design), speaker notes, simple conceptual explanations for
  presenters, live demo script, or judge Q&A for AI Harness Arena.
  Trigger in English or Romanian for pitch, prezentare, demo, jurizare, or
  explaining how the harness works.
---

# Hackathon Presentation & Harness Story (AI Harness Arena)

The judges award **55 out of 100 points** on Sunday afternoon (**30 pts Presentation** + **25 pts Harness Design**). Ground every claim strictly in the team's actual code and tournament logs.

## Mandatory Opening & Structure

### 1. Individual Intros (~30 sec per teammate, mandatory rule 12.2)
Do NOT just say "I am X from team Y." Each member tells the room:
- Name and what they study or do
- What they have built or worked on before
- What they enjoy doing

### 2. Core Story (Tailored to the 25-pt Harness Design Rubric)
Walk the judges clearly through **what the harness actually does** (not just which framework/library you imported):
1. **Architecture Overview (The "Mesh" -> "Specialist" Flow):** Explain how an empty container turns into a game specialist during the 10-minute `prep` phase and persists artifacts in `/memory`.
2. **Rule Understanding & State Tracking:** How you parse unseen rules, `sample_turn`, and `observation` into structured state.
3. **Move Verification & Tools/Simulators:** How your harness generates or runs validators, search/simulators, or solvers, and prevents illegal moves/warnings.
4. **Budget, Clock & Robustness:** How you control Haiku 5.5 `thinking`, token spend (`$0.75` cap), and 20s deadline fallbacks.
5. **Chat Offense & Defense (Mandatory Compliance):** Show in real code/replays how opponent `chat` informs the decision while resisting prompt injections and Guardrails traps, and how your own `chat` bluffs/negotiates in sync with your move.

### 3. Live Demo & Lessons Learned
- Show a concrete replay or live practice/tournament match illustrating a smart decision (e.g., simulator/tool call + chat bluff or injection defense).
- Share **what failed and what you fixed** between T1, T2, Round Robin, and T3 (judges explicitly score "what you learned").

### 4. Tough Judge Q&A Prep
Prepare crisp 20-second answers for:
- *"Show us in code where opponent chat enters your move decision and cannot be bypassed."*
- *"What happens when your generated solver/code crashes or Haiku times out at second 19?"*
- *"How did you keep match spend low while still reasoning deeply on hard games?"*
- *"Which of the 7 game families broke your first design, and how did you fix it?"*

## Validation
- Verify every module name, diagram, and replay cited in the slides matches the frozen repository code that judges have read access to.
