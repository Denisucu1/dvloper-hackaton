"""Runtime configuration. Everything is env-driven so the same image works locally and in the arena.

The arena sets LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / TEAM_ID / MEMORY_DIR and, in the arena only,
HTTP_PROXY / HTTPS_PROXY / NO_PROXY. Nothing here may ever require a secret to be baked into the image.
"""
from __future__ import annotations

import os

VERSION = "0.1.0-skeleton"

TEAM_ID = os.environ.get("TEAM_ID", "adn")

# /memory is shared by every container of the team and survives the whole event.
MEMORY_DIR = os.environ.get("MEMORY_DIR") or (
    "/memory" if os.path.isdir("/memory") else os.path.join(os.getcwd(), "memory")
)

# Defaults used before the arena sends its first `prep` / `start` (local practice, smoke tests).
LLM = {
    "base_url": os.environ.get("LLM_BASE_URL", "http://127.0.0.1:4000/v1"),
    "api_key": os.environ.get("LLM_API_KEY", "missing-key"),
    "model": os.environ.get("LLM_MODEL", "haiku"),
}

# Haiku 5.5 prices, USD per million tokens (rulebook 7.5).
PRICE_INPUT = 0.10
PRICE_OUTPUT = 0.50
PRICE_CACHE_READ = 0.01

# One fresh key per prep container ($0.75) and one per match, shared by all our containers in it.
PREP_BUDGET = float(os.environ.get("HARNESS_PREP_BUDGET", "0.75"))
MATCH_BUDGET = float(os.environ.get("HARNESS_MATCH_BUDGET", "0.75"))
# A match key is shared, so one container must never eat the whole budget by itself.
MATCH_CONTAINER_SHARE = float(os.environ.get("HARNESS_MATCH_SHARE", "0.40"))

# Seconds kept free before a clock runs out (network jitter, JSON, stdout flush).
SAFETY = float(os.environ.get("HARNESS_SAFETY", "3.0"))
# Never spend more than this fraction of a clock on model work.
PREP_SHARE = float(os.environ.get("HARNESS_PREP_SHARE", "0.55"))

# Let the model write a `choose()` policy during prep, validated on every sample_turn in an isolated sandbox.
GENPOLICY = os.environ.get("HARNESS_GENPOLICY", "1") == "1"

# Limits that keep prompts (and the bill) bounded.
MAX_RULES_CHARS = int(os.environ.get("HARNESS_MAX_RULES", "9000"))
# Observations carry the question at their end and, in Truth games, sometimes the whole document.
MAX_OBS_CHARS = int(os.environ.get("HARNESS_MAX_OBS", "12000"))
MAX_DOC_CHARS = int(os.environ.get("HARNESS_MAX_DOC", "40000"))
MAX_CHAT_LINES = int(os.environ.get("HARNESS_MAX_CHAT_LINES", "24"))
MAX_CHAT_CHARS = 200
# Deals: the audit of an acceptance. Seconds LEFT on the move clock that each mode needs, and the hard cap
# on the call itself, so the audit can never be what makes us late (rule 8.2: a late answer loses the game).
AUDIT_MIN_SECONDS = float(os.environ.get("HARNESS_AUDIT_MIN", "3.0"))
AUDIT_THINKING_SECONDS = float(os.environ.get("HARNESS_AUDIT_THINKING", "10.5"))
AUDIT_THINKING_CAP = float(os.environ.get("HARNESS_AUDIT_THINKING_CAP", "9.0"))
AUDIT_FAST_CAP = float(os.environ.get("HARNESS_AUDIT_FAST_CAP", "4.0"))
# Chat offense: every 4th line may be aimed at the opponent's model (rule 6.7 allows it). 0 switches it off.
CHAT_INJECT = os.environ.get("HARNESS_CHAT_INJECT", "1") == "1"

# Files, never directories, that hold per-game artifacts.
ARTIFACT_SPEC = "spec.json"
ARTIFACT_PLAYBOOK = "playbook.txt"
ARTIFACT_SAMPLES = "samples.json"
ARTIFACT_RULES = "rules.txt"
ARTIFACT_GENPOLICY = "policy.py"
ARTIFACT_RESULTS = "results.jsonl"
