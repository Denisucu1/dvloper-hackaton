"""Runtime configuration. Everything is env-driven so the same image works locally and in the arena.

The arena sets LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / TEAM_ID / MEMORY_DIR and, in the arena only,
HTTP_PROXY / HTTPS_PROXY / NO_PROXY. Nothing here may ever require a secret to be baked into the image.
"""
from __future__ import annotations

import os

VERSION = "0.1.0-skeleton"

TEAM_ID = os.environ.get("TEAM_ID", "adn")

# /memory is shared by every container of the team and survives the whole event.
MEMORY_DIR = os.environ.get("MEMORY_DIR") or os.path.join(os.getcwd(), "memory")

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

# Experimental: let the model write a `choose()` policy during prep, run it in a sandbox.
# Off by default: a wrong generated module is worse than the generic fallback.
GENPOLICY = os.environ.get("HARNESS_GENPOLICY", "0") == "1"

# Limits that keep prompts (and the bill) bounded.
MAX_RULES_CHARS = int(os.environ.get("HARNESS_MAX_RULES", "9000"))
MAX_CHAT_LINES = int(os.environ.get("HARNESS_MAX_CHAT_LINES", "24"))
MAX_CHAT_CHARS = 200

# Files, never directories, that hold per-game artifacts.
ARTIFACT_SPEC = "spec.json"
ARTIFACT_PLAYBOOK = "playbook.txt"
ARTIFACT_SAMPLES = "samples.json"
ARTIFACT_RULES = "rules.txt"
ARTIFACT_GENPOLICY = "policy.py"
ARTIFACT_RESULTS = "results.jsonl"
