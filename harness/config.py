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
# Claim checks run in the background with thinking on (6-7 s measured): a bound on how many per game and
# on how long one may take.
CRED_LLM = os.environ.get("HARNESS_CRED_LLM", "0") == "1"
CRED_MAX_CHECKS = int(os.environ.get("HARNESS_CRED_CHECKS", "6"))
CRED_EVERY = int(os.environ.get("HARNESS_CRED_EVERY", "3"))  # one check every N of our turns
CRED_TIMEOUT = float(os.environ.get("HARNESS_CRED_TIMEOUT", "25"))
# The per-game chat plan (harness/chatplan.py): one call with thinking, in prep or at the first start.
CHATPLAN = os.environ.get("HARNESS_CHATPLAN", "1") == "1"
CHATPLAN_THINKING = int(os.environ.get("HARNESS_CHATPLAN_THINKING", "2000"))
CHATPLAN_TIMEOUT = float(os.environ.get("HARNESS_CHATPLAN_TIMEOUT", "45"))
CHATPLAN_MIN_SECONDS = float(os.environ.get("HARNESS_CHATPLAN_MIN", "8"))
# Seconds of the study clock kept for `ready` itself when the plan has to be built at `start`.
CHATPLAN_RESERVE = float(os.environ.get("HARNESS_CHATPLAN_RESERVE", "12"))
# Families with no solver, where the model decides the move alone: bounded thinking when the clock allows.
# Empty string switches it off.
THINK_FAMILIES = tuple(item for item in os.environ.get("HARNESS_THINK_FAMILIES", "deals,truth,guardrails,unknown").split(",") if item)
# Post-game review (harness/lessons.py): one background call with thinking at `end`.
LESSONS = os.environ.get("HARNESS_LESSONS", "1") == "1"
LESSONS_THINKING = int(os.environ.get("HARNESS_LESSONS_THINKING", "2000"))
LESSONS_TIMEOUT = float(os.environ.get("HARNESS_LESSONS_TIMEOUT", "45"))
LESSONS_KEPT = int(os.environ.get("HARNESS_LESSONS_KEPT", "9"))
LESSONS_SHOWN = int(os.environ.get("HARNESS_LESSONS_SHOWN", "4"))
# The strategy brief (harness/strategy.py): one prep call with a large thinking budget.
STRATEGY = os.environ.get("HARNESS_STRATEGY", "1") == "1"
STRATEGY_THINKING = int(os.environ.get("HARNESS_STRATEGY_THINKING", "2000"))
STRATEGY_TIMEOUT = float(os.environ.get("HARNESS_STRATEGY_TIMEOUT", "240"))
STRATEGY_MIN_SECONDS = float(os.environ.get("HARNESS_STRATEGY_MIN", "40"))
STRATEGY_CHARS = int(os.environ.get("HARNESS_STRATEGY_CHARS", "2600"))
# Seconds of the study clock kept for the rest of `start` when the brief is built there (cold container).
STRATEGY_RESERVE = float(os.environ.get("HARNESS_STRATEGY_RESERVE", "35"))
MAX_PLAYBOOK_CHARS = int(os.environ.get("HARNESS_MAX_PLAYBOOK", "4200"))
# Voting (decide.py): on a turn where no code move exists, this many answers are asked for in parallel and
# the move most of them agree on is played. 1 switches voting off.
VOTERS = int(os.environ.get("HARNESS_VOTERS", "3"))
# The council (harness/council.py): its own cap on the clock, and how much clock must be left for the
# strategists to think.
COUNCIL_TIMEOUT = float(os.environ.get("HARNESS_COUNCIL_TIMEOUT", "8"))
COUNCIL_MIN_SECONDS = float(os.environ.get("HARNESS_COUNCIL_MIN", "3"))
COUNCIL_THINK_SECONDS = float(os.environ.get("HARNESS_COUNCIL_THINK", "12"))
VOTE_MIN_SECONDS = float(os.environ.get("HARNESS_VOTE_MIN", "8"))
# One retry on a busy gateway (429 / 5xx): how long to wait, and how much of the call's own timeout must
# still be left for the retry to be worth making.
RETRY_WAIT = float(os.environ.get("HARNESS_RETRY_WAIT", "1.2"))
RETRY_MIN_LEFT = float(os.environ.get("HARNESS_RETRY_MIN_LEFT", "4.0"))
# Prep: waits (seconds) between attempts of the spec call when the gateway gives nothing back.
PREP_SPEC_WAITS = tuple(float(x) for x in os.environ.get("HARNESS_PREP_SPEC_WAITS", "5,15,30,45").split(",") if x)
# Chat offense: every 4th line may be aimed at the opponent's model (rule 6.7 allows it). 0 switches it off.
CHAT_INJECT = os.environ.get("HARNESS_CHAT_INJECT", "1") == "1"
# One retry on a busy gateway (429 / 5xx): how long to wait, and how much of the call's own timeout must
# still be left for the retry to be worth making.
RETRY_WAIT = float(os.environ.get("HARNESS_RETRY_WAIT", "1.2"))
RETRY_MIN_LEFT = float(os.environ.get("HARNESS_RETRY_MIN_LEFT", "4.0"))
# Prep: waits (seconds) between attempts of the spec call when the gateway gives nothing back.
PREP_SPEC_WAITS = tuple(float(x) for x in os.environ.get("HARNESS_PREP_SPEC_WAITS", "5,15,30,45").split(",") if x)

# Files, never directories, that hold per-game artifacts.
ARTIFACT_SPEC = "spec.json"
ARTIFACT_PLAYBOOK = "playbook.txt"
ARTIFACT_SAMPLES = "samples.json"
ARTIFACT_RULES = "rules.txt"
ARTIFACT_GENPOLICY = "policy.py"
ARTIFACT_RESULTS = "results.jsonl"
