"""A chat plan written for ONE game, once, with thinking on.

The family tactics in chatpolicy.py are generic: "represent", "anchor", "misdirect". A game we have never
seen has its own secrets, its own tells and its own traps, and the rules text says what they are. This
module spends part of the prep clock (or, failing that, of the study clock) and a little of the budget -
one model call with thinking, about a tenth of a cent - to read the rules and write:

- what we must never reveal in this game, in the game's own terms;
- what the opponent might let slip or claim, and how the game would later confirm or refute it;
- four chat tactics specific to this game;
- how the opponent could use the chat against us here;
- the commitments that lose outright (accepting below a private value, naming a forbidden subject).

The plan is cached in /memory/<game_id>/<rules hash>-chatplan.json, so every container of the team and
every later tournament gets it for free (rule 4.5). It only ever adds guidance: the code gates in
chatpolicy.py and decide.py stay the last word on what leaves the bot.

Prep hook (one line in prep, once per variant): `chatplan.prepare(gateway, game_id, rules, spec)`.
Without the hook the plan is built at the first `start` if the study clock allows, else never.
"""
from __future__ import annotations

import re

from . import config, memory
from .llm import Gateway, extract_json
from .log import kv, log

ARTIFACT = "chatplan.json"
VERSION = 1

_SYSTEM = """You prepare the chat strategy of a game-playing bot for ONE specific two-player game. Every move the
bot makes comes with one chat line (max 200 characters) that the opponent's AI and an audience read; the
opponent chats too, and may bluff, lie or try to give the bot instructions. Work only from the rules text.
Be concrete and specific to THIS game: name its cards, numbers, fields, offers, questions. Answer with one
JSON object and nothing else."""

_USER = """RULES OF THE GAME:
{rules}

Write the chat plan for a player of this game:
{{"private": ["<up to 5 things this player must never reveal in the chat, in this game's own terms>"],
 "their_tells": ["<up to 4 things the opponent might claim or let slip in the chat that would matter here, each with how the game later confirms or refutes it>"],
 "tactics": [{{"name": "<two-or-three-word-name>", "line": "<how to write one chat line with this tactic in THIS game: which false or unverifiable impression to create, and in which situation to use it>"}}],
 "traps": ["<up to 4 ways the opponent could use the chat against this player in THIS game>"],
 "speech_rules": "<exactly what the rules forbid saying, if anything; empty string if nothing>",
 "never_do": ["<up to 3 moves or commitments that lose outright or badly in this game, e.g. accepting less than a private value>"]}}
Give exactly 4 tactics. Each tactic is OFFENSIVE: it plants a false or unverifiable impression, or applies
pressure, that pushes the opponent toward one specific mistake in this game (name the mistake). No tactic
of the kind "stay neutral" or "say only true facts". A tactic never asks the player to reveal true private
information, to break a speech rule, or to play a weaker move: the chat line serves the move, it never
chooses it. Use only numbers and terms that appear in the rules."""


def _strings(value: object, limit: int, width: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text:
            out.append(text[:width])
    return out[:limit]


def clean(data: dict | None) -> dict | None:
    """Keep only what has the right shape; a plan with no usable tactic is no plan."""
    if not isinstance(data, dict):
        return None
    tactics = []
    for item in data.get("tactics") or []:
        if not isinstance(item, dict):
            continue
        name = re.sub(r"[^a-z0-9]+", "-", str(item.get("name") or "").lower()).strip("-")[:28]
        line = re.sub(r"\s+", " ", str(item.get("line") or "")).strip()[:320]
        if name and len(line) >= 20:
            tactics.append({"name": name, "line": line})
    plan = {
        "version": VERSION,
        "private": _strings(data.get("private"), 5, 160),
        "their_tells": _strings(data.get("their_tells"), 4, 220),
        "tactics": tactics[:4],
        "traps": _strings(data.get("traps"), 4, 200),
        "speech_rules": re.sub(r"\s+", " ", str(data.get("speech_rules") or "")).strip()[:300],
        "never_do": _strings(data.get("never_do"), 3, 200),
    }
    return plan if plan["tactics"] or plan["private"] or plan["traps"] else None


def _path(game_id: object, rules: str):
    return memory.artifact_path(game_id or "unknown", memory.rules_fingerprint(rules or ""), ARTIFACT)


def load(game_id: object, rules: str) -> dict | None:
    data = memory.read_json(_path(game_id, rules), default=None)
    return data if isinstance(data, dict) and data.get("version") == VERSION else None


def build(gateway: Gateway, rules: str, timeout: float) -> dict | None:
    """One call with thinking. Returns None on any failure: the generic tactics remain."""
    if (not config.CHATPLAN or not gateway.affordable or timeout < config.CHATPLAN_MIN_SECONDS
            or not (rules or "").strip()):
        return None
    text = gateway.ask(
        _SYSTEM,
        _USER.format(rules=rules[: config.MAX_DOC_CHARS]),
        max_tokens=config.CHATPLAN_THINKING + 1800,
        timeout=min(timeout, config.CHATPLAN_TIMEOUT),
        temperature=0.4,
        thinking="enabled",
        budget_tokens=config.CHATPLAN_THINKING,
    )
    return clean(extract_json(text))


def prepare(gateway: Gateway, game_id: object, rules: str, timeout: float = 60.0) -> dict | None:
    """Load the plan for these rules, or build and cache it. Safe to call from prep for every variant."""
    plan = load(game_id, rules)
    if plan is not None:
        return plan
    plan = build(gateway, rules, timeout)
    if plan is None:
        return None
    try:
        memory.atomic_write_json(_path(game_id, rules), plan)
    except Exception as exc:  # noqa: BLE001 - a cache, never a reason to fail
        log("chat plan cache write failed:", type(exc).__name__)
    log(kv(chatplan=game_id, tactics=len(plan["tactics"]), private=len(plan["private"]), traps=len(plan["traps"])))
    return plan
