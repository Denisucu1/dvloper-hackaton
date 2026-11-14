"""Sparring partner for Colonel Blotto style games: allocates troops uniformly at random.

The kit's random bot returns "pass", which is illegal in allocation games, so it forfeits and tells us
nothing. This one reads the requested troop/field counts from the observation and returns a legal random
allocation. No model, no key.

    cd arena-lite
    python3 -m arena practice --games-dir games --game tg_blotto_probe \
        --bot-a "cmd:python3 ../main.py" --bot-b "cmd:python3 ../scripts/blotto_bot.py" \
        --env-file ../dev.env --best-of 1
"""
import json
import random
import re
import sys

_RNG = random.Random()


def allocate(observation: str) -> str:
    match = re.search(r"allocate\s+(\d+)\s+troops\s+across\s+(\d+)\s+fields", observation or "", re.I)
    if not match:
        return "pass"
    troops, fields = int(match.group(1)), int(match.group(2))
    if fields <= 1:
        return str(troops)
    # Uniform random composition: choose fields-1 separators among troops+fields-1 gaps.
    # Composition of `troops` into `fields` parts: separators among troops+fields-1 gaps (1..troops+fields-1).
    gaps = troops + fields
    cuts = sorted(_RNG.sample(range(1, gaps), fields - 1))
    parts, previous = [], 0
    for cut in cuts:
        parts.append(cut - previous - 1)
        previous = cut
    parts.append(gaps - previous - 1)
    return " ".join(str(part) for part in parts)


def say(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        kind = msg.get("type")
        if kind == "prep":
            say({"type": "ready"})
        elif kind == "start":
            say({"type": "ready", "chat": "Even split? Let us find out."})
        elif kind == "turn":
            say({"type": "action", "action": allocate(str(msg.get("observation", ""))),
                 "chat": "Troops are spread."})
        elif kind == "shutdown":
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
