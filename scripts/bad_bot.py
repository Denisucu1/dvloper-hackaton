"""Deliberately terrible bot: proves the bench gates actually catch bad behaviour.

- never chats (rule 6.2 violation),
- always sends an illegal action (two of them lose the game, rule 8.1),
- hangs on one turn if asked (timeout).
Used only by the bench self-check, never in the image.
"""
import json
import sys
import time

HANG = "--hang" in sys.argv
turns = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break
    msg = json.loads(line)
    kind = msg.get("type")
    if kind == "prep":
        print(json.dumps({"type": "ready"}), flush=True)
    elif kind == "start":
        print(json.dumps({"type": "ready"}), flush=True)
    elif kind == "turn":
        turns += 1
        if HANG and turns == 2:
            time.sleep(300)
        print(json.dumps({"type": "action", "action": "ZZZ"}), flush=True)
    elif kind == "shutdown":
        break
