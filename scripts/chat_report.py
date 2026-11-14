"""What did our chat do in a practice match? Reads arena-lite replays and the bot's stderr log.

    python scripts/chat_report.py arena-lite/results/practice/matches/<id>.json [more.json ...] [--bot bot-a]

For every match: the result, our chat lines (count, longest, any over 200), warnings, the tactic mix, what
the speech gate did, and for every line aimed at the opponent's model ("inject") the opponent's next move,
so "did the attack work" is read from the replay instead of guessed.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def report(path: Path, bot: str) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    ours = [line for line in doc.get("chat", []) if line.get("team") == bot and line.get("kind") == "chat"]
    lengths = [len(line["text"]) for line in ours] or [0]
    logs_dir = path.parent.parent / "logs"
    # Without --prep one log per match; with --prep each specialist logs to logs/warm/<bot>--<game>.log.
    sources = [logs_dir / f"{path.stem}-{bot}.log"]
    sources += [logs_dir / "warm" / f"{bot}--{game.get('game_id')}.log" for game in doc.get("games", [])]
    log = "\n".join(source.read_text(encoding="utf-8", errors="replace")
                    for source in dict.fromkeys(sources) if source.exists())
    tactics = Counter(re.findall(r"chat_tactic=([\w-]+)", log))
    gates = Counter(re.findall(r"chat_gate=([^=]+?) read=", log))
    out = {
        "match": path.stem,
        "games": [f"{game['game_id']}: {(game.get('result') or {}).get('reason', '?')}" for game in doc.get("games", [])],
        "score": doc.get("score"),
        "winner": doc.get("winner"),
        "our_lines": len(ours),
        "longest": max(lengths),
        "over_200": sum(1 for n in lengths if n > 200),
        "warnings": sum(int(v) for game in doc.get("games", []) for v in (game.get("warnings") or {}).values()),
        "tactics": dict(tactics),
        "gate": dict(gates),
        "audits": re.findall(r"audit rejected .*", log),
        "inject": [],
    }
    # Each inject line, and what the opponent did next. Every turn logs one tactic and sends one line, in
    # the same order, so the play-phase lines and the logged tactics pair up one to one.
    played = [line for line in ours if line.get("phase") == "play"]
    order = re.findall(r"chat_tactic=([\w-]+)", log)
    games = {game.get("index"): game for game in doc.get("games", [])}
    for line, tactic in zip(played, order):
        if tactic != "inject":
            continue
        game = games.get(line.get("game")) or {}
        seat = game.get("seats", [bot]).index(bot) if bot in game.get("seats", []) else 0
        answer = None
        for step in game.get("timeline", []):
            if step.get("step", 0) > line.get("step", 0):
                theirs = (step.get("actions") or {}).get(str(1 - seat))
                if theirs:
                    answer = theirs.get("action")
                    break
        out["inject"].append({"said": line["text"], "their_next_move": answer})
    return out


def main() -> int:
    args = sys.argv[1:]
    bot = "bot-a"
    if "--bot" in args:
        at = args.index("--bot")
        bot = args[at + 1]
        del args[at : at + 2]
    for name in args:
        print(json.dumps(report(Path(name), bot), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
