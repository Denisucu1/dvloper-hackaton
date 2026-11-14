"""Everything this container knows about the game it is currently playing.

A specialist keeps this alive across matches (rule 5.3): the spec, the playbook, the engine and the chat
policy survive `end`; only `shutdown` (or a closed stdin) ends the process.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config, memory
from .chatpolicy import ChatPolicy
from .engine import Engine
from .llm import Gateway
from .prep import PrepResult
from .spec import Spec, fallback_spec


@dataclass
class Session:
    gateway: Gateway = field(default_factory=Gateway)
    spec: Spec = field(default_factory=Spec)
    playbook: str = ""
    rules: str = ""
    genpolicy: str | None = None
    engine: Engine | None = None
    chat: ChatPolicy | None = None
    game: dict = field(default_factory=dict)
    prepared_game: str = ""
    turn: int = 0
    avoid: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    games_played: int = 0

    # ------------------------------------------------------------------ prep
    def adopt(self, result: PrepResult) -> None:
        self.spec = result.spec
        self.playbook = result.playbook
        self.rules = result.rules
        self.genpolicy = result.genpolicy
        self.prepared_game = result.spec.game_id or self.prepared_game
        self._rebuild()

    def cold_start(self, msg: dict) -> None:
        """No prep artifacts: build the cheapest useful spec straight from `start.rules`."""
        game_id = str(msg.get("game_id") or "unknown")
        rules = str(msg.get("rules") or "")
        self.spec = fallback_spec(game_id, rules, [], memory.rules_fingerprint(rules))
        self.playbook = ""
        self.rules = rules
        self.genpolicy = None
        self.prepared_game = game_id
        self._rebuild()

    def _rebuild(self) -> None:
        self.engine = Engine(self.spec, self.genpolicy, seed=self.spec.fingerprint)
        self.chat = ChatPolicy(self.spec, seed=self.spec.fingerprint)

    # ------------------------------------------------------------------ play
    def begin_game(self, msg: dict) -> None:
        self.game = msg
        self.turn = 0
        self.avoid = []
        if msg.get("rules"):
            self.rules = str(msg["rules"])

    def note_action(self, action: str) -> None:
        if action:
            self.avoid.append(action)
            del self.avoid[:-20]

    @property
    def game_id(self) -> str:
        return str(self.game.get("game_id") or self.prepared_game or "unknown")

    @property
    def seat(self) -> int:
        return int(self.game.get("seat", 0) or 0)

    def summary(self) -> dict:
        return {
            "team": config.TEAM_ID,
            "games_played": self.games_played,
            "game_id": self.game_id,
            "family": self.spec.family,
            "up_seconds": round(time.time() - self.started, 1),
            "llm": self.gateway.summary(),
        }
