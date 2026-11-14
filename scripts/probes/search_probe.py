"""Probe: search routing, tactics, determinism and budget.

Fixtures are tiny and self-contained (they never touch arena-lite):

- a 3x3 tic-tac-toe model for exact tactical wins/blocks and deterministic alpha-beta;
- a simultaneous rock-paper-scissors model for the joint (MCTS) route;
- a 30-action alternating model for the wide route and the wall-clock cap.

The probe records the chosen action/value and the node count, so a regression shows up as a named
check failure rather than a mystery slowdown. Score is the fraction of ten boolean checks that pass.
"""
from __future__ import annotations

import time

from harness import search

from scripts.probes.base import Probe, collect, register, summarize

_TTT_LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
_RPS_BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


class TicTacToe:
    """Minimal perfect-information fixture with the exact GameModel surface search expects."""

    def __init__(self, board: list[str] | None = None) -> None:
        self.board = list(board or ["."] * 9)

    @staticmethod
    def _index(action: str) -> int:
        return "ABC".index(action[0].upper()) + 3 * (int(action[1]) - 1)

    @staticmethod
    def _name(index: int) -> str:
        return "ABC"[index % 3] + str(index // 3 + 1)

    def update(self, observation: str) -> None:
        return None

    def legal_actions(self, seat: int) -> list[str]:
        return [self._name(index) for index, cell in enumerate(self.board) if cell == "."]

    def apply(self, seat: int, action: str) -> None:
        index = self._index(action)
        if self.board[index] != ".":
            raise ValueError("occupied")
        self.board[index] = "X" if seat == 0 else "O"

    def is_terminal(self) -> bool:
        return self._winner() is not None or "." not in self.board

    def to_move(self) -> list[int]:
        if self.is_terminal():
            return []
        return [0 if self.board.count("X") == self.board.count("O") else 1]

    def score(self, seat: int) -> float:
        winner = self._winner()
        if winner is None:
            return 0.5
        return 1.0 if winner == seat else 0.0

    def evaluate(self, seat: int) -> float:
        return 0.5 + 0.01 * (self.board.count("X" if seat == 0 else "O")
                             - self.board.count("O" if seat == 0 else "X"))

    def clone(self) -> "TicTacToe":
        return TicTacToe(self.board)

    def key(self) -> str:
        return "".join(self.board)

    def _winner(self) -> int | None:
        for a, b, c in _TTT_LINES:
            if self.board[a] != "." and self.board[a] == self.board[b] == self.board[c]:
                return 0 if self.board[a] == "X" else 1
        return None


class JointRps:
    """Both seats choose at once, so minimax has no tree: this must route to MCTS-joint."""

    def __init__(self, played: list[tuple[int, str]] | None = None) -> None:
        self.played = list(played or [])

    def update(self, observation: str) -> None:
        return None

    def legal_actions(self, seat: int) -> list[str]:
        return ["rock", "paper", "scissors"]

    def apply(self, seat: int, action: str) -> None:
        self.played.append((int(seat), str(action)))

    def is_terminal(self) -> bool:
        return len(self.played) >= 2

    def to_move(self) -> list[int]:
        return [] if self.is_terminal() else [0, 1]

    def score(self, seat: int) -> float:
        if not self.is_terminal():
            return 0.5
        mine = next(action for who, action in self.played if who == seat)
        theirs = next(action for who, action in self.played if who != seat)
        if _RPS_BEATS[mine] == theirs:
            return 1.0
        if _RPS_BEATS[theirs] == mine:
            return 0.0
        return 0.5

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "JointRps":
        return JointRps(self.played)


class WideGame:
    """30 actions on the first two plies, then terminal: exercises the wide-node path."""

    def __init__(self, used: list[str] | None = None) -> None:
        self.used = list(used or [])

    def update(self, observation: str) -> None:
        return None

    def legal_actions(self, seat: int) -> list[str]:
        return [f"m{index}" for index in range(30) if f"m{index}" not in self.used]

    def apply(self, seat: int, action: str) -> None:
        if action in self.used:
            raise ValueError("already used")
        self.used.append(str(action))

    def is_terminal(self) -> bool:
        return len(self.used) >= 2

    def to_move(self) -> list[int]:
        if self.is_terminal():
            return []
        return [0] if not self.used else [1]

    def score(self, seat: int) -> float:
        return 0.5

    def evaluate(self, seat: int) -> float:
        return 0.5

    def clone(self) -> "WideGame":
        return WideGame(self.used)


@register
class SearchProbe(Probe):
    name = "search.routing"
    component = "search"

    def run(self) -> dict:
        facts: dict = {}

        def takes_win() -> bool:
            action, value = search.best_action(
                model=TicTacToe(["X", "X", ".", "O", "O", ".", ".", ".", "."]),
                seat=0, seconds=0.5,
            )
            facts["win"] = {"action": action, "value": round(value, 4)}
            return action == "C1" and value >= 0.99

        def blocks_loss() -> bool:
            action, _value = search.best_action(
                model=TicTacToe(["O", "O", ".", "X", ".", ".", ".", ".", "X"]),
                seat=0, seconds=0.5,
            )
            facts["block"] = action
            return action == "C1"

        def terminal_has_no_move() -> bool:
            action, value = search.best_action(
                model=TicTacToe(["X", "X", "X", "O", "O", ".", ".", ".", "."]),
                seat=0, seconds=0.2,
            )
            return action is None and value == 0.5

        def empty_board_is_legal() -> bool:
            model = TicTacToe()
            action, _value = search.best_action(model=model, seat=0, seconds=0.2, max_depth=4)
            return action in model.legal_actions(0)

        def alphabeta_is_deterministic() -> bool:
            first = search.best_action(model=TicTacToe(), seat=0, seconds=1.0, max_depth=4)
            second = search.best_action(model=TicTacToe(), seat=0, seconds=1.0, max_depth=4)
            facts["determinism"] = [first[0], round(first[1], 6), second[0], round(second[1], 6)]
            return first[0] == second[0] and round(first[1], 6) == round(second[1], 6)

        def joint_route_is_reported() -> bool:
            route = search.route_for(JointRps(), 0)
            facts["joint_route"] = route
            return route == "mcts-joint"

        def joint_search_is_legal() -> bool:
            meta = search.best_action_meta(JointRps(), 0, seconds=0.3)
            facts["joint"] = {"action": meta.get("action"), "route": meta.get("route")}
            return meta.get("route") == "mcts-joint" and meta.get("action") in {"rock", "paper", "scissors"}

        def wide_route_is_reported() -> bool:
            route = search.route_for(WideGame(), 0)
            facts["wide_route"] = route
            return route == "alphabeta"

        def wide_search_is_bounded() -> bool:
            model = WideGame()
            started = time.monotonic()
            meta = search.best_action_meta(model, 0, seconds=0.15)
            elapsed = time.monotonic() - started
            facts["wide"] = {"action": meta.get("action"), "elapsed_s": round(elapsed, 3)}
            return meta.get("action") in model.legal_actions(0) and elapsed < 2.0

        def node_budget_is_enforced() -> bool:
            searcher = search.Searcher(TicTacToe(), 0, seconds=0.12, max_depth=20)
            started = time.monotonic()
            action, value = searcher.best()
            elapsed = time.monotonic() - started
            facts["budget"] = {
                "action": action, "value": round(value, 4), "nodes": searcher.nodes,
                "depth": searcher.depth_reached, "elapsed_s": round(elapsed, 3),
            }
            return action in TicTacToe().legal_actions(0) and 0 < searcher.nodes <= 400_000 and elapsed < 1.0

        checks = [
            ("takes the immediate win", takes_win),
            ("blocks the opponent's win", blocks_loss),
            ("terminal position has no move", terminal_has_no_move),
            ("empty board returns a legal move", empty_board_is_legal),
            ("alpha-beta is deterministic", alphabeta_is_deterministic),
            ("simultaneous position routes to MCTS-joint", joint_route_is_reported),
            ("joint search returns a legal move", joint_search_is_legal),
            ("wide position routes to alpha-beta", wide_route_is_reported),
            ("wide search returns a legal move in budget", wide_search_is_bounded),
            ("node/time cap bounds the search", node_budget_is_enforced),
        ]
        run = collect(checks)
        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"win {facts.get('win', {}).get('action')}, block {facts.get('block')}, "
            f"routes {facts.get('joint_route')}/{facts.get('wide_route')}, "
            f"{facts.get('budget', {}).get('nodes', 0)} nodes"
        )
        return summarize(run, notes=notes, details=details)
