"""Tabular CFR+ for small extensive-form games, with a self-contained Kuhn poker probe.

What is here
------------
- A tiny *generic* game-tree interface: `Terminal`, `Chance` and `Decision` dataclasses. A `Decision`
  names its player, its information set and the actions/children; a `Chance` node carries
  `(probability, child)` branches; a `Terminal` carries one utility per player.
- `CfrTrainer`: tabular counterfactual regret minimisation with CFR+ flooring (regrets are clamped at
  zero after every update) and linear strategy averaging (`weight = t`). Alternating updates are the
  default, which is the usual CFR+ variant and converges quickly on small games.
- `build_kuhn()` / `solve_kuhn()`: one-card (Kuhn) poker as a 12-information-set tree. `solve_kuhn`
  returns a `Policy` (average strategy per information set).
- `kuhn_exploitability()`: an exact best-response proxy for Kuhn (the game is small enough to enumerate
  pure strategies). This is a test instrument, not a runtime dependency.
- `CfrSolver`: a conservative `Solver`-shaped wrapper for Kuhn-shaped observation text. It declines
  anything it cannot parse; the harness integration is therefore safe to keep behind `HARNESS_CFR`.

Design boundaries
-----------------
- stdlib only; no training, no external solver.
- Runtime `CfrSolver.propose` uses a cached policy trained once (2000 CFR+ iterations, well under the
  per-turn budget). The offline `solve_kuhn` default is 20000 iterations for a tighter profile.
- The best-response proxy enumerates pure strategies and is intended for games the size of Kuhn. It
  raises `ValueError` rather than silently returning a wrong number when the cap is exceeded.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Any, Iterable, Mapping, Sequence, Union

# ---------------------------------------------------------------------------- game-tree interface
@dataclass(frozen=True)
class Terminal:
    """A finished game: one utility per player (zero-sum games use u0 + u1 == 0)."""

    values: tuple[float, ...]


@dataclass(frozen=True)
class Chance:
    """A chance node: `(probability, child)` branches. Probabilities must sum to one."""

    branches: tuple[tuple[float, "Node"], ...]


@dataclass(frozen=True)
class Decision:
    """A decision node: `player` chooses among `actions`; children are aligned with actions."""

    player: int
    infoset: str
    actions: tuple[str, ...]
    children: tuple["Node", ...]


Node = Union[Terminal, Chance, Decision]


# Convenience constructors keep example trees short and immutable.
def terminal(*values: float) -> Terminal:
    return Terminal(tuple(float(value) for value in values))


def chance(*branches: tuple[float, Node]) -> Chance:
    return Chance(tuple((float(probability), child) for probability, child in branches))


def decision(player: int, infoset: str, actions: Iterable[str], children: Iterable[Node]) -> Decision:
    return Decision(int(player), str(infoset), tuple(str(action) for action in actions), tuple(children))


# ---------------------------------------------------------------------------- regret matching
def regret_matching(regrets: Sequence[float]) -> list[float]:
    """Normalise positive cumulative regrets into a strategy; uniform when no regret is positive."""
    positive = [float(value) if math.isfinite(float(value)) and value > 0 else 0.0 for value in regrets]
    total = sum(positive)
    if total > 0.0:
        return [value / total for value in positive]
    count = len(positive)
    if count == 0:
        return []
    return [1.0 / count] * count


@dataclass
class Policy:
    """A behavioural strategy: information set -> {action: probability}."""

    strategies: dict[str, dict[str, float]] = field(default_factory=dict)
    seed: int | None = None
    iterations: int = 0

    def probs(self, infoset: str, actions: Sequence[str] | None = None) -> dict[str, float]:
        """Probabilities restricted to `actions` (all stored actions when omitted)."""
        stored = self.strategies.get(infoset) or {}
        wanted = [str(action) for action in actions] if actions is not None else list(stored)
        if not wanted:
            return {}
        values = [max(0.0, float(stored.get(action, 0.0))) for action in wanted]
        total = sum(values)
        if total <= 0.0:
            uniform = 1.0 / len(wanted)
            return {action: uniform for action in wanted}
        return {action: value / total for action, value in zip(wanted, values)}

    def probability(self, infoset: str, action: str) -> float:
        return self.probs(infoset, [action]).get(action, 0.0)

    def best(self, infoset: str, actions: Sequence[str] | None = None) -> str | None:
        probabilities = self.probs(infoset, actions)
        if not probabilities:
            return None
        return max(probabilities, key=probabilities.__getitem__)

    def sample(self, infoset: str, actions: Sequence[str] | None = None, rng: Any | None = None) -> str | None:
        probabilities = self.probs(infoset, actions)
        if not probabilities:
            return None
        generator = rng if rng is not None else random.Random(self.seed)
        roll = generator.random()
        running = 0.0
        last = None
        for action, probability in probabilities.items():
            last = action
            running += probability
            if roll < running:
                return action
        return last


def _strategies_of(policy: Policy | Mapping[str, Mapping[str, float]]) -> Mapping[str, Mapping[str, float]]:
    if isinstance(policy, Policy):
        return policy.strategies
    return policy


def _probabilities(
    strategies: Mapping[str, Mapping[str, float]],
    infoset: str,
    actions: Sequence[str],
) -> list[float]:
    stored = strategies.get(infoset) or {}
    values = [max(0.0, float(stored.get(action, 0.0))) for action in actions]
    total = sum(values)
    if total > 0.0:
        return [value / total for value in values]
    count = len(actions)
    return [1.0 / count] * count if count else []


# ---------------------------------------------------------------------------- generic CFR+
class CfrTrainer:
    """Tabular CFR+ over an explicit game tree.

    Regret and average-strategy tables are keyed by information-set id. The tree is validated while
    collecting infosets: a repeated infoset must expose the same player and the same action list, chance
    probabilities must sum to one, and every terminal must carry one utility per player.
    """

    def __init__(self, root: Node) -> None:
        self.root = root
        self.players = 0
        self.actions: dict[str, tuple[str, ...]] = {}
        self.infoset_players: dict[str, int] = {}
        self.regrets: dict[str, list[float]] = {}
        self.strategy_sum: dict[str, list[float]] = {}
        self.iterations_done = 0
        self._terminals: list[tuple[float, ...]] = []
        self._collect(root)
        if self.players <= 0:
            raise ValueError("game tree has no terminal values")
        for values in self._terminals:
            if len(values) != self.players:
                raise ValueError("every terminal must carry one utility per player")
        self._player_range = range(self.players)

    # ------------------------------------------------------------------ construction
    def _collect(self, node: Node) -> None:
        if isinstance(node, Terminal):
            values = tuple(float(value) for value in node.values)
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError("terminal values must be finite and non-empty")
            self._terminals.append(values)
            self.players = max(self.players, len(values))
            return
        if isinstance(node, Chance):
            if not node.branches:
                raise ValueError("chance node needs at least one branch")
            total = 0.0
            for probability, child in node.branches:
                probability = float(probability)
                if not math.isfinite(probability) or probability < 0.0:
                    raise ValueError("chance probabilities must be finite and non-negative")
                total += probability
                self._collect(child)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"chance probabilities must sum to 1, got {total}")
            return
        if isinstance(node, Decision):
            player = int(node.player)
            if player < 0:
                raise ValueError("decision player must be non-negative")
            self.players = max(self.players, player + 1)
            actions = tuple(str(action) for action in node.actions)
            if not actions or len(actions) != len(node.children):
                raise ValueError("decision needs one child per action")
            if len(set(actions)) != len(actions):
                raise ValueError(f"duplicate actions in infoset {node.infoset!r}")
            previous = self.actions.get(node.infoset)
            if previous is None:
                self.actions[node.infoset] = actions
                self.infoset_players[node.infoset] = player
                self.regrets[node.infoset] = [0.0] * len(actions)
                self.strategy_sum[node.infoset] = [0.0] * len(actions)
            else:
                if previous != actions:
                    raise ValueError(f"infoset {node.infoset!r} has inconsistent action lists")
                if self.infoset_players[node.infoset] != player:
                    raise ValueError(f"infoset {node.infoset!r} belongs to two players")
            for child in node.children:
                self._collect(child)
            return
        raise TypeError(f"unknown node type: {type(node).__name__}")

    # ------------------------------------------------------------------ training
    def _snapshot(self) -> dict[str, list[float]]:
        # Regrets are finite by construction (CFR+ clamps at zero), so a lean inline normalisation is
        # enough here; `regret_matching()` remains the defensive public helper.
        snapshot: dict[str, list[float]] = {}
        for infoset, regrets in self.regrets.items():
            positive = [value if value > 0.0 else 0.0 for value in regrets]
            total = sum(positive)
            if total > 0.0:
                snapshot[infoset] = [value / total for value in positive]
            else:
                count = len(regrets)
                snapshot[infoset] = [1.0 / count] * count if count else []
        return snapshot

    def _walk(
        self,
        node: Node,
        reach: list[float],
        chance: float,
        weight: float,
        strategy: dict[str, list[float]],
        update_player: int | None,
    ) -> Sequence[float]:
        if isinstance(node, Terminal):
            return node.values
        players = self._player_range
        if isinstance(node, Chance):
            total = [0.0] * self.players
            for probability, child in node.branches:
                values = self._walk(child, reach, chance * probability, weight, strategy, update_player)
                for player in players:
                    total[player] += probability * values[player]
            return total

        player = node.player
        sigma = strategy.get(node.infoset)
        children = node.children
        count = len(children)
        if sigma is None or len(sigma) != count:
            sigma = [1.0 / count] * count if count else []
        child_values: list[Sequence[float]] = []
        node_value = [0.0] * self.players
        own_reach = reach[player]
        # Mutate the single per-player reach entry and restore it, instead of copying the whole reach
        # vector for every child. This is the hot loop of the trainer (~1.1M nodes for 20k iterations).
        for index, child in enumerate(children):
            reach[player] = own_reach * sigma[index]
            values = self._walk(child, reach, chance, weight, strategy, update_player)
            reach[player] = own_reach
            child_values.append(values)
            probability = sigma[index]
            for other in players:
                node_value[other] += probability * values[other]

        if update_player is None or player == update_player:
            # Counterfactual reach for `player`: chance times every *other* player's reach. Our own
            # reach is deliberately excluded from the regret weight.
            opponent_reach = chance
            for other in players:
                if other != player:
                    opponent_reach *= reach[other]
            regrets = self.regrets[node.infoset]
            target = node_value[player]
            for index in range(count):
                delta = opponent_reach * (child_values[index][player] - target)
                regrets[index] = max(0.0, regrets[index] + delta)  # CFR+ flooring
            sums = self.strategy_sum[node.infoset]
            for index in range(count):
                sums[index] += weight * own_reach * sigma[index]  # linear averaging
        return node_value

    def run(
        self,
        iterations: int = 20000,
        *,
        alternating: bool = True,
        average_delay: int = 0,
    ) -> "CfrTrainer":
        """Run CFR+ for `iterations` rounds. Returns self so callers can chain `average_policy()`."""
        rounds = max(0, int(iterations))
        for iteration in range(1, rounds + 1):
            weight = float(max(0, iteration - int(average_delay)))
            if alternating:
                for player in range(self.players):
                    strategy = self._snapshot()
                    self._walk(self.root, [1.0] * self.players, 1.0, weight, strategy, player)
            else:
                strategy = self._snapshot()
                self._walk(self.root, [1.0] * self.players, 1.0, weight, strategy, None)
        self.iterations_done += rounds
        return self

    # ------------------------------------------------------------------ output
    def average_policy(self) -> Policy:
        strategies: dict[str, dict[str, float]] = {}
        for infoset, actions in self.actions.items():
            sums = self.strategy_sum[infoset]
            total = sum(sums)
            if total > 0.0:
                probabilities = [value / total for value in sums]
            else:
                probabilities = [1.0 / len(actions)] * len(actions)
            strategies[infoset] = {
                action: probabilities[index] for index, action in enumerate(actions)
            }
        return Policy(strategies=strategies, iterations=self.iterations_done)

    def current_policy(self) -> Policy:
        """The strategy implied by the current regrets (useful for debugging/tests)."""
        strategies = {
            infoset: {
                action: probabilities[index]
                for index, action in enumerate(actions)
            }
            for infoset, actions in self.actions.items()
            for probabilities in [regret_matching(self.regrets[infoset])]
        }
        return Policy(strategies=strategies, iterations=self.iterations_done)


def validate_tree(root: Node) -> int:
    """Return the number of players, or raise ValueError if the tree is malformed."""
    return CfrTrainer(root).players


# ---------------------------------------------------------------------------- Kuhn poker
_KUHN_CARDS = ("J", "Q", "K")
_KUHN_RANK = {card: rank for rank, card in enumerate(_KUHN_CARDS)}

# Public histories: "" is player 0's open, "k"/"b" are player 1's replies, "kb"/"b" alternatives,
# and the terminal histories are check-check, check-bet-call, check-bet-fold, bet-call, bet-fold.
_KUHN_ACTIONS: dict[str, tuple[str, ...]] = {
    "": ("check", "bet"),
    "k": ("check", "bet"),
    "kb": ("call", "fold"),
    "b": ("call", "fold"),
}
_KUHN_NEXT: dict[tuple[str, str], str] = {
    ("", "check"): "k",
    ("", "bet"): "b",
    ("k", "check"): "kk",
    ("k", "bet"): "kb",
    ("kb", "call"): "kbc",
    ("kb", "fold"): "kbf",
    ("b", "call"): "bc",
    ("b", "fold"): "bf",
}
_KUHN_TERMINAL = {"kk", "kbc", "kbf", "bc", "bf"}
_KUHN_ACTOR = {"": 0, "k": 1, "kb": 0, "b": 1}


def _kuhn_showdown(card0: str, card1: str, stake: float) -> Terminal:
    if _KUHN_RANK[card0] > _KUHN_RANK[card1]:
        return Terminal((stake, -stake))
    return Terminal((-stake, stake))


def _kuhn_node(card0: str, card1: str, history: str) -> Node:
    if history == "kk":
        return _kuhn_showdown(card0, card1, 1.0)
    if history in ("kbc", "bc"):
        return _kuhn_showdown(card0, card1, 2.0)
    if history == "kbf":
        return Terminal((-1.0, 1.0))  # player 0 folded after check-bet
    if history == "bf":
        return Terminal((1.0, -1.0))  # player 1 folded to the open bet
    player = _KUHN_ACTOR[history]
    actions = _KUHN_ACTIONS[history]
    card = card0 if player == 0 else card1
    return Decision(
        player=player,
        infoset=f"{player}|{card}|{history}",
        actions=actions,
        children=tuple(_kuhn_node(card0, card1, _KUHN_NEXT[(history, action)]) for action in actions),
    )


@lru_cache(maxsize=1)
def build_kuhn() -> Chance:
    """The 6-deal, 12-information-set Kuhn poker tree (immutable, cached)."""
    branches: list[tuple[float, Node]] = []
    for card0 in _KUHN_CARDS:
        for card1 in _KUHN_CARDS:
            if card0 == card1:
                continue
            branches.append((1.0 / 6.0, _kuhn_node(card0, card1, "")))
    return Chance(tuple(branches))


def kuhn_infosets(root: Node | None = None) -> dict[str, tuple[str, ...]]:
    """Information set -> actions, in depth-first order (12 for the standard Kuhn tree)."""
    tree = root if root is not None else build_kuhn()
    found: dict[str, tuple[str, ...]] = {}

    def walk(node: Node) -> None:
        if isinstance(node, Terminal):
            return
        if isinstance(node, Chance):
            for _, child in node.branches:
                walk(child)
            return
        found.setdefault(node.infoset, node.actions)
        for child in node.children:
            walk(child)

    walk(tree)
    return found


def solve_kuhn(iterations: int = 20000, seed: int | None = None) -> Policy:
    """Train CFR+ on Kuhn poker and return the average behavioural strategy.

    Deterministic: the tree and the traversal involve no randomness. `seed` is recorded on the returned
    `Policy` for reproducible sampling (`policy.sample(..., rng=random.Random(seed))`).
    """
    trainer = CfrTrainer(build_kuhn()).run(max(1, int(iterations)))
    policy = trainer.average_policy()
    policy.seed = seed
    policy.iterations = trainer.iterations_done
    return policy


def evaluate_policy(
    root: Node,
    policy: Policy | Mapping[str, Mapping[str, float]],
) -> tuple[float, ...]:
    """Expected utility per player when both seats follow `policy` (missing infosets play uniformly)."""
    strategies = _strategies_of(policy)
    players = _players_of(root)

    def walk(node: Node) -> tuple[float, ...]:
        if isinstance(node, Terminal):
            return node.values
        if isinstance(node, Chance):
            total = [0.0] * players
            for probability, child in node.branches:
                values = walk(child)
                for player in range(players):
                    total[player] += probability * values[player]
            return tuple(total)
        probabilities = _probabilities(strategies, node.infoset, node.actions)
        total = [0.0] * players
        for index, child in enumerate(node.children):
            values = walk(child)
            for player in range(players):
                total[player] += probabilities[index] * values[player]
        return tuple(total)

    return walk(root)


def _players_of(node: Node) -> int:
    """Infer the utility-vector length below `node` (terminal values are the ground truth)."""
    if isinstance(node, Terminal):
        return len(node.values)
    if isinstance(node, Chance):
        return max((_players_of(child) for _, child in node.branches), default=0)
    return max((_players_of(child) for child in node.children), default=0)


def _player_infosets(root: Node, player: int) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}

    def walk(node: Node) -> None:
        if isinstance(node, Terminal):
            return
        if isinstance(node, Chance):
            for _, child in node.branches:
                walk(child)
            return
        if node.player == player:
            found.setdefault(node.infoset, node.actions)
        for child in node.children:
            walk(child)

    walk(root)
    return found


def _eval_pure(
    node: Node,
    player: int,
    assignment: Mapping[str, str],
    strategies: Mapping[str, Mapping[str, float]],
) -> float:
    if isinstance(node, Terminal):
        return float(node.values[player])
    if isinstance(node, Chance):
        return sum(
            probability * _eval_pure(child, player, assignment, strategies)
            for probability, child in node.branches
        )
    if node.player == player:
        action = assignment[node.infoset]
        return _eval_pure(node.children[node.actions.index(action)], player, assignment, strategies)
    probabilities = _probabilities(strategies, node.infoset, node.actions)
    return sum(
        probabilities[index] * _eval_pure(child, player, assignment, strategies)
        for index, child in enumerate(node.children)
    )


def best_response_value(
    root: Node,
    policy: Policy | Mapping[str, Mapping[str, float]],
    player: int,
    max_pure: int = 20000,
) -> float:
    """Exact best-response value for `player` against `policy`, by enumerating pure strategies.

    Kuhn has six infosets per player with two actions each (64 pure strategies), so this is instant.
    A larger game raises ValueError instead of returning a wrong approximation; the runtime solver does
    not use this function.
    """
    player = int(player)
    strategies = _strategies_of(policy)
    infosets = _player_infosets(root, player)
    keys = list(infosets)
    if not keys:
        return evaluate_policy(root, policy)[player]
    combinations = 1
    for key in keys:
        combinations *= len(infosets[key])
    if combinations > int(max_pure):
        raise ValueError(f"best response would enumerate {combinations} pure strategies (cap {max_pure})")
    best = -math.inf
    for choices in product(*(infosets[key] for key in keys)):
        assignment = dict(zip(keys, choices))
        value = _eval_pure(root, player, assignment, strategies)
        if value > best:
            best = value
    return float(best)


def kuhn_exploitability(
    policy: Policy | Mapping[str, Mapping[str, float]],
    root: Node | None = None,
    max_pure: int = 20000,
) -> float:
    """Exploitability proxy for a two-player zero-sum policy: (BR0 + BR1) / 2.

    `BRi` is the best-response value for player i against the fixed opponent strategy. For a zero-sum
    game this is exactly (max u0 - min u0) / 2 and reaches 0 at a Nash equilibrium.
    """
    tree = root if root is not None else build_kuhn()
    best0 = best_response_value(tree, policy, 0, max_pure)
    best1 = best_response_value(tree, policy, 1, max_pure)
    return max(0.0, (best0 + best1) / 2.0)


# ---------------------------------------------------------------------------- observation wrapper
_ACTION_WORDS = {
    "check": "check",
    "checks": "check",
    "checked": "check",
    "pass": "check",
    "passes": "check",
    "bet": "bet",
    "bets": "bet",
    "betting": "bet",
    "raise": "bet",
    "raises": "bet",
    "call": "call",
    "calls": "call",
    "called": "call",
    "fold": "fold",
    "folds": "fold",
    "folded": "fold",
}
_CARD_WORDS = {
    "j": "J",
    "jack": "J",
    "q": "Q",
    "queen": "Q",
    "k": "K",
    "king": "K",
}
_CARD_PATTERNS = (
    re.compile(r"(?i)\byour\s+(?:private\s+|hidden\s+)?card\s*(?:is|:|=|-)?\s*(?:the\s+)?(jack|queen|king|[jqk])\b"),
    re.compile(r"(?i)\byou\s+(?:were\s+dealt|are\s+dealt|were\s+given|are\s+given|hold|have|get|draw|receive)\s+(?:the\s+)?(jack|queen|king|[jqk])\b"),
    re.compile(r"(?i)\bcard\s*[:=]\s*(jack|queen|king|[jqk])\b"),
    re.compile(r"(?i)\byour\s+hand\s*[:=]\s*(?:the\s+)?(jack|queen|king|[jqk])\b"),
)
_SEAT_PATTERN = re.compile(r"(?i)\byou\s+are\s+(?:player|seat)\s*([01])\b")
_HISTORY_SECTION = re.compile(r"(?i)\b(?:history|actions?\s+so\s+far|betting\s+so\s+far|so\s+far)\s*[:\-]\s*(.{0,300})")
_ACTION_TOKEN = re.compile(r"(?i)\b(checks?|checked|bets?|calls?|called|folds?|folded|pass(?:es)?)\b")
_LINE_PLAYER = re.compile(
    r"(?i)^(?:player|seat|p)\s*([01])\s*[:,.]?\s*(checks?|checked|bets?|calls?|called|folds?|folded|pass(?:es)?)\b"
)
_LINE_INDEX = re.compile(
    r"(?i)^([01])\s*[:.]\s*(checks?|checked|bets?|calls?|called|folds?|folded|pass(?:es)?)\b"
)
_LINE_WORD = re.compile(
    r"(?i)^(you|opponent|they|i)\s*[:,]?\s*(checks?|checked|bets?|calls?|called|folds?|folded|pass(?:es)?)\b"
)


def _line_tokens(text: str, seat: int) -> list[tuple[int, str]]:
    """History actions from lines that *start* with an actor, never from explanatory prose."""
    tokens: list[tuple[int, str]] = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*\u2022 ").strip()
        if not stripped:
            continue
        match = _LINE_PLAYER.match(stripped)
        if match:
            tokens.append((int(match.group(1)), match.group(2)))
            continue
        match = _LINE_INDEX.match(stripped)
        if match:
            tokens.append((int(match.group(1)), match.group(2)))
            continue
        match = _LINE_WORD.match(stripped)
        if match:
            who = match.group(1).lower()
            tokens.append((seat if who in ("you", "i") else 1 - seat, match.group(2)))
    return tokens


@dataclass(frozen=True)
class KuhnTurn:
    """A parsed Kuhn position: our seat, private card, public history and available actions."""

    seat: int
    card: str
    history: str
    actions: tuple[str, ...]

    @property
    def infoset(self) -> str:
        return f"{self.seat}|{self.card}|{self.history}"


def _parse_card(observation: str) -> str | None:
    for pattern in _CARD_PATTERNS:
        match = pattern.search(observation or "")
        if match:
            word = match.group(1).lower()
            card = _CARD_WORDS.get(word)
            if card:
                return card
    return None


def _expected_actor(history: str) -> int | None:
    """The seat that must act at a non-terminal public history, or None if it is terminal/invalid."""
    if history in _KUHN_TERMINAL or history not in _KUHN_ACTIONS:
        return None
    return _KUHN_ACTOR[history]


def _apply_tokens(tokens: list[tuple[int | None, str]]) -> str | None:
    history = ""
    for actor, word in tokens:
        expected = _expected_actor(history)
        if expected is None:
            return None
        if actor is not None and int(actor) != expected:
            return None
        action = _ACTION_WORDS.get(word.lower())
        if action is None or action not in _KUHN_ACTIONS[history]:
            return None
        history = _KUHN_NEXT[(history, action)]
    return history


def parse_kuhn_observation(
    observation: str,
    legal: Sequence[str] | None = None,
    seat: int = 0,
) -> KuhnTurn | None:
    """Parse a Kuhn-shaped observation, or return None when anything is uncertain.

    Supported text shapes include `Your card: K`, `You are Player 0`, history lines such as
    `Player 0 checks` / `Opponent bets`, and a `History: check, bet` section. The parser validates the
    action order against the Kuhn tree, so a partial or inconsistent history is declined rather than
    guessed.
    """
    text = observation or ""
    if not text.strip() or re.search(r"(?i)\bgame\s+(?:is\s+)?over\b", text):
        return None
    card = _parse_card(text)
    if card is None:
        return None
    seat_match = _SEAT_PATTERN.search(text)
    if seat_match:
        try:
            seat = int(seat_match.group(1))
        except (TypeError, ValueError):
            seat = int(seat)
    seat = int(seat)

    attributed = _line_tokens(text, seat)
    tokens: list[tuple[int | None, str]] = []
    if attributed:
        tokens = [(actor, word) for actor, word in attributed]
    else:
        section = _HISTORY_SECTION.search(text)
        if section:
            tokens = [(None, match.group(1)) for match in _ACTION_TOKEN.finditer(section.group(1))]
    if not tokens:
        # No history yet is valid only at the opening (player 0's first decision).
        history = ""
    else:
        parsed = _apply_tokens(tokens)
        if parsed is None:
            return None
        history = parsed
    if history in _KUHN_TERMINAL or history not in _KUHN_ACTIONS:
        return None
    actions = _KUHN_ACTIONS[history]
    if legal:
        available = {
            _ACTION_WORDS.get(str(action).strip().lower())
            for action in legal
        }
        actions = tuple(action for action in actions if action in available)
        if not actions:
            return None
    return KuhnTurn(seat=seat, card=card, history=history, actions=tuple(actions))


_RUNTIME_ITERATIONS = 2000
_KUHN_POLICY_CACHE: Policy | None = None
_SYSTEM_RNG = random.SystemRandom()


def kuhn_runtime_policy() -> Policy:
    """Cached policy for `CfrSolver.propose` (2000 CFR+ iterations, roughly 3 ms of work)."""
    global _KUHN_POLICY_CACHE
    if _KUHN_POLICY_CACHE is None:
        _KUHN_POLICY_CACHE = solve_kuhn(iterations=_RUNTIME_ITERATIONS)
    return _KUHN_POLICY_CACHE


def _canonical_legal(legal: Sequence[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in legal or []:
        text = str(item)
        canonical = _ACTION_WORDS.get(text.strip().lower())
        if canonical and canonical not in mapping:
            mapping[canonical] = text
    return mapping


class CfrSolver:
    """Kuhn-poker expert backed by the cached CFR+ policy. Declines every non-Kuhn observation."""

    name = "cfr"

    def __init__(self) -> None:
        self.rules_text = ""

    def configure(self, spec: object, rules: str = "", has_model: bool = False) -> None:
        self.rules_text = "\n".join(
            part
            for part in (
                str(rules or ""),
                str(getattr(spec, "winning_idea", "") or ""),
                str(getattr(spec, "notes", "") or ""),
            )
            if part
        )

    def matches(self, game_id: str, spec: object) -> bool:
        """Conservative: only explicitly Kuhn / one-card poker games."""
        try:
            identifier = " ".join(
                (
                    str(game_id or ""),
                    str(getattr(spec, "game_id", "") or ""),
                    str(getattr(spec, "winning_idea", "") or ""),
                    str(getattr(spec, "notes", "") or ""),
                    self.rules_text,
                )
            ).lower()
            if "kuhn" in identifier:
                return True
            if "poker" in identifier and any(
                cue in identifier for cue in ("one-card", "one card", "single card", "single-card")
            ):
                return True
            return False
        except Exception:  # noqa: BLE001
            return False

    def propose(
        self,
        *,
        observation: str,
        legal: list[str] | None,
        seat: int,
        turn: int = 0,
        timeout: float = 2.0,
    ) -> str | None:
        """A policy-sampled action for a parsed Kuhn turn, or None when the text is not understood."""
        try:
            if legal is not None and not legal:
                return None
            parsed = parse_kuhn_observation(observation, legal, seat)
            if parsed is None:
                return None
            mapping = _canonical_legal(legal)
            if legal:
                allowed = [action for action in parsed.actions if action in mapping]
            else:
                mapping = {action: action for action in parsed.actions}
                allowed = list(parsed.actions)
            if not allowed:
                return None
            chosen = kuhn_runtime_policy().sample(parsed.infoset, allowed, rng=_SYSTEM_RNG)
            if chosen is None:
                return None
            return mapping.get(chosen)
        except Exception:  # noqa: BLE001 - an expert bug must not cost a move
            return None


__all__ = [
    "CfrSolver",
    "CfrTrainer",
    "Chance",
    "Decision",
    "KuhnTurn",
    "Node",
    "Policy",
    "Terminal",
    "best_response_value",
    "build_kuhn",
    "chance",
    "decision",
    "evaluate_policy",
    "kuhn_exploitability",
    "kuhn_infosets",
    "kuhn_runtime_policy",
    "parse_kuhn_observation",
    "regret_matching",
    "solve_kuhn",
    "terminal",
    "validate_tree",
]
