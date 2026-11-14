import importlib.util
import os
import time
import unittest

from harness import search

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TicTacToe:
    """Minimal perfect-information fixture: just enough to test the search, not a good game."""

    LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6))

    def __init__(self, board=None, seat=0):
        self.board = list(board or ["."] * 9)
        self.seat = seat

    # --- arena surface -----------------------------------------------------
    @staticmethod
    def _cell(action):
        column = "ABC".index(action[0].upper())
        row = int(action[1]) - 1
        return row * 3 + column

    @staticmethod
    def _name(index):
        return "ABC"[index % 3] + str(index // 3 + 1)

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [self._name(i) for i, cell in enumerate(self.board) if cell == "."]

    def apply(self, seat, action):
        index = self._cell(action)
        if self.board[index] != ".":
            raise ValueError("occupied")
        self.board[index] = "X" if seat == 0 else "O"

    def is_terminal(self):
        return self._winner() is not None or "." not in self.board

    def to_move(self):
        if self.is_terminal():
            return []
        x = self.board.count("X")
        o = self.board.count("O")
        return [0 if x == o else 1]

    def score(self, seat):
        winner = self._winner()
        if winner is None:
            return 0.5
        return 1.0 if winner == seat else 0.0

    def evaluate(self, seat):
        other = 1 - seat
        mine = sum(1 for line in self.LINES if all(self.board[i] != ("O" if seat == 0 else "X") for i in line))
        theirs = sum(1 for line in self.LINES if all(self.board[i] != ("X" if seat == 0 else "O") for i in line))
        return 0.5 + 0.05 * (mine - theirs) / 8.0

    def clone(self):
        return TicTacToe(self.board, self.seat)

    def key(self):
        return "".join(self.board)

    def _winner(self):
        for line in self.LINES:
            a, b, c = (self.board[i] for i in line)
            if a != "." and a == b == c:
                return 0 if a == "X" else 1
        return None


def clean(**kwargs):
    return search.best_action(**kwargs)


class SearchTests(unittest.TestCase):
    def test_takes_the_win(self):
        board = ["X", "X", ".", "O", "O", ".", ".", ".", "."]
        action, value = clean(model=TicTacToe(board), seat=0, seconds=0.5)
        self.assertEqual(action, "C1")
        self.assertGreater(value, 0.9)

    def test_blocks_the_loss(self):
        board = ["O", "O", ".", "X", ".", ".", ".", ".", "X"]
        action, _ = clean(model=TicTacToe(board), seat=0, seconds=0.5)
        self.assertEqual(action, "C1")

    def test_returns_a_legal_move_from_empty_board(self):
        model = TicTacToe()
        action, value = clean(model=model, seat=0, seconds=0.3, max_depth=9)
        self.assertIn(action, model.legal_actions(0))
        self.assertTrue(0.0 <= value <= 1.0)

    def test_terminal_state_has_no_move(self):
        board = ["X", "X", "X", "O", "O", ".", ".", ".", "."]
        action, value = clean(model=TicTacToe(board), seat=0, seconds=0.2)
        self.assertIsNone(action)
        self.assertEqual(value, 0.5)

    def test_respects_the_time_budget(self):
        started = time.monotonic()
        clean(model=TicTacToe(), seat=0, seconds=0.2, max_depth=20)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_broken_model_never_raises(self):
        class Broken(TicTacToe):
            def apply(self, seat, action):
                raise RuntimeError("boom")

        action, _ = clean(model=Broken(), seat=0, seconds=0.2)
        self.assertIn(action, ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"])

    def test_none_model(self):
        self.assertEqual(clean(model=None, seat=0), (None, 0.5))

    def test_prefers_center_on_empty_board(self):
        # Not a correctness proof, only a sanity check that evaluation moves the search.
        action, _ = clean(model=TicTacToe(), seat=0, seconds=0.6, max_depth=4)
        self.assertIn(action, ["B2", "A1", "A3", "C1", "C3"])


class WidePick:
    """Two plies, 100 actions, the higher number wins: exactly the shape alpha-beta cannot handle."""

    def __init__(self, picks=None):
        self.picks = list(picks or [])

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [str(i) for i in range(100)]

    def apply(self, seat, action):
        self.picks.append((seat, int(action)))

    def is_terminal(self):
        return len(self.picks) >= 2

    def to_move(self):
        return [] if self.is_terminal() else [len(self.picks)]

    def score(self, seat):
        chosen = dict(self.picks)
        if len(chosen) < 2:
            return 0.5
        mine, theirs = chosen[seat], chosen[1 - seat]
        return 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        return WidePick(self.picks)


class BothPick:
    """Simultaneous: both seats pick 0..9 at once, higher wins. `to_move()` returns two seats."""

    def __init__(self, picks=None):
        self.picks = dict(picks or {})

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [str(i) for i in range(10)]

    def apply(self, seat, action):
        self.picks[int(seat)] = int(action)

    def is_terminal(self):
        return len(self.picks) >= 2

    def to_move(self):
        return [] if self.is_terminal() else [0, 1]

    def score(self, seat):
        if len(self.picks) < 2:
            return 0.5
        mine, theirs = self.picks[seat], self.picks[1 - seat]
        return 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        return BothPick(self.picks)


class RpsPick:
    """Simultaneous rock-paper-scissors: no dominant action, only a mixed equilibrium."""

    BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}

    def __init__(self, picks=None):
        self.picks = dict(picks or {})

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return ["rock", "paper", "scissors"]

    def apply(self, seat, action):
        self.picks[int(seat)] = str(action)

    def is_terminal(self):
        return len(self.picks) >= 2

    def to_move(self):
        return [] if self.is_terminal() else [0, 1]

    def score(self, seat):
        if len(self.picks) < 2:
            return 0.5
        mine, theirs = self.picks[seat], self.picks[1 - seat]
        if mine == theirs:
            return 0.5
        return 1.0 if self.BEATS[mine] == theirs else 0.0

    def evaluate(self, seat):
        return 0.5

    def clone(self):
        return RpsPick(self.picks)


class WideAndSimultaneousTests(unittest.TestCase):
    def test_wide_game_finds_the_top_move(self):
        self.assertEqual(search.route_for(WidePick(), 0), "alphabeta")
        action, value = search.best_action(WidePick(), 0, seconds=0.8)
        self.assertIsNotNone(action)
        self.assertGreaterEqual(int(action), 90)
        self.assertGreaterEqual(value, 0.5)

    def test_simultaneous_game_returns_a_legal_move(self):
        model = BothPick()
        action, value = search.best_action(model, 0, seconds=0.5)
        self.assertIn(action, model.legal_actions(0))
        self.assertTrue(0.0 <= value <= 1.0)
        # Against a uniform opponent the top action dominates; MCTS must find a strong one.
        self.assertGreaterEqual(int(action), 7)

    def test_wide_search_respects_the_time_budget(self):
        started = time.monotonic()
        search.best_action(WidePick(), 0, seconds=0.2)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_alpha_beta_is_deterministic(self):
        first = search.best_action(WidePick(), 0, seconds=0.4)[0]
        second = search.best_action(WidePick(), 0, seconds=0.4)[0]
        self.assertEqual(first, second)

    def test_joint_moves_are_not_frozen_across_turns(self):
        # Rock-paper-scissors has a genuine mixed equilibrium: 20 samples from a frozen seed would all be
        # identical, which is exactly what a round-robin opponent could learn.
        choices = {search.best_action(RpsPick(), 0, seconds=0.25)[0] for _ in range(20)}
        self.assertGreaterEqual(len(choices), 2)

    def test_terminal_simultaneous_state_has_no_move(self):
        action, value = search.best_action(BothPick({0: 3, 1: 4}), 0, seconds=0.2)
        self.assertIsNone(action)
        self.assertEqual(value, 0.5)

    def test_broken_wide_model_does_not_raise(self):
        class Broken(WidePick):
            def clone(self):
                raise RuntimeError("no clone")

        action, _ = search.best_action(Broken(), 0, seconds=0.2)
        self.assertTrue(action is None or isinstance(action, str))


class Place:
    """Wide placement game: any empty cell, line of `connect` in a row. Used to test MCTS on >20 moves."""

    def __init__(self, size=5, connect=4, board=None, seat=0):
        self.size = size
        self.connect = connect
        self.board = list(board or ["."] * (size * size))
        self.seat = seat

    @staticmethod
    def name(index, size):
        return "ABCDEFG"[index % size] + str(index // size + 1)

    @staticmethod
    def index(action, size):
        return (int(action[1:]) - 1) * size + "ABCDEFG".index(action[0].upper())

    def windows(self):
        n, k = self.size, self.connect
        out = []
        for r in range(n):
            for c in range(n):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = []
                    for i in range(k):
                        rr, cc = r + dr * i, c + dc * i
                        if 0 <= rr < n and 0 <= cc < n:
                            cells.append(rr * n + cc)
                    if len(cells) == k:
                        out.append(cells)
        return out

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        return [self.name(i, self.size) for i, cell in enumerate(self.board) if cell == "."]

    def apply(self, seat, action):
        index = self.index(action, self.size)
        if self.board[index] != ".":
            raise ValueError("occupied")
        self.board[index] = "X" if seat == 0 else "O"

    def winner(self):
        for cells in self.windows():
            marks = {self.board[i] for i in cells}
            if len(marks) == 1 and marks.pop() != ".":
                return 0 if "X" in {self.board[cells[0]]} else 1
        return None

    def is_terminal(self):
        return self.winner() is not None or "." not in self.board

    def to_move(self):
        if self.is_terminal():
            return []
        x = self.board.count("X")
        o = self.board.count("O")
        return [0 if x == o else 1]

    def score(self, seat):
        w = self.winner()
        if w is None:
            return 0.5
        return 1.0 if w == seat else 0.0

    def evaluate(self, seat):
        me, opp = ("X", "O") if seat == 0 else ("O", "X")
        mine = theirs = 0
        for cells in self.windows():
            marks = {self.board[i] for i in cells}
            if opp in marks:
                continue
            own = sum(1 for i in cells if self.board[i] == me)
            if own:
                mine += own
        for cells in self.windows():
            marks = {self.board[i] for i in cells}
            if me in marks:
                continue
            own = sum(1 for i in cells if self.board[i] == opp)
            if own:
                theirs += own
        total = max(1, len(self.windows()) * self.connect)
        return min(1.0, max(0.0, 0.5 + 0.4 * (mine - theirs) / total))

    def clone(self):
        return Place(self.size, self.connect, list(self.board), self.seat)


class WideTacticsTests(unittest.TestCase):
    def test_empty_board_prefers_the_centre(self):
        model = Place(size=5, connect=4)
        self.assertEqual(search.route_for(model, 0), "alphabeta")
        action, value = search.best_action(model, 0, seconds=1.0)
        self.assertEqual(action, "C3")

    def test_blocks_an_open_line_that_would_win(self):
        # O has A1-B1-C1; only D1 stops the fourth stone. X has three harmless stones elsewhere.
        board = ["."] * 36
        board[0], board[1], board[2] = "O", "O", "O"      # A1 B1 C1
        board[5], board[28], board[19] = "X", "X", "X"     # F6 E5 B4 (no line)
        model = Place(size=6, connect=4, board=board)
        self.assertEqual(search.route_for(model, 0), "alphabeta")
        action, _ = search.best_action(model, 0, seconds=1.0)
        self.assertEqual(action, "D1")

    def test_takes_an_immediate_win_before_searching(self):
        board = ["."] * 36
        board[0], board[1], board[2] = "X", "X", "X"      # we can win on D1
        board[5], board[28], board[19] = "O", "O", "O"
        model = Place(size=6, connect=4, board=board)
        action, value = search.best_action(model, 0, seconds=0.5)
        self.assertEqual(action, "D1")
        self.assertEqual(value, 1.0)


def _connect_windows(cols, rows, need):
    """Every length-`need` line on a `cols` x `rows` board, as index tuples."""
    out = []
    for row in range(rows):
        for col in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = []
                for step in range(need):
                    rr, cc = row + dr * step, col + dc * step
                    if not (0 <= rr < rows and 0 <= cc < cols):
                        cells = []
                        break
                    cells.append(rr * cols + cc)
                if cells:
                    out.append(tuple(cells))
    return out


class WideConnect:
    """9x7 five-in-a-row: the shape of scripts/lab/models/connect5.py (63 legal actions).

    Same board, rules and action names as the lab model, with a leaner evaluate(): the lab model rebuilds
    every window on every call (~0.25 ms), which a depth-3 search calls ~18k times. That is 5 s, and the
    timing test would then measure the fixture's scoring cost instead of the search's depth.
    """

    COLS, ROWS, NEED = 9, 7, 5
    COLUMNS = "ABCDEFGHI"
    WINDOWS = tuple(_connect_windows(COLS, ROWS, NEED))

    def __init__(self, board=None, current=0):
        self.board = list(board or ["."] * (self.COLS * self.ROWS))
        self.current = int(current)
        self.winner = self._winner()
        self.over = self.winner is not None or "." not in self.board

    # --- arena surface -----------------------------------------------------
    def update(self, observation):
        return None

    def legal_actions(self, seat):
        if self.over or int(seat) != self.current:
            return []
        return [self._name(i) for i, cell in enumerate(self.board) if cell == "."]

    def apply(self, seat, action):
        index = self._index(action)
        if self.board[index] != ".":
            raise ValueError("occupied")
        self.board[index] = "X" if int(seat) == 0 else "O"
        self.winner = self._winner()
        self.over = self.winner is not None or "." not in self.board
        if not self.over:
            self.current = 1 - int(seat)

    def is_terminal(self):
        return self.over

    def to_move(self):
        return [] if self.over else [self.current]

    def score(self, seat):
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == int(seat) else 0.0

    def evaluate(self, seat):
        me, opp = ("X", "O") if int(seat) == 0 else ("O", "X")
        mine = theirs = 0
        board = self.board
        for window in self.WINDOWS:
            marks = [board[i] for i in window]
            if opp not in marks:
                mine += 1
            if me not in marks:
                theirs += 1
        total = max(1, mine + theirs)
        return max(0.0, min(1.0, 0.5 + 0.4 * (mine - theirs) / total))

    def clone(self):
        return WideConnect(self.board, self.current)

    def key(self):
        return "".join(self.board) + str(self.current)

    @classmethod
    def _name(cls, index):
        return cls.COLUMNS[index % cls.COLS] + str(index // cls.COLS + 1)

    @classmethod
    def _index(cls, action):
        return (int(action[1:]) - 1) * cls.COLS + cls.COLUMNS.index(action[0].upper())

    def _winner(self):
        board = self.board
        for window in self.WINDOWS:
            first = board[window[0]]
            if first != "." and all(board[i] == first for i in window[1:]):
                return 0 if first == "X" else 1
        return None


class WideRootCapTests(unittest.TestCase):
    """The 63-action board that stalled at depth 2: root ranking, a cap, then real iterative deepening."""

    def test_sixty_three_actions_reach_depth_three(self):
        model = WideConnect()
        self.assertEqual(len(model.legal_actions(0)), 63)
        self.assertGreater(63, search.WIDE_ACTIONS)
        started = time.monotonic()
        meta = search.best_action_meta(model, 0, seconds=1.5, max_depth=8)
        elapsed = time.monotonic() - started
        self.assertEqual(meta["route"], "alphabeta")
        self.assertIn(meta["action"], model.legal_actions(0))
        self.assertGreaterEqual(meta["depth"], 3)
        self.assertLess(elapsed, 2.0)  # 1.5 s budget plus a small scheduling margin

    def test_root_shortlist_is_capped_and_deterministic(self):
        model = WideConnect()
        _, safe = search.tactical_root(model, 0, model.legal_actions(0))
        self.assertGreater(len(safe), search.WIDE_ACTIONS)
        first = search.root_shortlist(model, 0, safe)
        second = search.root_shortlist(model, 0, safe)
        self.assertEqual(first, second)
        self.assertEqual(len(first), search.WIDE_ROOT_ACTIONS)
        # The centre cells tie on one-ply score; the action string breaks the tie, so the order is stable.
        self.assertEqual(first[0], "E3")

    def test_cap_keeps_the_uncapped_choice(self):
        # Before the cap alpha-beta was full width at depth 2; reproduce that exact path as the baseline.
        model = WideConnect()
        _, safe = search.tactical_root(model, 0, model.legal_actions(0))
        before_action, before_value = search.Searcher(
            model, 0, seconds=1.5, max_depth=2, root_actions=safe).best()
        meta = search.best_action_meta(model, 0, seconds=1.5, max_depth=8)
        self.assertLessEqual(meta["root_actions"], search.WIDE_ROOT_ACTIONS)
        self.assertTrue(
            meta["action"] == before_action or meta["value"] >= before_value,
            (before_action, before_value, meta["action"], meta["value"]),
        )

    def test_flat_heuristic_keeps_the_full_root(self):
        # A model whose evaluate() cannot rank (WidePick returns 0.5 everywhere) must keep every action:
        # an alphabetical "top 24" would hide the winning pick 90+.
        model = WidePick()
        actions = model.legal_actions(0)
        self.assertEqual(search.root_shortlist(model, 0, actions), actions)


class DeferredWin:
    """Seat 0 has three first moves and wins only with 'c', two plies later.
    evaluate() always returns 1.0, which used to make the root believe it had mate at depth 1."""

    def __init__(self, first=None, second=False):
        self.first = first
        self.second = second

    def update(self, observation):
        return None

    def legal_actions(self, seat):
        if self.second:
            return []
        return ["a", "b", "c"] if self.first is None else ["x"]

    def apply(self, seat, action):
        if self.first is None:
            self.first = action
        else:
            self.second = True

    def is_terminal(self):
        return self.second

    def to_move(self):
        if self.second:
            return []
        return [1] if self.first is not None else [0]

    def score(self, seat):
        if not self.second:
            return 0.5
        winner = 0 if self.first == "c" else 1
        return 1.0 if seat == winner else 0.0

    def evaluate(self, seat):
        return 1.0  # the trap: a heuristic that claims to be winning

    def clone(self):
        return DeferredWin(self.first, self.second)

    def key(self):
        return f"{self.first}:{self.second}"


class KeylessDeferredWin(DeferredWin):
    def key(self):
        raise AttributeError


class SearchCorrectnessTests(unittest.TestCase):
    def test_heuristic_saturation_does_not_fake_mate(self):
        action, value = search.best_action(DeferredWin(), 0, seconds=1.0, max_depth=3)
        self.assertEqual(action, "c")
        self.assertEqual(value, 1.0)

    def test_key_and_no_key_agree(self):
        with_key, value_key = search.best_action(DeferredWin(), 0, seconds=0.5, max_depth=3)
        without_key, value_plain = search.best_action(KeylessDeferredWin(), 0, seconds=0.5, max_depth=3)
        self.assertEqual((with_key, value_key), (without_key, value_plain))

    def test_heuristic_is_bounded_below_mate(self):
        # evaluate() is never allowed to reach a terminal value on its own.
        class OnlyHeuristic(DeferredWin):
            def is_terminal(self):
                return False

            def score(self, seat):
                return 0.5

            def clone(self):
                # keep the class: the inherited clone() would bring real terminal states back
                return OnlyHeuristic(self.first, self.second)

        action, value = search.best_action(OnlyHeuristic(), 0, seconds=0.3, max_depth=2)
        self.assertIsNotNone(action)
        self.assertLess(value, 1.0)


class StandaloneLoadTests(unittest.TestCase):
    def test_search_module_loads_without_the_package(self):
        """The sandbox loads this file by path, so it must not use relative imports."""
        path = os.path.join(ROOT, "harness", "search.py")
        spec = importlib.util.spec_from_file_location("sandboxed_search", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        action, value = module.best_action(TicTacToe(["X", "X", ".", "O", "O", ".", ".", ".", "."]), 0, 0.3)
        self.assertEqual(action, "C1")
        self.assertGreater(value, 0.9)


if __name__ == "__main__":
    unittest.main()
