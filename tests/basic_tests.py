"""Distilled-repo test suite.

Covers the engine (board, positions, harmony, clash, rings, save/load)
plus the two agents that ship in the distilled build: basic_minimax
and cnn_basic. Self-contained: no import-time dependency on agents
that the distillation strips.
"""

import os
import random
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.PaiShoGame import (
    PaiShoGame, CIRCLE, ACCENT_TILES, SPECIAL_TILES, GATES, CENTER,
    BOARD_SIZE, VALID_SPACES, is_valid, garden_of,
)


# ---------- minimal test runner ----------

class _Result:
    def __init__(self, name, passed, detail, duration):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.duration = duration


class _Runner:
    def __init__(self):
        self.results = []

    def run(self, name, fn):
        t0 = time.perf_counter()
        try:
            fn()
            self.results.append(_Result(name, True, '', time.perf_counter() - t0))
        except AssertionError as e:
            self.results.append(_Result(name, False, str(e), time.perf_counter() - t0))
        except Exception:
            self.results.append(_Result(name, False, traceback.format_exc(), time.perf_counter() - t0))

    def report(self):
        lines = ['=' * 66, '  BASIC TESTS', '=' * 66]
        sections = {}
        for r in self.results:
            sec = r.name.split('/', 1)[0]
            sections.setdefault(sec, []).append(r)
        for sec, rs in sections.items():
            passed = sum(1 for r in rs if r.passed)
            lines.append(f'\n  [{sec}]  {passed}/{len(rs)} passed')
            for r in rs:
                label = r.name.split('/', 1)[-1]
                status = 'PASS' if r.passed else 'FAIL'
                lines.append(f'    [{status}] {label}  ({r.duration * 1000:.1f}ms)')
                if not r.passed and r.detail:
                    for dl in r.detail.strip().splitlines()[-5:]:
                        lines.append(f'           {dl}')
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines += ['', '=' * 66, f'  TOTAL: {passed}/{total} passed, {total - passed} failed', '=' * 66]
        return '\n'.join(lines)


# ---------- helpers ----------

def _fresh():
    return PaiShoGame()


def _bare():
    g = _fresh()
    for p in (1, 2):
        for f in CIRCLE + ACCENT_TILES + SPECIAL_TILES:
            g.hands[p][f] = 99
    return g


def _place(g, flower, player, r, c, growing=False):
    old = g.board.get((r, c))
    if old:
        g._z_toggle((r, c), old)
    tile = {'flower': flower, 'player': player, 'growing': growing}
    g.board[(r, c)] = tile
    g._z_toggle((r, c), tile)


# ---------- board / positions ----------

def test_board_size():
    assert BOARD_SIZE == 19


def test_center_is_valid():
    assert is_valid(*CENTER)


def test_corners_invalid():
    for r, c in [(0, 0), (0, 18), (18, 0), (18, 18)]:
        assert not is_valid(r, c)


def test_gates_are_valid():
    for r, c in GATES:
        assert (r, c) in VALID_SPACES


def test_garden_center_neutral():
    assert garden_of(*CENTER) == 'neutral'


# ---------- game state ----------

def test_initial_state():
    g = _fresh()
    assert g.board == {}
    assert g.current_player == 1
    assert g.winner is None
    assert g.bonus_turn is False


def test_initial_hands():
    g = _fresh()
    for p in (1, 2):
        for f in CIRCLE:
            assert g.hands[p][f] == 2
        for f in ACCENT_TILES + SPECIAL_TILES:
            assert g.hands[p][f] == 1


def test_clone_is_independent():
    g = _bare()
    _place(g, 'Rose', 1, 5, 5)
    c = g.clone()
    c.board[(5, 5)]['flower'] = 'Jade'
    c.hands[1]['Rose'] = 42
    assert g.board[(5, 5)]['flower'] == 'Rose'
    assert g.hands[1]['Rose'] != 42


def test_plant_at_gate_sets_growing():
    g = _bare()
    gr, gc = GATES[0]
    g.step(('plant', 'Rose', gr, gc))
    assert g.board[(gr, gc)]['growing'] is True


def test_legal_actions_start_with_plants_only():
    g = _fresh()
    actions = g.get_legal_actions()
    assert actions, 'expected legal plant actions at game start'
    assert all(a[0] == 'plant' for a in actions)


def test_save_load_roundtrip():
    g = _bare()
    _place(g, 'Rose', 1, 5, 9)
    _place(g, 'Jasmine', 2, 13, 9)
    g.bonus_turn = True
    g.current_player = 2
    d = {
        'board': {f'{r},{c}': t for (r, c), t in g.board.items()},
        'hands': {'1': g.hands[1], '2': g.hands[2]},
        'current_player': g.current_player,
        'winner': g.winner,
        'message': g.message,
        'bonus_turn': g.bonus_turn,
    }
    g2 = PaiShoGame.from_dict(d)
    assert g2.board == g.board
    assert g2.hands == g.hands
    assert g2.current_player == 2
    assert g2.bonus_turn is True


# ---------- harmony / clash ----------

def test_harmony_same_row_neighbours_on_circle():
    g = _bare()
    _place(g, 'Rose', 1, 5, 5)
    _place(g, 'Chrysanthemum', 1, 5, 8)
    assert len(g.find_harmonies(1)) == 1


def test_harmony_blocked_by_obstacle():
    g = _bare()
    _place(g, 'Rose', 1, 5, 5)
    _place(g, 'Chrysanthemum', 1, 5, 9)
    _place(g, 'Lily', 2, 5, 7)
    assert len(g.find_harmonies(1)) == 0


def test_harmony_requires_same_row_or_col():
    g = _bare()
    _place(g, 'Rose', 1, 5, 5)
    _place(g, 'Chrysanthemum', 1, 7, 8)
    assert len(g.find_harmonies(1)) == 0


def test_growing_tile_does_not_form_harmony():
    g = _bare()
    _place(g, 'Rose', 1, 5, 5, growing=True)
    _place(g, 'Chrysanthemum', 1, 5, 8)
    assert len(g.find_harmonies(1)) == 0


def test_clash_detected_on_circle_distance_3():
    g = _bare()
    _place(g, 'Rose', 1, 5, 5)
    _place(g, 'Jasmine', 2, 5, 9)
    assert len(g.find_clashes()) == 1


# ---------- random game smoke ----------

def test_random_game_completes():
    random.seed(123)
    g = _fresh()
    for _ in range(300):
        if g.winner is not None:
            break
        legal = g.get_legal_actions()
        if not legal:
            break
        g.step(random.choice(legal))
    # No crash = pass. Winner may or may not be set.
    assert g.current_player in (1, 2)


# ---------- basic_minimax ----------

def test_basic_minimax_returns_legal_action():
    from ai.classical.basic_minimax import BasicMinimaxAgent
    random.seed(7)
    g = _fresh()
    agent = BasicMinimaxAgent(player=1, time_budget=0.3, max_depth=1)
    a = agent.choose_action(g)
    assert a in g.get_legal_actions()


def test_basic_minimax_handles_bonus_turn():
    from ai.classical.basic_minimax import BasicMinimaxAgent
    g = _bare()
    g.step(('plant', 'Rose', GATES[0][0], GATES[0][1]))
    g.step(('plant', 'Jasmine', GATES[1][0], GATES[1][1]))
    g.bonus_turn = True
    agent = BasicMinimaxAgent(player=g.current_player, time_budget=0.3, max_depth=1)
    a = agent.choose_action(g)
    assert a is None or a in g.get_legal_actions()


def test_basic_minimax_evaluate_terminal():
    from ai.classical.basic_minimax import BasicMinimaxAgent
    g = _fresh()
    g.winner = 1
    agent = BasicMinimaxAgent(player=1)
    assert agent.evaluate(g, 1) == float('inf')
    assert agent.evaluate(g, 2) == float('-inf')


# ---------- cnn_basic ----------

def test_cnn_encode_board_shape():
    from ai.rl.cnn_basic import encode_board, IN_CHANNELS
    g = _fresh()
    enc = encode_board(g, 1)
    assert enc.shape == (IN_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def test_cnn_choose_action_returns_legal():
    from ai.rl.cnn_basic import CNNBasicAgent
    random.seed(11)
    g = _fresh()
    agent = CNNBasicAgent(player=1, load=False)
    a = agent.choose_action(g)
    assert a in g.get_legal_actions()


def test_cnn_save_load_roundtrip(tmp_path_factory=None):
    import tempfile
    import torch
    from ai.rl.cnn_basic import CNNBasicAgent
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'cnn.pt')
        a = CNNBasicAgent(player=1, load=False)
        a.save_model(p)
        assert os.path.exists(p)
        b = CNNBasicAgent(player=1, load=False)
        b.load_model(p)
        for k in a.net.state_dict():
            assert torch.equal(a.net.state_dict()[k], b.net.state_dict()[k])


# ---------- registry ----------

def test_registry_contains_expected_agents():
    from ai.registry import AGENTS
    keys = {a['key'] for a in AGENTS}
    for required in ('random', 'basic_minimax', 'cnn_basic'):
        assert required in keys, f'missing agent: {required}'


# ---------- dispatch ----------

TESTS = [
    ('Board/board_size',                 test_board_size),
    ('Board/center_valid',               test_center_is_valid),
    ('Board/corners_invalid',            test_corners_invalid),
    ('Board/gates_valid',                test_gates_are_valid),
    ('Board/garden_center_neutral',      test_garden_center_neutral),

    ('State/initial_state',              test_initial_state),
    ('State/initial_hands',              test_initial_hands),
    ('State/clone_independent',          test_clone_is_independent),
    ('State/plant_sets_growing',         test_plant_at_gate_sets_growing),
    ('State/legal_actions_plants_only',  test_legal_actions_start_with_plants_only),
    ('State/save_load_roundtrip',        test_save_load_roundtrip),

    ('Harmony/same_row_neighbours',      test_harmony_same_row_neighbours_on_circle),
    ('Harmony/blocked_by_obstacle',      test_harmony_blocked_by_obstacle),
    ('Harmony/needs_same_row_or_col',    test_harmony_requires_same_row_or_col),
    ('Harmony/growing_tile_excluded',    test_growing_tile_does_not_form_harmony),
    ('Harmony/clash_distance_3',         test_clash_detected_on_circle_distance_3),

    ('Smoke/random_game_completes',      test_random_game_completes),

    ('BasicMinimax/returns_legal',       test_basic_minimax_returns_legal_action),
    ('BasicMinimax/bonus_turn',          test_basic_minimax_handles_bonus_turn),
    ('BasicMinimax/evaluate_terminal',   test_basic_minimax_evaluate_terminal),

    ('CNNBasic/encode_shape',            test_cnn_encode_board_shape),
    ('CNNBasic/choose_legal',            test_cnn_choose_action_returns_legal),
    ('CNNBasic/save_load_roundtrip',     test_cnn_save_load_roundtrip),

    ('Registry/has_expected_agents',     test_registry_contains_expected_agents),
]


if __name__ == '__main__':
    runner = _Runner()
    for name, fn in TESTS:
        runner.run(name, fn)
    print(runner.report())
    failed = sum(1 for r in runner.results if not r.passed)
    sys.exit(1 if failed else 0)
