"""PSN v2 round-trips on both engines (Rust when built)."""

import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'engine'))

from engine_select import ENGINE_CHOICES, game_class  # noqa: E402
from PythonEngine.notation import game_to_psn, parse_psn_turns, psn_to_action, psn_to_game  # noqa: E402


def test_boat_and_skip_tokens():
    assert psn_to_action('Boat@8,10>9,11') == ['plant', 'Boat', 8, 10, 9, 11]
    assert psn_to_action('Boat@8,10') == ['plant', 'Boat', 8, 10]
    tags, turns = parse_psn_turns('[Opening "Rose"]\n\n1. 9,1-10,2+Rock@5,5  8,2-8,6\n')
    assert tags['Opening'] == 'Rose'
    assert turns == [[['arrange', 9, 1, 10, 2], ['plant', 'Rock', 5, 5]], [['arrange', 8, 2, 8, 6]]]


def test_random_games_round_trip(cls):
    rng = random.Random(11)
    for g in range(60):
        accents = {1: ['Rock', 'Rock', 'Boat', 'Wheel'], 2: ['Knotweed', 'Boat', 'Boat', 'Wheel']}
        game = cls(accents=accents, opening=rng.choice(['Rose', 'Lily', None]))
        for _ in range(rng.randint(10, 120)):
            legal = sorted((tuple(a) for a in game.get_legal_actions()), key=repr)
            if not legal:
                break
            game.step(rng.choice(legal))
        text = game_to_psn(game, 'A', 'B')
        replay, tags = psn_to_game(text, cls)
        for key in ('board', 'hands', 'current_player', 'winner', 'end_reason', 'bonus_turn', 'setup', 'history'):
            assert getattr(replay, key) == getattr(game, key), f"game {g}: {key} differs after PSN round-trip\n{text}"


_ACCENTS = {1: ['Rock', 'Rock', 'Boat', 'Wheel'], 2: ['Knotweed', 'Boat', 'Boat', 'Wheel']}


def _find_game_matching(cls, rng, stop_pred, max_games=300, max_steps=200):
    """Play random games until, right after some step, `stop_pred(game, action)` is
    true; return that game (stopped exactly there). None if the budget runs out."""
    for _ in range(max_games):
        game = cls(accents=_ACCENTS, opening=rng.choice(['Rose', 'Lily', None]))
        for _ in range(max_steps):
            legal = sorted((tuple(a) for a in game.get_legal_actions()), key=repr)
            if not legal or game.winner is not None:
                break
            action = rng.choice(legal)
            game.step(action)
            if stop_pred(game, action):
                return game
            if game.winner is not None:
                break
    return None


def test_explicit_final_skip_round_trip(cls):
    """A recorded history ending in an explicit ('skip_bonus',) must round-trip as
    an explicit skip, not be confused with a bonus still pending (Task 11 fix round 1)."""
    rng = random.Random(101)
    game = _find_game_matching(cls, rng, lambda g, a: a[0] == 'skip_bonus')
    assert game is not None, "couldn't find a random game ending in an explicit skip_bonus"
    assert tuple(game.history[-1]) == ('skip_bonus',)
    text = game_to_psn(game, 'A', 'B')
    replay, tags = psn_to_game(text, cls)
    for key in ('board', 'hands', 'current_player', 'bonus_turn', 'history'):
        assert getattr(replay, key) == getattr(game, key), f"{key} differs after PSN round-trip\n{text}"


def test_pending_final_bonus_round_trip(cls):
    """A bonus earned but neither played nor skipped when the record ends must be
    written as a trailing lone '+' and left pending on replay (Task 11 fix round 1)."""
    rng = random.Random(102)
    game = _find_game_matching(cls, rng, lambda g, a: a[0] != 'skip_bonus' and g.bonus_turn)
    assert game is not None, "couldn't find a random game ending with a pending bonus"
    assert game.bonus_turn is True
    text = game_to_psn(game, 'A', 'B')
    last_line = [ln for ln in text.splitlines() if ln and not ln.startswith('[')][-1]
    assert last_line.rstrip().endswith('+'), f"expected a trailing '+' on the last turn\n{text}"
    replay, tags = psn_to_game(text, cls)
    assert replay.bonus_turn is True
    assert replay.history == game.history
    assert replay.current_player == game.current_player
    assert replay.board == game.board
    assert replay.hands == game.hands


def test_resign_round_trip(cls):
    """resign() never appears in history, so the [Termination] tag is what lets
    psn_to_game reconstruct a resigned game's winner/end_reason (Task 11 fix round 1)."""
    rng = random.Random(103)
    game = None
    for _ in range(50):
        candidate = cls(accents=_ACCENTS, opening=rng.choice(['Rose', 'Lily', None]))
        for _ in range(rng.randint(5, 30)):
            legal = sorted((tuple(a) for a in candidate.get_legal_actions()), key=repr)
            if not legal or candidate.winner is not None:
                break
            candidate.step(rng.choice(legal))
        if candidate.winner is None:
            game = candidate
            break
    assert game is not None, "couldn't find an unfinished random game to resign"
    game.resign(2)
    text = game_to_psn(game, 'A', 'B')
    assert '[Termination "resign"]' in text
    replay, tags = psn_to_game(text, cls)
    assert replay.winner == game.winner == 1
    assert replay.end_reason == game.end_reason == 'resign'
    assert replay.history == game.history


def test_export_needs_full_history(cls):
    """A from_dict game carrying a position but not the history that led to it can't be
    exported faithfully: game_to_psn raises ValueError rather than writing a bad record."""
    game = cls(opening='Rose')
    game.step(('arrange', 17, 9, 16, 9))
    state = game.to_save_dict()['state']
    no_history = cls.from_dict(state)
    try:
        game_to_psn(no_history)
    except ValueError:
        pass
    else:
        raise AssertionError("game_to_psn should raise ValueError without the full history")
    bad_history = cls.from_dict(state)
    bad_history.history = [['arrange', 5, 5, 6, 6]]
    try:
        game_to_psn(bad_history)
    except ValueError:
        pass
    else:
        raise AssertionError("game_to_psn should raise ValueError when the history does not replay")


if __name__ == '__main__':
    cases = [('boat_and_skip_tokens', test_boat_and_skip_tokens, ())]
    for name in ENGINE_CHOICES:
        try:
            cls = game_class(name)
        except ImportError:
            print(f"[skip] {name} engine not built")
            continue
        cases.append((f'random_games_round_trip[{name}]', test_random_games_round_trip, (cls,)))
        cases.append((f'explicit_final_skip_round_trip[{name}]', test_explicit_final_skip_round_trip, (cls,)))
        cases.append((f'pending_final_bonus_round_trip[{name}]', test_pending_final_bonus_round_trip, (cls,)))
        cases.append((f'resign_round_trip[{name}]', test_resign_round_trip, (cls,)))
        cases.append((f'export_needs_full_history[{name}]', test_export_needs_full_history, (cls,)))
    failures = 0
    for name, fn, args in cases:
        try:
            fn(*args)
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001 - report every case
            failures += 1
            print(f"[FAIL] {name}: {e}")
    sys.exit(1 if failures else 0)
