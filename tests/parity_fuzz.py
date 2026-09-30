"""Lockstep parity fuzzer: plays random games on the Python and Rust engines at
the same time, feeding both the same random legal action each ply, and stops at
the first ply where anything observable differs.

Run: python tests/parity_fuzz.py --games 300 --seed 1
"""

import argparse
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'engine'))

from engine_select import game_class  # noqa: E402
from PythonEngine.PaiShoGame import ACCENT_TILES, CIRCLE, GATES  # noqa: E402


def random_setup(rng):
    accents = {}
    for p in (1, 2):
        while True:
            chosen = [rng.choice(ACCENT_TILES) for _ in range(4)]
            if all(chosen.count(a) <= 2 for a in ACCENT_TILES):
                accents[p] = chosen
                break
    return {'accents': accents, 'opening': rng.choice(CIRCLE + [None])}


def new_pair(py_cls, rs_cls, rng):
    kwargs = random_setup(rng)
    try:
        return py_cls(**kwargs), rs_cls(**kwargs), kwargs
    except TypeError:  # engines from before the setup options existed
        return py_cls(), rs_cls(), {}


def _pairs(pairs):
    return sorted(tuple(sorted((tuple(a), tuple(b)))) for a, b in pairs)


def snapshot(game):
    return {
        'board': game.board,
        'hands': game.hands,
        'current_player': game.current_player,
        'bonus_turn': game.bonus_turn,
        'winner': game.winner,
        'end_reason': getattr(game, 'end_reason', None),
        'setup': getattr(game, 'setup', None),
        'message': game.message,
        'history': game.history,
        'history_players': list(getattr(game, 'history_players', [])),
        'harmonies': (_pairs(game.find_harmonies(1)), _pairs(game.find_harmonies(2))),
        'legal': sorted((tuple(a) for a in game.get_legal_actions()), key=repr),
    }


def first_difference(a, b):
    for key in a:
        if a[key] != b[key]:
            return key, a[key], b[key]
    return None


def play(py, rs, rng, max_plies):
    for ply in range(max_plies + 1):
        sp, sr = snapshot(py), snapshot(rs)
        diff = first_difference(sp, sr)
        bad = sorted(p for p, t in py.board.items() if t['growing'] != (p in GATES))
        if bad and not diff:
            diff = ('growing_invariant', bad, 'every growing tile must be on a gate and vice versa')
        clashes = py.find_clashes()
        if clashes and not diff:
            diff = ('clash_invariant', clashes, 'the board must never contain a clash')
        if diff:
            return ply, diff
        if py.winner is not None or not sp['legal'] or ply == max_plies:
            return None
        action = rng.choice(sp['legal'])
        py.step(action)
        try:
            rs.step(action)
        except ValueError as e:
            return ply, ('step', action, f'rust raised: {e}')
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--games', type=int, default=300)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max-plies', type=int, default=400)
    args = ap.parse_args(argv)

    py_cls = game_class('python')
    try:
        rs_cls = game_class('rust')
    except ImportError as e:
        print(f"parity fuzz needs the Rust engine: {e}")
        return 2

    rng = random.Random(args.seed)
    for g in range(args.games):
        py, rs, kwargs = new_pair(py_cls, rs_cls, rng)
        result = play(py, rs, rng, args.max_plies)
        if result:
            ply, (field, pyv, rsv) = result
            print(f"DIVERGENCE: game {g} (seed {args.seed}), ply {ply}, setup {kwargs}, field {field!r}")
            print(f"  python: {pyv!r}")
            print(f"  rust:   {rsv!r}")
            try:
                from PythonEngine.notation import game_to_psn
                print(game_to_psn(py))
            except Exception as e:  # noqa: BLE001 - PSN is a debugging aid only
                print(f"(no PSN: {e})")
            return 1
        if (g + 1) % 50 == 0:
            print(f"{g + 1}/{args.games} games identical")
    print(f"OK: {args.games} games, zero divergence")
    return 0


if __name__ == '__main__':
    sys.exit(main())
