"""Engine rules contract.

Runs every JSON scenario in tests/fixtures/rules/ against each available rules
engine (Python always; Rust when its bridge is built), so both engines are held
to the same data-described rules. See the fixture vocabulary in
docs/superpowers/plans/2026-09-27-phase0-official-rules-m01-m03.md (Task 1).

Run: python tests/test_rules_contract.py [--engine python|rust] [-k substring]
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'engine'))

from engine_select import ENGINE_CHOICES, game_class  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'rules')


def available_engines():
    engines = []
    for name in ENGINE_CHOICES:
        try:
            game_class(name)
        except ImportError:
            print(f"[skip] the {name} engine is not built")
            continue
        engines.append(name)
    return engines


def tile(spec):
    """'Rose/1' -> a blooming Rose of player 1; 'Rose/1/g' -> growing; None stays None."""
    if spec is None or isinstance(spec, dict):
        return spec
    parts = spec.split('/')
    return {'flower': parts[0], 'player': int(parts[1]), 'growing': len(parts) > 2 and parts[2] == 'g'}


def pos(key):
    r, c = key.split(',')
    return int(r), int(c)


def pair_set(pairs):
    return sorted(tuple(sorted((tuple(p), tuple(q)))) for p, q in pairs)


def blank_game(cls):
    try:
        return cls(opening=None)
    except TypeError:  # engines from before the setup options existed
        return cls()


def setup_kwargs(setup):
    kwargs = {}
    if 'accents' in setup:
        accents = setup['accents']
        kwargs['accents'] = None if accents is None else {int(k): v for k, v in accents.items()}
    if 'opening' in setup:
        kwargs['opening'] = setup['opening']
    return kwargs


def build_game(cls, fx):
    setup = fx.get('setup', {})
    if 'state' not in fx:
        return cls(**setup_kwargs(setup))
    state = dict(fx['state'])
    state['board'] = {k: tile(v) for k, v in state.get('board', {}).items()}
    defaults = blank_game(cls).hands
    hands = {str(p): dict(defaults[p]) for p in (1, 2)}
    for p, overrides in state.get('hands', {}).items():
        hands[str(p)].update(overrides)
    state['hands'] = hands
    state.setdefault('current_player', 1)
    if setup:
        state['setup'] = setup
    return cls.from_dict(state)


def _norm_setup(s):
    return {'accents': {str(k): list(v) for k, v in s['accents'].items()}, 'opening': s['opening']}


def check_expect(game, exp, where):
    def fail(msg):
        raise AssertionError(f"{where}: {msg}")

    for key in ('winner', 'current_player', 'bonus_turn'):
        if key in exp and getattr(game, key) != exp[key]:
            fail(f"{key} = {getattr(game, key)!r}, expected {exp[key]!r}")
    if 'end_reason' in exp and getattr(game, 'end_reason', None) != exp['end_reason']:
        fail(f"end_reason = {getattr(game, 'end_reason', None)!r}, expected {exp['end_reason']!r}")
    if 'board' in exp:
        board = game.board
        for key, spec in exp['board'].items():
            got = board.get(pos(key))
            if got != tile(spec):
                fail(f"board[{key}] = {got!r}, expected {tile(spec)!r}")
    if 'hands' in exp:
        for p, counts in exp['hands'].items():
            for name, n in counts.items():
                got = game.hands[int(p)].get(name, 0)
                if got != n:
                    fail(f"hands[{p}][{name}] = {got}, expected {n}")
    for p, pairs in exp.get('harmonies', {}).items():
        got, want = pair_set(game.find_harmonies(int(p))), pair_set(pairs)
        if got != want:
            fail(f"player {p} harmonies = {got}, expected {want}")
    for p, n in exp.get('midline', {}).items():
        got = game.count_midline_harmonies(int(p))
        if got != n:
            fail(f"player {p} midline harmonies = {got}, expected {n}")
    for p, want in exp.get('ring', {}).items():
        got = game.check_harmony_ring(int(p))
        if got != want:
            fail(f"player {p} ring = {got}, expected {want}")
    if 'setup' in exp and _norm_setup(game.setup) != _norm_setup(exp['setup']):
        fail(f"setup = {game.setup!r}, expected {exp['setup']!r}")
    if 'history' in exp and [list(a) for a in game.history] != exp['history']:
        fail(f"history = {game.history!r}, expected {exp['history']!r}")
    if 'history_players' in exp and list(game.history_players) != exp['history_players']:
        fail(f"history_players = {game.history_players!r}, expected {exp['history_players']!r}")


def run_checks(cls, game, fx):
    for i, chk in enumerate(fx.get('checks', [])):
        where = f"check #{i}"
        if any(k in chk for k in ('legal_includes', 'legal_excludes', 'legal_count')):
            legal = {tuple(a) for a in game.get_legal_actions()}
            missing = [tuple(a) for a in chk.get('legal_includes', []) if tuple(a) not in legal]
            if missing:
                raise AssertionError(f"{where}: legal actions lack {missing}")
            present = [tuple(a) for a in chk.get('legal_excludes', []) if tuple(a) in legal]
            if present:
                raise AssertionError(f"{where}: legal actions wrongly include {present}")
            if 'legal_count' in chk and len(legal) != chk['legal_count']:
                raise AssertionError(f"{where}: {len(legal)} legal actions, expected {chk['legal_count']}")
        if 'legal_actions_under_ms' in chk:
            # Time a cold call: a clone would share the legal-actions cache that the
            # legal_* checks above already filled, so rebuild the state from a save.
            probe = cls.from_save_dict(json.loads(json.dumps(game.to_save_dict())))
            start = time.perf_counter()
            probe.get_legal_actions()
            ms = (time.perf_counter() - start) * 1000
            if ms > chk['legal_actions_under_ms']:
                raise AssertionError(f"{where}: get_legal_actions took {ms:.0f} ms (limit {chk['legal_actions_under_ms']})")
        for key, want in chk.get('destinations_include', {}).items():
            got = {tuple(d) for d in game.valid_destinations(*pos(key))}
            missing = [tuple(d) for d in want if tuple(d) not in got]
            if missing:
                raise AssertionError(f"{where}: destinations of {key} lack {missing} (got {sorted(got)})")
        for key, want in chk.get('destinations_exclude', {}).items():
            got = {tuple(d) for d in game.valid_destinations(*pos(key))}
            present = [tuple(d) for d in want if tuple(d) in got]
            if present:
                raise AssertionError(f"{where}: destinations of {key} wrongly include {present}")
        if 'step' in chk:
            action = chk['step'] if chk.get('as_list') else tuple(chk['step'])
            if chk.get('raises'):
                try:
                    game.step(action)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"{where}: step{tuple(action)} should have raised ValueError")
            else:
                game.step(action)
        if 'resign' in chk:
            game.resign(chk['resign'])
        if chk.get('save_roundtrip'):
            data = json.loads(json.dumps(game.to_save_dict('A', 'B')))
            copy = cls.from_save_dict(data)
            for key in ('board', 'hands', 'current_player', 'winner', 'bonus_turn', 'end_reason', 'setup', 'history'):
                if getattr(copy, key) != getattr(game, key):
                    raise AssertionError(f"{where}: save round-trip changed {key}: "
                                         f"{getattr(game, key)!r} -> {getattr(copy, key)!r}")
        if 'expect' in chk:
            check_expect(game, chk['expect'], where)


def run_fixture(cls, fx):
    if fx.get('constructor_raises'):
        try:
            build_game(cls, fx)
        except ValueError:
            return
        raise AssertionError("building the game should have raised ValueError")
    run_checks(cls, build_game(cls, fx), fx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', choices=ENGINE_CHOICES)
    ap.add_argument('-k', default='', help='only fixtures whose file name contains this')
    args = ap.parse_args(argv)
    engines = [args.engine] if args.engine else available_engines()
    files = sorted(f for f in glob.glob(os.path.join(FIXTURE_DIR, '*.json')) if args.k in os.path.basename(f))
    failures = 0
    for engine in engines:
        cls = game_class(engine)
        for path in files:
            name = os.path.basename(path)
            with open(path, encoding='utf-8') as f:
                fx = json.load(f)
            try:
                run_fixture(cls, fx)
                print(f"[PASS] {engine:6} {name}")
            except Exception as e:  # noqa: BLE001 - report every fixture, keep going
                failures += 1
                print(f"[FAIL] {engine:6} {name}: {e}")
                if not isinstance(e, AssertionError):
                    traceback.print_exc()
    total = len(engines) * len(files)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
