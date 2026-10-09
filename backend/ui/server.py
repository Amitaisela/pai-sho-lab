import os
import sys
import json
import hmac
import subprocess
import threading
from functools import wraps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# engine_select.py is a loose module directly under engine/, not a package matched by
# pyproject.toml's [tool.setuptools.packages.find] include patterns, so the editable
# install's finder doesn't expose it the way it does PythonEngine/Agents/ui - add its
# directory explicitly, same as every other cross-directory import in this file.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'engine'))
# Our own backend/ first, so `ui.*` always resolves to the code next to this file
# (an editable install elsewhere could otherwise shadow it).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, jsonify, request, send_from_directory, Response, render_template, redirect
from ui.training_manager import (
    start_training, stop_training, get_status as training_status,
    get_log_tail as training_log_tail,
    get_model_info, PROJECT_ROOT,
    wait_for_change as training_wait_for_change,
)
from ui.simulate_manager import (
    start_simulation, stop_simulation, get_status as simulate_status,
    get_log_tail as simulate_log_tail,
    wait_for_change as simulate_wait_for_change,
)
from PythonEngine.PaiShoGame import (PaiShoGame, VALID_SPACES, GATES, CENTER, FLOWER, CIRCLE,
                              ACCENT_TILES, SPECIAL_TILES)
from PythonEngine.notation import game_to_psn, psn_to_game
from engine_select import DEFAULT_ENGINE, engine_name_of, game_class
from Agents.registry import get_agent, playable_agents, trainable_agents
from Agents import elo
from ui import play
# _clamp_params is not used here; tests/test_integration.py reaches it as srv._clamp_params
from ui.bots import BotCache, clamp_params as _clamp_params, choose_action as _bot_choose_action  # noqa: F401

_FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))

app = Flask(
    __name__,
    template_folder=os.path.join(_FRONTEND_DIR, 'templates'),
    static_folder=os.path.join(_FRONTEND_DIR, 'static'),
)


class _BadRequest(Exception):
    """Raised by request validators; converted to a 400 by the error handler below."""


@app.errorhandler(_BadRequest)
def _handle_bad_request(e):
    return jsonify({'error': str(e)}), 400


_VALID_SET = set(VALID_SPACES)
_KNOWN_TILES = set(FLOWER) | set(SPECIAL_TILES) | set(ACCENT_TILES)

_API_TOKEN = os.environ.get('MUSHIBOT_API_TOKEN', '').strip()


def _require_api_token(view):
    """Gate a state-mutating endpoint behind MUSHIBOT_API_TOKEN, if set.

    Training/simulation start-stop and config writes had no access control
    beyond network position: fine for a laptop, not fine for the Tailscale
    deployment this app documents, where anyone reachable on the tailnet -
    not just the intended operator - could stop someone else's training run
    or overwrite their hyperparameter config. Leaving MUSHIBOT_API_TOKEN
    unset keeps the previous (open) behavior for local single-user use.
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        if _API_TOKEN:
            supplied = request.headers.get('X-API-Token', '')
            if not hmac.compare_digest(supplied, _API_TOKEN):
                return jsonify({'error': 'unauthorized'}), 401
        return view(*args, **kwargs)
    return wrapped


def _json_body(*required):
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        raise _BadRequest("request body must be a JSON object")
    for k in required:
        if k not in d:
            raise _BadRequest(f"missing field: {k}")
    return d


def _optional_json_body():
    """Like _json_body() but a missing/non-JSON body is treated as {} instead
    of a 400 - for endpoints that have historically accepted a bodiless POST
    (all fields optional/defaulted). A body that *was* sent but isn't a JSON
    object is still rejected with 400, instead of reaching `.get()` on
    something that isn't a dict and 500ing.
    """
    d = request.get_json(silent=True)
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise _BadRequest("request body must be a JSON object")
    return d


def _require_int(value, name):
    """Reject anything but a real int (bools, floats, numeric strings included)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _BadRequest(f"{name} must be an integer")
    return value


def _coord(d, r_key, c_key):
    try:
        r, c = int(d[r_key]), int(d[c_key])
    except (KeyError, TypeError, ValueError):
        raise _BadRequest(f"{r_key}/{c_key} must be integers")
    if (r, c) not in _VALID_SET:
        raise _BadRequest(f"({r},{c}) is not a valid board space")
    return r, c


def _flower_name(d, key='flower'):
    f = d.get(key)
    if f not in _KNOWN_TILES:
        raise _BadRequest(f"unknown flower: {f!r}")
    return f


# All per-visitor state below is keyed by `gid` (a client-generated id, one
# per browser tab; see index.html). Before this, `agent_names`/`elo_session`
# were single global dicts and `games`/`_bot_agents` used a hardcoded
# 'default' key everywhere, so two concurrent tabs/visitors silently shared
# and clobbered one game. Every route that reads/writes this state now takes
# (or defaults) a `gid`, and callers that predate per-tab ids (e.g.
# simulator.py's --mode flask) keep working unchanged via the 'default' key.
games = {}


def _default_agent_names():
    return {'1': None, '2': None}


_agent_names_by_gid = {}


def _get_agent_names(gid):
    return _agent_names_by_gid.setdefault(gid, _default_agent_names())


def _default_elo_session():
    return {
        'p1_key': None,
        'p2_key': None,
        'p1_human_name': '',
        'p2_human_name': '',
        'rated': True,
    }


_elo_sessions_by_gid = {}


def _get_elo_session(gid):
    return _elo_sessions_by_gid.setdefault(gid, _default_elo_session())


_recorded_games = set()
_last_elo_result = {}

_history_stacks = {}


def _push_snapshot(gid, snapshot):
    stacks = _history_stacks.setdefault(gid, {'undo': [], 'redo': []})
    stacks['undo'].append(snapshot)
    stacks['redo'].clear()

_bot_cache = BotCache()


def _lab_gate_gid(gid):
    """401 unless `gid` is a registered player game (play.META) or no API
    token is configured. Lab/legacy games (any gid outside play.META) used to
    be reachable by anyone who guessed or was handed the id - new_game,
    bot_move, set_agents, set_state/load/import_psn, undo/redo, plant/
    arrange/skip_bonus/resign and elo/session(POST) all mutate server state
    (including, via api_state's _maybe_record_elo, the Elo leaderboard) with
    no access control beyond network position otherwise. Registered player
    games keep their own seat-token rules unchanged - this never fires for
    them, token configured or not.
    """
    if gid in play.META:
        return None
    # Read live rather than the frozen _API_TOKEN snapshot (unlike
    # _require_api_token, which only ever runs with the token the process
    # started with) so this matches play.py's api_lab_check, which the Lab
    # gate's own UI polls dynamically.
    expected = os.environ.get('MUSHIBOT_API_TOKEN', '').strip()
    if not expected:
        return None
    supplied = request.headers.get('X-API-Token', '')
    if not hmac.compare_digest(supplied, expected):
        return jsonify({'error': 'unauthorized'}), 401
    return None


def _seat_guard(action):
    """Run play.guard() before a mutating endpoint, under the game's lock.

    Registered player games (play.META) are locked to their seat tokens; Lab and
    legacy games (any other gid) require the API token instead, when one is
    configured (see _lab_gate_gid). The gid comes from the URL, or from the
    body for /api/new_game. A successful mutation refreshes the game's idle
    timer.
    """
    def deco(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                body = {}
            gid = kwargs.get('gid') or body.get('gid') or 'default'
            blocked = _lab_gate_gid(gid)
            if blocked:
                return blocked
            with play.game_lock(gid):
                err = play.guard(gid, action, body)
                if err:
                    return err
                resp = view(*args, **kwargs)
            status = resp[1] if isinstance(resp, tuple) else getattr(resp, 'status_code', 200)
            if status < 400:
                play.touch(gid)
            return resp
        return wrapped
    return deco


def serialize(game: PaiShoGame) -> dict:
    board_s = {f"{r},{c}": t for (r, c), t in game.board.items()}
    h1 = game.find_harmonies(1)
    h2 = game.find_harmonies(2)
    return {
        'engine': engine_name_of(game),
        'board': board_s,
        'hands': {'1': game.hands[1], '2': game.hands[2]},
        'current_player': game.current_player,
        'winner': game.winner,
        'message': game.message,
        'bonus_turn': game.bonus_turn,
        'valid_spaces': VALID_SPACES,
        'gates': GATES,
        'center': list(CENTER),
        'flower_data': FLOWER,
        'harmony_circle': CIRCLE,
        'harmonies': {
            '1': [list(map(list, pair)) for pair in h1],
            '2': [list(map(list, pair)) for pair in h2],
        },
        'history': game.history,
        'history_players': list(getattr(game, 'history_players', [])),
        'setup': {'accents': {str(p): list(v) for p, v in game.setup['accents'].items()},
                  'opening': game.setup['opening']},
        'end_reason': game.end_reason,
        'can_skip_bonus': ('skip_bonus',) in {tuple(a) for a in game.get_legal_actions()},
    }


@app.route('/')
def root():
    return render_template('home.html')


@app.route('/game/<gid>')
def game_page(gid):
    return render_template('game.html', gid=gid)


@app.route('/learn')
def learn_page():
    return render_template('learn.html')


@app.route('/bots')
def bots_page():
    return render_template('bots.html')


@app.route('/developers')
def developers_page():
    return render_template('developers.html')


@app.route('/lab')
def lab_index_page():
    return render_template('lab/index.html')


@app.route('/lab/board')
def lab_board_page():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates', 'lab'), 'board.html')


@app.route('/api/new_game', methods=['POST'])
@_seat_guard('new_game')
def api_new_game():
    d = _optional_json_body()
    gid = d.get('gid') or 'default'
    try:
        setup = {}
        if d.get('accents') is not None:
            setup['accents'] = {int(k): v for k, v in d['accents'].items()}
        if 'opening' in d:
            setup['opening'] = d['opening']
        games[gid] = game_class(d.get('engine', DEFAULT_ENGINE))(**setup)
    except (ValueError, ImportError, TypeError, AttributeError) as e:
        return jsonify({'error': str(e)}), 400
    _history_stacks[gid] = {'undo': [], 'redo': []}
    _recorded_games.discard(gid)
    _last_elo_result.pop(gid, None)
    return jsonify({'game_id': gid, 'state': serialize(games[gid])})


def _resolve_agent_key(gid, slot):
    session = _get_elo_session(gid)
    k = session.get(f'p{slot}_key')
    if not k:
        return None
    if k == 'human':
        name = session.get(f'p{slot}_human_name', '') or 'Guest'
        return elo.human_key(name)
    return k


def _maybe_record_elo(gid, game):
    if gid in play.META:          # player games (human/pass/bot) are never rated in Phase 1
        return
    if gid in _recorded_games:
        return
    if game.winner is None:
        return
    if not _get_elo_session(gid).get('rated', True):
        _recorded_games.add(gid)
        return
    p1_key = _resolve_agent_key(gid, '1')
    p2_key = _resolve_agent_key(gid, '2')
    if not p1_key or not p2_key:
        _recorded_games.add(gid)
        return
    # Unnamed guests aren't recorded, to keep the leaderboard clean.
    if p1_key == 'human:Guest' or p2_key == 'human:Guest':
        _recorded_games.add(gid)
        return
    winner = game.winner if game.winner in (1, 2) else None
    result = elo.record_game(p1_key, p2_key, winner)
    _recorded_games.add(gid)
    _last_elo_result[gid] = result


@app.route('/api/state/<gid>')
def api_state(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    _maybe_record_elo(gid, g)
    payload = {'state': serialize(g)}
    if gid in _last_elo_result:
        payload['elo_result'] = _last_elo_result[gid]
    return jsonify(payload)


@app.route('/api/plant/<gid>', methods=['POST'])
@_seat_guard('move')
def api_plant(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    if g.winner is not None:
        return jsonify({'error': 'game over'}), 400

    d = _json_body('flower', 'row', 'col')
    flower = _flower_name(d)
    r, c = _coord(d, 'row', 'col')
    kwargs = {}
    if 'displace_row' in d and 'displace_col' in d:
        dr, dc = _coord(d, 'displace_row', 'displace_col')
        kwargs['displace_r'] = dr
        kwargs['displace_c'] = dc
    snap = g.clone()
    try:
        g.plant(flower, r, c, **kwargs)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    _push_snapshot(gid, snap)

    return jsonify({'state': serialize(g)})


@app.route('/api/arrange/<gid>', methods=['POST'])
@_seat_guard('move')
def api_arrange(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    if g.winner is not None:
        return jsonify({'error': 'game over'}), 400

    d = _json_body('from_row', 'from_col', 'to_row', 'to_col')
    fr, fc = _coord(d, 'from_row', 'from_col')
    tr, tc = _coord(d, 'to_row', 'to_col')
    snap = g.clone()
    try:
        g.arrange(fr, fc, tr, tc)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    _push_snapshot(gid, snap)

    return jsonify({'state': serialize(g)})


@app.route('/api/skip_bonus/<gid>', methods=['POST'])
@_seat_guard('move')
def api_skip_bonus(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    if g.winner is not None:
        return jsonify({'error': 'game over'}), 400
    snap = g.clone()
    try:
        g.step(('skip_bonus',))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    _push_snapshot(gid, snap)
    return jsonify({'state': serialize(g)})


@app.route('/api/resign/<gid>', methods=['POST'])
@_seat_guard('resign')
def api_resign(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    d = _optional_json_body()
    # In a seated game the default is the caller's own seat, never "whoever is to move".
    default = play.seat_of(gid, d) or g.current_player
    player = _require_int(d.get('player', default), 'player')
    snap = g.clone()
    try:
        g.resign(player)
    except (TypeError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    _push_snapshot(gid, snap)
    return jsonify({'state': serialize(g)})


@app.route('/api/valid_moves/<gid>', methods=['POST'])
def api_valid_moves(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    d = _json_body('row', 'col')
    r, c = _coord(d, 'row', 'col')
    # Derived from get_legal_actions() rather than valid_destinations(), which
    # ignores the pending bonus (and other legality gates) and would otherwise
    # surface illegal 'arrange' hints on bonus turns.
    seen, moves = set(), []
    for a in g.get_legal_actions():
        if a[0] == 'arrange' and (a[1], a[2]) == (r, c) and (a[3], a[4]) not in seen:
            seen.add((a[3], a[4]))
            moves.append([a[3], a[4]])
    return jsonify({'moves': moves})


@app.route('/api/valid_boat_displacement/<gid>', methods=['POST'])
def api_valid_boat_displacement(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    d = _json_body('target_row', 'target_col')
    tr, tc = _coord(d, 'target_row', 'target_col')
    legal = [tuple(a) for a in g.get_legal_actions()]
    moves = [[a[4], a[5]] for a in legal if a[0] == 'plant' and a[1] == 'Boat' and len(a) == 6 and (a[2], a[3]) == (tr, tc)]
    direct = ('plant', 'Boat', tr, tc) in legal
    return jsonify({'moves': moves, 'direct': direct})


@app.route('/api/valid_plant_moves/<gid>', methods=['POST'])
def api_valid_plant_moves(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    d = _json_body()
    tile = d.get('tile', '')
    seen, moves = set(), []
    for a in g.get_legal_actions():
        if a[0] == 'plant' and a[1] == tile and (a[2], a[3]) not in seen:
            seen.add((a[2], a[3]))
            moves.append([a[2], a[3]])
    return jsonify({'moves': moves})


@app.route('/api/set_state/<gid>', methods=['POST'])
@_seat_guard('set_state')
def api_set_state(gid):
    d = _json_body('board', 'hands', 'current_player')
    try:
        games[gid] = game_class(d.get('engine', DEFAULT_ENGINE)).from_dict(d)
    except (KeyError, TypeError, ValueError, ImportError) as e:
        return jsonify({'error': f'invalid state: {e}'}), 400
    return jsonify({'status': 'success'})


@app.route('/api/set_agents', methods=['POST'])
def api_set_agents():
    d = _optional_json_body()
    gid = d.get('gid') or 'default'
    blocked = _lab_gate_gid(gid)
    if blocked:
        return blocked
    names = _get_agent_names(gid)
    names['1'] = d.get('p1', 'Player 1')
    names['2'] = d.get('p2', 'Player 2')
    session = _get_elo_session(gid)
    if 'p1_key' in d:
        session['p1_key'] = d.get('p1_key')
    if 'p2_key' in d:
        session['p2_key'] = d.get('p2_key')
    if 'rated' in d:
        session['rated'] = bool(d.get('rated'))
    if 'p1_human_name' in d:
        session['p1_human_name'] = d.get('p1_human_name', '') or ''
    if 'p2_human_name' in d:
        session['p2_human_name'] = d.get('p2_human_name', '') or ''
    return jsonify({'status': 'success'})


@app.route('/api/agents', methods=['GET'])
def api_get_agents():
    gid = request.args.get('gid') or 'default'
    return jsonify(_get_agent_names(gid))


@app.route('/leaderboard')
def leaderboard_page_redirect():
    return redirect('/lab/leaderboard', code=301)


@app.route('/lab/leaderboard')
def leaderboard_page():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates', 'lab'), 'leaderboard.html')


@app.route('/api/elo/leaderboard', methods=['GET'])
def api_elo_leaderboard():
    return jsonify({
        'rows': elo.get_leaderboard(),
        'history': elo.get_history(),
    })


@app.route('/api/elo/rating', methods=['GET'])
def api_elo_rating():
    key = request.args.get('key', '')
    if not key:
        return jsonify({'error': 'key required'}), 400
    return jsonify({'key': key, 'rating': elo.get_rating(key)})


@app.route('/api/elo/session', methods=['GET', 'POST'])
def api_elo_session():
    if request.method == 'POST':
        d = _optional_json_body()
        gid = d.get('gid') or 'default'
        blocked = _lab_gate_gid(gid)
        if blocked:
            return blocked
        session = _get_elo_session(gid)
        for field in ('p1_key', 'p2_key', 'p1_human_name', 'p2_human_name'):
            if field in d:
                session[field] = d[field] or ''
        if 'rated' in d:
            session['rated'] = bool(d['rated'])
    else:
        gid = request.args.get('gid') or 'default'
        session = _get_elo_session(gid)
    return jsonify(dict(session))


@app.route('/api/save/<gid>', methods=['GET'])
def api_save_game(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    names = _get_agent_names(gid)
    p1 = names.get('1') or 'Player 1'
    p2 = names.get('2') or 'Player 2'
    save_data = g.to_save_dict(p1_name=p1, p2_name=p2)
    save_data['engine'] = engine_name_of(g)
    filename = f"{p1}_vs_{p2}.json"
    return Response(
        json.dumps(save_data, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.route('/api/export_psn/<gid>', methods=['GET'])
def api_export_psn(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    names = _get_agent_names(gid)
    p1 = names.get('1') or 'Player 1'
    p2 = names.get('2') or 'Player 2'
    text = game_to_psn(g, p1_name=p1, p2_name=p2)
    filename = f"{p1}_vs_{p2}.psn"
    return Response(
        text,
        mimetype='text/plain',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.route('/api/import_psn/<gid>', methods=['POST'])
@_seat_guard('set_state')
def api_import_psn(gid):
    text = None
    requested_engine = DEFAULT_ENGINE
    if request.is_json:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'request body must be a JSON object'}), 400
        text = data.get('psn')
        requested_engine = data.get('engine', DEFAULT_ENGINE)
    if text is None:
        text = request.get_data(as_text=True)
    if not text or not text.strip():
        return jsonify({'error': 'empty PSN'}), 400
    try:
        game, tags = psn_to_game(text, game_class(requested_engine))
    except ImportError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'parse error: {e}'}), 400
    games[gid] = game
    _history_stacks[gid] = {'undo': [], 'redo': []}
    names = _get_agent_names(gid)
    if tags.get('Player1'):
        names['1'] = tags['Player1']
    if tags.get('Player2'):
        names['2'] = tags['Player2']
    return jsonify({'state': serialize(games[gid]), 'tags': tags})


@app.route('/api/load/<gid>', methods=['POST'])
@_seat_guard('set_state')
def api_load_game(gid):
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or 'state' not in data:
        return jsonify({'error': 'invalid save file'}), 400
    try:
        games[gid] = game_class(data.get('engine', DEFAULT_ENGINE)).from_save_dict(data)
    except (KeyError, TypeError, ValueError, ImportError) as e:
        return jsonify({'error': f'invalid save file: {e}'}), 400
    _history_stacks[gid] = {'undo': [], 'redo': []}
    names = _get_agent_names(gid)
    if data.get('p1'):
        names['1'] = data['p1']
    if data.get('p2'):
        names['2'] = data['p2']
    return jsonify({'state': serialize(games[gid])})


def _is_bot_turn(gid, game):
    """True when `gid` is a registered bot game and its bot seat is to move -
    undo/redo there skip past bot plies so one tap returns to the human's turn."""
    meta = play.META.get(gid)
    if not meta or meta['mode'] != 'bot' or game.winner is not None:
        return False
    seat = meta['seats'].get(game.current_player)
    return bool(seat and seat['kind'] == 'bot')


@app.route('/api/undo/<gid>', methods=['POST'])
@_seat_guard('undo')
def api_undo(gid):
    stacks = _history_stacks.get(gid)
    if not stacks or not stacks['undo']:
        return jsonify({'error': 'nothing to undo'}), 400
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    stacks['redo'].append(g.clone())
    games[gid] = stacks['undo'].pop()
    while _is_bot_turn(gid, games[gid]) and stacks['undo']:
        stacks['redo'].append(games[gid])
        games[gid] = stacks['undo'].pop()
    return jsonify({'state': serialize(games[gid]),
                    'can_undo': len(stacks['undo']) > 0,
                    'can_redo': len(stacks['redo']) > 0})


@app.route('/api/redo/<gid>', methods=['POST'])
@_seat_guard('undo')
def api_redo(gid):
    stacks = _history_stacks.get(gid)
    if not stacks or not stacks['redo']:
        return jsonify({'error': 'nothing to redo'}), 400
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    stacks['undo'].append(g.clone())
    games[gid] = stacks['redo'].pop()
    while _is_bot_turn(gid, games[gid]) and stacks['redo']:
        stacks['undo'].append(games[gid])
        games[gid] = stacks['redo'].pop()
    return jsonify({'state': serialize(games[gid]),
                    'can_undo': len(stacks['undo']) > 0,
                    'can_redo': len(stacks['redo']) > 0})


def _forget_game(gid):
    """Drop server-side per-game state for a game play.py evicted."""
    _history_stacks.pop(gid, None)
    _agent_names_by_gid.pop(gid, None)
    _elo_sessions_by_gid.pop(gid, None)
    _recorded_games.discard(gid)
    _last_elo_result.pop(gid, None)
    _bot_cache.drop_game(gid)


play.init(games, serialize, game_class,
          lambda gid, game, bot_type, params: _bot_choose_action(_bot_cache, gid, game, bot_type, params),
          on_evict=_forget_game)
app.register_blueprint(play.bp)

_TUTORIAL_STEPS_PATH = os.path.join(_FRONTEND_DIR, 'static', 'tutorial', 'steps.json')
_tutorial_steps_cache = None


def _tutorial_steps():
    global _tutorial_steps_cache
    if _tutorial_steps_cache is None:
        with open(_TUTORIAL_STEPS_PATH, encoding='utf-8') as f:
            _tutorial_steps_cache = json.load(f)
    return _tutorial_steps_cache


@app.route('/api/tutorial/<int:n>', methods=['POST'])
def api_tutorial(n):
    steps = _tutorial_steps()
    if not 1 <= n <= len(steps):
        return jsonify({'error': 'no such tutorial step'}), 404
    gid, err = play.create_tutorial_game(steps[n - 1]['state'])
    if err:
        return err
    return jsonify({'game_id': gid})


@app.route('/api/bot_move/<gid>', methods=['POST'])
@_seat_guard('bot_move')
def api_bot_move(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    if g.winner is not None:
        return jsonify({'error': 'game over', 'state': serialize(g)}), 400

    d = _optional_json_body()
    bot_type = d.get('bot', 'random')
    params = d.get('params', {})
    meta = play.META.get(gid)
    if meta is not None:
        # A house-bot game always plays its own seat's bot; the client can't swap it.
        bot_type, params = meta['seats'][g.current_player]['bot'], {}

    snap = g.clone()
    action = _bot_choose_action(_bot_cache, gid, g, bot_type, params)
    if action is None:
        return jsonify({'error': 'no legal moves', 'state': serialize(g)}), 400

    try:
        g.step(action)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    _push_snapshot(gid, snap)
    return jsonify({'state': serialize(g), 'action': list(action)})


@app.route('/train')
def train_page_redirect():
    return redirect('/lab/train', code=301)


@app.route('/lab/train')
def train_page():
    train = []
    model_infos = {}
    configs = {}
    for a in trainable_agents():
        train.append({
            "key": a["key"],
            "display_name": a["display_name"],
            "description": a.get("description", ""),
            "architecture": a.get("architecture", ""),
            "training_params": a["training_params"],
            "config_file": a.get("config_file"),
        })
        model_infos[a["key"]] = get_model_info(a["key"])
        if a.get("config_file"):
            path = os.path.join(PROJECT_ROOT, a["config_file"])
            if os.path.exists(path):
                with open(path, 'r') as f:
                    configs[a["key"]] = {
                        "content": f.read(),
                        "path": os.path.relpath(path, PROJECT_ROOT),
                    }
    init_data = {
        "train": train,
        "model_infos": model_infos,
        "configs": configs,
        "status": training_status(),
    }
    return render_template('lab/train.html', init_data=init_data)


@app.route('/guide')
def guide_page_redirect():
    return redirect('/lab/guide', code=301)


@app.route('/lab/guide')
def guide_page():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates', 'lab'), 'guide.html')


@app.route('/rules')
def rules_page_redirect():
    # rules.html's content now lives, restyled, inside learn.html; this just
    # stops linking to the old URL.
    return redirect('/learn', code=301)


@app.route('/api/example/<filename>')
def download_example(filename):
    ALLOWED = {
        'basic_minimax.py': os.path.join(PROJECT_ROOT, 'Agents', 'classical', 'basic_minimax.py'),
        'cnn_basic.py': os.path.join(PROJECT_ROOT, 'Agents', 'rl', 'cnn_basic.py'),
        'cnn_basic_training.py': os.path.join(PROJECT_ROOT, 'Agents', 'training', 'cnn_basic_training.py'),
        'registry.py': os.path.join(PROJECT_ROOT, 'Agents', 'registry.py'),
    }
    path = ALLOWED.get(filename)
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(os.path.dirname(path), os.path.basename(path),
                               as_attachment=True)


@app.route('/api/training/start', methods=['POST'])
@_require_api_token
def api_training_start():
    d = _optional_json_body()
    model = d.get('model')
    params = d.get('params', {})
    try:
        status = start_training(model, params)
        return jsonify(status)
    except ValueError as e:
        return jsonify({'error': str(e)}), 409


@app.route('/api/training/stop', methods=['POST'])
@_require_api_token
def api_training_stop():
    return jsonify(stop_training())


@app.route('/api/training/status', methods=['GET'])
def api_training_status():
    return jsonify(training_status())


@app.route('/api/training/stream')
def api_training_stream():
    try:
        since_seq = int(request.args.get('since', 0))
    except (TypeError, ValueError):
        since_seq = 0
    def generate():
        last_ep = -1
        last_status = None
        sent_seq = since_seq
        while True:
            status = training_status()
            ep = status.get("current_episode", 0)
            st = status.get("status", "idle")
            new_lines, seq = training_log_tail(sent_seq)
            if ep != last_ep or st != last_status or new_lines:
                payload = dict(status)
                payload["new_log_lines"] = new_lines
                payload["log_seq"] = seq
                payload.pop("log_lines", None)
                yield f"data: {json.dumps(payload)}\n\n"
                last_ep = ep
                last_status = st
                sent_seq = seq
            if st in ("idle", "completed", "error"):
                break
            # Block until the manager signals a state change (or 2s timeout
            # as a heartbeat / disconnect-detection floor).
            training_wait_for_change(2.0)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/training/model_info', methods=['GET'])
def api_training_model_info():
    model = request.args.get('model', '')
    return jsonify(get_model_info(model))


def _config_path_for(model_key):
    entry = get_agent(model_key)
    if not entry or not entry.get("config_file"):
        return None
    return os.path.join(PROJECT_ROOT, entry["config_file"])


@app.route('/api/training/config', methods=['GET'])
def api_training_config_read():
    model = request.args.get('model', '')
    path = _config_path_for(model)
    if not path:
        return jsonify({'error': 'No config file for this model'}), 404
    if not os.path.exists(path):
        return jsonify({'error': 'Config file not found', 'path': path}), 404
    with open(path, 'r') as f:
        content = f.read()
    return jsonify({'content': content, 'path': os.path.relpath(path, PROJECT_ROOT)})


@app.route('/api/training/config', methods=['POST'])
@_require_api_token
def api_training_config_write():
    d = _optional_json_body()
    model = d.get('model', '')
    content = d.get('content', '')
    path = _config_path_for(model)
    if not path:
        return jsonify({'error': 'No config file for this model'}), 404
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    return jsonify({'ok': True, 'path': os.path.relpath(path, PROJECT_ROOT)})


@app.route('/api/agents/registry')
def api_agents_registry():
    play = []
    for a in playable_agents():
        play.append({
            "key": a["key"],
            "display_name": a["display_name"],
            "description": a.get("description", ""),
            "play_params": a.get("play_params", []),
        })
    train = []
    for a in trainable_agents():
        train.append({
            "key": a["key"],
            "display_name": a["display_name"],
            "description": a.get("description", ""),
            "architecture": a.get("architecture", ""),
            "training_params": a["training_params"],
            "config_file": a.get("config_file"),
        })
    return jsonify({"play": play, "train": train})


@app.route('/simulate')
def simulate_page_redirect():
    return redirect('/lab/simulate', code=301)


@app.route('/lab/simulate')
def simulate_page():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates', 'lab'), 'simulate.html')


@app.route('/api/simulate/start', methods=['POST'])
@_require_api_token
def api_simulate_start():
    d = _optional_json_body()
    try:
        status = start_simulation(
            p1_model=d.get('p1_model'),
            p1_params=d.get('p1_params', {}) or {},
            p2_model=d.get('p2_model'),
            p2_params=d.get('p2_params', {}) or {},
            n_games=d.get('n_games', 1),
            save_results=bool(d.get('save_results', False)),
            save_games=bool(d.get('save_games', False)),
            save_period=int(d.get('save_period', 1) or 1),
            verbose=bool(d.get('verbose', True)),
            rated=bool(d.get('rated', True)),
            max_steps=int(d.get('max_steps', 1000) or 1000),
            engine=d.get('engine', DEFAULT_ENGINE),
        )
        return jsonify(status)
    except ValueError as e:
        return jsonify({'error': str(e)}), 409


@app.route('/api/simulate/stop', methods=['POST'])
@_require_api_token
def api_simulate_stop():
    return jsonify(stop_simulation())


@app.route('/api/simulate/status', methods=['GET'])
def api_simulate_status():
    return jsonify(simulate_status())


@app.route('/api/simulate/stream')
def api_simulate_stream():
    try:
        since_seq = int(request.args.get('since', 0))
    except (TypeError, ValueError):
        since_seq = 0
    def generate():
        last_game = -1
        last_status = None
        sent_seq = since_seq
        while True:
            status = simulate_status()
            st = status.get("status", "idle")
            g = status.get("current_game", 0)
            new_lines, seq = simulate_log_tail(sent_seq)
            if g != last_game or st != last_status or new_lines:
                payload = dict(status)
                payload["new_log_lines"] = new_lines
                payload["log_seq"] = seq
                payload.pop("log_lines", None)
                yield f"data: {json.dumps(payload)}\n\n"
                last_game = g
                last_status = st
                sent_seq = seq
            if st in ("idle", "completed", "error"):
                break
            simulate_wait_for_change(2.0)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


_test_state = {
    "process": None,
    "status": "idle",
    "output": "",
    "lock": threading.Lock(),
}


@app.route('/api/tests/run', methods=['POST'])
@_require_api_token
def api_tests_run():
    with _test_state["lock"]:
        if _test_state["status"] == "running":
            return jsonify({"error": "Tests are already running"}), 409
        _test_state["status"] = "running"
        _test_state["output"] = ""

    def _run():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.join("tests", "test.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
                bufsize=1,
            )
            with _test_state["lock"]:
                _test_state["process"] = proc
            out_lines = []
            for line in iter(proc.stdout.readline, ''):
                out_lines.append(line)
            proc.wait()
            with _test_state["lock"]:
                _test_state["output"] = ''.join(out_lines)
                _test_state["status"] = "completed" if proc.returncode == 0 else "error"
                _test_state["process"] = None
        except Exception as e:
            with _test_state["lock"]:
                _test_state["output"] = str(e)
                _test_state["status"] = "error"
                _test_state["process"] = None

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "running"})


@app.route('/api/tests/status', methods=['GET'])
def api_tests_status():
    with _test_state["lock"]:
        return jsonify({
            "status": _test_state["status"],
            "output": _test_state["output"],
        })


@app.after_request
def add_cache_headers(response):
    # No cross-origin API access is needed: the UI and the API are served
    # from the same Flask app/origin. A wildcard CORS header here would only
    # let any third-party webpage make authenticated-by-network-position
    # requests (start/stop training, overwrite config) against this server
    # from a visiting browser - so we deliberately don't set one.
    if request.path.startswith('/static/tiles/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    if debug and host not in ('127.0.0.1', 'localhost'):
        # Werkzeug's interactive debugger is remote code execution if it's
        # reachable from anywhere but localhost (e.g. the Tailscale-exposed
        # Docker deployment binds HOST=0.0.0.0). Refuse rather than trust
        # the env var blindly.
        print(f"WARNING: FLASK_DEBUG is set but HOST={host!r} is not loopback; "
              f"refusing to enable the interactive debugger. Set HOST=127.0.0.1 "
              f"to use debug mode.")
        debug = False
    print(f"\nSkud Pai Sho running at http://{host}:{port}\n")
    app.run(debug=debug, use_reloader=False, port=port, host=host, threaded=True)
