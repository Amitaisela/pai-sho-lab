import os
import sys
import json
import random
import subprocess
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Flask, jsonify, request, send_from_directory, Response, render_template
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
                              ACCENT_TILES, SPECIAL_TILES, garden_of)
from PythonEngine.notation import game_to_psn, psn_to_game
from Agents.registry import get_agent, playable_agents, trainable_agents
from Agents.agent_loader import instantiate, act
from Agents import elo

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


def _json_body(*required):
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        raise _BadRequest("request body must be a JSON object")
    for k in required:
        if k not in d:
            raise _BadRequest(f"missing field: {k}")
    return d


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


games = {}
agent_names = {'1': None, '2': None}

elo_session = {
    'p1_key': None,
    'p2_key': None,
    'p1_human_name': '',
    'p2_human_name': '',
    'rated': True,
}
_recorded_games = set()
_last_elo_result = {}

_history_stacks = {}


def _save_snapshot(gid):
    g = games.get(gid)
    if not g:
        return
    if gid not in _history_stacks:
        _history_stacks[gid] = {'undo': [], 'redo': []}
    _history_stacks[gid]['undo'].append(g.clone())
    _history_stacks[gid]['redo'].clear()

_bot_agents = {}


def serialize(game: PaiShoGame) -> dict:
    board_s = {f"{r},{c}": t for (r, c), t in game.board.items()}
    h1 = game.find_harmonies(1)
    h2 = game.find_harmonies(2)
    return {
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
    }


@app.route('/')
def root():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'index.html')


@app.route('/api/new_game', methods=['POST'])
def api_new_game():
    games['default'] = PaiShoGame()
    _history_stacks['default'] = {'undo': [], 'redo': []}
    _recorded_games.discard('default')
    _last_elo_result.pop('default', None)
    return jsonify({'game_id': 'default', 'state': serialize(games['default'])})


def _resolve_agent_key(slot):
    k = elo_session.get(f'p{slot}_key')
    if not k:
        return None
    if k == 'human':
        name = elo_session.get(f'p{slot}_human_name', '') or 'Guest'
        return elo.human_key(name)
    return k


def _maybe_record_elo(gid, game):
    if gid in _recorded_games:
        return
    if not game.winner:
        return
    if not elo_session.get('rated', True):
        _recorded_games.add(gid)
        return
    p1_key = _resolve_agent_key('1')
    p2_key = _resolve_agent_key('2')
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
def api_plant(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    if g.winner:
        return jsonify({'error': 'game over'}), 400

    d = _json_body('flower', 'row', 'col')
    flower = _flower_name(d)
    r, c = _coord(d, 'row', 'col')
    kwargs = {}
    if 'displace_row' in d and 'displace_col' in d:
        dr, dc = _coord(d, 'displace_row', 'displace_col')
        kwargs['displace_r'] = dr
        kwargs['displace_c'] = dc
    _save_snapshot(gid)
    try:
        g.plant(flower, r, c, **kwargs)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'state': serialize(g)})


@app.route('/api/arrange/<gid>', methods=['POST'])
def api_arrange(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    if g.winner:
        return jsonify({'error': 'game over'}), 400

    d = _json_body('from_row', 'from_col', 'to_row', 'to_col')
    fr, fc = _coord(d, 'from_row', 'from_col')
    tr, tc = _coord(d, 'to_row', 'to_col')
    _save_snapshot(gid)
    try:
        g.arrange(fr, fc, tr, tc)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'state': serialize(g)})


@app.route('/api/valid_moves/<gid>', methods=['POST'])
def api_valid_moves(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    d = _json_body('row', 'col')
    r, c = _coord(d, 'row', 'col')
    moves = g.valid_destinations(r, c)
    return jsonify({'moves': moves})


@app.route('/api/valid_boat_displacement/<gid>', methods=['POST'])
def api_valid_boat_displacement(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    d = _json_body('target_row', 'target_col')
    tr, tc = _coord(d, 'target_row', 'target_col')
    target_tile = g.board.get((tr, tc))
    if not target_tile:
        return jsonify({'moves': []})

    valid_set = set(map(tuple, VALID_SPACES))
    gate_set = set(map(tuple, GATES))
    flower = target_tile['flower']
    fcol = FLOWER[flower]['color'] if flower in FLOWER else None

    moves = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = tr + dr, tc + dc
        if (nr, nc) not in valid_set or (nr, nc) in gate_set or (nr, nc) in g.board:
            continue
        if fcol is not None:
            gdn = garden_of(nr, nc)
            if (fcol == 'red' and gdn == 'white') or (fcol == 'white' and gdn == 'red'):
                continue
        moves.append([nr, nc])
    return jsonify({'moves': moves})


@app.route('/api/valid_plant_moves/<gid>', methods=['POST'])
def api_valid_plant_moves(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    d = _json_body()
    tile = d.get('tile', '')
    player = g.current_player

    if tile in ('Rock', 'Wheel', 'Knotweed'):
        moves = [list(pos) for pos in VALID_SPACES if pos not in set(GATES) and pos not in g.board]
    elif tile == 'Boat':
        moves = [list(pos) for pos, t in g.board.items()
                 if t['player'] != player and not t['growing']]
    elif tile in SPECIAL_TILES:
        moves = [list(gate) for gate in GATES if gate not in g.board]
    else:
        moves = [list(gate) for gate in GATES if gate not in g.board]

    return jsonify({'moves': moves})


@app.route('/api/set_state/<gid>', methods=['POST'])
def api_set_state(gid):
    d = _json_body('board', 'hands', 'current_player')
    try:
        games[gid] = PaiShoGame.from_dict(d)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({'error': f'invalid state: {e}'}), 400
    return jsonify({'status': 'success'})


@app.route('/api/set_agents', methods=['POST'])
def api_set_agents():
    global agent_names
    d = request.get_json(silent=True) or {}
    agent_names = {'1': d.get('p1', 'Player 1'), '2': d.get('p2', 'Player 2')}
    if 'p1_key' in d:
        elo_session['p1_key'] = d.get('p1_key')
    if 'p2_key' in d:
        elo_session['p2_key'] = d.get('p2_key')
    if 'rated' in d:
        elo_session['rated'] = bool(d.get('rated'))
    if 'p1_human_name' in d:
        elo_session['p1_human_name'] = d.get('p1_human_name', '') or ''
    if 'p2_human_name' in d:
        elo_session['p2_human_name'] = d.get('p2_human_name', '') or ''
    return jsonify({'status': 'success'})


@app.route('/api/agents', methods=['GET'])
def api_get_agents():
    return jsonify(agent_names)


@app.route('/leaderboard')
def leaderboard_page():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'leaderboard.html')


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
        d = request.json or {}
        for field in ('p1_key', 'p2_key', 'p1_human_name', 'p2_human_name'):
            if field in d:
                elo_session[field] = d[field] or ''
        if 'rated' in d:
            elo_session['rated'] = bool(d['rated'])
    return jsonify(dict(elo_session))


@app.route('/api/save/<gid>', methods=['GET'])
def api_save_game(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    p1 = agent_names.get('1', 'Player 1') or 'Player 1'
    p2 = agent_names.get('2', 'Player 2') or 'Player 2'
    save_data = g.to_save_dict(p1_name=p1, p2_name=p2)
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
    p1 = agent_names.get('1', 'Player 1') or 'Player 1'
    p2 = agent_names.get('2', 'Player 2') or 'Player 2'
    text = game_to_psn(g, p1_name=p1, p2_name=p2)
    filename = f"{p1}_vs_{p2}.psn"
    return Response(
        text,
        mimetype='text/plain',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.route('/api/import_psn/<gid>', methods=['POST'])
def api_import_psn(gid):
    text = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = data.get('psn')
    if text is None:
        text = request.get_data(as_text=True)
    if not text or not text.strip():
        return jsonify({'error': 'empty PSN'}), 400
    try:
        game, tags = psn_to_game(text, PaiShoGame)
    except Exception as e:
        return jsonify({'error': f'parse error: {e}'}), 400
    games[gid] = game
    _history_stacks[gid] = {'undo': [], 'redo': []}
    if tags.get('Player1'):
        agent_names['1'] = tags['Player1']
    if tags.get('Player2'):
        agent_names['2'] = tags['Player2']
    return jsonify({'state': serialize(games[gid]), 'tags': tags})


@app.route('/api/load/<gid>', methods=['POST'])
def api_load_game(gid):
    data = request.json
    if not data or 'state' not in data:
        return jsonify({'error': 'invalid save file'}), 400
    games[gid] = PaiShoGame.from_save_dict(data)
    _history_stacks[gid] = {'undo': [], 'redo': []}
    if data.get('p1'):
        agent_names['1'] = data['p1']
    if data.get('p2'):
        agent_names['2'] = data['p2']
    return jsonify({'state': serialize(games[gid])})


@app.route('/api/undo/<gid>', methods=['POST'])
def api_undo(gid):
    stacks = _history_stacks.get(gid)
    if not stacks or not stacks['undo']:
        return jsonify({'error': 'nothing to undo'}), 400
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    stacks['redo'].append(g.clone())
    games[gid] = stacks['undo'].pop()
    return jsonify({'state': serialize(games[gid]),
                    'can_undo': len(stacks['undo']) > 0,
                    'can_redo': len(stacks['redo']) > 0})


@app.route('/api/redo/<gid>', methods=['POST'])
def api_redo(gid):
    stacks = _history_stacks.get(gid)
    if not stacks or not stacks['redo']:
        return jsonify({'error': 'nothing to redo'}), 400
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    stacks['undo'].append(g.clone())
    games[gid] = stacks['redo'].pop()
    return jsonify({'state': serialize(games[gid]),
                    'can_undo': len(stacks['undo']) > 0,
                    'can_redo': len(stacks['redo']) > 0})


def _get_bot_agent(bot_type, params):
    key = bot_type.lower()
    if key in _bot_agents:
        return _bot_agents[key]

    entry = get_agent(key)
    if not entry or entry["kind"] != "class":
        return None

    try:
        agent = instantiate(entry, params=params)
    except Exception:
        return None

    _bot_agents[key] = agent
    return agent


def _bot_choose_action(game, bot_type, params):
    key = bot_type.lower()
    legal_actions = game.get_legal_actions()
    if not legal_actions:
        return None

    entry = get_agent(key)
    if not entry:
        return random.choice(legal_actions)

    kind = entry["kind"]

    if kind == "class":
        agent = _get_bot_agent(key, params)
        if agent is None:
            return random.choice(legal_actions)
        # Every play_param is re-applied each turn, falling back to its own
        # default when absent, so a stale override never lingers on the
        # cached agent instance.
        synced = {pd["key"]: params.get(pd["key"], pd["default"]) for pd in entry.get("play_params", [])}
        return act(entry, game, agent=agent, params=synced, verbose=False, legal_actions=legal_actions)

    if kind in ("inline", "function"):
        return act(entry, game, params=params, verbose=False, legal_actions=legal_actions)

    return random.choice(legal_actions)


@app.route('/api/bot_move/<gid>', methods=['POST'])
def api_bot_move(gid):
    g = games.get(gid)
    if not g:
        return jsonify({'error': 'not found'}), 404
    if g.winner:
        return jsonify({'error': 'game over', 'state': serialize(g)}), 400

    d = request.json
    bot_type = d.get('bot', 'random')
    params = d.get('params', {})

    _save_snapshot(gid)
    action = _bot_choose_action(g, bot_type, params)
    if action is None:
        return jsonify({'error': 'no legal moves', 'state': serialize(g)}), 400

    try:
        g.step(action)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'state': serialize(g), 'action': list(action)})


@app.route('/train')
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
    return render_template('train.html', init_data=init_data)


@app.route('/guide')
def guide_page():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'guide.html')


@app.route('/rules')
def rules_page():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'rules.html')


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
def api_training_start():
    d = request.json
    model = d.get('model')
    params = d.get('params', {})
    try:
        status = start_training(model, params)
        return jsonify(status)
    except ValueError as e:
        return jsonify({'error': str(e)}), 409


@app.route('/api/training/stop', methods=['POST'])
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
def api_training_config_write():
    d = request.json
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
def simulate_page():
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'simulate.html')


@app.route('/api/simulate/start', methods=['POST'])
def api_simulate_start():
    d = request.json or {}
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
        )
        return jsonify(status)
    except ValueError as e:
        return jsonify({'error': str(e)}), 409


@app.route('/api/simulate/stop', methods=['POST'])
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
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    if request.path.startswith('/static/tiles/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    print(f"\nSkud Pai Sho running at http://{host}:{port}\n")
    app.run(debug=debug, use_reloader=False, port=port, host=host, threaded=True)
