"""Player-facing game registry: modes, seats with secret tokens, seeks, live list.

Everything is in memory (Phase 2 adds a database). Person-vs-person games are locked
to their seat tokens; pass-and-play and Lab games keep the old open behaviour.

server.py imports this as `ui.play` (the same way it imports its other sibling
modules) and calls `init()` once; its mutating endpoints call `guard()` under
`game_lock()` before touching a registered game.
"""
import contextlib
import hmac
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from flask import Blueprint, jsonify, request

from Agents.registry import house_bots

bp = Blueprint('play', __name__)

META = {}            # game_id -> {'mode', 'seats', 'created', 'last_active'}
_SEEKS = {}          # seek_id -> {'nickname', 'client_id', 'created', 'polled', 'result'}
_lock = threading.RLock()
_game_locks = {}     # game_id -> Lock serialising guard + mutation per registered game
_deps = {}
MAX_GAMES = 200
MAX_LIVE_PER_IP = 5          # concurrently live (unfinished) games created by one IP
SEEK_TTL = 6                 # a seek not polled in this long is dropped; clients poll every 1s
IDLE_SECONDS = 1800          # 30 min: default idle eviction for an active game
OPEN_CHALLENGE_SECONDS = 600  # 10 min: a challenge whose open seat nobody joined
FINISHED_SECONDS = 120        # 2 min: a game that already has a winner
TUTORIAL_IDLE_SECONDS = 600   # 10 min: an abandoned tutorial game (tab closed mid-step)
_TUTORIAL_FOR_IP = {}         # ip -> gid of that IP's current (only) tutorial game
_RATE = defaultdict(deque)   # ip -> recent create timestamps
RATE_LIMIT = (20, 60)        # at most 20 creates per 60 s per IP
_MODES = ('human', 'pass', 'bot')


class _CreateBlocked(Exception):
    """Raised by _create_game() when creation is refused for a reason that
    isn't a bad setup (server full, or this IP already has too many live
    games) - carries the status/message _create_or_error() turns into a
    response."""

    def __init__(self, msg, status):
        super().__init__(msg)
        self.msg = msg
        self.status = status


def init(games, serialize, game_class, bot_move, on_evict=None):
    """Wire in server.py's state. `on_evict(gid)` (optional) drops server-side
    per-game state (undo stacks, cached bot agents, ...) when a game is evicted."""
    _deps.update(games=games, serialize=serialize, game_class=game_class,
                 bot_move=bot_move, on_evict=on_evict)


def best_engine():
    try:
        import RustEngine  # noqa: F401
        return 'rust'
    except ImportError:
        return 'python'


def _err(msg, status):
    return jsonify({'error': msg}), status


def _body():
    d = request.get_json(silent=True)
    return d if isinstance(d, dict) else None


def _rate_limited():
    ip = request.remote_addr or '?'
    now = time.time()
    with _lock:
        q = _RATE[ip]
        while q and now - q[0] > RATE_LIMIT[1]:
            q.popleft()
        if len(q) >= RATE_LIMIT[0]:
            return True
        q.append(now)
        return False


def _nickname(d):
    n = d.get('nickname')
    if n is None:
        return 'Guest'
    if not isinstance(n, str):
        return None
    n = ''.join(ch for ch in n if ch.isprintable()).strip()[:24]
    return n or 'Guest'


def _seat_choice(d):
    s = d.get('seat', 'random')
    if s == 'random':
        return secrets.choice([1, 2])
    if isinstance(s, bool) or s not in (1, 2):
        return None
    return s


def _setup(d):
    """Only the setup keys api_new_game accepts: accents and opening."""
    raw = d.get('setup')
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError('setup must be an object')
    out = {}
    if raw.get('accents') is not None:
        if not isinstance(raw['accents'], dict):
            raise ValueError('setup.accents must be an object')
        out['accents'] = {int(k): v for k, v in raw['accents'].items()}
    if 'opening' in raw:
        out['opening'] = raw['opening']
    return out


def _seat(name, kind='human', bot=None, level=None, token=True):
    return {'name': name, 'token': secrets.token_urlsafe(16) if token else None,
            'kind': kind, 'bot': bot, 'level': level}


def _live_count_for_ip(ip):
    games = _deps['games']
    n = 0
    for gid, m in META.items():
        if ip is None or m.get('creator_ip') != ip or m.get('tutorial'):
            continue
        g = games.get(gid)
        if g is not None and g.winner is None:
            n += 1
    return n


def _create_game(mode, seats, setup=None, game=None, ip=None, count_against_cap=True,
                  tutorial=False):
    """Create and register a game; returns its id. Raises ValueError/TypeError
    for a bad setup, or _CreateBlocked when the server is full or (when
    `count_against_cap`) `ip` already has MAX_LIVE_PER_IP live games. If
    `game` is given, it's registered as-is (e.g. built via from_dict) instead
    of being constructed from `setup`. `count_against_cap=False` is for
    tutorial games (see create_tutorial_game): they get their own one-per-IP
    replacement policy instead of competing with a learner's real games for
    the per-IP live-game cap."""
    if game is None:
        game = _deps['game_class'](best_engine())(**(setup or {}))
    with _lock:
        evict_idle()
        if len(META) >= MAX_GAMES:
            raise _CreateBlocked('the server is full, try again in a few minutes', 503)
        if count_against_cap and _live_count_for_ip(ip) >= MAX_LIVE_PER_IP:
            raise _CreateBlocked(
                'too many games in progress from this connection — finish one first', 429)
        gid = secrets.token_urlsafe(6)
        while gid in _deps['games'] or gid in META:
            gid = secrets.token_urlsafe(6)
        now = time.time()
        _deps['games'][gid] = game
        META[gid] = {'mode': mode, 'seats': seats, 'created': now, 'last_active': now,
                     'creator_ip': ip, 'tutorial': tutorial}
        return gid


def _seat_of_token(meta, token):
    if not isinstance(token, str) or not token:
        return None
    for p, s in meta['seats'].items():
        if s['token'] and hmac.compare_digest(s['token'], token):
            return p
    return None


# ---------------------------------------------------------------- server hooks

def game_lock(gid):
    """A per-game lock for registered games (Lab games get a no-op context), so a
    seat check and the mutation it authorises can't interleave with another request
    on the same game (e.g. a double-submitted move landing on the opponent's turn,
    or two bot_move calls both stepping)."""
    with _lock:
        if gid not in META:
            return contextlib.nullcontext()
        return _game_locks.setdefault(gid, threading.Lock())


def seat_of(gid, body):
    """The seat number the body's seat_token holds in a registered game, else None."""
    meta = META.get(gid)
    if meta is None:
        return None
    return _seat_of_token(meta, (body or {}).get('seat_token'))


def guard(gid, action, body):
    """Return an error (response, status) for a mutation the caller may not make, else None."""
    meta = META.get(gid)
    if meta is None:
        return None                                   # Lab / legacy game: unchanged behaviour
    game = _deps['games'].get(gid)
    if action in ('undo', 'set_state', 'new_game') and meta['mode'] == 'human':
        return _err('not allowed in a game against a person', 403)
    if action == 'bot_move':
        seat = meta['seats'].get(game.current_player) if game is not None else None
        if meta['mode'] != 'bot' or not seat or seat['kind'] != 'bot' or game.winner is not None:
            return _err("it isn't the bot's turn", 409)
        return None
    if meta['mode'] == 'pass':
        return None
    who = _seat_of_token(meta, (body or {}).get('seat_token'))
    if who is None:
        return _err('not your seat', 403)
    if action == 'move' and game is not None and game.current_player != who:
        return _err('not your turn', 403)
    if action == 'resign':
        player = (body or {}).get('player', who)
        if isinstance(player, bool) or player != who:
            return _err('you can only resign your own seat', 403)
    return None


def touch(gid):
    with _lock:
        meta = META.get(gid)
        if meta is not None:
            meta['last_active'] = time.time()


def _drop_expired_seeks(now):
    for sid in [s for s, rec in _SEEKS.items() if now - rec['polled'] > SEEK_TTL]:
        del _SEEKS[sid]


def _idle_limit_for(meta, game, idle_seconds):
    """The idle_seconds a game actually gets: a finished game only gets
    FINISHED_SECONDS past its last activity (the winning move, since GET no
    longer refreshes last_active for spectators), a tutorial game only gets
    TUTORIAL_IDLE_SECONDS (it's normally replaced the moment the next step
    loads - see create_tutorial_game - so this only catches an abandoned
    tab), an unfilled challenge only gets OPEN_CHALLENGE_SECONDS, and
    everything else gets the full idle budget."""
    if game is not None and game.winner is not None:
        return min(idle_seconds, FINISHED_SECONDS)
    if meta.get('tutorial'):
        return min(idle_seconds, TUTORIAL_IDLE_SECONDS)
    if meta['mode'] == 'human' and any(
            s['kind'] == 'human' and s['name'] is None for s in meta['seats'].values()):
        return min(idle_seconds, OPEN_CHALLENGE_SECONDS)
    return idle_seconds


def _evict_game(gid):
    """Drop one registered game's server-side state: META, the per-game lock,
    the server's own game object, and (via on_evict) any undo stack or
    cached bot agent. Caller holds _lock."""
    games = _deps.get('games', {})
    on_evict = _deps.get('on_evict')
    META.pop(gid, None)
    _game_locks.pop(gid, None)
    games.pop(gid, None)
    if on_evict:
        on_evict(gid)


def evict_idle(now=None, idle_seconds=IDLE_SECONDS):
    """Drop games idle longer than their effective limit (and any whose game
    object is already gone) from META and the server's games, plus expired
    seeks. See _idle_limit_for() for the finished-game / tutorial / open-
    challenge shorter limits."""
    now = time.time() if now is None else now
    games = _deps.get('games', {})
    with _lock:
        stale = [gid for gid, m in META.items()
                 if gid not in games
                 or now - m['last_active'] > _idle_limit_for(m, games.get(gid), idle_seconds)]
        for gid in stale:
            _evict_game(gid)
        for ip in [ip for ip, g in _TUTORIAL_FOR_IP.items() if g not in META]:
            del _TUTORIAL_FOR_IP[ip]
        _drop_expired_seeks(now)
        for ip in [ip for ip, q in _RATE.items() if not q or now - q[-1] > RATE_LIMIT[1]]:
            del _RATE[ip]
    return stale


# ---------------------------------------------------------------------- routes

def _create_or_error(mode, seats, d, ip=None):
    try:
        gid = _create_game(mode, seats, _setup(d), ip=ip)
    except (ValueError, TypeError, ImportError, AttributeError) as e:
        return None, _err(f'invalid setup: {e}', 400)
    except _CreateBlocked as e:
        return None, _err(e.msg, e.status)
    return gid, None


@bp.route('/api/challenge', methods=['POST'])
def api_challenge():
    d = _body()
    if d is None:
        return _err('request body must be a JSON object', 400)
    name, seat = _nickname(d), _seat_choice(d)
    if name is None or seat is None:
        return _err("nickname must be text and seat 1, 2 or 'random'", 400)
    if _rate_limited():
        return _err('too many new games, slow down', 429)
    seats = {seat: _seat(name), 3 - seat: _seat(None)}
    gid, err = _create_or_error('human', seats, d, ip=request.remote_addr)
    if err:
        return err
    return jsonify({'game_id': gid, 'seat': seat, 'seat_token': seats[seat]['token'],
                    'join_url': f'/game/{gid}'})


@bp.route('/api/join/<gid>', methods=['POST'])
def api_join(gid):
    d = _body()
    if d is None:
        d = {}
    name = _nickname(d)
    if name is None:
        return _err('nickname must be text', 400)
    with _lock:
        meta = META.get(gid)
        if meta is None or gid not in _deps['games']:
            return _err('no such game', 404)
        free = next((p for p, s in sorted(meta['seats'].items())
                     if s['kind'] == 'human' and s['name'] is None), None)
        if meta['mode'] != 'human' or free is None:
            return _err('this game has no free seat', 409)
        meta['seats'][free]['name'] = name
        meta['last_active'] = time.time()
        token = meta['seats'][free]['token']
    return jsonify({'game_id': gid, 'seat': free, 'seat_token': token})


@bp.route('/api/seek', methods=['POST'])
def api_seek():
    d = _body()
    if d is None:
        return _err('request body must be a JSON object', 400)
    name, client = _nickname(d), d.get('client_id')
    if name is None or not isinstance(client, str) or not client or len(client) > 64:
        return _err('nickname must be text and client_id a non-empty string', 400)
    requested_sid = d.get('seek_id')
    if requested_sid is not None and not isinstance(requested_sid, str):
        return _err('seek_id must be a string', 400)
    if _rate_limited():
        return _err('too many seeks, slow down', 429)
    with _lock:
        now = time.time()
        _drop_expired_seeks(now)
        target = None
        if requested_sid:
            rec = _SEEKS.get(requested_sid)
            if rec is not None and rec['result'] is None and rec['client_id'] != client:
                target = (requested_sid, rec)
        else:
            waiting = sorted(((sid, rec) for sid, rec in _SEEKS.items()
                              if rec['result'] is None and rec['client_id'] != client),
                             key=lambda x: x[1]['created'])
            if waiting:
                target = waiting[0]
        if target:
            sid, rec = target
            theirs = secrets.choice([1, 2])
            mine = 3 - theirs
            seats = {theirs: _seat(rec['nickname']), mine: _seat(name)}
            gid, err = _create_or_error('human', seats, {}, ip=request.remote_addr)
            if err:
                return err
            rec['result'] = {'game_id': gid, 'seat': theirs, 'seat_token': seats[theirs]['token']}
            return jsonify({'game_id': gid, 'seat': mine, 'seat_token': seats[mine]['token']})
        sid = secrets.token_urlsafe(9)
        _SEEKS[sid] = {'nickname': name, 'client_id': client, 'created': now,
                       'polled': now, 'result': None}
    return jsonify({'seek_id': sid})


def _own_seek(sid):
    """The seek record if the query's client_id owns it, else None (caller holds _lock)."""
    rec = _SEEKS.get(sid)
    client = request.args.get('client_id', '')
    if rec is None or not client or not hmac.compare_digest(rec['client_id'], client):
        return None
    return rec


@bp.route('/api/seek/<sid>', methods=['GET'])
def api_seek_poll(sid):
    with _lock:
        _drop_expired_seeks(time.time())
        rec = _own_seek(sid)
        if rec is None:
            return _err('seek expired', 404)
        rec['polled'] = time.time()
        if rec['result'] is None:
            return jsonify({'status': 'waiting'})
        return jsonify({'status': 'paired', **rec['result']})


@bp.route('/api/seek/<sid>', methods=['DELETE'])
def api_seek_cancel(sid):
    with _lock:
        if _own_seek(sid) is not None:
            del _SEEKS[sid]
    return '', 204


@bp.route('/api/seeks', methods=['GET'])
def api_seeks():
    with _lock:
        now = time.time()
        _drop_expired_seeks(now)
        rows = [{'seek_id': sid, 'nickname': rec['nickname'], 'age_s': int(now - rec['created'])}
                for sid, rec in sorted(_SEEKS.items(), key=lambda x: x[1]['created'])
                if rec['result'] is None]
    return jsonify(rows)


@bp.route('/api/bot_game', methods=['POST'])
def api_bot_game():
    d = _body()
    if d is None:
        return _err('request body must be a JSON object', 400)
    level = d.get('level')
    bots = {b['house_level']: b for b in house_bots()}
    if isinstance(level, bool) or not isinstance(level, int) or level not in bots or not 1 <= level <= 6:
        return _err('level must be an integer from 1 to 6', 400)
    name, seat = _nickname(d), _seat_choice(d)
    if name is None or seat is None:
        return _err("nickname must be text and seat 1, 2 or 'random'", 400)
    if _rate_limited():
        return _err('too many new games, slow down', 429)
    bot = bots[level]
    seats = {seat: _seat(name),
             3 - seat: _seat(bot['label'], kind='bot', bot=bot['key'], level=level, token=False)}
    gid, err = _create_or_error('bot', seats, d, ip=request.remote_addr)
    if err:
        return err
    return jsonify({'game_id': gid, 'seat': seat, 'seat_token': seats[seat]['token']})


@bp.route('/api/pass_game', methods=['POST'])
def api_pass_game():
    d = _body()
    if d is None:
        d = {}
    if _rate_limited():
        return _err('too many new games, slow down', 429)
    seats = {1: _seat('Player 1', token=False), 2: _seat('Player 2', token=False)}
    gid, err = _create_or_error('pass', seats, d, ip=request.remote_addr)
    if err:
        return err
    return jsonify({'game_id': gid})


def create_tutorial_game(state):
    """Register a pass-mode game from a tutorial `from_dict` state (steps.json).

    Each connection keeps at most one tutorial game: loading a step (Next,
    or "Start this step over") replaces whatever tutorial game that IP had
    before, and tutorial games never count against MAX_LIVE_PER_IP - walking
    all 7 steps (or retrying one) must never compete with that IP's real
    games for the per-IP live-game cap, which previously made the tutorial
    fail outright after 5 steps. A tutorial game nobody replaces (the tab
    closed mid-step) is still swept by evict_idle() after
    TUTORIAL_IDLE_SECONDS. Returns (gid, error_response); never raises - a
    bad state becomes a 400."""
    try:
        game = _deps['game_class'](best_engine()).from_dict(state)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        return None, _err(f'invalid tutorial state: {e}', 400)
    if _rate_limited():
        return None, _err('too many new games, slow down', 429)
    ip = request.remote_addr
    seats = {1: _seat('Player 1', token=False), 2: _seat('Player 2', token=False)}
    try:
        with _lock:
            prev = _TUTORIAL_FOR_IP.get(ip)
            if prev is not None:
                _evict_game(prev)
            gid = _create_game('pass', seats, game=game, ip=ip,
                                count_against_cap=False, tutorial=True)
            _TUTORIAL_FOR_IP[ip] = gid
    except _CreateBlocked as e:
        return None, _err(e.msg, e.status)
    return gid, None


def _public_seats(meta):
    return {p: {'name': s['name'], 'kind': s['kind'], 'level': s['level']}
            for p, s in meta['seats'].items()}


@bp.route('/api/game/<gid>', methods=['GET'])
def api_game(gid):
    with _lock:
        meta = META.get(gid)
        game = _deps['games'].get(gid)
        if meta is None or game is None:
            return _err('this game has ended or expired', 404)
        seats = _public_seats(meta)
        mode = meta['mode']
        # A spectator polling (or holding) the id never keeps a game's idle
        # timer alive by itself - only a seated player's own poll (one
        # holding a valid seat_token) or a successful move does. Otherwise
        # the game cap is trivially exhaustible: anyone could keep any game
        # (including one they're not even in) alive forever just by polling
        # its id.
        who = _seat_of_token(meta, request.args.get('seat_token'))
    if who is not None:
        touch(gid)
    return jsonify({'state': _deps['serialize'](game), 'mode': mode, 'seats': seats,
                    'over': game.winner is not None})


BOT_LIVE_IDLE_SECONDS = 120   # a bot game nobody has polled in this long is probably abandoned


@bp.route('/api/live', methods=['GET'])
def api_live():
    """Newest 20 games in progress. Pass-and-play games (tokenless, so anyone holding
    the id could move in them) and challenges with an open seat (anyone holding the id
    could take it) are never advertised. A game with no moves yet is just noise (nothing
    to watch), and a bot game nobody has actively polled in a while is likely a tab the
    human walked away from."""
    games = _deps['games']
    now = time.time()
    with _lock:
        rows = []
        for gid, m in sorted(META.items(), key=lambda x: x[1]['last_active'], reverse=True):
            g = games.get(gid)
            if g is None or g.winner is not None or m['mode'] == 'pass':
                continue
            if any(s['name'] is None for s in m['seats'].values()):
                continue
            if len(g.history) == 0:
                continue
            if m['mode'] == 'bot' and now - m['last_active'] > BOT_LIVE_IDLE_SECONDS:
                continue
            rows.append({'game_id': gid, 'mode': m['mode'],
                         'names': [m['seats'][1]['name'], m['seats'][2]['name']],
                         'moves': len(g.history)})
            if len(rows) == 20:
                break
    return jsonify(rows)


@bp.route('/api/house_bots', methods=['GET'])
def api_house_bots():
    return jsonify([{'level': b['house_level'], 'label': b['label'], 'key': b['key']}
                    for b in house_bots()])


@bp.route('/api/lab/check', methods=['GET'])
def api_lab_check():
    """Owner ruling 37: the Lab link is shown only when MUSHIBOT_API_TOKEN is
    configured on the server AND this browser holds a matching token. When no
    token is configured at all (local dev), the response says so via
    `token_required: false` — the /lab page itself uses that to stay open
    rather than showing an unusable "enter a token that doesn't exist" gate;
    the Lab's routes remain reachable by URL either way, just never
    advertised in the menu.

    Every page calls this on load to decide whether to show the Lab link, so
    a visitor without the token is the common case, not an error - it always
    answers 200 (ok distinguishes "authenticated" from "not"), never 401, so
    that normal case doesn't leave a console error on every page load."""
    expected = os.environ.get('MUSHIBOT_API_TOKEN', '').strip()
    if not expected:
        # No token configured at all (local dev): nothing to authenticate against.
        return jsonify({'ok': False, 'token_required': False})
    supplied = request.headers.get('X-API-Token', '')
    if hmac.compare_digest(supplied.encode(), expected.encode()):
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'token_required': True})
