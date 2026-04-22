import json
import os
import threading
import time

from ai.registry import get_agent

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RATINGS_PATH = os.path.join(PROJECT_ROOT, 'elo_ratings.json')
HISTORY_PATH = os.path.join(PROJECT_ROOT, 'elo_history.json')

DEFAULT_RATING = 1200
K = 32

_lock = threading.Lock()


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _atomic_write(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _load_ratings():
    return _load_json(RATINGS_PATH, {})


def _save_ratings(ratings):
    _atomic_write(RATINGS_PATH, ratings)


def _load_history():
    return _load_json(HISTORY_PATH, [])


def _append_history(entry):
    history = _load_history()
    history.append(entry)
    _atomic_write(HISTORY_PATH, history)


def is_human_key(agent_key):
    return agent_key.startswith('human:')


def human_key(name):
    name = (name or '').strip()
    return f"human:{name}" if name else ''


def _weights_mtime(agent_key):
    if is_human_key(agent_key):
        return None
    entry = get_agent(agent_key)
    if not entry:
        return None
    rel = entry.get('model_path')
    if not rel:
        return None
    abspath = os.path.join(PROJECT_ROOT, rel)
    if not os.path.exists(abspath):
        return None
    try:
        return os.path.getmtime(abspath)
    except OSError:
        return None


def _check_and_reset(ratings, agent_key):
    """If weights changed since last rating, archive history and reset. Returns True on reset."""
    if is_human_key(agent_key):
        return False
    cur_mtime = _weights_mtime(agent_key)
    rec = ratings.get(agent_key)
    if rec is None:
        return False
    stored = rec.get('weights_mtime')
    if cur_mtime is None or stored is None:
        if cur_mtime is not None and stored is None:
            rec['weights_mtime'] = cur_mtime
        return False
    if abs(cur_mtime - stored) < 1e-6:
        return False
    _append_history({
        'agent': agent_key,
        'rating': rec.get('rating', DEFAULT_RATING),
        'games': rec.get('games', 0),
        'wins': rec.get('wins', 0),
        'losses': rec.get('losses', 0),
        'draws': rec.get('draws', 0),
        'archived_at': time.time(),
        'reason': 'weights_mtime_changed',
    })
    ratings[agent_key] = {
        'rating': DEFAULT_RATING,
        'games': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'weights_mtime': cur_mtime,
        'last_updated': time.time(),
    }
    return True


def _ensure_record(ratings, agent_key):
    if agent_key not in ratings:
        ratings[agent_key] = {
            'rating': DEFAULT_RATING,
            'games': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'weights_mtime': _weights_mtime(agent_key),
            'last_updated': time.time(),
        }
    else:
        _check_and_reset(ratings, agent_key)


def get_rating(agent_key):
    with _lock:
        ratings = _load_ratings()
        _ensure_record(ratings, agent_key)
        _save_ratings(ratings)
        return ratings[agent_key]['rating']


def _expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def record_game(p1_key, p2_key, winner):
    """Update ELO for both players. winner is 1, 2, or None for a draw."""
    if not p1_key or not p2_key:
        return None
    with _lock:
        ratings = _load_ratings()
        _ensure_record(ratings, p1_key)
        _ensure_record(ratings, p2_key)

        r1 = ratings[p1_key]['rating']
        r2 = ratings[p2_key]['rating']

        if winner == 1:
            s1, s2 = 1.0, 0.0
        elif winner == 2:
            s1, s2 = 0.0, 1.0
        else:
            s1, s2 = 0.5, 0.5

        e1 = _expected(r1, r2)
        e2 = _expected(r2, r1)

        new_r1 = r1 + K * (s1 - e1)
        new_r2 = r2 + K * (s2 - e2)

        for key, new_r, score in ((p1_key, new_r1, s1), (p2_key, new_r2, s2)):
            rec = ratings[key]
            rec['rating'] = round(new_r, 1)
            rec['games'] = rec.get('games', 0) + 1
            if score == 1.0:
                rec['wins'] = rec.get('wins', 0) + 1
            elif score == 0.0:
                rec['losses'] = rec.get('losses', 0) + 1
            else:
                rec['draws'] = rec.get('draws', 0) + 1
            rec['last_updated'] = time.time()

        _save_ratings(ratings)

        return {
            'p1_key': p1_key,
            'p2_key': p2_key,
            'p1_before': round(r1, 1),
            'p2_before': round(r2, 1),
            'p1_after': ratings[p1_key]['rating'],
            'p2_after': ratings[p2_key]['rating'],
            'p1_delta': round(ratings[p1_key]['rating'] - r1, 1),
            'p2_delta': round(ratings[p2_key]['rating'] - r2, 1),
            'winner': winner,
        }


def get_leaderboard(include_humans=True, include_bots=True):
    with _lock:
        ratings = _load_ratings()
        dirty = False
        for key in list(ratings.keys()):
            if _check_and_reset(ratings, key):
                dirty = True
        if dirty:
            _save_ratings(ratings)

        rows = []
        for key, rec in ratings.items():
            human = is_human_key(key)
            if human and not include_humans:
                continue
            if not human and not include_bots:
                continue
            display = key.split(':', 1)[1] if human else key
            rows.append({
                'key': key,
                'display_name': display,
                'is_human': human,
                'rating': rec.get('rating', DEFAULT_RATING),
                'games': rec.get('games', 0),
                'wins': rec.get('wins', 0),
                'losses': rec.get('losses', 0),
                'draws': rec.get('draws', 0),
                'last_updated': rec.get('last_updated'),
            })
        rows.sort(key=lambda r: r['rating'], reverse=True)
        return rows


def get_history():
    with _lock:
        return _load_history()
