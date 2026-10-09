"""Bot move selection, shared by the site and the Flask Lab server (lives in ui/ so the public
mirror, which ships no site_app, keeps it). M2.3 moves the `act()` call into a worker pool."""
from Agents.agent_loader import act, instantiate
from Agents.registry import get_agent
import random
import threading


def clamp_params(entry, params):
    """Clamp every numeric play param to its registry min/max. Unknown keys pass through."""
    if not isinstance(params, dict):
        return {}
    out = dict(params)
    for pd in entry.get('play_params', []):
        if pd.get('type') != 'number':
            continue
        k = pd['key']
        if k not in out:
            continue
        v = out[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        lo, hi = pd.get('min'), pd.get('max')
        if lo is not None and v < lo:
            v = lo
        if hi is not None and v > hi:
            v = hi
        out[k] = v
    return out


class BotCache:
    def __init__(self):
        self._agents = {}          # (gid, bot_type) -> agent instance
        self._lock = threading.Lock()   # bot threads insert while the loop drops games

    def get_agent(self, gid, bot_type, params):
        key = (gid, bot_type.lower())
        with self._lock:
            agent = self._agents.get(key)
        if agent is not None:
            return agent
        entry = get_agent(bot_type.lower())
        if not entry or entry['kind'] != 'class':
            return None
        try:
            agent = instantiate(entry, params=clamp_params(entry, params))
        except Exception:
            return None
        with self._lock:
            return self._agents.setdefault(key, agent)

    def drop_game(self, gid):
        with self._lock:
            for key in [k for k in self._agents if k[0] == gid]:
                del self._agents[key]


def choose_action(cache, gid, game, bot_type, params):
    key = bot_type.lower()
    legal = game.get_legal_actions()
    if not legal:
        return None
    entry = get_agent(key)
    if not entry:
        return random.choice(legal)
    params = clamp_params(entry, params)
    if entry['kind'] == 'class':
        agent = cache.get_agent(gid, key, params)
        if agent is None:
            return random.choice(legal)
        synced = {pd['key']: params.get(pd['key'], pd['default']) for pd in entry.get('play_params', [])}
        return act(entry, game, agent=agent, params=synced, verbose=False, legal_actions=legal)
    if entry['kind'] in ('inline', 'function'):
        return act(entry, game, params=params, verbose=False, legal_actions=legal)
    return random.choice(legal)
