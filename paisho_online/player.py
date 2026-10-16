"""Plays a registered agent on the site. The bot never remembers the position: every time it must
move it rebuilds the game from the setup and the full action list the site sent, so a reconnect,
a repeated message or a restart of the site cannot desynchronise it."""
import random
import threading
import time

import requests

from Agents.agent_loader import act, instantiate
from Agents.elo import RULES_VERSION
from Agents.registry import get_agent
from backend.simulator import parse_model_spec
from engine.engine_select import game_class
from paisho_online.client import ApiError, RulesMismatch


def rebuild(initial, actions, engine='python'):
    """The game after `actions`, built from the site's `initial` setup."""
    try:
        cls = game_class(engine)
    except ImportError:
        cls = game_class('python')                       # the site used Rust; the rules are identical
    if 'setup' not in initial:
        raise ValueError('this game starts from a custom position, which the connector does not support')
    setup = initial['setup']
    game = cls(accents={int(k): v for k, v in setup['accents'].items()}, opening=setup['opening'])
    for action in actions:
        game.step(tuple(action))
    return game


class OnlineBot:
    def __init__(self, client, agent_spec, *, accept=True, seek=None, rated=False, challenge=None,
                 vs_house=None, max_games=None, reconnect_delay=1.0, log=print):
        self.client, self.log = client, log
        self.key, self.params = parse_model_spec(agent_spec)
        self.entry = get_agent(self.key)
        if self.entry is None:
            raise ValueError(f'unknown agent: {self.key!r}')
        self.agent = instantiate(self.entry, params=self.params) if self.entry['kind'] == 'class' else None
        self.accept, self.seek, self.rated = accept, seek, rated
        self.challenge, self.vs_house = challenge, vs_house
        self.max_games, self.reconnect_delay = max_games, reconnect_delay
        self.finished = 0
        self._playing = {}                                # gid -> thread
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- the loop
    def run(self):
        """Connect, handle events, reconnect with backoff. Returns once `max_games` games have finished."""
        delay = self.reconnect_delay
        self._check_rules()
        while not self._done():
            try:
                for event in self.client.events():
                    delay = self.reconnect_delay
                    self._on_event(event)
                    if self._done():
                        break
            except RulesMismatch:
                raise
            except ApiError as e:
                if e.status in (401, 403):
                    raise
                self.log(f'api error: {e}; retrying in {delay:.1f}s')
                time.sleep(delay)
                delay = min(delay * 2, 30)
            except (requests.RequestException, ValueError) as e:
                self.log(f'connection lost ({type(e).__name__}); reconnecting in {delay:.1f}s')
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                if not self._done():
                    time.sleep(self.reconnect_delay)        # the site closed the stream (a newer one replaced it)
        for t in list(self._playing.values()):
            t.join(timeout=10)

    def _done(self):
        return self.max_games is not None and self.finished >= self.max_games

    def _check_rules(self):
        meta = self.client.meta()
        self._compare(meta['rules_version'])

    @staticmethod
    def _compare(site_version):
        if site_version != RULES_VERSION:
            raise RulesMismatch(f'site rules_version {site_version}, your clone {RULES_VERSION}: '
                                'update your clone (git pull) before playing online')

    # ---------------------------------------------------------------- events
    def _on_event(self, event):
        kind = event.get('type')
        if kind == 'hello':
            self._compare(event['rules_version'])
            self.log(f"connected as {event.get('username')}")
            self._start_activity()
        elif kind == 'challenge' and self.accept:
            ch = event['challenge']
            try:
                self.client.accept(ch['id'])
                self.log(f"accepted a challenge from {ch['challenger']['name']}")
            except ApiError as e:
                self.log(f'could not accept: {e}')
        elif kind == 'gameStart':
            self._start_game(event['game']['id'])
        elif kind == 'gameFinish':
            self.finished += 1
            game = event['game']
            self.log(f"game {game['id']} over: winner {game['winner']} ({game['end_reason']})")
            if not self._done():
                self._start_activity()

    def _start_activity(self):
        """What to do when idle: seek, challenge or start a house game, per the options."""
        try:
            if self.seek:
                self.client.seek(self.seek, self.rated)
            elif self.challenge:
                self.client.challenge(self.challenge[0], self.challenge[1], self.rated)
            elif self.vs_house:
                self.client.vs_house(self.vs_house[0], self.vs_house[1])
        except ApiError as e:
            self.log(f'could not start a game: {e}')

    # ---------------------------------------------------------------- one game
    def _start_game(self, gid):
        with self._lock:
            if gid in self._playing and self._playing[gid].is_alive():
                return
            t = threading.Thread(target=self._play, args=(gid,), daemon=True)
            self._playing[gid] = t
        t.start()

    def _play(self, gid):
        """Follow one game. If its stream drops before the game is over, reopen it (with backoff) and
        rebuild from the fresh gameFull; the site ignores a repeated move for a ply already played."""
        delay, failures = self.reconnect_delay, 0
        while not self._done():
            try:
                if self._follow(gid):
                    return
                failures = 0                              # the stream worked, then closed: reopen it
                delay = self.reconnect_delay
            except ApiError as e:
                if e.status in (401, 403, 404, 409):
                    self.log(f'game {gid}: {e}; giving up on this game')
                    return
                failures += 1
                self.log(f'game {gid}: api error ({e})')
            except (requests.RequestException, ValueError) as e:
                failures += 1
                self.log(f'game {gid}: stream lost ({type(e).__name__})')
            if failures >= 8:
                self.log(f'game {gid}: giving up after repeated failures')
                return
            time.sleep(delay)
            delay = min(delay * 2, 30)

    def _follow(self, gid):
        """Read one game stream to its end. True when the game is over."""
        seat = initial = engine = None
        last_ply = -1
        for event in self.client.game_stream(gid):
            if event['type'] == 'gameFull':
                seat, initial, engine = event['seat'], event['initial'], event.get('engine', 'python')
                state = event['state']
            elif event['type'] == 'gameState' and seat is not None:
                state = event
            else:
                continue
            if state['winner'] is not None:
                return True
            legal = state.get('legal_actions')
            if state['current_player'] == seat and legal and state['ply'] != last_ply:
                last_ply = state['ply']
                self._move(gid, initial, engine, seat, state, legal)
        return False

    def _move(self, gid, initial, engine, seat, state, legal):
        action = self._choose(initial, engine, seat, state, legal)
        try:
            self.client.move(gid, action, state['ply'])
        except ApiError as e:
            if e.code != 'illegal':
                self.log(f'move rejected: {e}')
                return
            self.log(f'the site rejected {action} as illegal; sending a random legal move')
            try:
                self.client.move(gid, random.choice(legal), state['ply'])
            except ApiError as e2:
                self.log(f'move rejected: {e2}')

    def _choose(self, initial, engine, seat, state, legal):
        """The agent's move if it is legal, else a random legal one (an agent bug must not stall the game)."""
        legal_set = {tuple(a) for a in legal}
        try:
            game = rebuild(initial, state['actions'], engine)
            params = dict(self.params)
            keys = {p['key'] for p in self.entry.get('play_params', [])}
            clock = state.get('clock')
            if 'time_budget' in keys and 'time_budget' not in params and clock:
                left = clock['remaining_ms'][str(seat)] / 1000
                params['time_budget'] = max(0.1, min(5.0, left / 30))
            action = act(self.entry, game, agent=self.agent, params=params, verbose=False)
            if action is not None and tuple(action) in legal_set:
                return list(action)
            self.log(f'agent returned {action!r}, which is not legal here; playing a random legal move')
        except Exception as e:  # noqa: BLE001
            self.log(f'agent failed ({type(e).__name__}: {e}); playing a random legal move')
        return random.choice(legal)
