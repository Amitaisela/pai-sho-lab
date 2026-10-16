"""A tiny in-process stand-in for the site's bot API, enough to drive the connector in tests.
Streams use chunked transfer encoding like the real server, so clients see lines as they are sent."""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'engine'))

from Agents.elo import RULES_VERSION  # noqa: E402
from engine.engine_select import game_class  # noqa: E402

TC = {'base_s': 180, 'inc_s': 2}


class FakeSite:
    def __init__(self, rules_version=None, illegal_first=False, rate_limit_move_once=False, close_first=False,
                 duplicate_state=False, always_429_no_retry_after=False, drop_game_stream_first=False):
        self.rules_version = RULES_VERSION if rules_version is None else rules_version
        self.illegal_first, self.rate_limit_move_once, self.close_first = illegal_first, rate_limit_move_once, close_first
        self.duplicate_state, self.always_429_no_retry_after = duplicate_state, always_429_no_retry_after
        self.drop_game_stream_first = drop_game_stream_first
        self.game_connections = 0
        game = game_class('python')()
        self.initial = {'setup': {'accents': {str(p): list(v) for p, v in game.setup['accents'].items()},
                                  'opening': game.setup['opening']}}
        self.legal = [list(a) for a in game.get_legal_actions()]
        self.moves, self.rejected, self.requests = [], [], []
        self.rate_limited = 0
        self.event_connections = 0
        self.move_event, self.finished = threading.Event(), threading.Event()
        site = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            # ---- plumbing
            def _json(self, status, body, headers=None):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _start_stream(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()

            def _chunk(self, data):
                self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                self.wfile.flush()

            def _send(self, obj):
                self._chunk((json.dumps(obj) + '\n').encode())

            def _end_stream(self):
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
                self.close_connection = True

            # ---- routes
            def do_GET(self):
                site.requests.append(('GET', self.path))
                try:
                    if self.path == '/api/bot/meta':
                        return self._json(200, {'api_version': 1, 'rules_version': site.rules_version,
                                                'time_controls': ['1+1', '3+2', '5+3'], 'limits': {}})
                    if self.path == '/api/stream/event':
                        return self._events()
                    if self.path == '/api/bot/game/stream/g1':
                        return self._game()
                    return self._json(404, {'error': 'nope', 'code': 'not_found'})
                except OSError:      # the client went away; Windows raises ConnectionAbortedError
                    pass

            def _events(self):
                site.event_connections += 1
                self._start_stream()
                self._send({'type': 'hello', 'api_version': 1, 'rules_version': site.rules_version,
                            'username': 'fakebot'})
                if site.close_first and site.event_connections == 1:
                    return self._end_stream()                       # a dropped connection
                self._send({'type': 'gameStart', 'game': {
                    'id': 'g1', 'seat': 1, 'opponent': {'name': 'other', 'bot': True, 'house': False},
                    'time_control': TC, 'rated': False}})
                while not site.finished.is_set():
                    self._chunk(b'\n')                              # keepalive
                    time.sleep(0.05)
                self._send({'type': 'gameFinish', 'game': {'id': 'g1', 'winner': 1, 'end_reason': 'resign',
                                                            'rated': False, 'rating': None}})
                self._end_stream()

            def _state(self, ply, actions, current, winner, legal):
                s = {'type': 'gameState', 'ply': ply, 'actions': actions, 'current_player': current,
                     'bonus_turn': False, 'winner': winner, 'end_reason': 'resign' if winner else None,
                     'clock': None}
                if legal is not None:
                    s['legal_actions'] = legal
                return s

            def _game(self):
                site.game_connections += 1
                self._start_stream()
                if site.drop_game_stream_first and site.game_connections == 1:
                    return self._end_stream()                       # the game stream dropped at once
                self._send({'type': 'gameFull', 'id': 'g1', 'seat': 1, 'rules_version': site.rules_version,
                            'engine': 'python', 'initial': site.initial,
                            'players': {'1': {'name': 'fakebot', 'bot': True, 'house': False},
                                        '2': {'name': 'other', 'bot': True, 'house': False}},
                            'time_control': TC, 'rated': False,
                            'state': self._state(0, [], 1, None, site.legal)})
                if site.duplicate_state:                            # the same state again, twice
                    self._send(self._state(0, [], 1, None, site.legal))
                    self._send(self._state(0, [], 1, None, site.legal))
                    time.sleep(0.3)
                if not site.move_event.wait(10):
                    return self._end_stream()
                self._send(self._state(1, [site.moves[0]['action']], 2, 1, None))
                site.finished.set()
                self._end_stream()

            def do_POST(self):
                site.requests.append(('POST', self.path))
                n = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(n) if n else b''
                body = json.loads(raw) if raw.strip() else {}
                if self.path == '/api/bot/game/g1/move':
                    if site.always_429_no_retry_after:
                        return self._json(429, {'error': 'too many games', 'code': 'too_many_games'})
                    if site.rate_limit_move_once and site.rate_limited == 0:
                        site.rate_limited += 1
                        return self._json(429, {'error': 'slow down', 'code': 'rate_limited'}, {'Retry-After': '0'})
                    if site.illegal_first and not site.rejected:
                        site.rejected.append(body)
                        return self._json(400, {'error': 'illegal move', 'code': 'illegal', 'strikes': 1,
                                                'forfeited': False})
                    site.moves.append(body)
                    site.move_event.set()
                    return self._json(200, {'ok': True, 'ply': 1})
                return self._json(200, {'ok': True, 'id': 'c1', 'game_id': 'g1'})

            do_DELETE = do_POST

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.finished.set()
        self.move_event.set()
        self.server.shutdown()
        self.server.server_close()
