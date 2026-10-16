"""A small client for the site's bot API: HTTPS + streaming NDJSON, bearer token. Needs only `requests`."""
import json
import time

import requests

TIMEOUT = (5, 30)        # connect, read: the server sends a blank keepalive line every 6 s


class ApiError(Exception):
    def __init__(self, status, body):
        self.status = status
        self.body = body if isinstance(body, dict) else {}
        self.code = self.body.get('code')
        super().__init__(f'{status} {self.body.get("error", body)}')


class RulesMismatch(Exception):
    """The site and this clone disagree on the rules: playing would only produce rejected moves."""


def _json(response):
    try:
        return response.json()
    except ValueError:
        return {'error': response.text[:200]}


class BotApiClient:
    def __init__(self, base_url, token, session=None, retries=3):
        self.base = base_url.rstrip('/')
        self.retries = retries
        self.http = session or requests.Session()
        self.http.headers['Authorization'] = f'Bearer {token.strip().strip(chr(34) + chr(39))}'
        self.http.headers['User-Agent'] = 'pai-sho-lab-connector'

    def _request(self, method, path, **kw):
        """JSON in, JSON out. A 429 that carries Retry-After (the rate limiter) is waited out and retried;
        a 429 without it (a free-tier limit) is raised, never retried in a loop."""
        for attempt in range(self.retries + 1):
            r = self.http.request(method, self.base + path, timeout=TIMEOUT, **kw)
            if r.status_code == 429 and 'Retry-After' in r.headers and attempt < self.retries:
                time.sleep(min(float(r.headers['Retry-After'] or 1), 10))
                continue
            break
        if r.status_code >= 400:
            raise ApiError(r.status_code, _json(r))
        return _json(r)

    def _stream(self, path):
        with self.http.get(self.base + path, stream=True, timeout=TIMEOUT) as r:
            if r.status_code >= 400:
                raise ApiError(r.status_code, _json(r))
            for line in r.iter_lines(decode_unicode=True):
                if line:                                   # blank lines are keepalives
                    yield json.loads(line)

    # ---- reads
    def meta(self):
        return self._request('GET', '/api/bot/meta')

    def events(self):
        return self._stream('/api/stream/event')

    def game_stream(self, gid):
        return self._stream(f'/api/bot/game/stream/{gid}')

    # ---- actions
    def move(self, gid, action, ply):
        return self._request('POST', f'/api/bot/game/{gid}/move', json={'action': list(action), 'ply': ply})

    def resign(self, gid):
        return self._request('POST', f'/api/bot/game/{gid}/resign')

    def challenge(self, target, time_control, rated=False):
        return self._request('POST', f'/api/challenge/{target}', json={'time_control': time_control, 'rated': rated})

    def accept(self, cid):
        return self._request('POST', f'/api/challenge/{cid}/accept')

    def decline(self, cid, reason=None):
        return self._request('POST', f'/api/challenge/{cid}/decline', json={'reason': reason or 'generic'})

    def seek(self, time_control, rated=False):
        return self._request('POST', '/api/bot/seek', json={'time_control': time_control, 'rated': rated})

    def cancel_seek(self):
        return self._request('DELETE', '/api/bot/seek')

    def vs_house(self, level, time_control, color='random'):
        return self._request('POST', '/api/bot/vs-house',
                             json={'level': level, 'time_control': time_control, 'color': color})
