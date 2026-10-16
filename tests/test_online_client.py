"""Connector tests against an in-process fake of the site's bot API (tests/fake_site.py)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'engine'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from Agents.elo import RULES_VERSION  # noqa: E402
from fake_site import FakeSite  # noqa: E402
from paisho_online.client import ApiError, BotApiClient, RulesMismatch  # noqa: E402
from paisho_online.player import OnlineBot, rebuild  # noqa: E402


def _quiet(*a, **k):
    pass


def _bot(site, **kw):
    kw.setdefault('max_games', 1)
    kw.setdefault('reconnect_delay', 0.05)
    return OnlineBot(BotApiClient(site.url, 'mbt_test'), 'random', log=_quiet, **kw)


def test_meta_and_the_event_stream_skip_keepalives():
    with FakeSite() as site:
        c = BotApiClient(site.url, 'mbt_test')
        assert c.meta()['api_version'] == 1
        events = c.events()
        assert next(events)['type'] == 'hello'
        assert next(events)['type'] == 'gameStart'


def test_a_429_with_retry_after_is_retried_and_other_errors_raise_api_error():
    with FakeSite(rate_limit_move_once=True) as site:
        c = BotApiClient(site.url, 'mbt_test')
        assert c.move('g1', site.legal[0], 0)['ok'] is True            # first answer 429, second 200
        assert site.rate_limited == 1 and len(site.moves) == 1
        try:
            c._request('GET', '/nope')
        except ApiError as e:
            assert e.status == 404 and e.code == 'not_found'
        else:
            raise AssertionError('404 should raise ApiError')


def test_rebuild_replays_the_actions_onto_the_initial_setup():
    with FakeSite() as site:
        game = rebuild(site.initial, [site.legal[0]])
        assert len(game.history) == 1 and tuple(game.history[0]) == tuple(site.legal[0])


def test_the_bot_plays_one_legal_move_at_ply_zero_and_stops_after_one_game():
    with FakeSite() as site:
        _bot(site).run()
        assert len(site.moves) == 1
        assert site.moves[0]['action'] in site.legal and site.moves[0]['ply'] == 0


def test_the_bot_refuses_to_play_on_a_rules_version_mismatch():
    """Review Focus 1: no move is ever sent."""
    with FakeSite(rules_version=999) as site:
        try:
            _bot(site).run()
        except RulesMismatch as e:
            assert '999' in str(e) and str(RULES_VERSION) in str(e)
        else:
            raise AssertionError('expected RulesMismatch')
        assert site.moves == [] and ('GET', '/api/bot/game/stream/g1') not in site.requests


def test_an_illegal_rejection_is_followed_by_a_random_legal_move():
    """Review Focus 3."""
    with FakeSite(illegal_first=True) as site:
        _bot(site).run()
        assert len(site.rejected) == 1 and len(site.moves) == 1
        assert site.moves[0]['action'] in site.legal


def test_the_bot_reconnects_after_a_dropped_event_stream():
    """Review Focus 2."""
    with FakeSite(close_first=True) as site:
        _bot(site).run()
        assert site.event_connections >= 2 and len(site.moves) == 1


def test_an_agent_that_throws_still_gets_a_legal_move_sent():
    import paisho_online.player as player
    with FakeSite() as site:
        real = player.act
        player.act = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom'))
        try:
            _bot(site).run()
        finally:
            player.act = real
        assert len(site.moves) == 1 and site.moves[0]['action'] in site.legal


def test_a_rules_mismatch_in_the_hello_event_also_stops_the_bot():
    with FakeSite() as site:
        bot = _bot(site)
        try:
            bot._on_event({'type': 'hello', 'rules_version': 999, 'username': 'fakebot'})
        except RulesMismatch as e:
            assert '999' in str(e) and str(RULES_VERSION) in str(e)
        else:
            raise AssertionError('expected RulesMismatch')
        assert site.moves == []


def _with_act(fn, site, **kw):
    import paisho_online.player as player
    real = player.act
    player.act = fn
    try:
        _bot(site, **kw).run()
    finally:
        player.act = real


def test_an_agent_that_returns_none_or_an_illegal_move_gets_a_random_legal_move_sent():
    for answer in (None, ['plant', 'Rose', 99, 99]):
        with FakeSite() as site:
            _with_act(lambda *a, **k: answer, site)
            assert len(site.moves) == 1 and site.moves[0]['action'] in site.legal


def test_a_repeated_game_state_never_produces_a_second_move_for_the_same_ply():
    with FakeSite(duplicate_state=True) as site:
        _bot(site).run()
        assert len(site.moves) == 1


def test_a_429_without_retry_after_is_reported_and_not_retried():
    with FakeSite(always_429_no_retry_after=True) as site:
        c = BotApiClient(site.url, 'mbt_test')
        try:
            c.move('g1', site.legal[0], 0)
        except ApiError as e:
            assert e.status == 429 and e.code == 'too_many_games'
        else:
            raise AssertionError('expected ApiError')
        assert site.requests.count(('POST', '/api/bot/game/g1/move')) == 1


def test_the_bot_reopens_a_dropped_game_stream_and_still_moves():
    with FakeSite(drop_game_stream_first=True) as site:
        _bot(site).run()
        assert site.game_connections >= 2 and len(site.moves) == 1


def test_the_token_is_never_logged():
    lines = []
    with FakeSite(illegal_first=True, close_first=True) as site:
        bot = OnlineBot(BotApiClient(site.url, 'mbt_secret_token'), 'random', max_games=1,
                        reconnect_delay=0.05, log=lambda *a, **k: lines.append(' '.join(map(str, a))))
        bot.run()
    assert lines and not any('mbt_secret_token' in x for x in lines)


TESTS = [
    test_meta_and_the_event_stream_skip_keepalives,
    test_a_429_with_retry_after_is_retried_and_other_errors_raise_api_error,
    test_rebuild_replays_the_actions_onto_the_initial_setup,
    test_the_bot_plays_one_legal_move_at_ply_zero_and_stops_after_one_game,
    test_the_bot_refuses_to_play_on_a_rules_version_mismatch,
    test_an_illegal_rejection_is_followed_by_a_random_legal_move,
    test_the_bot_reconnects_after_a_dropped_event_stream,
    test_an_agent_that_throws_still_gets_a_legal_move_sent,
    test_a_rules_mismatch_in_the_hello_event_also_stops_the_bot,
    test_an_agent_that_returns_none_or_an_illegal_move_gets_a_random_legal_move_sent,
    test_a_repeated_game_state_never_produces_a_second_move_for_the_same_ply,
    test_a_429_without_retry_after_is_reported_and_not_retried,
    test_the_bot_reopens_a_dropped_game_stream_and_still_moves,
    test_the_token_is_never_logged,
]

if __name__ == '__main__':
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS {t.__name__}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'FAIL {t.__name__}: {e!r}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    sys.exit(1 if failed else 0)
