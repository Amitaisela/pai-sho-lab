"""Play a registered agent on the Pai Sho Lab site through its bot API.

  python scripts/play_online.py basic_minimax --url https://SITE --token $PAISHO_TOKEN --seek 3+2
  python scripts/play_online.py random --url https://SITE --accept             (answer challenges)
  python scripts/play_online.py cnn_basic --url https://SITE --vs-house 2 3+2  (practice against a house bot)
  python scripts/play_online.py --check --url https://SITE                      (compatibility check)

The token comes from --token or the PAISHO_TOKEN environment variable. Create a bot and a token on
the site's "My bots" page first. Time controls are given by name, as the site lists them (for example 3+2).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'engine'))

import requests  # noqa: E402

from Agents.elo import RULES_VERSION  # noqa: E402
from paisho_online.client import ApiError, BotApiClient, RulesMismatch  # noqa: E402
from paisho_online.player import OnlineBot  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description='Play an agent on the Pai Sho Lab site.')
    p.add_argument('agent', nargs='?', help='registered agent, optionally name:key=value,key=value')
    p.add_argument('--url', default=os.environ.get('PAISHO_URL', ''), help='the site, e.g. https://example.org (or PAISHO_URL)')
    p.add_argument('--token', default=os.environ.get('PAISHO_TOKEN', ''), help='bot API token (or PAISHO_TOKEN)')
    p.add_argument('--check', action='store_true', help='only check that this clone and the site agree on the rules')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--accept', action='store_true', help='accept incoming challenges (default when nothing else is chosen)')
    mode.add_argument('--seek', metavar='TC', help='join the bot pool, e.g. 3+2')
    mode.add_argument('--challenge', nargs=2, metavar=('BOT', 'TC'), help='challenge another bot')
    mode.add_argument('--vs-house', nargs=2, metavar=('LEVEL', 'TC'), help='play house bot LEVEL (1-6), unrated')
    p.add_argument('--rated', action='store_true', help='rated games (bot vs bot of different owners only)')
    p.add_argument('--games', type=int, default=None, help='stop after this many finished games')
    return p


def _tc_names(meta):
    names = []
    for t in meta.get('time_controls') or []:
        names.append(t.get('name') if isinstance(t, dict) else t)
    return [n for n in names if n]


def _print(*args):
    """print, flushed at once: with stdout piped (a log file, tee, a supervisor) Python would
    otherwise hold the lines back until exit, and a bot in a long game would look silent."""
    print(*args, flush=True)


def main(argv=None, log=_print):
    args = build_parser().parse_args(argv)
    if not args.url:
        log('give the site address with --url (or PAISHO_URL)')
        return 2
    token = args.token.strip().strip('"\'').strip()
    if not token:
        log('no token: pass --token or set PAISHO_TOKEN (create one under "My bots" on the site)')
        return 2
    client = BotApiClient(args.url, token)
    try:
        if args.check:
            meta = client.meta()
            log(f"site api_version {meta['api_version']}, rules_version {meta['rules_version']}; "
                f"this clone's rules_version is {RULES_VERSION}")
            if meta['rules_version'] != RULES_VERSION:
                log('MISMATCH: update your clone (git pull) before playing online')
                return 3
            return 0
        if not args.agent:
            log('name an agent to play, for example: basic_minimax')
            return 2
        tc = args.seek or (args.challenge[1] if args.challenge else None) or (args.vs_house[1] if args.vs_house else None)
        vs_house = None
        if args.vs_house:
            try:
                vs_house = (int(args.vs_house[0]), args.vs_house[1])
            except ValueError:
                log(f'--vs-house level must be a number (1-6), not {args.vs_house[0]!r}')
                return 2
        if tc is not None:
            names = _tc_names(client.meta())
            if names and tc not in names:
                log(f'unknown time control {tc!r}; this site offers: {", ".join(names)}')
                return 2
        try:
            bot = OnlineBot(
                client, args.agent, accept=not (args.seek or args.challenge or args.vs_house) or args.accept,
                seek=args.seek, rated=args.rated, challenge=tuple(args.challenge) if args.challenge else None,
                vs_house=vs_house, max_games=args.games, log=log)
        except ValueError as e:
            log(str(e))
            return 2
        bot.run()
        return 0
    except RulesMismatch as e:
        log(str(e))
        return 3
    except ApiError as e:
        if e.status in (401, 403):
            log(f'token rejected ({e.status}); check it, or create a new one on the "My bots" page')
        else:
            log(f'the site refused: {e}')
        return 2
    except requests.RequestException as e:
        log(f'could not reach {args.url}: {type(e).__name__}')
        return 2
    except KeyboardInterrupt:
        log('stopped')
        return 0


if __name__ == '__main__':
    sys.exit(main())
