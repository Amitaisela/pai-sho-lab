"""Pai Sho Notation (PSN) v2 - a PGN-like text format for Skud Pai Sho games.

    [Event "..."]
    [Date "YYYY-MM-DD HH:MM:SS"]
    [Player1 "..."]            # the Guest, who moves first
    [Player2 "..."]            # the Host
    [GuestAccents "Rock,Wheel,Knotweed,Boat"]
    [HostAccents "Rock,Wheel,Knotweed,Boat"]
    [Opening "Rose"]           # or "Informal"
    [Result "1-0" | "0-1" | "1/2-1/2" | "*"]
    [Termination "resign"]     # only present when game.end_reason is not None
    [Turns "N"]

    1. Jasmine@9,1  17,9-16,9
    2. 16,9-12,9+Rock@5,5  1,9-2,9+

One token per turn, numbered in Guest/Host pairs. A turn token is the main action,
plus "+<bonus action>" if a Harmony Bonus was used. A bare main action always means
any bonus earned was skipped, EXCEPT when the token ends in a lone trailing "+" with
no bonus action after it - that marks a bonus that was still pending (neither played
nor skipped) when the record ends, which can only happen on the game's last turn.

    Plant:              <Tile>@<row>,<col>
    Boat on a flower:   Boat@<row>,<col>><to_row>,<to_col>
    Arrange:            <fr>,<fc>-<tr>,<tc>
"""

import re
import time

_PLANT_RE = re.compile(r'^([A-Za-z]+)@(\d+),(\d+)(?:>(\d+),(\d+))?$')
_ARRANGE_RE = re.compile(r'^(\d+),(\d+)-(\d+),(\d+)$')
_TAG_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')
_RESULTS = ('1-0', '0-1', '1/2-1/2', '*')


def action_to_psn(action):
    kind = action[0]
    if kind == 'plant':
        token = f"{action[1]}@{action[2]},{action[3]}"
        return token + (f">{action[4]},{action[5]}" if len(action) == 6 else '')
    if kind == 'arrange':
        _, fr, fc, tr, tc = action
        return f"{fr},{fc}-{tr},{tc}"
    raise ValueError(f"No PSN token for action {action!r}")


def psn_to_action(token):
    m = _PLANT_RE.match(token)
    if m:
        action = ['plant', m.group(1), int(m.group(2)), int(m.group(3))]
        if m.group(4) is not None:
            action += [int(m.group(4)), int(m.group(5))]
        return action
    m = _ARRANGE_RE.match(token)
    if m:
        return ['arrange'] + [int(m.group(i)) for i in range(1, 5)]
    raise ValueError(f"Invalid PSN move token: {token!r}")


def _result_code(game):
    return {1: "1-0", 2: "0-1", 0: "1/2-1/2"}.get(game.winner, "*")


def _turn_tokens(game):
    """Replay `game`'s history from its setup and group it into one token per turn.

    If a bonus is still pending (earned but neither played nor skipped) once history
    is exhausted, that can only be the final turn's bonus; mark it with a trailing
    lone '+' so psn_to_game can tell it apart from a skipped bonus.

    Raises ValueError if the history doesn't replay from the setup, or doesn't
    arrive at the game's current board.
    """
    setup = game.setup
    replay = type(game)(accents=setup['accents'], opening=setup['opening'])
    tokens = []
    for i, action in enumerate(game.history):
        try:
            action = tuple(action)
            if action[0] != 'skip_bonus':
                if replay.bonus_turn:
                    tokens[-1] += '+' + action_to_psn(action)
                else:
                    tokens.append(action_to_psn(action))
            replay.step(action)
        except Exception as e:  # noqa: BLE001 - any replay failure means the history can't be exported
            raise ValueError(f"cannot export PSN: history action #{i} {action!r} does not replay from the "
                             f"game's setup ({e}); game_to_psn needs the full history since setup") from e
    if replay.board != game.board:
        raise ValueError("cannot export PSN: the history does not reproduce the game's board; "
                         "game_to_psn needs the full history since setup")
    if replay.bonus_turn:
        tokens[-1] += '+'
    return tokens


def game_to_psn(game, p1_name='Player 1', p2_name='Player 2', event='Pai Sho Lab Game'):
    """Export `game` as PSN v2.

    PSN records moves, not positions, so this replays `game.history` from
    `game.setup` and needs that history to be complete since the setup. A game
    built with from_dict() from a mid-game position (without its full history)
    can't be exported faithfully: that raises ValueError instead of writing a
    record that would replay to a different position.
    """
    setup = game.setup
    tokens = _turn_tokens(game)
    lines = [
        f'[Event "{event}"]',
        f'[Date "{time.strftime("%Y-%m-%d %H:%M:%S")}"]',
        f'[Player1 "{p1_name}"]',
        f'[Player2 "{p2_name}"]',
        f'[GuestAccents "{",".join(setup["accents"][1])}"]',
        f'[HostAccents "{",".join(setup["accents"][2])}"]',
        f'[Opening "{setup["opening"] or "Informal"}"]',
        f'[Result "{_result_code(game)}"]',
    ]
    if game.end_reason is not None:
        lines.append(f'[Termination "{game.end_reason}"]')
    lines += [
        f'[Turns "{len(tokens)}"]',
        '',
    ]
    for i in range(0, len(tokens), 2):
        lines.append(f"{i // 2 + 1}. {'  '.join(tokens[i:i + 2])}")
    lines.append('')
    return '\n'.join(lines)


def parse_psn_turns(text):
    """Parse PSN into (tags, turns); each turn is [main_action], [main_action, bonus_action],
    or [main_action, None] for a bonus that was still pending when the record ends
    (a bare trailing '+' with no bonus token after it)."""
    tags, turns = {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tm = _TAG_RE.match(line)
        if tm:
            tags[tm.group(1)] = tm.group(2)
            continue
        for tok in re.sub(r'^\d+\.\s*', '', line).split():
            if tok in _RESULTS:
                continue
            if '+' in tok:
                main, _, bonus = tok.partition('+')
                turns.append([psn_to_action(main), psn_to_action(bonus) if bonus else None])
            else:
                turns.append([psn_to_action(tok)])
    return tags, turns


def parse_psn(text):
    """Parse PSN into (tags, actions), flattened; skipped bonuses and a pending
    trailing bonus are not listed."""
    tags, turns = parse_psn_turns(text)
    return tags, [a for turn in turns for a in turn if a is not None]


def _setup_from_tags(tags):
    accents = None
    if 'GuestAccents' in tags and 'HostAccents' in tags:
        accents = {1: tags['GuestAccents'].split(','), 2: tags['HostAccents'].split(',')}
    opening = tags.get('Opening')
    return accents, (None if opening in (None, 'Informal') else opening)


def psn_to_game(text, game_cls):
    """Replay PSN into a fresh game and return (game, tags). A bare turn token
    always means any bonus earned was skipped, so skip_bonus is stepped whenever
    a bonus is pending after one - including on the last turn. A turn recorded as
    [main_action, None] (a trailing '+' with no bonus token) leaves that bonus
    pending, unplayed, exactly as the source game left it. Finally, if the game is
    still unfinished after replaying every turn and [Termination] says "resign",
    resign() is called for the loser implied by [Result] - resign() itself never
    appears in history, so it can't be replayed as an ordinary turn."""
    tags, turns = parse_psn_turns(text)
    accents, opening = _setup_from_tags(tags)
    game = game_cls(accents=accents, opening=opening)
    for turn in turns:
        game.step(tuple(turn[0]))
        if len(turn) > 1:
            bonus = turn[1]
            if bonus is not None:
                game.step(tuple(bonus))
            # else: bonus explicitly left pending (trailing '+') - do nothing more.
        elif game.bonus_turn:
            game.step(('skip_bonus',))
    if game.winner is None and tags.get('Termination') == 'resign':
        loser = 2 if tags.get('Result') == '1-0' else 1
        game.resign(loser)
    return game, tags
