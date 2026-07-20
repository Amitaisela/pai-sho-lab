"""Pai Sho Notation (PSN) — a PGN-like text format for Skud Pai Sho games.

Format:
    [Event "..."]
    [Date "YYYY-MM-DD HH:MM:SS"]
    [Player1 "..."]
    [Player2 "..."]
    [Result "1-0" | "0-1" | "1/2-1/2" | "*"]
    [Turns "N"]

    1. Rose@9,1  Jade@9,17
    2. 9,1-10,2  9,17-8,16
    ...

Move syntax:
    Plant:   <Flower>@<row>,<col>
    Arrange: <fr>,<fc>-<tr>,<tc>
"""
import re
import time


_PLANT_RE = re.compile(r'^([A-Za-z]+)@(\d+),(\d+)$')
_ARRANGE_RE = re.compile(r'^(\d+),(\d+)-(\d+),(\d+)$')
_TAG_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')


def action_to_psn(action):
    kind = action[0]
    if kind == 'plant':
        _, flower, r, c = action
        return f"{flower}@{r},{c}"
    if kind == 'arrange':
        _, fr, fc, tr, tc = action
        return f"{fr},{fc}-{tr},{tc}"
    raise ValueError(f"Unknown action kind: {kind!r}")


def psn_to_action(token):
    m = _PLANT_RE.match(token)
    if m:
        flower, r, c = m.group(1), int(m.group(2)), int(m.group(3))
        return ['plant', flower, r, c]
    m = _ARRANGE_RE.match(token)
    if m:
        fr, fc, tr, tc = (int(m.group(i)) for i in range(1, 5))
        return ['arrange', fr, fc, tr, tc]
    raise ValueError(f"Invalid PSN move token: {token!r}")


def _result_code(game):
    if game.winner == 1:
        return "1-0"
    if game.winner == 2:
        return "0-1"
    if game.winner == 0:
        return "1/2-1/2"
    return "*"


def game_to_psn(game, p1_name='Player 1', p2_name='Player 2', event='Pai-Sho-Lab Game'):
    lines = [
        f'[Event "{event}"]',
        f'[Date "{time.strftime("%Y-%m-%d %H:%M:%S")}"]',
        f'[Player1 "{p1_name}"]',
        f'[Player2 "{p2_name}"]',
        f'[Result "{_result_code(game)}"]',
        f'[Turns "{len(game.history)}"]',
        '',
    ]
    tokens = [action_to_psn(a) for a in game.history]
    for i in range(0, len(tokens), 2):
        turn = i // 2 + 1
        p1 = tokens[i]
        p2 = tokens[i + 1] if i + 1 < len(tokens) else ''
        lines.append(f"{turn}. {p1}  {p2}".rstrip())
    lines.append('')
    return '\n'.join(lines)


def parse_psn(text):
    """Parse PSN text into (tags, actions). Does not replay the game."""
    tags = {}
    actions = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tm = _TAG_RE.match(line)
        if tm:
            tags[tm.group(1)] = tm.group(2)
            continue
        body = re.sub(r'^\d+\.\s*', '', line)
        for tok in body.split():
            if tok in ('1-0', '0-1', '1/2-1/2', '*'):
                continue
            actions.append(psn_to_action(tok))
    return tags, actions


def psn_to_game(text, game_cls):
    """Replay a PSN text into a fresh game and return (game, tags)."""
    tags, actions = parse_psn(text)
    game = game_cls()
    for a in actions:
        game.step(tuple(a))
    return game, tags
