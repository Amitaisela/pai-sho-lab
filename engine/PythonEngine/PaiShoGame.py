import random
from collections import deque

BOARD_SIZE = 19
RADIUS = 9
CENTER = (RADIUS, RADIUS)
GATES = [(1, RADIUS), (BOARD_SIZE - 2, RADIUS), (RADIUS, 1), (RADIUS, BOARD_SIZE - 2)]
BEHIND_GATES = [(0, RADIUS), (BOARD_SIZE - 1, RADIUS), (RADIUS, 0), (RADIUS, BOARD_SIZE - 1)]
CIRCLE = ['Rose', 'Chrysanthemum', 'Rhododendron', 'Jasmine', 'Lily', 'Jade']
_CIRCLE_IDX = {f: i for i, f in enumerate(CIRCLE)}


def _circle_distance(i, j):
    d = abs(i - j)
    return min(d, 6 - d)


_HARMONY_PAIRS = frozenset(
    (a, b)
    for i, a in enumerate(CIRCLE)
    for j, b in enumerate(CIRCLE)
    if _circle_distance(i, j) == 1
)
_CLASH_PAIRS = frozenset(
    (a, b)
    for i, a in enumerate(CIRCLE)
    for j, b in enumerate(CIRCLE)
    if _circle_distance(i, j) == 3
)

FLOWER = {
    'Rose': {'color': 'red', 'move': 3},
    'Chrysanthemum': {'color': 'red', 'move': 4},
    'Rhododendron': {'color': 'red', 'move': 5},
    'Jasmine': {'color': 'white', 'move': 3},
    'Lily': {'color': 'white', 'move': 4},
    'Jade': {'color': 'white', 'move': 5},
}

ACCENT_TILES = ['Rock', 'Wheel', 'Knotweed', 'Boat']
SPECIAL_TILES = ['Orchid', 'WhiteLotus']
SPECIAL_MOVEMENT = {'Orchid': 6, 'WhiteLotus': 2}

BASIC_PER_KIND = 3          # rulebook p.6: 3 of each Basic Flower Tile
ACCENTS_PER_PLAYER = 4      # rulebook p.8: each player uses 4 of their 8 accent tiles
MAX_PER_ACCENT = 2          # rulebook p.6: 2 of each accent tile in a set
GUEST_GATE = (BOARD_SIZE - 2, RADIUS)   # (17, 9) - Player 1, the Guest, moves first
HOST_GATE = (1, RADIUS)                 # (1, 9)  - Player 2, the Host
SAVE_VERSION = 2
_NEIGHBOURS_8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # N, then clockwise


def normalize_accents(accents):
    """Validate a {player: [4 accent names]} choice (int or str keys) and return it
    with int keys, each list in ACCENT_TILES order. None is the rulebook's beginner
    default: one of each accent."""
    if accents is None:
        return {1: list(ACCENT_TILES), 2: list(ACCENT_TILES)}
    out = {}
    for p in (1, 2):
        chosen = accents.get(p, accents.get(str(p)))
        if chosen is None or len(chosen) != ACCENTS_PER_PLAYER:
            raise ValueError(f"accents for player {p} must list exactly {ACCENTS_PER_PLAYER} tiles")
        for name in chosen:
            if name not in ACCENT_TILES:
                raise ValueError(f"unknown accent tile: {name!r}")
        for name in ACCENT_TILES:
            if chosen.count(name) > MAX_PER_ACCENT:
                raise ValueError(f"at most {MAX_PER_ACCENT} of each accent tile per player (player {p}: {name})")
        out[p] = sorted(chosen, key=ACCENT_TILES.index)
    return out


def resolve_opening(opening):
    """None (informal start), a basic flower name, or 'random' (drawn with the
    global `random` module, so seeding it makes the choice reproducible)."""
    if opening is None:
        return None
    if opening == 'random':
        return random.choice(CIRCLE)
    if opening in CIRCLE:
        return opening
    raise ValueError("opening must be a basic flower name, 'random', or None")


def is_valid(r, c):
    return (r - RADIUS) ** 2 + (c - RADIUS) ** 2 <= RADIUS * RADIUS


VALID_SPACES = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if
                is_valid(r, c) and (r, c) not in BEHIND_GATES]
_VALID_SPACES_SET = set(VALID_SPACES)
_GATES_SET = set(GATES)


def garden_of(r, c):
    dr, dc = r - RADIUS, c - RADIUS
    if dr == 0 or dc == 0: return 'neutral'
    if abs(dr) + abs(dc) < 7:
        is_red = (dr < 0) != (dc < 0)
        return 'red' if is_red else 'white'
    if dr < 0 and abs(dc) <= -dr - 7: return 'neutral'
    if dr > 0 and abs(dc) <= dr - 7:  return 'neutral'
    if dc < 0 and abs(dr) <= -dc - 7: return 'neutral'
    if dc > 0 and abs(dr) <= dc - 7:  return 'neutral'
    return 'neutral'


_GARDEN_OF = {(r, c): garden_of(r, c) for r, c in VALID_SPACES}


def _garden_blocks(flower, pos):
    """True if `flower` may not end a move on `pos`: basic flowers can't stop in the
    opposite-coloured garden (rulebook p.9). Special and accent tiles are never blocked."""
    info = FLOWER.get(flower)
    if info is None:
        return False
    garden = _GARDEN_OF.get(pos, 'neutral')
    return (info['color'] == 'red' and garden == 'white') or (info['color'] == 'white' and garden == 'red')


def _segment_touches_center(a, b):
    """True if the harmony line a-b passes through or ends on the center point."""
    (r1, c1), (r2, c2) = a, b
    if r1 == r2 == RADIUS:
        return min(c1, c2) <= RADIUS <= max(c1, c2)
    if c1 == c2 == RADIUS:
        return min(r1, r2) <= RADIUS <= max(r1, r2)
    return False


def _ray_parity(a, b):
    """1 if the harmony line a-b crosses the ray running right from the center just
    below row 9 (y = 9 + epsilon, x > 9), else 0."""
    (r1, c1), (r2, c2) = a, b
    if c1 != c2 or c1 <= RADIUS:
        return 0
    return 1 if min(r1, r2) <= RADIUS < max(r1, r2) else 0


def _has_odd_cycle(edges):
    """True if some cycle of `edges` crosses the center ray an odd number of times,
    i.e. surrounds the center. Crossing parity is linear over GF(2), so an odd simple
    cycle exists iff an odd fundamental cycle does: a weighted union-find finds one
    in O(E), with no cap on ring size (bug B1)."""
    parent, parity = {}, {}

    def find(x):
        p = 0
        while parent[x] != x:
            p ^= parity[x]
            x = parent[x]
        return x, p

    for a, b in edges:
        for v in (a, b):
            if v not in parent:
                parent[v], parity[v] = v, 0
        (ra, pa), (rb, pb) = find(a), find(b)
        w = _ray_parity(a, b)
        if ra == rb:
            if pa ^ pb ^ w:
                return True
        else:
            parent[ra], parity[ra] = rb, pa ^ pb ^ w
    return False


# Fixed seed so the hash is reproducible across sessions and Q-tables stay loadable.
# A private generator: importing the engine must never reseed the caller's global random.
_ZRNG = random.Random(42)
_ALL_FLOWERS = CIRCLE + ACCENT_TILES + SPECIAL_TILES
_ZOBRIST = {
    (pos, flower, player, growing): _ZRNG.getrandbits(64)
    for pos in VALID_SPACES
    for flower in _ALL_FLOWERS
    for player in (1, 2)
    for growing in (True, False)
}


def _validate_history_action(action):
    """Same shape/tile-name checks as the Rust bridge's parse_action, so a
    malformed history entry raises ValueError in both engines (used by from_dict)."""
    kind, n = action[0], len(action)
    if kind == 'plant' and n in (4, 6):
        if action[1] not in _ALL_FLOWERS:
            raise ValueError(f"unknown tile name: {action[1]!r}")
        return
    if kind == 'arrange' and n == 5:
        return
    if kind == 'skip_bonus' and n == 1:
        return
    if kind in ('plant', 'arrange'):
        raise ValueError(f"Malformed {kind} action with {n} elements")
    raise ValueError(f"Unknown action type: {kind}")


class PaiShoGame:
    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        # An externally replaced history has unknown movers - matches Rust's
        # set_history (crates/pybind/src/lib.rs). Internal code that means to
        # keep history_players valid (clone, from_dict, from_save_dict, the
        # no-stranding sim) always re-assigns history_players right after
        # assigning history, so this clear never sticks for them.
        self._history = list(value)
        self.history_players = []

    def __init__(self, accents=None, opening='random'):
        self.reset(accents=accents, opening=opening)

    def reset(self, accents=None, opening='random'):
        self._setup = {'accents': normalize_accents(accents), 'opening': resolve_opening(opening)}
        self.board = {}
        self._zhash = 0
        self.hands = {}
        for p in (1, 2):
            hand = {f: BASIC_PER_KIND for f in CIRCLE}
            for a in ACCENT_TILES:
                hand[a] = self._setup['accents'][p].count(a)
            for s in SPECIAL_TILES:
                hand[s] = 1
            self.hands[p] = hand
        self.current_player = 1
        self.winner = None
        self.end_reason = None
        self.bonus_turn = False
        self._harmony_cache = {}
        self._legal_actions_cache = None
        self.history = []
        self.history_players = []
        flower = self._setup['opening']
        if flower is not None:
            for player, gate in ((1, GUEST_GATE), (2, HOST_GATE)):
                self._put(gate, {'flower': flower, 'player': player, 'growing': True})
                self.hands[player][flower] -= 1
        self.message = 'Player 1: Plant in a Gate or Arrange a tile'

    @property
    def setup(self):
        """The pre-game options in effect: {'accents': {1: [...], 2: [...]}, 'opening': name | None}."""
        return {'accents': {p: list(v) for p, v in self._setup['accents'].items()},
                'opening': self._setup['opening']}

    def _put(self, pos, tile):
        self.board[pos] = tile
        self._z_toggle(pos, tile)

    def _remove(self, pos):
        tile = self.board.pop(pos)
        self._z_toggle(pos, tile)
        return tile

    def clone(self):
        # Built without __init__ so cloning never draws a random opening (that would
        # shift the global random stream agents and the parity fuzzer rely on).
        new_game = PaiShoGame.__new__(PaiShoGame)
        new_game.board = {pos: dict(tile) for pos, tile in self.board.items()}
        new_game.hands = {pid: dict(hand) for pid, hand in self.hands.items()}
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.end_reason = self.end_reason
        new_game.message = self.message
        new_game.bonus_turn = self.bonus_turn
        new_game._setup = {'accents': {p: list(v) for p, v in self._setup['accents'].items()},
                           'opening': self._setup['opening']}
        new_game._zhash = self._zhash
        new_game._harmony_cache = dict(self._harmony_cache)
        new_game._legal_actions_cache = self._legal_actions_cache
        new_game.history = [list(a) for a in self.history]
        new_game.history_players = list(self.history_players)
        return new_game

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise TypeError("from_dict expects a dict")
        if not isinstance(d.get('board'), dict):
            raise TypeError("'board' must be an object mapping 'r,c' -> tile")
        if not isinstance(d.get('hands'), dict):
            raise TypeError("'hands' must be an object")
        if d.get('current_player') not in (1, 2):
            raise ValueError("'current_player' must be 1 or 2")
        game = cls(opening=None)
        try:
            game.board = {
                tuple(map(int, pos_str.split(','))): tile
                for pos_str, tile in d['board'].items()
            }
        except (AttributeError, ValueError) as e:
            raise ValueError(f"malformed board key: {e}")
        hands = d['hands']
        game.hands = {
            1: hands.get('1', hands.get(1, {})),
            2: hands.get('2', hands.get(2, {})),
        }
        game.current_player = d['current_player']
        game.winner = d.get('winner')
        game.bonus_turn = d.get('bonus_turn', False)
        setup = d.get('setup') or {}
        opening = setup.get('opening')
        if opening is not None and opening not in CIRCLE:
            raise ValueError("opening must be a basic flower name or None")
        game._setup = {'accents': normalize_accents(setup.get('accents')), 'opening': opening}
        end_reason = d.get('end_reason')
        if end_reason is not None and end_reason not in ('ring', 'last_basic_flower', 'no_moves', 'resign'):
            raise ValueError(f"unknown end_reason: {end_reason!r}")
        game.end_reason = end_reason
        game._zhash = 0
        for pos, t in game.board.items():
            game._zhash ^= _ZOBRIST.get((pos, t['flower'], t['player'], t['growing']), 0)
        history = d.get('history', [])
        for a in history:
            _validate_history_action(tuple(a))
        game.history = [list(a) for a in history]
        players = d.get('history_players')
        game.history_players = list(players) if players is not None and len(players) == len(game.history) else []
        message = d.get('message')
        game.message = message if isinstance(message, str) else game._state_message()
        return game

    @staticmethod
    def _circle_dist(f1, f2):
        i = _CIRCLE_IDX.get(f1)
        j = _CIRCLE_IDX.get(f2)
        if i is None or j is None:
            return -1
        d = abs(i - j)
        return min(d, 6 - d)

    def is_harmonious(self, f1, f2):
        return (f1, f2) in _HARMONY_PAIRS

    def is_clash(self, f1, f2):
        return (f1, f2) in _CLASH_PAIRS

    def _clear_line_between(self, r1, c1, r2, c2, board=None):
        """No tile and no gate strictly between two same-row/column points (rulebook p.7)."""
        board = self.board if board is None else board
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if (r1, c) in board or (r1, c) in _GATES_SET:
                    return False
        elif c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if (r, c1) in board or (r, c1) in _GATES_SET:
                    return False
        return True

    def _z_toggle(self, pos, tile):
        self._zhash ^= _ZOBRIST.get((pos, tile['flower'], tile['player'], tile['growing']), 0)

    def _board_sig(self):
        return self._zhash

    def _state_sig(self):
        h = self._board_sig()
        h ^= hash(('turn', self.current_player, bool(self.bonus_turn)))
        for pid in (1, 2):
            for flower, count in self.hands[pid].items():
                h ^= hash((pid, flower, count))
        return h

    def board_sig(self):
        """Hash of the tiles on the board (position, tile, owner, growing). Equal positions
        reached by different move orders share it. Only comparable within one engine/process."""
        return self._board_sig()

    def state_sig(self):
        """board_sig plus side to move, bonus flag and both hands: the transposition key."""
        return self._state_sig()

    def find_harmonies(self, player, custom_board=None):
        if custom_board is None:
            sig = self._board_sig()
            entry = self._harmony_cache.get(player)
            if entry is not None and entry[0] == sig:
                return entry[1]
        board = custom_board if custom_board is not None else self.board

        rock_rows, rock_cols, drained = set(), set(), set()
        for pos, t in board.items():
            if t['flower'] == 'Rock':
                rock_rows.add(pos[0])
                rock_cols.add(pos[1])
            elif t['flower'] == 'Knotweed':
                for dr, dc in _NEIGHBOURS_8:
                    drained.add((pos[0] + dr, pos[1] + dc))

        def in_harmony(p1, p2):
            if p1[0] != p2[0] and p1[1] != p2[1]:
                return False
            # A Rock cancels harmonies lying along the row or column it sits on (rulebook p.10).
            if (p1[0] == p2[0] and p1[0] in rock_rows) or (p1[1] == p2[1] and p1[1] in rock_cols):
                return False
            return self._clear_line_between(p1[0], p1[1], p2[0], p2[1], board)

        owned = [(pos, t['flower']) for pos, t in board.items()
                 if t['player'] == player and not t['growing'] and t['flower'] in _CIRCLE_IDX and pos not in drained]
        result = []
        for i, (p1, f1) in enumerate(owned):
            for p2, f2 in owned[i + 1:]:
                if (f1, f2) in _HARMONY_PAIRS and in_harmony(p1, p2):
                    result.append((p1, p2))
        # A White Lotus (either owner's) harmonizes with every basic flower; the harmony
        # belongs to the basic flower's owner (rulebook p.12).
        lotuses = [pos for pos, t in board.items()
                   if t['flower'] == 'WhiteLotus' and not t['growing'] and pos not in drained]
        for p1, _ in owned:
            for p2 in lotuses:
                if in_harmony(p1, p2):
                    result.append((p1, p2))

        if custom_board is None:
            self._harmony_cache[player] = (sig, result)
        return result

    def find_clashes(self, custom_board=None):
        board = custom_board if custom_board is not None else self.board
        items = list(board.items())
        result = []
        for i, (p1, t1) in enumerate(items):
            if t1["growing"]:
                continue
            for p2, t2 in items[i + 1:]:
                if t2["growing"]:
                    continue
                r1, c1 = p1
                r2, c2 = p2
                if r1 == r2 or c1 == c2:
                    if (t1['flower'], t2['flower']) in _CLASH_PAIRS:
                        if self._clear_line_between(r1, c1, r2, c2, board):
                            result.append((p1, p2))
        return result

    def count_midline_harmonies(self, player):
        """Harmonies whose two tiles lie in adjacent quadrants with neither tile on a
        midline (row 9 / column 9) - the last-basic-flower tiebreak (rulebook p.15)."""
        count = 0
        for (r1, c1), (r2, c2) in self.find_harmonies(player):
            if RADIUS in (r1, c1, r2, c2):
                continue
            if (r1 < RADIUS) != (r2 < RADIUS) or (c1 < RADIUS) != (c2 < RADIUS):
                count += 1
        return count

    def check_harmony_ring(self, player):
        """A chain of the player's harmonies surrounding the center point without touching it."""
        edges = [(a, b) for a, b in self.find_harmonies(player) if not _segment_touches_center(a, b)]
        return _has_odd_cycle(edges)

    def valid_destinations(self, fr, fc):
        return self._destinations(fr, fc, first_only=False)

    def _has_any_destination(self, fr, fc):
        """Same answer as bool(valid_destinations(fr, fc)), stopping at the first."""
        return bool(self._destinations(fr, fc, first_only=True))

    def _destinations(self, fr, fc, first_only):
        tile = self.board.get((fr, fc))
        if not tile or tile['flower'] in ACCENT_TILES or self._is_trapped(fr, fc, tile):
            return []
        flower = tile['flower']
        limit = SPECIAL_MOVEMENT[flower] if flower in SPECIAL_MOVEMENT else FLOWER[flower]['move']
        wild = {1: self._is_wild(1), 2: self._is_wild(2)}
        moved = {'flower': flower, 'player': tile['player'], 'growing': False}

        dests = []
        queue = deque([(fr, fc, 0)])
        visited = {(fr, fc): 0}
        # The mover is lifted off the board for the search and restored afterwards.
        source_tile = self.board.pop((fr, fc))
        saved_zhash = self._zhash
        try:
            while queue:
                cr, cc, dist = queue.popleft()
                if dist == limit:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r, c = cr + dr, cc + dc
                    if (r, c) not in _VALID_SPACES_SET:
                        continue
                    nd = dist + 1
                    if visited.get((r, c), float('inf')) <= nd:
                        continue
                    visited[(r, c)] = nd
                    occ = self.board.get((r, c))
                    if (r, c) in _GATES_SET:
                        # A tile may pass through an open Gate but never stop on one (rulebook p.9).
                        if occ is None:
                            queue.append((r, c, nd))
                        continue
                    if occ is not None:
                        if (self._can_capture(source_tile, occ, wild) and not _garden_blocks(flower, (r, c))
                                and self._landing_is_clash_free(fr, fc, r, c, moved, occ)):
                            dests.append((r, c))
                            if first_only:
                                return dests
                        continue
                    if not _garden_blocks(flower, (r, c)) and self._landing_is_clash_free(fr, fc, r, c, moved, None):
                        dests.append((r, c))
                        if first_only:
                            return dests
                    queue.append((r, c, nd))
        finally:
            self.board[(fr, fc)] = source_tile
            self._zhash = saved_zhash
        return dests

    def _is_wild(self, player):
        """A player's Orchid is wild while they have a blooming White Lotus (rulebook p.12)."""
        return any(t['flower'] == 'WhiteLotus' and t['player'] == player and not t['growing'] and pos not in _GATES_SET
                   for pos, t in self.board.items())

    def _is_trapped(self, r, c, tile):
        """Orchid trap: an opponent's blooming flower tile next to a blooming Orchid can't arrange."""
        if tile['growing']:
            return False
        enemy = 3 - tile['player']
        for dr, dc in _NEIGHBOURS_8:
            adj = self.board.get((r + dr, c + dc))
            if adj and adj['flower'] == 'Orchid' and adj['player'] == enemy and not adj['growing']:
                return True
        return False

    def _can_capture(self, mover, target, wild):
        """Rulebook p.9/p.12: basic flowers capture the opposing basic they clash with; a wild
        (raging) Orchid captures any opposing flower tile; a wild (vulnerable) Orchid can be
        captured by any opposing flower tile. Accents and growing tiles are never captured."""
        if target['player'] == mover['player'] or target['growing'] or target['flower'] in ACCENT_TILES:
            return False
        if mover['flower'] == 'Orchid' and wild[mover['player']]:
            return True
        if target['flower'] == 'Orchid' and wild[target['player']]:
            return True
        if mover['flower'] in FLOWER and target['flower'] in FLOWER:
            return self.is_clash(mover['flower'], target['flower'])
        return False

    def _landing_is_clash_free(self, fr, fc, r, c, moved, captured):
        """With the mover already lifted off (fr, fc), would it landing on (r, c) leave no clash?"""
        self.board[(r, c)] = moved
        try:
            return not self._check_clash_after_move(fr, fc, r, c)
        finally:
            if captured is None:
                del self.board[(r, c)]
            else:
                self.board[(r, c)] = captured

    def _wheel_rotation(self, wr, wc):
        """{src: dest} moves a Wheel placed at (wr, wc) would make, or None if a Wheel may
        not be placed there: never on a gate or next to a Rock, and only if every rotated
        tile stays on the board, off the gates, out of the opposite garden, and the result
        has no clash (rulebook p.10)."""
        if (wr, wc) not in _VALID_SPACES_SET or (wr, wc) in _GATES_SET or (wr, wc) in self.board:
            return None
        ring = [(wr + dr, wc + dc) for dr, dc in _NEIGHBOURS_8]
        moves = {}
        for i, src in enumerate(ring):
            t = self.board.get(src)
            if t is None:
                continue
            dest = ring[(i + 1) % 8]
            if t['flower'] == 'Rock':
                return None
            if src in _GATES_SET or dest in _GATES_SET or dest not in _VALID_SPACES_SET:
                return None
            if _garden_blocks(t['flower'], dest):
                return None
            moves[src] = dest
        if not moves:
            return moves
        new_board = {p: t for p, t in self.board.items() if p not in moves}
        for src, dest in moves.items():
            new_board[dest] = self.board[src]
        new_board[(wr, wc)] = {'flower': 'Wheel', 'player': self.current_player, 'growing': False}
        if self.find_clashes(custom_board=new_board):
            return None
        return moves

    def _boat_displacements(self, r, c):
        """Empty surrounding points a Boat played on the flower at (r, c) may move it to:
        on the board, not a gate, garden rule for basics, and no resulting clash."""
        target = self.board[(r, c)]
        out = []
        for dr, dc in _NEIGHBOURS_8:
            d = (r + dr, c + dc)
            if d not in _VALID_SPACES_SET or d in _GATES_SET or d in self.board or _garden_blocks(target['flower'], d):
                continue
            new_board = dict(self.board)
            new_board[d] = target
            new_board[(r, c)] = {'flower': 'Boat', 'player': self.current_player, 'growing': False}
            if not self.find_clashes(custom_board=new_board):
                out.append(d)
        return out

    def _accent_actions(self, player):
        """Every legal accent placement for `player` (the rulebook p.10 rules above)."""
        hand = self.hands[player]
        empty = [p for p in VALID_SPACES if p not in _GATES_SET and p not in self.board]
        out = []
        for accent in ('Rock', 'Knotweed'):
            if hand.get(accent, 0) > 0:
                out += [('plant', accent, r, c) for r, c in empty]
        if hand.get('Wheel', 0) > 0:
            out += [('plant', 'Wheel', r, c) for r, c in empty if self._wheel_rotation(r, c) is not None]
        if hand.get('Boat', 0) > 0:
            for (r, c), t in list(self.board.items()):
                if t['flower'] in ACCENT_TILES:
                    # Both tiles leave the game; emptying the point must not uncover a clash.
                    if not self.find_clashes(custom_board={p: v for p, v in self.board.items() if p != (r, c)}):
                        out.append(('plant', 'Boat', r, c))
                elif not t['growing']:
                    out += [('plant', 'Boat', r, c, dr, dc) for dr, dc in self._boat_displacements(r, c)]
        return out

    def _place_accent(self, flower, r, c, displace):
        """Board/hand effects of an already-validated accent placement."""
        player = self.current_player
        if flower == 'Boat':
            target = self._remove((r, c))
            if target['flower'] not in ACCENT_TILES:
                self._put(displace, dict(target))
                self._put((r, c), {'flower': 'Boat', 'player': player, 'growing': False})
            # A Boat played on an accent: both leave the game and the point stays empty.
        elif flower == 'Wheel':
            moves = self._wheel_rotation(r, c)
            self._put((r, c), {'flower': 'Wheel', 'player': player, 'growing': False})
            moved = {dest: self._remove(src) for src, dest in moves.items()}
            for dest, t in moved.items():
                self._put(dest, t)
        else:
            self._put((r, c), {'flower': flower, 'player': player, 'growing': False})
        self.hands[player][flower] -= 1

    def _check_clash_after_move(self, fr, fc, tr, tc):
        """Incremental clash check after a move from (fr, fc) to (tr, tc).

        Assumes self.board already reflects the in-progress move. Because the
        board was clash-free before the move, only two new clashes are possible:
        one caused by the mover's new position, or one unblocked by removing
        the source tile.
        """
        moved = self.board.get((tr, tc))
        if moved and not moved['growing'] and moved['flower'] in _CIRCLE_IDX:
            mf = moved['flower']
            for pos, t in self.board.items():
                if pos == (tr, tc) or t['growing'] or t['flower'] not in _CIRCLE_IDX:
                    continue
                pr, pc = pos
                if pr == tr or pc == tc:
                    if (mf, t['flower']) in _CLASH_PAIRS:
                        if self._clear_line_between(tr, tc, pr, pc):
                            return True

        row_cands = []
        col_cands = []
        for pos, t in self.board.items():
            if t['growing'] or t['flower'] not in _CIRCLE_IDX:
                continue
            if pos[0] == fr:
                row_cands.append((pos, t['flower']))
            if pos[1] == fc:
                col_cands.append((pos, t['flower']))

        for cands in (row_cands, col_cands):
            for i, (p1, f1) in enumerate(cands):
                for p2, f2 in cands[i + 1:]:
                    if (f1, f2) in _CLASH_PAIRS:
                        if self._clear_line_between(p1[0], p1[1], p2[0], p2[1]):
                            return True

        return False

    # ---- turn structure -------------------------------------------------

    def get_legal_actions(self):
        if self.winner is not None:
            return []
        key = (self._zhash, self.current_player, self.bonus_turn,
               tuple(sorted(self.hands[1].items())), tuple(sorted(self.hands[2].items())))
        cached = self._legal_actions_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        actions = [a for a in self._iter_candidate_actions() if self._opponent_can_move_after(a)]
        if self.bonus_turn:
            actions.append(('skip_bonus',))
        self._legal_actions_cache = (key, actions, frozenset(actions))
        return actions

    def _iter_candidate_actions(self):
        """Plant/arrange/bonus actions for the side to move (rulebook p.8), before the
        no-stranding filter. A generator so callers that only need one can stop early."""
        player = self.current_player
        hand = self.hands[player]
        empty_gates = [g for g in GATES if g not in self.board]
        basics = [('plant', f, r, c) for f in CIRCLE if hand.get(f, 0) > 0 for r, c in empty_gates]
        if self.bonus_turn:
            if not self._has_growing(player):
                yield from basics
            for s in SPECIAL_TILES:
                if hand.get(s, 0) > 0:
                    for r, c in empty_gates:
                        yield ('plant', s, r, c)
            yield from self._accent_actions(player)
            return
        yield from basics
        for (fr, fc), t in list(self.board.items()):
            if t['player'] == player and t['flower'] not in ACCENT_TILES:
                for tr, tc in self.valid_destinations(fr, fc):
                    yield ('arrange', fr, fc, tr, tc)

    def _has_growing(self, player):
        return any(t['player'] == player and t['growing'] for t in self.board.values())

    def step(self, action):
        if self.winner is not None:
            raise ValueError("Game is already over.")
        action = tuple(action)
        actor = self.current_player
        self.get_legal_actions()
        if action not in self._legal_actions_cache[2]:
            raise ValueError(f"Illegal action: {action!r}")
        before = None
        if action[0] == 'arrange':
            before = self._pairs_after_relocation(self.current_player, (action[1], action[2]), (action[3], action[4]))
        self._apply_effects(action)
        self.history.append(list(action))
        self.history_players.append(actor)
        self._end_turn(action, before)
        return self.winner is not None

    def plant(self, flower, r, c, displace_r=None, displace_c=None):
        """Plant a flower or place an accent; a Boat played on a flower needs (displace_r, displace_c)."""
        action = ('plant', flower, r, c)
        if displace_r is not None:
            action += (displace_r, displace_c)
        self.step(action)

    def arrange(self, fr, fc, tr, tc):
        self.step(('arrange', fr, fc, tr, tc))

    def skip_bonus(self):
        self.step(('skip_bonus',))

    def _apply_effects(self, action):
        """Board and hand changes for a legal `action`, with no turn bookkeeping.
        Used by step() and by what-if simulations."""
        kind = action[0]
        if kind == 'arrange':
            _, fr, fc, tr, tc = action
            cap = self.board.get((tr, tc))
            if cap:
                self._remove((tr, tc))
                self.hands[cap['player']][cap['flower']] += 0  # captured tiles leave the game for good (rulebook p.9)
            moved = dict(self._remove((fr, fc)))
            moved['growing'] = False
            self._put((tr, tc), moved)
        elif kind == 'plant':
            flower, r, c = action[1], action[2], action[3]
            if flower in ACCENT_TILES:
                self._place_accent(flower, r, c, (action[4], action[5]) if len(action) == 6 else None)
            else:
                self._put((r, c), {'flower': flower, 'player': self.current_player, 'growing': True})
                self.hands[self.current_player][flower] -= 1

    def _pairs_after_relocation(self, player, src, dst):
        """The player's current harmony pairs, with the tile on `src` relabelled as on `dst`."""
        def moved(p):
            return dst if p == src else p
        return {frozenset((moved(a), moved(b))) for a, b in self.find_harmonies(player)}

    def _end_turn(self, action, before):
        actor = self.current_player
        ring1, ring2 = self.check_harmony_ring(1), self.check_harmony_ring(2)
        if ring1 or ring2:
            self.winner = 0 if (ring1 and ring2) else (1 if ring1 else 2)
            self._finish('ring')
            return
        if action[0] == 'plant' and action[1] in _CIRCLE_IDX and self._basic_count(actor) == 0:
            self._score_by_midlines()
            self._finish('last_basic_flower')
            return
        if action[0] == 'arrange' and not self.bonus_turn:
            if {frozenset(p) for p in self.find_harmonies(actor)} - before:
                self.bonus_turn = True
                self.message = (f'Player {actor}: Harmony! Bonus: place an accent, plant a special flower, '
                                f'plant a basic flower (if none are growing), or skip.')
                return
        self.bonus_turn = False
        self.current_player = 3 - actor
        self.message = f'Player {self.current_player}: Plant in a Gate or Arrange a tile'
        if not self._has_any_legal_action():
            self._score_by_midlines()
            self._finish('no_moves')

    def _opponent_can_move_after(self, action):
        """No stranding (rules page): after `action`, as if the turn ended there, the
        opponent must still be able to Plant (an open gate and a basic flower in reserve)
        or Arrange some tile. Fast path without simulation when a gate stays open."""
        opponent = 3 - self.current_player
        if self._basic_count(opponent) > 0:
            open_gates = {g for g in GATES if g not in self.board}
            if action[0] == 'plant' and (action[2], action[3]) in _GATES_SET:
                open_gates.discard((action[2], action[3]))
            elif action[0] == 'arrange' and (action[1], action[2]) in _GATES_SET:
                open_gates.add((action[1], action[2]))
            if open_gates:
                return True
        sim = self._sim_copy()
        sim._apply_effects(action)
        return sim._can_move(opponent)

    def _sim_copy(self):
        """A throwaway copy for what-if simulation: board, hands, turn and hash, with no
        history or caches (cheaper than clone()). The board copy is shallow on purpose:
        _apply_effects and valid_destinations never mutate a tile dict in place, they
        only move tile dicts between points or put fresh ones."""
        sim = PaiShoGame.__new__(PaiShoGame)
        sim.board = dict(self.board)
        sim.hands = {pid: dict(hand) for pid, hand in self.hands.items()}
        sim.current_player = self.current_player
        sim.winner = self.winner
        sim.end_reason = self.end_reason
        sim.message = ''
        sim.bonus_turn = self.bonus_turn
        sim._setup = self._setup
        sim._zhash = self._zhash
        sim._harmony_cache = {}
        sim._legal_actions_cache = None
        sim.history = []
        sim.history_players = []
        return sim

    def _can_move(self, player):
        """Whether `player` would have any Plant or Arrange on a normal turn."""
        if self._basic_count(player) > 0 and any(g not in self.board for g in GATES):
            return True
        return any(t['player'] == player and t['flower'] not in ACCENT_TILES and self._has_any_destination(r, c)
                   for (r, c), t in list(self.board.items()))

    def _has_any_legal_action(self):
        return bool(self.get_legal_actions())

    def resign(self, player):
        """Forfeit (rulebook p.15): `player` resigns and the opponent wins."""
        if self.winner is not None:
            raise ValueError("Game is already over.")
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        self.winner = 3 - player
        self._finish('resign')

    def _basic_count(self, player):
        return sum(self.hands[player].get(f, 0) for f in CIRCLE)

    def _score_by_midlines(self):
        c1, c2 = self.count_midline_harmonies(1), self.count_midline_harmonies(2)
        self.winner = 1 if c1 > c2 else 2 if c2 > c1 else 0

    def _finish(self, reason):
        self.end_reason = reason
        self.bonus_turn = False
        self.message = self._end_message()

    def _end_message(self):
        w, reason = self.winner, self.end_reason
        if reason == 'ring':
            return 'Tie: both players formed a Harmony Ring.' if w == 0 else f'Player {w} wins by Harmony Ring.'
        if reason == 'resign':
            return 'Game over.' if w == 0 else f'Player {3 - w} resigned. Player {w} wins.'
        c1, c2 = self.count_midline_harmonies(1), self.count_midline_harmonies(2)
        result = 'Tie' if w == 0 else f'Player {w} wins'
        cause = (f'Player {self.current_player} planted their last basic flower.' if reason == 'last_basic_flower'
                 else f'Player {self.current_player} has no legal move.')
        return f'{result} on midline harmonies. {cause} Midline-crossing harmonies — P1: {c1}, P2: {c2}.'

    def _state_message(self):
        """The message both engines show for the current state (used when a dict carries none)."""
        if self.winner is not None:
            return self._end_message() if self.end_reason is not None else 'Game over.'
        if self.bonus_turn:
            return (f'Player {self.current_player}: Harmony! Bonus: place an accent, plant a special flower, '
                    f'plant a basic flower (if none are growing), or skip.')
        return f'Player {self.current_player}: Plant in a Gate or Arrange a tile'

    def to_save_dict(self, p1_name='Player 1', p2_name='Player 2'):
        import time
        board_s = {f"{r},{c}": t for (r, c), t in self.board.items()}
        accents = self._setup['accents']
        return {
            'version': SAVE_VERSION,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'p1': p1_name,
            'p2': p2_name,
            'history': self.history,
            'history_players': list(self.history_players),
            'state': {
                'board': board_s,
                'hands': {'1': self.hands[1], '2': self.hands[2]},
                'current_player': self.current_player,
                'winner': self.winner,
                'end_reason': self.end_reason,
                'message': self.message,
                'bonus_turn': self.bonus_turn,
                'setup': {'accents': {'1': list(accents[1]), '2': list(accents[2])},
                          'opening': self._setup['opening']},
            },
        }

    @classmethod
    def from_save_dict(cls, data):
        version = data.get('version')
        if not isinstance(version, int) or isinstance(version, bool) or version > SAVE_VERSION:
            raise ValueError(f"unsupported save version {version!r}; this build reads v{SAVE_VERSION}")
        if version < SAVE_VERSION:
            raise ValueError(f"save format v{version} predates the official rules; only v{SAVE_VERSION} saves can be loaded")
        game = cls.from_dict(data['state'])
        game.history = [list(a) for a in data.get('history', [])]
        players = data.get('history_players')
        game.history_players = list(players) if players is not None and len(players) == len(game.history) else []
        return game
