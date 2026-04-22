import random
import requests
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

# Fixed seed so the hash is reproducible across sessions and Q-tables stay loadable.
random.seed(42)
_ALL_FLOWERS = CIRCLE + ACCENT_TILES + SPECIAL_TILES
_ZOBRIST = {
    (pos, flower, player, growing): random.getrandbits(64)
    for pos in VALID_SPACES
    for flower in _ALL_FLOWERS
    for player in (1, 2)
    for growing in (True, False)
}


class PaiShoGame:
    def __init__(self):
        self.board = None
        self.hands = None
        self.current_player = None
        self.winner = None
        self.message = None
        self.bonus_turn = False
        self._harmony_cache = {}
        self._zhash = 0
        self._legal_actions_cache = None
        self.history = []
        self.reset()

    def reset(self):
        hand = {f: 2 for f in CIRCLE}
        for t in ACCENT_TILES + SPECIAL_TILES:
            hand[t] = 1
        self.board = {}
        self._zhash = 0
        self.hands = {1: dict(hand), 2: dict(hand)}
        self.current_player = 1
        self.winner = None
        self.bonus_turn = False
        self.message = 'Player 1: Plant a flower onto an open Gate'
        self._harmony_cache = {}
        self._legal_actions_cache = None
        self.history = []

    def clone(self):
        new_game = PaiShoGame()
        new_game.board = {pos: dict(tile) for pos, tile in self.board.items()}
        new_game.hands = {pid: dict(hand) for pid, hand in self.hands.items()}
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.message = self.message
        new_game.bonus_turn = self.bonus_turn
        new_game._zhash = self._zhash
        new_game._harmony_cache = dict(self._harmony_cache)
        new_game.history = [list(a) for a in self.history]
        return new_game

    @classmethod
    def from_dict(cls, d):
        game = cls()
        game.board = {
            tuple(map(int, pos_str.split(','))): tile
            for pos_str, tile in d['board'].items()
        }
        hands = d['hands']
        game.hands = {
            1: hands.get('1', hands.get(1, {})),
            2: hands.get('2', hands.get(2, {})),
        }
        game.current_player = d['current_player']
        game.winner = d['winner']
        game.message = d.get('message', '')
        game.bonus_turn = d.get('bonus_turn', False)
        game.history = [list(a) for a in d.get('history', [])]
        game._zhash = 0
        for pos, t in game.board.items():
            game._zhash ^= _ZOBRIST.get((pos, t['flower'], t['player'], t['growing']), 0)
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

    def _clear_line_between(self, r1, c1, r2, c2):
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if (r1, c) in self.board or (r1, c) in _GATES_SET:
                    return False
        elif c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if (r, c1) in self.board or (r, c1) in _GATES_SET:
                    return False
        return True

    def _z_toggle(self, pos, tile):
        self._zhash ^= _ZOBRIST.get((pos, tile['flower'], tile['player'], tile['growing']), 0)

    def _board_sig(self):
        return self._zhash

    def _state_sig(self):
        h = self._board_sig()
        h ^= hash(self.current_player)
        h ^= hash(self.bonus_turn)
        for pid in (1, 2):
            for flower, count in self.hands[pid].items():
                h ^= hash((pid, flower, count))
        return h

    def find_harmonies(self, player, custom_board=None):
        if custom_board is None:
            sig = self._board_sig()
            entry = self._harmony_cache.get(player)
            if entry is not None and entry[0] == sig:
                return entry[1]

        board = custom_board if custom_board is not None else self.board

        rock_rows = set()
        rock_cols = set()
        drained = set()
        for pos, t in board.items():
            f = t['flower']
            if f == 'Rock':
                rock_rows.add(pos[0])
                rock_cols.add(pos[1])
            elif f == 'Knotweed':
                r, c = pos
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr or dc:
                            drained.add((r + dr, c + dc))

        def rock_affected(pos):
            return pos[0] in rock_rows or pos[1] in rock_cols

        owned = [(pos, t) for pos, t in board.items()
                 if t['player'] == player and not t['growing']
                 and t['flower'] in _CIRCLE_IDX
                 and pos not in drained and not rock_affected(pos)]

        result = []
        for i, (p1, t1) in enumerate(owned):
            for p2, t2 in owned[i + 1:]:
                r1, c1 = p1
                r2, c2 = p2
                if r1 == r2 or c1 == c2:
                    if (t1['flower'], t2['flower']) in _HARMONY_PAIRS:
                        if self._clear_line_between(r1, c1, r2, c2):
                            result.append((p1, p2))

        wl_tiles = [(pos, t) for pos, t in board.items()
                    if t['flower'] == 'WhiteLotus' and not t['growing']
                    and pos not in drained and not rock_affected(pos)]

        for pos_f, t_f in owned:
            r_f, c_f = pos_f
            for pos_wl, _ in wl_tiles:
                r_wl, c_wl = pos_wl
                if r_f == r_wl or c_f == c_wl:
                    if self._clear_line_between(r_f, c_f, r_wl, c_wl):
                        pair = (pos_f, pos_wl)
                        if pair not in result and (pos_wl, pos_f) not in result:
                            result.append(pair)

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
                        if self._clear_line_between(r1, c1, r2, c2):
                            result.append((p1, p2))
        return result

    def count_midline_harmonies(self, player):
        """Count the player's harmonies whose two tiles sit on opposite sides
        of one of the board's midlines (row 9 or column 9). Tiles that lie
        exactly on a midline do not count as crossing it.
        """
        count = 0
        mid = RADIUS
        for (r1, c1), (r2, c2) in self.find_harmonies(player):
            if r1 == r2 and min(c1, c2) < mid < max(c1, c2):
                count += 1
            elif c1 == c2 and min(r1, r2) < mid < max(r1, r2):
                count += 1
        return count

    def _basic_flowers_exhausted(self):
        """True if either player has no basic flowers left in their reserve."""
        for pid in (1, 2):
            if sum(self.hands[pid].get(f, 0) for f in CIRCLE) == 0:
                return pid
        return None

    def _finish_by_midline_harmonies(self, exhausted_player):
        c1 = self.count_midline_harmonies(1)
        c2 = self.count_midline_harmonies(2)
        trigger = (f'Player {exhausted_player} planted their last basic flower. '
                   f'Midline-crossing harmonies — P1: {c1}, P2: {c2}.')
        if c1 > c2:
            self.winner = 1
            self.message = f'Player 1 wins by Last Basic Flower rule. {trigger}'
        elif c2 > c1:
            self.winner = 2
            self.message = f'Player 2 wins by Last Basic Flower rule. {trigger}'
        else:
            self.winner = 0
            self.message = f'Tie by Last Basic Flower rule. {trigger}'
        self.bonus_turn = False

    def check_harmony_ring(self, player):
        harmonies = self.find_harmonies(player)
        if len(harmonies) < 4:
            return False
        adj = {}
        for p1, p2 in harmonies:
            adj.setdefault(p1, []).append(p2)
            adj.setdefault(p2, []).append(p1)

        cx, cy = 9.0, 9.0

        def enclosed(cycle):
            n = len(cycle)
            inside = False
            j = n - 1
            for i in range(n):
                x1, y1 = cycle[j][1], cycle[j][0]
                x2, y2 = cycle[i][1], cycle[i][0]
                if (y1 > cy) != (y2 > cy):
                    x_int = x1 + (cy - y1) * (x2 - x1) / (y2 - y1)
                    if cx < x_int:
                        inside = not inside
                j = i
            return inside

        found = [False]

        def dfs(start, cur, path, visited):
            if found[0] or len(path) > 10:
                return
            for nb in adj.get(cur, []):
                if nb == start and len(path) >= 4:
                    if enclosed(path):
                        found[0] = True
                    return
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(start, nb, path, visited)
                    path.pop()
                    visited.remove(nb)
                    if found[0]:
                        return

        for start in list(adj.keys()):
            if found[0]:
                break
            dfs(start, start, [start], {start})
        return found[0]

    def valid_destinations(self, fr, fc):
        tile = self.board.get((fr, fc))
        if not tile: return []

        flower = tile['flower']
        player = tile['player']

        if flower in ACCENT_TILES:
            return []

        # Orchid trap: tiles adjacent to an enemy blooming Orchid are frozen (gates excluded).
        if (fr, fc) not in _GATES_SET:
            enemy = 3 - player
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    adj = self.board.get((fr + dr, fc + dc))
                    if adj and adj['flower'] == 'Orchid' and adj['player'] == enemy and not adj['growing']:
                        return []

        if flower in SPECIAL_MOVEMENT:
            limit = SPECIAL_MOVEMENT[flower]
            fcol = None
        else:
            limit = FLOWER[flower]['move']
            fcol = FLOWER[flower]['color']

        dests = set()
        queue = deque([(fr, fc, 0)])
        visited = {(fr, fc): 0}

        # Pop the source tile once for the whole BFS; restored in the finally block.
        # Avoids cloning the board for every candidate destination.
        source_tile = self.board.pop((fr, fc))
        saved_zhash = self._zhash

        try:
            while queue:
                curr_r, curr_c, dist = queue.popleft()
                if dist == limit: continue

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r, c = curr_r + dr, curr_c + dc
                    if (r, c) not in _VALID_SPACES_SET or (r, c) in _GATES_SET: continue

                    new_dist = dist + 1
                    if visited.get((r, c), float('inf')) <= new_dist: continue

                    visited[(r, c)] = new_dist
                    occ = self.board.get((r, c))

                    if occ:
                        if occ['player'] != player:
                            can_capture = False
                            if flower in SPECIAL_TILES:
                                if flower == 'Orchid':
                                    # Orchid captures any flower only if the mover has a blooming White Lotus outside a gate.
                                    can_capture = any(
                                        t['flower'] == 'WhiteLotus' and not t['growing']
                                        and pos2 not in _GATES_SET
                                        for pos2, t in self.board.items()
                                        if t['player'] == player
                                    )
                            else:
                                if occ['flower'] == 'Orchid':
                                    # Orchid is capturable only when its own owner has a blooming White Lotus outside a gate.
                                    orchid_owner = occ['player']
                                    can_capture = any(
                                        t['flower'] == 'WhiteLotus' and not t['growing']
                                        and pos2 not in _GATES_SET
                                        for pos2, t in self.board.items()
                                        if t['player'] == orchid_owner
                                    )
                                elif occ['flower'] not in ACCENT_TILES and occ['flower'] != 'WhiteLotus':
                                    can_capture = self.is_clash(flower, occ['flower'])

                            if can_capture:
                                if fcol is None or not (
                                        (fcol == 'red' and _GARDEN_OF.get((r, c), 'neutral') == 'white') or
                                        (fcol == 'white' and _GARDEN_OF.get((r, c), 'neutral') == 'red')
                                ):
                                    captured_tile = self.board.pop((r, c))
                                    self.board[(r, c)] = {'flower': flower, 'player': player, 'growing': False}

                                    no_clash = not self._check_clash_after_move(fr, fc, r, c)

                                    del self.board[(r, c)]
                                    self.board[(r, c)] = captured_tile

                                    if no_clash:
                                        dests.add((r, c))
                        continue

                    garden = _GARDEN_OF.get((r, c), 'neutral')
                    if fcol is None or not (
                            (fcol == 'red' and garden == 'white') or
                            (fcol == 'white' and garden == 'red')
                    ):
                        self.board[(r, c)] = {'flower': flower, 'player': player, 'growing': False}

                        no_clash = not self._check_clash_after_move(fr, fc, r, c)

                        del self.board[(r, c)]

                        if no_clash:
                            dests.add((r, c))

                    queue.append((r, c, new_dist))

        finally:
            self.board[(fr, fc)] = source_tile
            self._zhash = saved_zhash

        return list(dests)

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

    def get_legal_actions(self):
        if self.winner is not None:
            return []

        player = self.current_player
        cur_hand_key = tuple(sorted(self.hands[player].items()))
        key = (self._zhash, player, self.bonus_turn, cur_hand_key)
        cached = self._legal_actions_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        actions = []

        empty_gates = [g for g in GATES if g not in self.board]
        if empty_gates:
            for flower, count in self.hands[player].items():
                if count > 0 and flower in _CIRCLE_IDX:
                    for r, c in empty_gates:
                        actions.append(('plant', flower, r, c))

        if self.bonus_turn:
            for flower in ACCENT_TILES:
                if self.hands[player].get(flower, 0) > 0:
                    if flower == 'Boat':
                        for pos, t in self.board.items():
                            if t['player'] != player:
                                actions.append(('plant', flower, pos[0], pos[1]))
                    else:
                        for r, c in VALID_SPACES:
                            if (r, c) not in _GATES_SET and (r, c) not in self.board:
                                actions.append(('plant', flower, r, c))

            for flower in SPECIAL_TILES:
                if self.hands[player].get(flower, 0) > 0:
                    for r, c in empty_gates:
                        actions.append(('plant', flower, r, c))

        owned_tiles = [
            (pos, t) for pos, t in self.board.items()
            if t['player'] == player and t['flower'] not in ACCENT_TILES
        ]
        for (fr, fc), tile in owned_tiles:
            if self.bonus_turn and tile['growing']:
                continue
            dests = self.valid_destinations(fr, fc)
            for tr, tc in dests:
                actions.append(('arrange', fr, fc, tr, tc))

        self._legal_actions_cache = (key, actions)
        return actions

    def step(self, action):
        if self.winner is not None:
            raise ValueError("Game is already over.")

        action_type = action[0]

        if action_type == 'plant':
            _, flower, r, c = action
            self.plant(flower, r, c)
        elif action_type == 'arrange':
            _, fr, fc, tr, tc = action
            self.arrange(fr, fc, tr, tc)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        return self.winner is not None

    def plant(self, flower, r, c, displace_r=None, displace_c=None):
        pre_move_harmonies = {p: len(self.find_harmonies(p)) for p in [1, 2]}

        if flower in ACCENT_TILES:
            self._plant_accent(flower, r, c, pre_move_harmonies, displace_r, displace_c)
            self.history.append(['plant', flower, r, c])
            return

        if (r, c) not in _GATES_SET:
            raise ValueError('Must plant in a Gate')
        if (r, c) in self.board:
            raise ValueError('Gate occupied')
        if self.hands[self.current_player].get(flower, 0) <= 0:
            raise ValueError('No tiles of that type left')

        tile = {'flower': flower, 'player': self.current_player, 'growing': True}
        self.board[(r, c)] = tile
        self._z_toggle((r, c), tile)
        self.hands[self.current_player][flower] -= 1
        self.history.append(['plant', flower, r, c])
        self._end_turn(pre_move_harmonies)

    def _plant_accent(self, flower, r, c, pre_move_harmonies, displace_r=None, displace_c=None):
        player = self.current_player

        if flower == 'Boat':
            target = self.board.get((r, c))
            if not target or target['player'] == player:
                raise ValueError('Boat must target an enemy tile')

            if target['flower'] in ACCENT_TILES:
                self._z_toggle((r, c), target)
                del self.board[(r, c)]
            else:
                def _legal_displacement(nr, nc):
                    if (nr, nc) not in _VALID_SPACES_SET or (nr, nc) in _GATES_SET:
                        return False
                    if (nr, nc) in self.board:
                        return False
                    if target['flower'] in FLOWER:
                        fcol = FLOWER[target['flower']]['color']
                        g = _GARDEN_OF.get((nr, nc), 'neutral')
                        if (fcol == 'red' and g == 'white') or (fcol == 'white' and g == 'red'):
                            return False
                    return True

                if displace_r is not None and displace_c is not None:
                    if not _legal_displacement(displace_r, displace_c):
                        raise ValueError('Invalid displacement position')
                    displaced = dict(target)
                    self.board[(displace_r, displace_c)] = displaced
                    self._z_toggle((displace_r, displace_c), displaced)
                else:
                    # AI fallback when the caller didn't pick a landing square: first legal neighbor.
                    for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + ddr, c + ddc
                        if _legal_displacement(nr, nc):
                            displaced = dict(target)
                            self.board[(nr, nc)] = displaced
                            self._z_toggle((nr, nc), displaced)
                            break
                self._z_toggle((r, c), target)
                del self.board[(r, c)]
                boat_tile = {'flower': 'Boat', 'player': player, 'growing': False}
                self.board[(r, c)] = boat_tile
                self._z_toggle((r, c), boat_tile)

            self.hands[player]['Boat'] -= 1

        else:
            if (r, c) in _GATES_SET:
                raise ValueError('Cannot place accent tile on a gate')
            if (r, c) not in _VALID_SPACES_SET:
                raise ValueError('Invalid position')
            if (r, c) in self.board:
                raise ValueError('Space occupied')

            accent_tile = {'flower': flower, 'player': player, 'growing': False}
            self.board[(r, c)] = accent_tile
            self._z_toggle((r, c), accent_tile)
            if flower == 'Wheel':
                self._apply_wheel(r, c)
            self.hands[player][flower] -= 1

        self._end_turn(pre_move_harmonies)

    def _apply_wheel(self, wr, wc):
        """Rotate tiles in the 8 cells around (wr, wc) one step clockwise."""
        surrounds = [
            (wr - 1, wc), (wr - 1, wc + 1), (wr, wc + 1), (wr + 1, wc + 1),
            (wr + 1, wc), (wr + 1, wc - 1), (wr, wc - 1), (wr - 1, wc - 1)
        ]

        tiles_at = [(pos, self.board[pos]) for pos in surrounds if pos in self.board]
        if not tiles_at:
            return

        pos_index = {pos: i for i, pos in enumerate(surrounds)}
        moves = {pos: surrounds[(pos_index[pos] + 1) % 8] for pos, _ in tiles_at}

        sources = set(moves.keys())
        dest_set = set(moves.values())

        if len(dest_set) != len(moves):
            return

        non_moved = {k: v for k, v in self.board.items()
                     if k not in sources and k != (wr, wc)}

        for src, dest in moves.items():
            if dest not in _VALID_SPACES_SET or dest in _GATES_SET:
                return
            if dest in non_moved:
                return
            t = self.board[src]
            if t['flower'] in FLOWER:
                fcol = FLOWER[t['flower']]['color']
                if ((fcol == 'red' and _GARDEN_OF.get(dest, 'neutral') == 'white') or
                        (fcol == 'white' and _GARDEN_OF.get(dest, 'neutral') == 'red')):
                    return

        new_board = dict(non_moved)
        new_board[(wr, wc)] = self.board[(wr, wc)]
        for src, dest in moves.items():
            new_board[dest] = self.board[src]

        if self.find_clashes(custom_board=new_board):
            return

        moved_data = {dest: self.board[src] for src, dest in moves.items()}
        for src in sources:
            self._z_toggle(src, self.board[src])
            del self.board[src]
        for dest, t in moved_data.items():
            self.board[dest] = t
            self._z_toggle(dest, t)

    def arrange(self, fr, fc, tr, tc):
        pre_move_harmonies = {p: len(self.find_harmonies(p)) for p in [1, 2]}

        tile = self.board.get((fr, fc))
        if not tile or tile['player'] != self.current_player:
            raise ValueError('Not your tile')
        if (tr, tc) not in self.valid_destinations(fr, fc):
            raise ValueError('Invalid move')

        cap = self.board.get((tr, tc))
        if cap:
            self._z_toggle((tr, tc), cap)
            self.hands[cap['player']][cap['flower']] += 0

        source_tile = self.board[(fr, fc)]
        self._z_toggle((fr, fc), source_tile)
        moved = dict(self.board.pop((fr, fc)))
        moved['growing'] = False
        self.board[(tr, tc)] = moved
        self._z_toggle((tr, tc), moved)
        self.history.append(['arrange', fr, fc, tr, tc])
        self._end_turn(pre_move_harmonies)

    def _end_turn(self, pre_move_harmonies=None):
        players_harmonies = {}
        for pl in [self.current_player, 3 - self.current_player]:
            status = self.check_harmony_ring(pl)
            if status:
                players_harmonies[pl] = status

        if players_harmonies.get(self.current_player):
            self.winner = self.current_player
            self.message = f'Player {self.current_player} wins by Harmony Ring rule.'
            self.bonus_turn = False
            return
        elif players_harmonies.get(3 - self.current_player):
            self.winner = 3 - self.current_player
            self.message = f'Player {3 - self.current_player} wins by Harmony Ring rule.'
            self.bonus_turn = False
            return

        exhausted = self._basic_flowers_exhausted()
        if exhausted is not None:
            self._finish_by_midline_harmonies(exhausted)
            return

        if not self.bonus_turn and pre_move_harmonies is not None:
            current_harmonies = len(self.find_harmonies(self.current_player))
            if current_harmonies > pre_move_harmonies.get(self.current_player, 0):
                self.bonus_turn = True
                self.message = (f'Player {self.current_player}: Harmony! '
                                f'Bonus turn - plant, arrange, or place an accent/special tile.')
                return

        self.bonus_turn = False
        self.current_player = 3 - self.current_player
        self.message = f"Player {self.current_player}: Plant in a Gate or Arrange a tile"

    def current_state_web(self):
        """Push the current game state to the local Flask server."""
        board_s = {f"{r},{c}": t for (r, c), t in self.board.items()}
        hands_s = {'1': self.hands[1], '2': self.hands[2]}

        payload = {
            'board': board_s,
            'hands': hands_s,
            'current_player': self.current_player,
            'winner': self.winner,
            'message': self.message,
            'bonus_turn': self.bonus_turn,
        }

        server_url = "http://127.0.0.1:5000"

        try:
            requests.post(f"{server_url}/api/new_game")
            requests.post(f"{server_url}/api/set_state/default", json=payload)

        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {server_url}. Make sure ui/server.py is running.")

    def to_save_dict(self, p1_name='Player 1', p2_name='Player 2'):
        import time
        board_s = {f"{r},{c}": t for (r, c), t in self.board.items()}
        return {
            'version': 1,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'p1': p1_name,
            'p2': p2_name,
            'history': self.history,
            'state': {
                'board': board_s,
                'hands': {'1': self.hands[1], '2': self.hands[2]},
                'current_player': self.current_player,
                'winner': self.winner,
                'message': self.message,
                'bonus_turn': self.bonus_turn,
            },
        }

    @classmethod
    def from_save_dict(cls, data):
        game = cls.from_dict(data['state'])
        game.history = [list(a) for a in data.get('history', [])]
        return game
