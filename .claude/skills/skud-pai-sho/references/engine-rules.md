# How MushiBot's engine implements Skud Pai Sho

The authoritative code is `engine/PythonEngine/PaiShoGame.py`. `engine/RustEngine/crates/engine/src/` (board.rs, moves.rs, harmony.rs, game.rs, …) is a 1:1 port exposed as `RustEngine.PaiShoGame`, and the two are kept in behavioural parity. Line numbers below are for the Python file, as of Phase 0's official-rules pass. They drift over time, so grep for the function name if one looks off.

## Contents
1. Coordinates, board, and garden map
2. Tiles, hands, and the Circle
3. Turn flow, the Harmony Bonus, and no-stranding
4. Movement (`valid_destinations`)
5. Capture matrix
6. Harmony and clash detection
7. Accent tiles as coded
8. Special flowers as coded
9. Game end
10. State flags worth knowing

## 1. Coordinates, board, and garden map
- `(row, col)`, with row 0 at the top and col 0 at the left. `BOARD_SIZE = 19`, `RADIUS = 9`, `CENTER = (9, 9)`.
- **Playable points** are `VALID_SPACES` (L91): the points with `(r-9)² + (c-9)² <= 81`, **minus** `BEHIND_GATES = (0,9) (18,9) (9,0) (9,18)`. That leaves 249 points, 4 of them gates (245 non-gate playable points). `is_valid(r,c)` (L87) is only the circle test and still accepts those 4 behind-gate points, so check membership in `VALID_SPACES` instead.
- **Gates:** `GATES = [(1,9), (17,9), (9,1), (9,17)]` (L8), north, south, west and east. `GUEST_GATE = (17,9)` is Player 1's; `HOST_GATE = (1,9)` is Player 2's. Each is the outermost playable point on its axis. A tile may pass through an open gate while moving but a move may never end on one.
- **Gardens** (`garden_of`, L97):
  - row 9 or col 9: `'neutral'` (these are the midlines);
  - `|r-9| + |c-9| < 7`: **red** when `(r-9)` and `(c-9)` have opposite signs (top-right and bottom-left quadrants), **white** when the signs match (top-left and bottom-right);
  - everything else: `'neutral'`.
  - There are 30 red points, 30 white, and 185 neutral (not counting gates).

Generated from the code (`G` gate, `R` red, `W` white, `.` neutral, blank = not playable):

```
     col:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
row  0:
row  1:                   .  .  .  .  G  .  .  .  .
row  2:                .  .  .  .  .  .  .  .  .  .  .
row  3:             .  .  .  .  .  .  .  .  .  .  .  .  .
row  4:          .  .  .  .  .  .  W  .  R  .  .  .  .  .  .
row  5:       .  .  .  .  .  .  W  W  .  R  R  .  .  .  .  .  .
row  6:       .  .  .  .  .  W  W  W  .  R  R  R  .  .  .  .  .
row  7:       .  .  .  .  W  W  W  W  .  R  R  R  R  .  .  .  .
row  8:       .  .  .  W  W  W  W  W  .  R  R  R  R  R  .  .  .
row  9:       G  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  G
row 10:       .  .  .  R  R  R  R  R  .  W  W  W  W  W  .  .  .
row 11:       .  .  .  .  R  R  R  R  .  W  W  W  W  .  .  .  .
row 12:       .  .  .  .  .  R  R  R  .  W  W  W  .  .  .  .  .
row 13:       .  .  .  .  .  .  R  R  .  W  W  .  .  .  .  .  .
row 14:          .  .  .  .  .  .  R  .  W  .  .  .  .  .  .
row 15:             .  .  .  .  .  .  .  .  .  .  .  .  .
row 16:                .  .  .  .  .  .  .  .  .  .  .
row 17:                   .  .  .  .  G  .  .  .  .
row 18:
```

Regenerated for this pass by iterating `VALID_SPACES` and `garden_of(r, c)` (unchanged from before Phase 0, since neither function changed).

## 2. Tiles, hands, and the Circle
- `CIRCLE = ['Rose','Chrysanthemum','Rhododendron','Jasmine','Lily','Jade']` (L10) is cyclic, in the order R3 R4 R5 W3 W4 W5.
  - **Harmony** = circle distance 1: Rose–Chrysanthemum, Chrysanthemum–Rhododendron, Rhododendron–Jasmine, Jasmine–Lily, Lily–Jade, Jade–Rose.
  - **Clash** = circle distance 3: Rose–Jasmine, Chrysanthemum–Lily, Rhododendron–Jade.
  - `is_harmonious(f1,f2)` and `is_clash(f1,f2)` look these up.
- `FLOWER` (L32) holds colour and move distance: Rose/Chrys/Rhodo are red 3/4/5, and Jasmine/Lily/Jade are white 3/4/5.
- `ACCENT_TILES = ['Rock','Wheel','Knotweed','Boat']`, `SPECIAL_TILES = ['Orchid','WhiteLotus']`, `SPECIAL_MOVEMENT = {'Orchid': 6, 'WhiteLotus': 2}`.
- **Setup** (`reset`, L203, via `normalize_accents`/`resolve_opening`, L54-84):
  - Each player's hand starts with `BASIC_PER_KIND = 3` of each basic flower, 1 White Lotus, 1 Orchid, and however many of each accent their `accents` choice gave them.
  - `accents=None` (the default) picks one of each accent per player. Otherwise it must be a `{player: [4 names]}` mapping — exactly `ACCENTS_PER_PLAYER = 4` tiles, at most `MAX_PER_ACCENT = 2` of any one type — or `normalize_accents` raises `ValueError`.
  - `opening='random'` (the default) draws a basic flower via the global `random` module (reproducible if you seed it); a flower name pins it; `opening=None` is the Informal Start, with no tile pre-placed.
  - When there is an opening flower, it's placed **growing** in both `GUEST_GATE (17,9)` and `HOST_GATE (1,9)`, and drawn from each player's hand.
  - **Player 1 (the Guest) always moves first.** `game.setup` reports `{'accents': {...}, 'opening': ...}` back.

## 3. Turn flow, the Harmony Bonus, and no-stranding
`get_legal_actions()` (L661) returns `[]` once `winner` is set, and is cached by (Zobrist hash, player, bonus flag, both hands).

**Normal turn** (`_iter_candidate_actions`, L675):
- `('plant', basic, gate)` for each basic flower in hand × each empty gate. The planted tile is `growing: True`.
- `('arrange', fr,fc,tr,tc)` for every owned non-accent tile, **including growing tiles in gates**, to each square in `valid_destinations`.

**Bonus turn** (`bonus_turn == True`). All of the following are offered, then filtered by no-stranding like any other action:
- plant a basic flower into an empty gate, **only if the player currently has no growing flower tiles** (rulebook p.8) — `_has_growing`, L697;
- Rock or Knotweed on any empty non-gate playable point;
- Wheel on any empty non-gate point whose full 8-tile rotation is legal (`_wheel_rotation`, L534 — see §7);
- Boat on any accent tile (removing both) or any blooming flower (yours or the opponent's), with a legal 8-neighbour displacement offered per flower (`_accent_actions`, L581, `_boat_displacements`, L565);
- Orchid or WhiteLotus into an empty gate;
- arrange any owned **non-growing, non-accent** tile.
- `('skip_bonus',)` is always offered on a bonus turn — the bonus is genuinely optional, matching the rulebook.

**How the bonus is granted** (`_end_turn`, L757): after an Arrange by a player who is *not already* in a bonus, the bonus is offered if the player's harmony pairs (relabelling the moved tile) gained at least one pair that wasn't there before the move (`_pairs_after_relocation`, L751) — i.e. any **new** harmony, matching the rulebook, not merely a rising net count. Planting does not grant a bonus. A bonus never chains into another bonus, and the player may decline it with `skip_bonus()`.

**No stranding** (`_opponent_can_move_after`, L781): every candidate action, plant/arrange/accent/bonus alike, is filtered out unless the opponent would still have *some* legal Plant or Arrange as if the turn ended right there. This check runs on every action individually, which is stricter than checking only at the actual end of turn (an engine choice — see `deviations.md`).

`plant()` itself does not check `bonus_turn` — only action generation restricts accents and specials to bonus turns, so a direct `plant()` call can bypass that restriction.

## 4. Movement (`valid_destinations`, L435)
- Accent tiles never move via Arrange.
- **Orchid trap:** a tile that is 8-adjacent to an **enemy blooming Orchid** cannot be Arranged (`_is_trapped`, L498). This applies to any flower tile type; an accent (e.g. Boat, Wheel) can still move it.
- **BFS** over orthogonal steps up to the move limit (basic: `FLOWER[f]['move']`; special: 6 for Orchid, 2 for WhiteLotus).
  - The source tile is lifted off the board during the search.
  - **A gate can be passed through but never landed on** (L471-475): an empty gate lets the search continue past it; an occupied one blocks it, same as any other occupied point.
  - Otherwise, occupied points cannot be passed through or landed on except to capture (see §5).
- **Garden rule:** a red basic flower cannot *land* on a white point, and a white one cannot land on a red point (`_garden_blocks`, L113). Passing through is allowed. Special flowers ignore gardens.
- **Clash rule:** every candidate landing is tested with `_check_clash_after_move` (L620), which rejects the move if it would produce **or uncover** a clash along the mover's new row/column or along the vacated source's row/column.
- `arrange()` (L724) validates against `valid_destinations`, removes any captured tile, sets the moved tile to `growing: False`, and records history.

## 5. Capture matrix (inside the BFS, `_can_capture`, L509)
A mover can capture only **enemy, non-growing, non-accent** tiles, and only by landing on them. The garden rule applies to the landing, and the post-move clash check must pass.

| Mover ↓ / Target → | Clashing basic flower | Other basic flower | White Lotus | Orchid | Accent |
|---|---|---|---|---|---|
| Basic flower | ✅ | ❌ | ❌ | ✅ only if the Orchid's **owner** has a blooming Lotus off-gate (vulnerable) | ❌ |
| Orchid | ✅ if the mover has a blooming Lotus off-gate (raging) | same | same | same | ❌ (Orchid captures flower tiles by Arrange, never accents — see §7 for Boat-on-accent removal, which is separate from capture) |
| White Lotus | ❌ | ❌ | ❌ | ❌ | ❌ |

A raging Orchid captures **any** opposing flower tile via Arrange. Growing tiles sit in gates, and gates are never Arranged onto, so growing tiles are never captured by movement.

**Captured tiles are gone for good.** L739 is `self.hands[cap['player']][cap['flower']] += 0`, which is intentional and matches the rulebook ("removed from the game"). Do not change it to `+= 1`.

## 6. Harmony and clash detection
`find_harmonies(player)` (L356) returns a list of `(pos, pos)` pairs. The result is cached per player by Zobrist hash.
- Candidates are the player's own **blooming basic flowers**, excluding any that are **drained** (8-adjacent to a Knotweed of either owner).
- **Rock:** a harmony is cancelled when it lies *along* a Rock's row or column, i.e. the pair shares a row that holds a Rock, or shares a column that holds a Rock (L377). A tile in a Rock's row can still harmonize along its column, and the other way round.
- A pair harmonizes if it is on the same row or column, is a harmony pair, and `_clear_line_between` finds **no tile and no gate square, even an empty one**, strictly between them (L328).
- **White Lotus pairs:** each of the player's candidate basics is paired with each blooming, non-drained White Lotus of **either owner** on a clear line. So the harmony belongs to the owner of the basic flower. Two Lotuses never harmonize with each other.

`find_clashes()` (L401) finds every clashing pair of blooming tiles, of either player, on a clear line. Rock and Knotweed do **not** affect clashes; the Rock only blocks a line physically, like any tile.

## 7. Accent tiles as coded (`_place_accent`, L601; legality in `_accent_actions`, L581)
- **Rock, Knotweed:** placed on any empty non-gate playable point. Their effects are passive (§6).
- **Wheel:** legality is decided up front by `_wheel_rotation` (L534): the target point must be empty, off gates, and not adjacent to a Rock; the resulting rotation of all 8 surrounding tiles one step clockwise (N→NE→E→SE→S→SW→W→NW→N, including accents and gate tiles) must keep every tile on the board, off gates, out of the wrong garden, and produce no clash — otherwise the Wheel simply isn't offered there at all. This matches the rulebook: a Wheel is never placed where it would be illegal.
- **Boat:** offered on any accent tile (removes both it and the Boat, leaving the point empty) or any blooming flower — the flower's owner doesn't matter. On a flower, the Boat replaces it and the flower is offered a displacement to any of its 8 legal surrounding points (on the board, not a gate, garden rule respected, no resulting clash); if a caller uses `plant()` directly without a displacement, the first legal neighbour (orthogonal before diagonal) is used automatically, matching what `step()` does for AI/replayed play.

After every accent the turn ends through `_end_turn`. Since the player was already in a bonus, no new bonus is granted.

## 8. Special flowers as coded
- They are planted only into gates, only on a bonus turn, and grow like basics.
- **White Lotus:** moves 2, ignores gardens, never clashes, and cannot be captured by basic flowers. See §6 for its harmonies.
- **Orchid:** moves 6, ignores gardens, and never clashes or harmonizes. Its trap is covered in §4, and its capture rules in §5.
  - "Blooming Lotus" is judged by the **owner's** Lotus being `not growing and not in a gate` (`_is_wild`, L493).

## 9. Game end (`_end_turn`, L757)
Checked after every plant, arrange or accent, in this order:
1. **Harmony ring:** `check_harmony_ring` (L430) for the mover, then the opponent.
   - It needs at least 4 harmonies, builds a graph of harmony edges (excluding any that pass through the center), and uses a weighted union-find over crossing-parity with a fixed ray from the center to detect a cycle enclosing it — with no cap on ring size.
   - **If both players have a ring, it's a tie** (`winner = 0`), matching the rulebook.
2. **Last basic flower:** if the player who just planted a basic flower now has 0 left in hand, the game ends at once (`_basic_count`, L838; the rulebook's own trigger). Winner is decided by `count_midline_harmonies` (L419): the number of harmonies whose two tiles sit strictly on opposite sides of row 9 or col 9 (a tile on a midline doesn't count). A tie sets `winner = 0`.
3. Otherwise, a Harmony Bonus is granted or declined, or the turn passes; if the next player then has **no legal action**, the game ends immediately (`end_reason = 'no_moves'`) and is scored the same way as Last Basic Flower.

`game.resign(player)` (L829) is a method, never an action: it sets `winner = 3 - player` and `end_reason = 'resign'` directly. It never appears in `history`.

## 10. State flags worth knowing
- `growing` is `True` only when a flower is planted. It becomes `False` on the tile's first arrange or when a Boat/Wheel moves it.
- `winner`: `None` (in progress), 1, 2, or 0 (tie). `end_reason`: `None | 'ring' | 'last_basic_flower' | 'no_moves' | 'resign'`. `message` is the human-readable status the UI shows.
- `history` is a list of `['plant', f, r, c]` / `['plant', 'Boat', r, c, dr, dc]` / `['arrange', fr, fc, tr, tc]` / `['skip_bonus']` entries and is the basis for PSN. `history_players` is the parallel list of which player (1 or 2) took each entry.
- `game.setup` is `{'accents': {1: [...], 2: [...]}, 'opening': name|None}` and is preserved across `clone()`/`from_dict()`/`from_save_dict()`.
- `random.seed(42)` runs at import, to build the Zobrist table, which reseeds Python's global `random` as a side effect. The module also does `import requests` at the top, for `current_state_web`, so the engine has a dependency on `requests`.
