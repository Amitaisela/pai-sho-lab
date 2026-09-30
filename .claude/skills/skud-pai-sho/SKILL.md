---
name: skud-pai-sho
description: Rules of Skud Pai Sho plus how MushiBot's engine implements them — board coordinates, gates, red/white gardens, the Circle of Harmony, harmony/clash/capture, accent and special tiles, harmony-ring win detection, PSN notation, the PaiShoGame API, and how to add an AI agent through Agents/registry.py. Use this whenever the conversation touches Pai Sho gameplay or rules (even casually, e.g. "can a rose take a jasmine?", "why can't this tile move?"), engine code in engine/PythonEngine or engine/RustEngine, move legality or win conditions, agent/evaluation/training code in Agents/, saved games or PSN, rules text in the UI (rules.html, guide.html), or explaining the game to a player — even if the user doesn't say "rules" or "skill".
---

# Skud Pai Sho (MushiBot)

Two sources of truth exist:

1. **The official rulebook** — `SkudPaiShoRules_20220314_compressed.pdf` (The Garden Gate, March 2022), distilled in `references/official-rules.md`.
2. **MushiBot's engine** — `engine/PythonEngine/PaiShoGame.py`, with `engine/RustEngine/` as a 1:1 port. This is what agents, the UI, and the simulator actually play.

As of Phase 0's official-rules pass, the engine now matches the rulebook on nearly everything. A short list of intentional engine choices remains — see `references/deviations.md` — plus a "Fixed in Phase 0" history of what changed. Don't "fix" one of the remaining differences toward the rulebook without asking: any engine change must land in both Python and Rust with parity tests.

## The game in one screen

- **Goal:** form a **Harmony Ring** — a closed chain of your harmonies that encloses the board's center point (9,9) without touching it. If both players complete one on the same move, it's a tie.
- **Setup:** each player has 3 of each basic flower, 4 chosen accents (up to 2 of any one type; `accents=None` defaults to one of each), 1 White Lotus, 1 Orchid. The Guest (Player 1) moves first. `opening='random'|<flower name>|None`: a basic flower starts growing in both gates, or the game starts informally (no tile pre-placed) when `opening=None`.
- **Board:** circular 19×19 grid, tiles sit on intersections. 4 **Gates** (entry points) on the cardinal edges — a tile may pass through an open gate but never stop on one. Central **Red** and **White Gardens**; everything else is **Neutral**.
- **Basic flowers** — named by colour + move distance: Rose R3, Chrysanthemum R4, Rhododendron R5, Jasmine W3, Lily W4, Jade W5. A basic flower may not *end* its move in the opposite-coloured garden.
- **Circle of Harmony:** R3 – R4 – R5 – W3 – W4 – W5 – (back to R3). Neighbours on the circle **harmonize**; opposites (R3↔W3, R4↔W4, R5↔W5) **clash**; everything else is neutral.
- **Harmony:** two of *your own* blooming harmonious flowers on the same row/column with nothing (no tile, no gate) between them. Rock cancels harmonies along its row/column; Knotweed drains the 8 points around it.
- **Clash:** a clashing pair lined up the same way. No move may ever produce **or uncover** a clash.
- **Turn:** either **Plant** a basic flower into an empty Gate (it is *Growing* there), or **Arrange** — move one of your flowers orthogonally up to its move distance, turning allowed, no jumping. Landing on an enemy flower it clashes with **captures** it (or, for a wild Orchid, any opposing flower); captured tiles leave the game for good. Accents and growing tiles are never captured.
- **Harmony Bonus:** an Arrange that creates a **new** harmony earns **one optional** bonus: place an accent, plant a special flower into a gate, or — only if none of your flowers are growing — plant a basic flower into a gate. You may skip it instead. A bonus never earns another bonus.
- **Wheel/Boat:** a Wheel can't be placed next to a Rock, and its 8-tile clockwise rotation must be fully legal (no off-board, no gate, no wrong garden, no clash) or it can't be placed there at all. A Boat goes on any blooming flower (which moves to an empty surrounding point) or any accent (both tiles removed), yours or the opponent's.
- **No stranding:** no move may leave the opponent with no legal move on their next turn.
- **Other endings:** a player planting their last basic flower, or having no legal move, ends the game and is scored by **midline-crossing harmonies** (ties possible); a player may also `resign()`.

## Engine essentials (what code sees)

- Coordinates are `(row, col)`, row 0 at the top, center `(9,9)`. Playable: `VALID_SPACES` (249 points, 4 of them gates; `is_valid()` alone is *not* enough — it also accepts the 4 unplayable points behind the gates).
- `GATES = [(1,9), (17,9), (9,1), (9,17)]`. `(17,9)` is the Guest's (Player 1's) gate, `(1,9)` is the Host's (Player 2's).
- Gardens: `garden_of(r,c)` — row 9 / col 9 are neutral; otherwise `|r-9|+|c-9| < 7` is **red** in the top-right & bottom-left quadrants, **white** in top-left & bottom-right; the rest is neutral. 30 red points, 30 white, 185 neutral (excluding gates).
- Tile names are exact strings: `'Rose' 'Chrysanthemum' 'Rhododendron' 'Jasmine' 'Lily' 'Jade' 'Rock' 'Wheel' 'Knotweed' 'Boat' 'Orchid' 'WhiteLotus'`.
- Board: `{(r,c): {'flower': str, 'player': 1|2, 'growing': bool}}`. Player 1 (the Guest) moves first.
- Actions (see `Agents/actions.py` for helpers instead of unpacking by fixed length):
  - `('plant', flower, r, c)` — a basic flower, special flower, Rock/Wheel/Knotweed, or a Boat placed on an accent.
  - `('plant', 'Boat', r, c, dr, dc)` — a Boat placed on a flower at `(r,c)`, which moves to `(dr, dc)`.
  - `('arrange', fr, fc, tr, tc)`.
  - `('skip_bonus',)` — decline a pending Harmony Bonus.
  - `game.resign(player)` is a method, never an action.
- `game.get_legal_actions()` → list (27 actions at a fresh `PaiShoGame()` with the default random opening: 12 plants — 6 basic flowers × the 2 still-empty gates — plus 15 arranges of the opening tile already growing in the mover's gate). `game.step(action)` → `True` if the game ended, raises `ValueError` on an illegal action or a finished game. `game.winner` is `None`/1/2/0 (0 = tie); `game.end_reason` is `None|'ring'|'last_basic_flower'|'no_moves'|'resign'`. `PaiShoGame` is mutable — **always `game.clone()` before simulating**.
- `game.setup` is `{'accents': {1: [...], 2: [...]}, 'opening': name|None}`; `game.history_players` parallels `game.history` with the acting player for each entry.
- To set up a test position, use `PaiShoGame.from_dict({...})` rather than editing `game.board` in place. Harmony results and legal actions are cached by a Zobrist hash, and a direct dict edit doesn't update the hash, so you would get stale answers.

## Which reference to read

| You need… | Read |
|---|---|
| The real rules, a rule's exact wording, rulebook page | `references/official-rules.md` |
| Exactly how the engine decides legality, harmonies, captures, accents, endings; the ASCII board/garden map | `references/engine-rules.md` |
| Whether engine and rulebook agree on X; the remaining intentional differences and Phase-0 fix history | `references/deviations.md` |
| PaiShoGame API, PSN v2 format, engine selection, registry fields, adding/training/validating an agent, tests | `references/dev-api.md` |

Line numbers cited in the references drift as code changes — if one looks wrong, grep for the function name rather than trusting the number.
