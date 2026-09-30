# Engine vs rulebook

As of Phase 0's official-rules pass, the engine implements the rulebook closely. This file lists only the **intentional engine choices** that remain — confirmed on purpose, not bugs — plus a short history of what Phase 0 fixed.

Any future change to one of these still has to land in **both** the Python and Rust engines, with parity tests, and shouldn't be made without asking first.

## Remaining intentional differences

| # | Topic | Official rulebook | MushiBot engine | Why |
|---|---|---|---|---|
| 1 | Seating | The Host offers the game; no rule says who moves first beyond "Guest first" (p.8) | **Player 1 is always the Guest**, and the Guest always moves first | Fixed seat assignment keeps agent code, Elo, and PSN's `Player1`/`Player2` tags unambiguous |
| 2 | Gate coordinates | Abstract "four gate points" | Fixed at `(1,9)` Host, `(17,9)` Guest, `(9,1)`, `(9,17)` | A concrete coordinate system every other piece of code (agents, UI, PSN) depends on |
| 3 | No-stranding check | Implied by "you may not leave your opponent without a legal move" | Applied to **every candidate action individually**, as if it ended the turn there — including mid-bonus actions | Stricter than checking only at actual end-of-turn, but simpler to implement correctly and never lets an illegal-by-the-rulebook stranding slip through |
| 4 | Ring parity / detection | "A connected chain of harmonies that surrounds the center point without touching it" | A weighted union-find over crossing parity against a fixed ray from the center, with no cap on ring size | An exact, cycle-detection-free way to test "encloses the center" that's cheap to run after every move |
| 5 | PSN trailing `+` | Not part of the rulebook (PSN is MushiBot's own notation) | A turn token ending in a lone trailing `+` marks a Harmony Bonus that was earned but left neither played nor skipped (only possible on the game's last recorded turn) | Distinguishes "the bonus was skipped" from "the bonus was still pending when the record was cut off", both of which are legal end states |
| 6 | Captured tiles | Removed from the game (p.9) | Removed; `+= 0` at the capture site in `arrange()` | Matches the rulebook exactly — confirmed intentional; never change to `+= 1` |

## Fixed in Phase 0

Before Phase 0, the engine had house rules that diverged from the rulebook (3 of each basic flower became 2, no accent selection, no gate pass-through, a stricter bonus trigger, and so on — all now matched, see `engine-rules.md`), plus five known bugs, all fixed in both engines:

- **B1 — Ring detection.** The old `check_harmony_ring` capped ring size at 10 nodes and could `return` out of a DFS early, missing valid rings reachable through a node's other neighbours. It's now the exact O(V+E) weighted union-find over crossing parity described in `engine-rules.md` §9, with no cap and no early-return blind spot.
- **B2 — Boat displacement not recorded.** A Boat's displacement square (chosen via the web UI) used to live only in the live board, so `history`/PSN recorded just `plant Boat@r,c` and replaying diverged from the original game. The 6-element `('plant', 'Boat', r, c, dr, dc)` action shape now carries the displacement through `history`, saves, and PSN.
- **B3 — `growing` not cleared by Wheel/Boat.** A tile moved out of a gate by a Wheel, or displaced by a Boat, used to keep `growing: True` — silently exempting it from harmonies, clashes, and capture. The new Wheel/Boat rules only ever touch **blooming** tiles and never move a tile onto or off a gate, so the `growing ⇔ on a gate` invariant now holds structurally (asserted by the parity fuzzer).
- **B4 — No "no legal moves" handling.** `get_legal_actions()` could silently return `[]` mid-game with no resolution. The engine now forbids any move that would strand the opponent (§3) and, on the rare position where a player still ends up with no legal action, ends the game with `end_reason='no_moves'`, scored the same way as Last Basic Flower.
- **B5 — Midline counting.** `count_midline_harmonies` used to count a harmony lying *on* row 9 or column 9 (e.g. tiles at (9,3) and (9,12)) as midline-crossing. Such a harmony passes through the center and now correctly doesn't count — both tiles must be strictly off both midlines.
- **Boat-on-accent clash.** Removing an accent via Boat could previously leave a clash uncovered with no check. The removal is now validated the same way as any other board change: it must not uncover a clash.
