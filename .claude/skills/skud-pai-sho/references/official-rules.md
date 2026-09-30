# Skud Pai Sho — official rules (distilled)

Source: *Skud Pai Sho — History and Rules of Play*, by @SkudPaiSho, rulebook version March 2022 (`SkudPaiShoRules_20220314_compressed.pdf` in the repo root). Page numbers refer to that PDF. Online play and community: SkudPaiSho.com ("The Garden Gate"), discord.gg/thegardengate. Skud Pai Sho is a fan-made game inspired by the Pai Sho board seen in *Avatar: The Last Airbender*.

This file describes the **rulebook**. MushiBot's engine differs in places — see `deviations.md`.

## Contents
1. Players & setup
2. The board
3. The tiles
4. Harmony and clashing
5. Beginning the game
6. Playing a turn (Plant / Arrange / Harmony Bonus)
7. Basic flower tiles
8. Accent tiles
9. Special flower tiles
10. Clash traps
11. Ending the game

## 1. Players & setup (p.5)
- Two players. The **Host** plays light tiles, the **Guest** plays dark tiles.
- Each player keeps unplayed tiles off-board, face up: the **Tile Reserve**.
- Traditionally the Host offers the game; players sit so a Red Garden is at the lower left of the Central Gardens.

## 2. The board (p.6)
- Tiles are played on intersections, called **points**. All visible intersections are playable.
- **Gates:** the points fully inside the four red triangles around the edge. There are exactly 4 gate points.
- **Neutral Garden:** the large brown areas around the edges.
- **Red Gardens / White Gardens:** the large red / white areas in the center. Together they are the **Central Gardens**.
- **Midlines:** the four lines separating the board's quadrants (used by the last-tile ending).

## 3. The tiles (p.6)
Per player:
- **Basic Flower Tiles, 3 of each:**
  - Rose (R3), Chrysanthemum (R4), Rhododendron (R5)
  - Jasmine (W3), Lily (W4), White Jade (W5)

  The shorthand gives colour and movement distance.
- **Accent Tiles, 2 of each:** Rock, Wheel, Knotweed, Boat. Only 4 of the 8 are used per game (see §5).
- **Special Flower Tiles, 1 of each:** White Lotus, Orchid.

"Flower Tile" means any basic or special flower.

## 4. Harmony and clashing (p.7, p.9)
- A flower tile in a **Gate** is **Growing**. A flower tile anywhere else is **Blooming**.
- **Harmony:** two harmonious **Blooming** flower tiles belonging to the **same player**, on the same line (row or column), with **no other tiles or Gates between them**.
- **Clash:** two clashing Blooming flower tiles of **either player** lined up the same way. **No tiles may ever clash.** A move that would result in a clash is illegal.
- **Circle of Harmony** (p.9): R3 – R4 – R5 – W3 – W4 – W5 – back to R3.
  - Adjacent entries harmonize.
  - Opposite entries clash: R3↔W3, R4↔W4, R5↔W5.
  - Tiles are neutral with themselves and with tiles two steps away on the circle: they can line up, but form no harmony.

## 5. Beginning the game (p.8)
1. Each player chooses **4 of their Accent Tiles** for the game and sets the other 4 aside. Traditionally the Host chooses first. Beginner tip: take one of each.
2. The **Guest** chooses a Basic Flower Tile. **Each player begins with a tile of that type in opposite Gates.**
3. Players alternate turns, **Guest first**.

Players may instead agree to an **Informal Start**: the first tile played is simply a normal turn, with no restriction on which Gate is used.

## 6. Playing a turn (p.8)
On each turn, choose **Plant** or **Arrange**.

- **Plant:** if there is an open Gate, place a Basic Flower Tile from your reserve into it.
- **Arrange:** move one of your Flower Tiles on the board along a clear path of 1 or more points, up to its maximum movement distance.
  - Moves go up, down, left and right along grid lines, never diagonally.
  - Tiles cannot move over other tiles.
  - A tile may change direction during its move.
  - A tile **may move through a Gate, but not onto it** (p.9, R4 example).
  - Casual note: if you have tiles in all the Gates, you're expected to move one out (not enforced in competitive play).

### Harmony Bonus
If Arranging forms any **new Harmony** (one between tiles that were not in Harmony at the start of the turn), you may take **one optional** Harmony Bonus action:
- place an Accent Tile from your reserve;
- Plant a Special Flower into an open Gate;
- **only if you have no Growing Flower Tiles:** Plant a Basic Flower Tile into an open Gate.

## 7. Basic flower tiles (p.9)
- They are Planted, and Arranged to form Harmonies.
- Their maximum move distance is the number in their name.
- They **cannot end movement completely inside an opposite-coloured Garden**: red flowers cannot end in a White Garden, and white flowers cannot end in a Red Garden. Passing through is allowed.
- They form Harmony with adjacent tiles on the Circle of Harmony, and Clash with the opposite-colour, same-number tile.
- **Capture:** a basic flower can capture a tile it Clashes with by landing on it. **Captured tiles are removed from the game.**
- Example (p.9): a W3 standing in the White Garden can capture an opponent's R3 by landing on it. The W3 itself is protected from capture by that R3, because the R3 cannot move into the White Garden.
- The garden rule therefore also governs captures: landing on a tile inside an opposite-coloured Garden is not allowed.

## 8. Accent tiles (p.10-11)
General rules:
- Accent tiles are played **only as a Harmony Bonus**.
- They change the board through a lasting effect or a one-time action.
- They **may not be moved when Arranging**.
- An accent **cannot**:
  - be placed in a Gate;
  - be played in a way that causes tiles to Clash;
  - be used to move a tile into an opposite-coloured Garden, into a Gate, out of a Gate, or off the board.

The four accents:
- **Rock:** played on an open point. It cancels Harmonies of either player formed **along the vertical and horizontal grid lines it lies on**, and it blocks lines like any other tile.
  - Rulebook example: a tile in the Rock's column can still harmonize horizontally, as long as there is no Rock on that row.
  - A Rock **cannot be moved by a Wheel**.
- **Knotweed:** played on an open point. It cancels Harmonies of either player formed by tiles on any of its **8 surrounding points**.
- **Wheel:** played on an open point that does **not** surround a Rock, and only where it will not illegally move a tile. When played, it rotates the tiles on all 8 surrounding points **one point clockwise** around the Wheel. It cannot be placed where the rotation would:
  - move a Rock;
  - move a basic flower into an opposite-coloured Garden;
  - move a tile out of or onto a Gate;
  - move a tile off the board;
  - cause a Clash.
- **Boat:** played on any **Accent Tile** or **Blooming Flower Tile**.
  - **On a Blooming Flower Tile:** the Boat replaces it, and the flower moves to **one of the 8 surrounding points**. That point must be legal: not in an opposite-coloured Garden, not into a Clash, not into a Gate, not off the board.
  - **On an Accent Tile:** both the Accent and the Boat are removed from the game, as captured tiles would be, and the point is left open. On p.11, playing the Boat onto a Rock in this way completes the Guest's Harmony Ring.

## 9. Special flower tiles (p.12)
Special flowers are **Planted only as a Harmony Bonus**, into an open Gate.

### White Lotus
- Moves up to **2** spaces.
- Forms Harmony with **all Basic Flower Tiles of either player**.
- A Harmony involving a White Lotus **belongs to the owner of the Basic Flower Tile**, not to the owner of the Lotus. So your opponent's Lotus lined up with your basic flower is your harmony.
- Two players' White Lotus tiles lined up together are **not** in Harmony: a Lotus only harmonizes with basic flowers.

### Orchid
- Moves up to **6** spaces.
- **Traps** the opponent's Flower Tiles on any of the **8 surrounding points**. Trapped tiles cannot be moved when Arranging, but they can still be moved by an Accent Tile.
- While its owner has a **Blooming White Lotus** on the board, the Orchid is **wild**:
  - **Vulnerable:** it can be captured by any of the opponent's Flower Tiles.
  - **Raging:** it can capture any of the opponent's Flower Tiles.
- Without a Blooming White Lotus, an Orchid can only be captured by an opponent's **wild** Orchid.

## 10. Clash traps (p.13)
A move can be illegal because it would uncover a Clash, not just create one. A tile standing between two clashing tiles cannot move away. A tile that has clashing tiles on both sides may be unable to move at all ("Clash trapped both ways").

Clashing is described as an advanced rule. Casual players may resolve a Clash noticed late on the next turn, or agree to play without the Clashing rule.

## 11. Ending the game (p.15)
The game ends in one of three ways.

1. **A Harmony Ring is formed.** It is a connected chain of Harmonies that **surrounds the center point of the board without touching it**. The player who forms it wins. If both players have rings, the game is a **tie**. A ring can have any number of tiles; 4 is the simplest.
2. **A player plants their last Basic Flower Tile** onto the board. The game ends, and the player with more **midline-crossing Harmonies** wins.
   - Midlines are the four lines separating the board's quadrants.
   - Harmonies that cross the center of the board, or that have a tile on a midline, do not count.
   - Equal counts give a **tie**.
3. **Forfeit.** You cannot win by forfeit unless you give your opponent a gift they deem worthy enough, traditionally teaware or fine tea.

## Quick reference (p.16)
- **Basic flowers:** Planted into a Gate on a turn, or as a Harmony Bonus if you have no Growing flowers. Their name is their colour and move distance. They cannot end movement in the opposite-coloured Garden. They harmonize with adjacent tiles on the circle, clash with opposite tiles, and can capture tiles they clash with.
- **Accents:** played only as a Harmony Bonus. They cannot be played in a Gate, cause a Clash, or move a tile into or out of a Gate, into an opposite Garden, or off the board.
- **Special flowers:** Planted into a Gate only as a Harmony Bonus.
