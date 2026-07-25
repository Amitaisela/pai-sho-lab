//! Board-wide harmony and clash scans: which pairs of tiles currently form a
//! harmony or a clash, and how many of a player's harmonies cross the
//! board's midlines. Ported from `PaiShoGame.find_harmonies`,
//! `PaiShoGame.find_clashes`, and `PaiShoGame.count_midline_harmonies`.
//!
//! Unlike Python's dict-iteration order, `HashMap` iteration order here is
//! arbitrary — callers and tests that care about the returned pairs compare
//! them as an unordered collection (sort first), the same convention
//! Milestone 4's `valid_destinations` tests already use for `Vec<Position>`.

use std::collections::{HashMap, HashSet};

use crate::board::{Position, RADIUS};
use crate::flower::{is_clash, is_harmonious, Flower};
use crate::game::{clear_line_between, Board};
use crate::piece::Piece;
use crate::player::Player;
use crate::tile::{AccentTile, SpecialTile, Tile};

impl Board {
    /// All harmony pairs among `player`'s own non-growing circle flowers,
    /// plus every non-growing circle flower of `player`'s paired with any
    /// non-growing White Lotus tile on the board (either player's) it
    /// shares a row/column with. A `Rock` tile excludes every tile sharing
    /// its row or column from participating at all (owner-independent); a
    /// `Knotweed` tile excludes every tile in its 8 surrounding cells.
    /// Ported from `PaiShoGame.find_harmonies` (the `custom_board=None`,
    /// caching path is dropped — see this plan's Global Constraints).
    pub fn find_harmonies(&self, player: Player) -> Vec<(Position, Position)> {
        let mut rock_rows: HashSet<i32> = HashSet::new();
        let mut rock_cols: HashSet<i32> = HashSet::new();
        let mut drained: HashSet<Position> = HashSet::new();

        for (&pos, piece) in &self.pieces {
            match piece.tile {
                Tile::Accent(AccentTile::Rock) => {
                    rock_rows.insert(pos.row);
                    rock_cols.insert(pos.col);
                }
                Tile::Accent(AccentTile::Knotweed) => {
                    for dr in -1..=1 {
                        for dc in -1..=1 {
                            if dr != 0 || dc != 0 {
                                drained.insert(Position::new(pos.row + dr, pos.col + dc));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let rock_affected = |pos: Position| rock_rows.contains(&pos.row) || rock_cols.contains(&pos.col);

        let owned: Vec<(Position, Flower)> = self
            .pieces
            .iter()
            .filter_map(|(&pos, piece)| {
                if piece.player == player && !piece.growing && !drained.contains(&pos) && !rock_affected(pos) {
                    if let Tile::Flower(f) = piece.tile {
                        return Some((pos, f));
                    }
                }
                None
            })
            .collect();

        let mut result: Vec<(Position, Position)> = Vec::new();
        for i in 0..owned.len() {
            for j in (i + 1)..owned.len() {
                let (p1, f1) = owned[i];
                let (p2, f2) = owned[j];
                if (p1.row == p2.row || p1.col == p2.col)
                    && is_harmonious(f1, f2)
                    && clear_line_between(&self.pieces, p1, p2)
                {
                    result.push((p1, p2));
                }
            }
        }

        let wl_tiles: Vec<Position> = self
            .pieces
            .iter()
            .filter_map(|(&pos, piece)| {
                if piece.tile == Tile::Special(SpecialTile::WhiteLotus)
                    && !piece.growing
                    && !drained.contains(&pos)
                    && !rock_affected(pos)
                {
                    Some(pos)
                } else {
                    None
                }
            })
            .collect();

        for &(pos_f, _) in &owned {
            for &pos_wl in &wl_tiles {
                if (pos_f.row == pos_wl.row || pos_f.col == pos_wl.col) && clear_line_between(&self.pieces, pos_f, pos_wl) {
                    let pair = (pos_f, pos_wl);
                    let reversed = (pos_wl, pos_f);
                    if !result.contains(&pair) && !result.contains(&reversed) {
                        result.push(pair);
                    }
                }
            }
        }

        result
    }

    /// Every clash pair among *all* non-growing circle flowers on the
    /// board, regardless of owner — with no Rock/Knotweed exemption (unlike
    /// `find_harmonies`). Ported from `PaiShoGame.find_clashes`
    /// (`custom_board=None` path).
    pub fn find_clashes(&self) -> Vec<(Position, Position)> {
        find_clashes_on(&self.pieces)
    }

    /// How many of `player`'s harmonies (from `find_harmonies`) have their
    /// two tiles straddling row 9 or column 9 — a tile exactly on the
    /// midline does not count as crossing it. Ported from
    /// `PaiShoGame.count_midline_harmonies`.
    pub fn count_midline_harmonies(&self, player: Player) -> i32 {
        let mid = RADIUS;
        let mut count = 0;
        for (p1, p2) in self.find_harmonies(player) {
            let crosses_row_midline = p1.row == p2.row && p1.col.min(p2.col) < mid && mid < p1.col.max(p2.col);
            let crosses_col_midline = p1.col == p2.col && p1.row.min(p2.row) < mid && mid < p1.row.max(p2.row);
            if crosses_row_midline || crosses_col_midline {
                count += 1;
            }
        }
        count
    }

    /// True if `player` has 4 or more harmonies whose positions form a
    /// cycle enclosing the board center. Ported from
    /// `PaiShoGame.check_harmony_ring`.
    pub fn check_harmony_ring(&self, player: Player) -> bool {
        let harmonies = self.find_harmonies(player);
        if harmonies.len() < 4 {
            return false;
        }

        let mut adjacency: HashMap<Position, Vec<Position>> = HashMap::new();
        for &(p1, p2) in &harmonies {
            adjacency.entry(p1).or_default().push(p2);
            adjacency.entry(p2).or_default().push(p1);
        }

        let mut found = false;
        for &start in adjacency.keys().collect::<Vec<_>>().iter().copied() {
            if found {
                break;
            }
            let mut path = vec![start];
            let mut visited: HashSet<Position> = HashSet::from([start]);
            ring_dfs(&adjacency, start, start, &mut path, &mut visited, &mut found);
        }
        found
    }
}

/// Whole-board clash scan over an arbitrary board snapshot (not necessarily
/// the live `Board`) — Task 7 needs this to check a hypothetical
/// post-rotation board before committing to it. Ported from
/// `PaiShoGame.find_clashes`.
pub(crate) fn find_clashes_on(pieces: &HashMap<Position, Piece>) -> Vec<(Position, Position)> {
    let items: Vec<(Position, Piece)> = pieces.iter().map(|(&pos, &piece)| (pos, piece)).collect();
    let mut result = Vec::new();
    for i in 0..items.len() {
        let (p1, t1) = items[i];
        if t1.growing {
            continue;
        }
        for &(p2, t2) in items.iter().skip(i + 1) {
            if t2.growing {
                continue;
            }
            if p1.row != p2.row && p1.col != p2.col {
                continue;
            }
            if let (Tile::Flower(f1), Tile::Flower(f2)) = (t1.tile, t2.tile) {
                if is_clash(f1, f2) && clear_line_between(pieces, p1, p2) {
                    result.push((p1, p2));
                }
            }
        }
    }
    result
}

/// Even-odd ray-casting point-in-polygon test: does the closed path
/// `cycle` (each consecutive pair, including the wraparound last-to-first
/// edge, treated as a polygon edge) enclose the board center? Ported from
/// `PaiShoGame.check_harmony_ring`'s nested `enclosed` closure. Uses
/// `Position.col` as the x-axis and `Position.row` as the y-axis, matching
/// Python's `x1, y1 = cycle[j][1], cycle[j][0]`.
fn ring_encloses_center(cycle: &[Position]) -> bool {
    let cx = RADIUS as f64;
    let cy = RADIUS as f64;
    let n = cycle.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (x1, y1) = (cycle[j].col as f64, cycle[j].row as f64);
        let (x2, y2) = (cycle[i].col as f64, cycle[i].row as f64);
        if (y1 > cy) != (y2 > cy) {
            let x_intersect = x1 + (cy - y1) * (x2 - x1) / (y2 - y1);
            if cx < x_intersect {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Depth-first search for a cycle of length >= 4 back to `start`, capped at
/// path length 10 (matches Python's `len(path) > 10` guard). Every cycle
/// found is tested with `ring_encloses_center`; `found` is set and search
/// stops as soon as one qualifies. Ported from `PaiShoGame.check_harmony_ring`'s
/// nested `dfs` closure.
fn ring_dfs(
    adjacency: &HashMap<Position, Vec<Position>>,
    start: Position,
    current: Position,
    path: &mut Vec<Position>,
    visited: &mut HashSet<Position>,
    found: &mut bool,
) {
    if *found || path.len() > 10 {
        return;
    }
    let neighbors = match adjacency.get(&current) {
        Some(n) => n.clone(),
        None => return,
    };
    for neighbor in neighbors {
        if neighbor == start && path.len() >= 4 {
            if ring_encloses_center(path) {
                *found = true;
            }
            return;
        }
        if !visited.contains(&neighbor) {
            visited.insert(neighbor);
            path.push(neighbor);
            ring_dfs(adjacency, start, neighbor, path, visited, found);
            path.pop();
            visited.remove(&neighbor);
            if *found {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(tile: Tile, player: Player) -> Piece {
        Piece { tile, player, growing: false }
    }

    // Normalizes both the pair-to-pair order of a result vec AND the
    // within-pair element order before comparing. Needed because
    // `HashMap` iteration order (unlike Python's insertion-ordered dicts)
    // is randomized per-run: for pairs whose two elements both come from
    // scanning `self.pieces` in arbitrary order (the flower-flower loop in
    // `find_harmonies`, and `find_clashes_on`), which element lands first
    // in the tuple is arbitrary too, not just which pair comes first in
    // the vec. (The White-Lotus-harmony pairs are NOT affected by this —
    // that loop always emits `(flower_pos, white_lotus_pos)` — but
    // normalizing them as well is harmless since both sides of every
    // assertion go through this same helper.)
    fn sorted(pairs: Vec<(Position, Position)>) -> Vec<(Position, Position)> {
        let mut pairs: Vec<(Position, Position)> = pairs
            .into_iter()
            .map(|(a, b)| if (a.row, a.col) <= (b.row, b.col) { (a, b) } else { (b, a) })
            .collect();
        pairs.sort_by_key(|&(a, b)| (a.row, a.col, b.row, b.col));
        pairs
    }

    #[test]
    fn plain_circle_adjacent_harmony() {
        // g.board[(9,5)]={'flower':'Rose','player':1,...}; g.board[(9,9)]={'flower':'Chrysanthemum','player':1,...}
        // g.find_harmonies(1) == [((9,5),(9,9))]
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Chrysanthemum), Player::One));
        // Both tiles come from the flower-flower loop, so `HashMap` iteration
        // order can put either one first in the returned tuple — sort both
        // sides to compare as an unordered pair (see `sorted`'s doc comment).
        assert_eq!(
            sorted(board.find_harmonies(Player::One)),
            sorted(vec![(Position::new(9, 5), Position::new(9, 9))])
        );
    }

    #[test]
    fn enemy_white_lotus_still_harmonizes() {
        // g.board[(9,5)]={'flower':'Rose','player':1,...}; g.board[(9,9)]={'flower':'WhiteLotus','player':2,...}
        // g.find_harmonies(1) == [((9,5),(9,9))] — the OTHER player's blooming WL still counts.
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::WhiteLotus), Player::Two));
        assert_eq!(board.find_harmonies(Player::One), vec![(Position::new(9, 5), Position::new(9, 9))]);
    }

    #[test]
    fn rock_drains_a_same_row_flower_out_of_harmonies() {
        // g.board[(9,3)]={'flower':'Rock',...}; g.board[(9,5)]={'flower':'Rose','player':1,...};
        // g.board[(9,6)]={'flower':'WhiteLotus','player':2,...}; g.find_harmonies(1) == []
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 3), piece(Tile::Accent(AccentTile::Rock), Player::One));
        board.pieces.insert(Position::new(9, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 6), piece(Tile::Special(SpecialTile::WhiteLotus), Player::Two));
        assert_eq!(board.find_harmonies(Player::One), Vec::new());
    }

    #[test]
    fn knotweed_drains_an_adjacent_flower_out_of_harmonies() {
        // g.board[(9,5)]={'flower':'Knotweed','player':2,...}; g.board[(9,6)]={'flower':'Rose','player':1,...}
        // (adjacent to the Knotweed); g.board[(9,8)]={'flower':'WhiteLotus','player':1,...}; g.find_harmonies(1) == []
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 5), piece(Tile::Accent(AccentTile::Knotweed), Player::Two));
        board.pieces.insert(Position::new(9, 6), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 8), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert_eq!(board.find_harmonies(Player::One), Vec::new());
    }

    #[test]
    fn harmony_blocked_by_an_intervening_piece() {
        // g.board[(9,5)]={'flower':'Rose','player':1,...}; g.board[(9,7)]={'flower':'Lily','player':2,...};
        // g.board[(9,9)]={'flower':'WhiteLotus','player':1,...}; g.find_harmonies(1) == []
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 7), piece(Tile::Flower(Flower::Lily), Player::Two));
        board.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert_eq!(board.find_harmonies(Player::One), Vec::new());
    }

    #[test]
    fn rectangle_ring_shape_produces_four_harmonies_via_two_white_lotus_corners() {
        // Computed from: g.board[(5,5)]={'flower':'Rose','player':1,...}; g.board[(5,13)]={'flower':'WhiteLotus','player':1,...};
        // g.board[(13,13)]={'flower':'Jasmine','player':1,...}; g.board[(13,5)]={'flower':'WhiteLotus','player':1,...}
        // g.find_harmonies(1) == [((5,5),(5,13)), ((5,5),(13,5)), ((13,13),(5,13)), ((13,13),(13,5))]
        let mut board = Board::new();
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(5, 13), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.pieces.insert(Position::new(13, 13), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(13, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        let expected = sorted(vec![
            (Position::new(5, 5), Position::new(5, 13)),
            (Position::new(5, 5), Position::new(13, 5)),
            (Position::new(13, 13), Position::new(5, 13)),
            (Position::new(13, 13), Position::new(13, 5)),
        ]);
        assert_eq!(sorted(board.find_harmonies(Player::One)), expected);
        assert_eq!(board.count_midline_harmonies(Player::One), 4);
    }

    #[test]
    fn find_clashes_has_no_rock_exemption() {
        // g.board[(9,3)]={'flower':'Rock','player':1,...}; g.board[(9,5)]={'flower':'Rose','player':1,...};
        // g.board[(9,9)]={'flower':'Jasmine','player':2,...}; g.find_clashes() == [((9,5),(9,9))]
        // (Rock shares row 9 with both flowers, but find_clashes has no Rock/Knotweed exemption at all.)
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 3), piece(Tile::Accent(AccentTile::Rock), Player::One));
        board.pieces.insert(Position::new(9, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        // Both tiles come from `find_clashes_on`'s single arbitrary-order scan
        // of `pieces`, so `HashMap` iteration order can put either one first
        // in the returned tuple — sort both sides to compare as an unordered
        // pair (see `sorted`'s doc comment).
        assert_eq!(
            sorted(board.find_clashes()),
            sorted(vec![(Position::new(9, 5), Position::new(9, 9))])
        );
    }

    #[test]
    fn count_midline_harmonies_excludes_a_pair_on_one_side_of_the_midline() {
        // Same shape as the rectangle-ring test but shifted to sit entirely in one quadrant
        // (rows/cols 2..6, never crossing row/col 9): g.count_midline_harmonies(1) == 0
        let mut board = Board::new();
        board.pieces.insert(Position::new(2, 2), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(2, 6), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert_eq!(board.count_midline_harmonies(Player::One), 0);
    }

    #[test]
    fn rectangle_ring_via_two_white_lotus_corners_encloses_center() {
        // Same board as Task 1's rectangle_ring_shape_produces_four_harmonies_via_two_white_lotus_corners.
        // g.check_harmony_ring(1) == True
        let mut board = Board::new();
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(5, 13), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.pieces.insert(Position::new(13, 13), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(13, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(board.check_harmony_ring(Player::One));
    }

    #[test]
    fn same_shape_but_one_white_lotus_still_growing_breaks_the_ring() {
        // Same board, but the (5,13) WhiteLotus is still growing — excluded from find_harmonies,
        // so only 2 harmonies remain (< 4). g.check_harmony_ring(1) == False
        let mut board = Board::new();
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(
            Position::new(5, 13),
            Piece { tile: Tile::Special(SpecialTile::WhiteLotus), player: Player::One, growing: true },
        );
        board.pieces.insert(Position::new(13, 13), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(13, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(!board.check_harmony_ring(Player::One));
    }

    #[test]
    fn rectangle_not_enclosing_center_is_not_a_ring() {
        // Same shape, shifted entirely into one quadrant (rows/cols 2..6): 4 harmonies, but the
        // rectangle they form does not contain (9,9). g.check_harmony_ring(1) == False
        let mut board = Board::new();
        board.pieces.insert(Position::new(2, 2), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(2, 6), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.pieces.insert(Position::new(6, 6), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(6, 2), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(!board.check_harmony_ring(Player::One));
    }

    #[test]
    fn fewer_than_four_harmonies_short_circuits_to_false() {
        // Only 3 pieces on the board (2 harmonies max, both sharing the WhiteLotus corner) — never
        // reaches 4, so check_harmony_ring returns False without needing a cycle at all.
        let mut board = Board::new();
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(5, 13), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.pieces.insert(Position::new(13, 13), piece(Tile::Flower(Flower::Jasmine), Player::One));
        assert!(!board.check_harmony_ring(Player::One));
    }
}
