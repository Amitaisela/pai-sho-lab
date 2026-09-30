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
use crate::tile::{AccentTile, SpecialTile, Tile, NEIGHBOURS_8};

impl Board {
    /// All harmony pairs among `player`'s own non-growing circle flowers,
    /// plus every non-growing circle flower of `player`'s paired with any
    /// non-growing White Lotus tile on the board (either player's) it
    /// shares a row/column with. A `Rock` tile cancels any harmony that
    /// would lie along its row or column (owner-independent); a
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
                    for (dr, dc) in NEIGHBOURS_8 {
                        drained.insert(Position::new(pos.row + dr, pos.col + dc));
                    }
                }
                _ => {}
            }
        }
        let in_harmony = |a: Position, b: Position| -> bool {
            if a.row != b.row && a.col != b.col {
                return false;
            }
            if (a.row == b.row && rock_rows.contains(&a.row)) || (a.col == b.col && rock_cols.contains(&a.col)) {
                return false;
            }
            clear_line_between(&self.pieces, a, b)
        };

        let owned: Vec<(Position, Flower)> = self
            .pieces
            .iter()
            .filter_map(|(&pos, p)| match p.tile {
                Tile::Flower(f) if p.player == player && !p.growing && !drained.contains(&pos) => Some((pos, f)),
                _ => None,
            })
            .collect();
        let mut result = Vec::new();
        for i in 0..owned.len() {
            for j in (i + 1)..owned.len() {
                let (p1, f1) = owned[i];
                let (p2, f2) = owned[j];
                if is_harmonious(f1, f2) && in_harmony(p1, p2) {
                    result.push((p1, p2));
                }
            }
        }
        let lotuses: Vec<Position> = self
            .pieces
            .iter()
            .filter(|(pos, p)| p.tile == Tile::Special(SpecialTile::WhiteLotus) && !p.growing && !drained.contains(pos))
            .map(|(&pos, _)| pos)
            .collect();
        for &(p1, _) in &owned {
            for &p2 in &lotuses {
                if in_harmony(p1, p2) {
                    result.push((p1, p2));
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
        for (a, b) in self.find_harmonies(player) {
            if [a.row, a.col, b.row, b.col].contains(&mid) {
                continue;
            }
            if (a.row < mid) != (b.row < mid) || (a.col < mid) != (b.col < mid) {
                count += 1;
            }
        }
        count
    }

    /// A chain of `player`'s harmonies surrounding the center without touching it.
    /// Ported from `PaiShoGame.check_harmony_ring`.
    pub fn check_harmony_ring(&self, player: Player) -> bool {
        let edges: Vec<(Position, Position)> =
            self.find_harmonies(player).into_iter().filter(|&(a, b)| !segment_touches_center(a, b)).collect();
        has_odd_cycle(&edges)
    }
}

/// Whole-board clash scan over an arbitrary board snapshot (not necessarily
/// the live `Board`), used to vet hypothetical boards after a Wheel rotation
/// or a Boat placement before offering them. Ported from
/// `PaiShoGame.find_clashes(custom_board=...)`.
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

fn segment_touches_center(a: Position, b: Position) -> bool {
    let m = RADIUS;
    (a.row == m && b.row == m && a.col.min(b.col) <= m && m <= a.col.max(b.col))
        || (a.col == m && b.col == m && a.row.min(b.row) <= m && m <= a.row.max(b.row))
}

/// 1 if segment a-b crosses the ray y = 9 + epsilon, x > 9. Ported from `_ray_parity`.
fn ray_parity(a: Position, b: Position) -> u8 {
    if a.col != b.col || a.col <= RADIUS {
        return 0;
    }
    u8::from(a.row.min(b.row) <= RADIUS && RADIUS < a.row.max(b.row))
}

/// Weighted union-find over GF(2) crossing parity; ported from `_has_odd_cycle`.
fn has_odd_cycle(edges: &[(Position, Position)]) -> bool {
    let mut parent: HashMap<Position, Position> = HashMap::new();
    let mut parity: HashMap<Position, u8> = HashMap::new();
    fn find(parent: &HashMap<Position, Position>, parity: &HashMap<Position, u8>, mut x: Position) -> (Position, u8) {
        let mut p = 0;
        while parent[&x] != x {
            p ^= parity[&x];
            x = parent[&x];
        }
        (x, p)
    }
    for &(a, b) in edges {
        for v in [a, b] {
            parent.entry(v).or_insert(v);
            parity.entry(v).or_insert(0);
        }
        let (ra, pa) = find(&parent, &parity, a);
        let (rb, pb) = find(&parent, &parity, b);
        let w = ray_parity(a, b);
        if ra == rb {
            if pa ^ pb ^ w == 1 {
                return true;
            }
        } else {
            parent.insert(ra, rb);
            parity.insert(ra, pa ^ pb ^ w);
        }
    }
    false
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

    #[test]
    fn square_around_center_is_a_ring_and_square_beside_it_is_not() {
        let p = Position::new;
        let around = [(p(3, 6), p(3, 12)), (p(3, 12), p(15, 12)), (p(15, 12), p(15, 6)), (p(15, 6), p(3, 6))];
        assert!(has_odd_cycle(&around));
        let beside = [(p(3, 10), p(3, 14)), (p(3, 14), p(15, 14)), (p(15, 14), p(15, 10)), (p(15, 10), p(3, 10))];
        assert!(!has_odd_cycle(&beside));
    }
}
