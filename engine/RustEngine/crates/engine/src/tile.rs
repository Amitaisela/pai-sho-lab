//! Tile kinds occupying a board cell: the four accent tiles and the two
//! special tiles (the six circle flowers live in the `flower` module).
//! Ported from `engine/PythonEngine/PaiShoGame.py`'s `ACCENT_TILES`,
//! `SPECIAL_TILES`, and `SPECIAL_MOVEMENT`.

use crate::flower::{Color, Flower};

/// The four accent tiles. They don't move via the orthogonal BFS movement
/// system at all (`PaiShoGame.valid_destinations` returns `[]` immediately
/// for any tile in `ACCENT_TILES`) — each instead has its own one-shot
/// placement effect (Wheel rotates neighbors, Boat displaces an enemy tile,
/// etc.), implemented in a later milestone once there's a board to act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccentTile {
    Rock,
    Wheel,
    Knotweed,
    Boat,
}

impl AccentTile {
    pub const ALL: [AccentTile; 4] = [
        AccentTile::Rock,
        AccentTile::Wheel,
        AccentTile::Knotweed,
        AccentTile::Boat,
    ];
}

/// The two special tiles. Unlike the six circle flowers, they move without
/// any garden-color restriction (`fcol = None` in
/// `PaiShoGame.valid_destinations` for special tiles).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialTile {
    Orchid,
    WhiteLotus,
}

impl SpecialTile {
    pub const ALL: [SpecialTile; 2] = [SpecialTile::Orchid, SpecialTile::WhiteLotus];

    /// How many orthogonal steps this tile can move. Ported from
    /// `PaiShoGame.SPECIAL_MOVEMENT`.
    pub fn move_range(self) -> i32 {
        match self {
            SpecialTile::Orchid => 6,
            SpecialTile::WhiteLotus => 2,
        }
    }
}

/// Whichever kind of tile occupies a board cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tile {
    Flower(Flower),
    Accent(AccentTile),
    Special(SpecialTile),
}

impl Tile {
    /// How many orthogonal steps this tile can move, or `None` if it
    /// doesn't move via the BFS system at all (accent tiles).
    pub fn move_range(self) -> Option<i32> {
        match self {
            Tile::Flower(f) => Some(f.move_range()),
            Tile::Special(s) => Some(s.move_range()),
            Tile::Accent(_) => None,
        }
    }

    /// This tile's color, if it has one that restricts movement (a flower
    /// may not move onto or be planted in a garden of the opposite color),
    /// or `None` if it has no such restriction (accent tiles don't move at
    /// all; special tiles move without a color restriction — see
    /// `SpecialTile`'s doc comment).
    pub fn color(self) -> Option<Color> {
        match self {
            Tile::Flower(f) => Some(f.color()),
            Tile::Accent(_) | Tile::Special(_) => None,
        }
    }
}

/// The four orthogonal step directions every mover's BFS expands through,
/// as `(row, col)` deltas. Ported from the
/// `for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]` loop in
/// `PaiShoGame.valid_destinations`.
pub const ORTHOGONAL_OFFSETS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accent_tile_all_lists_four_tiles_in_order() {
        assert_eq!(AccentTile::ALL.len(), 4);
        assert_eq!(AccentTile::ALL[0], AccentTile::Rock);
        assert_eq!(AccentTile::ALL[3], AccentTile::Boat);
    }

    #[test]
    fn special_tile_all_lists_two_tiles_in_order() {
        assert_eq!(SpecialTile::ALL.len(), 2);
        assert_eq!(SpecialTile::ALL[0], SpecialTile::Orchid);
        assert_eq!(SpecialTile::ALL[1], SpecialTile::WhiteLotus);
    }

    #[test]
    fn special_tile_move_ranges_match_python_oracle() {
        // Ported from PaiShoGame.SPECIAL_MOVEMENT = {'Orchid': 6, 'WhiteLotus': 2}.
        assert_eq!(SpecialTile::Orchid.move_range(), 6);
        assert_eq!(SpecialTile::WhiteLotus.move_range(), 2);
    }

    #[test]
    fn tile_flower_variant_move_range_matches_flower() {
        assert_eq!(Tile::Flower(Flower::Rose).move_range(), Some(3));
        assert_eq!(Tile::Flower(Flower::Jade).move_range(), Some(5));
    }

    #[test]
    fn tile_special_variant_move_range_matches_special() {
        assert_eq!(Tile::Special(SpecialTile::Orchid).move_range(), Some(6));
        assert_eq!(Tile::Special(SpecialTile::WhiteLotus).move_range(), Some(2));
    }

    #[test]
    fn tile_accent_variant_has_no_move_range() {
        for accent in AccentTile::ALL {
            assert_eq!(Tile::Accent(accent).move_range(), None, "{accent:?} should not move");
        }
    }

    #[test]
    fn tile_flower_variant_color_matches_flower() {
        assert_eq!(Tile::Flower(Flower::Jasmine).color(), Some(Color::White));
        assert_eq!(Tile::Flower(Flower::Rose).color(), Some(Color::Red));
    }

    #[test]
    fn tile_accent_and_special_variants_have_no_color() {
        for accent in AccentTile::ALL {
            assert_eq!(Tile::Accent(accent).color(), None, "{accent:?} should have no color restriction");
        }
        for special in SpecialTile::ALL {
            assert_eq!(Tile::Special(special).color(), None, "{special:?} should have no color restriction");
        }
    }

    #[test]
    fn orthogonal_offsets_are_the_four_cardinal_directions() {
        assert_eq!(ORTHOGONAL_OFFSETS, [(-1, 0), (1, 0), (0, -1), (0, 1)]);
    }
}
