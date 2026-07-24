//! The six circle flowers and their cyclic-ring relationship. Ported from
//! `engine/PythonEngine/PaiShoGame.py`'s `CIRCLE`, `_circle_distance`,
//! `_HARMONY_PAIRS`, and `_CLASH_PAIRS`.

/// The six flowers arranged around the harmony/clash ring, in ring order.
/// Order matters: it's the order Python's `CIRCLE` list uses, and
/// `circle_index` depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Flower {
    Rose,
    Chrysanthemum,
    Rhododendron,
    Jasmine,
    Lily,
    Jade,
}

/// This flower's color. A flower may not move onto or be planted in a
/// garden of the opposite color (a red flower is barred from white
/// gardens and vice versa). Ported from `FLOWER[...]['color']` in
/// `PaiShoGame.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Red,
    White,
}

impl Flower {
    pub const ALL: [Flower; 6] = [
        Flower::Rose,
        Flower::Chrysanthemum,
        Flower::Rhododendron,
        Flower::Jasmine,
        Flower::Lily,
        Flower::Jade,
    ];

    /// This flower's position on the ring (Rose = 0 .. Jade = 5).
    pub fn circle_index(self) -> i32 {
        match self {
            Flower::Rose => 0,
            Flower::Chrysanthemum => 1,
            Flower::Rhododendron => 2,
            Flower::Jasmine => 3,
            Flower::Lily => 4,
            Flower::Jade => 5,
        }
    }

    /// This flower's color — see the `Color` enum's doc comment for the
    /// garden-restriction rule it implies. Ported from
    /// `FLOWER[...]['color']`.
    pub fn color(self) -> Color {
        match self {
            Flower::Rose | Flower::Chrysanthemum | Flower::Rhododendron => Color::Red,
            Flower::Jasmine | Flower::Lily | Flower::Jade => Color::White,
        }
    }

    /// How many orthogonal steps this flower can move. Ported from
    /// `FLOWER[...]['move']`.
    pub fn move_range(self) -> i32 {
        match self {
            Flower::Rose => 3,
            Flower::Chrysanthemum => 4,
            Flower::Rhododendron => 5,
            Flower::Jasmine => 3,
            Flower::Lily => 4,
            Flower::Jade => 5,
        }
    }
}

/// Shortest distance between two ring positions, wrapping around the
/// 6-flower ring. Ported from `PaiShoGame._circle_distance`.
pub fn circle_distance(i: i32, j: i32) -> i32 {
    let d = (i - j).abs();
    d.min(6 - d)
}

/// True if `f1` and `f2` sit one step apart on the ring. Ported from
/// `PaiShoGame.is_harmonious` (via `_HARMONY_PAIRS`).
pub fn is_harmonious(f1: Flower, f2: Flower) -> bool {
    circle_distance(f1.circle_index(), f2.circle_index()) == 1
}

/// True if `f1` and `f2` sit directly opposite each other on the ring.
/// Ported from `PaiShoGame.is_clash` (via `_CLASH_PAIRS`).
pub fn is_clash(f1: Flower, f2: Flower) -> bool {
    circle_distance(f1.circle_index(), f2.circle_index()) == 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_lists_six_flowers_in_circle_order() {
        assert_eq!(Flower::ALL.len(), 6);
        assert_eq!(Flower::ALL[0], Flower::Rose);
        assert_eq!(Flower::ALL[5], Flower::Jade);
    }

    #[test]
    fn circle_index_matches_python_circle_order() {
        // Matches the index of each name in Python's CIRCLE list.
        assert_eq!(Flower::Rose.circle_index(), 0);
        assert_eq!(Flower::Chrysanthemum.circle_index(), 1);
        assert_eq!(Flower::Rhododendron.circle_index(), 2);
        assert_eq!(Flower::Jasmine.circle_index(), 3);
        assert_eq!(Flower::Lily.circle_index(), 4);
        assert_eq!(Flower::Jade.circle_index(), 5);
    }

    #[test]
    fn circle_distance_adjacent_is_one() {
        assert_eq!(circle_distance(0, 1), 1);
        assert_eq!(circle_distance(4, 5), 1);
    }

    #[test]
    fn circle_distance_wraps_around_the_ring() {
        // Rose (0) and Jade (5) are ring-adjacent, distance 1, not 5.
        assert_eq!(circle_distance(0, 5), 1);
    }

    #[test]
    fn circle_distance_opposite_is_three() {
        assert_eq!(circle_distance(0, 3), 3);
        assert_eq!(circle_distance(1, 4), 3);
    }

    #[test]
    fn circle_distance_same_index_is_zero() {
        assert_eq!(circle_distance(2, 2), 0);
    }

    #[test]
    fn harmony_pairs_match_python_oracle() {
        // The 12 ordered pairs in PaiShoGame._HARMONY_PAIRS, computed by
        // running engine/PythonEngine/PaiShoGame.py directly.
        let harmonious = [
            (Flower::Chrysanthemum, Flower::Rhododendron),
            (Flower::Chrysanthemum, Flower::Rose),
            (Flower::Jade, Flower::Lily),
            (Flower::Jade, Flower::Rose),
            (Flower::Jasmine, Flower::Lily),
            (Flower::Jasmine, Flower::Rhododendron),
            (Flower::Lily, Flower::Jade),
            (Flower::Lily, Flower::Jasmine),
            (Flower::Rhododendron, Flower::Chrysanthemum),
            (Flower::Rhododendron, Flower::Jasmine),
            (Flower::Rose, Flower::Chrysanthemum),
            (Flower::Rose, Flower::Jade),
        ];
        for (f1, f2) in harmonious {
            assert!(is_harmonious(f1, f2), "{f1:?}/{f2:?} should be harmonious");
        }
    }

    #[test]
    fn clash_pairs_match_python_oracle() {
        // The 6 ordered pairs in PaiShoGame._CLASH_PAIRS, computed by
        // running engine/PythonEngine/PaiShoGame.py directly.
        let clashing = [
            (Flower::Chrysanthemum, Flower::Lily),
            (Flower::Jade, Flower::Rhododendron),
            (Flower::Jasmine, Flower::Rose),
            (Flower::Lily, Flower::Chrysanthemum),
            (Flower::Rhododendron, Flower::Jade),
            (Flower::Rose, Flower::Jasmine),
        ];
        for (f1, f2) in clashing {
            assert!(is_clash(f1, f2), "{f1:?}/{f2:?} should clash");
        }
    }

    #[test]
    fn same_flower_is_neither_harmonious_nor_clash() {
        for f in Flower::ALL {
            assert!(!is_harmonious(f, f), "{f:?} should not be harmonious with itself");
            assert!(!is_clash(f, f), "{f:?} should not clash with itself");
        }
    }

    #[test]
    fn every_flower_has_exactly_two_harmony_partners_and_one_clash_partner() {
        for f in Flower::ALL {
            let harmony_count = Flower::ALL.iter().filter(|&&other| is_harmonious(f, other)).count();
            let clash_count = Flower::ALL.iter().filter(|&&other| is_clash(f, other)).count();
            assert_eq!(harmony_count, 2, "{f:?} should have 2 harmony partners");
            assert_eq!(clash_count, 1, "{f:?} should have 1 clash partner");
        }
    }

    #[test]
    fn flower_colors_match_python_oracle() {
        // Ported from FLOWER[...]['color'] in engine/PythonEngine/PaiShoGame.py.
        assert_eq!(Flower::Rose.color(), Color::Red);
        assert_eq!(Flower::Chrysanthemum.color(), Color::Red);
        assert_eq!(Flower::Rhododendron.color(), Color::Red);
        assert_eq!(Flower::Jasmine.color(), Color::White);
        assert_eq!(Flower::Lily.color(), Color::White);
        assert_eq!(Flower::Jade.color(), Color::White);
    }

    #[test]
    fn flower_move_ranges_match_python_oracle() {
        // Ported from FLOWER[...]['move'] in engine/PythonEngine/PaiShoGame.py.
        assert_eq!(Flower::Rose.move_range(), 3);
        assert_eq!(Flower::Chrysanthemum.move_range(), 4);
        assert_eq!(Flower::Rhododendron.move_range(), 5);
        assert_eq!(Flower::Jasmine.move_range(), 3);
        assert_eq!(Flower::Lily.move_range(), 4);
        assert_eq!(Flower::Jade.move_range(), 5);
    }
}
