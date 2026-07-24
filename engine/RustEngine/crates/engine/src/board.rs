//! Board shape: which (row, col) cells are valid, and the gate / center
//! landmarks used throughout the rest of the engine.

pub const BOARD_SIZE: i32 = 19;
pub const RADIUS: i32 = 9;

/// A board cell, addressed by (row, col). Row 0 is the top of the board,
/// matching `engine/PythonEngine/PaiShoGame.py`'s `(r, c)` convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub row: i32,
    pub col: i32,
}

impl Position {
    pub const fn new(row: i32, col: i32) -> Self {
        Position { row, col }
    }
}

pub const CENTER: Position = Position::new(RADIUS, RADIUS);

pub const GATES: [Position; 4] = [
    Position::new(1, RADIUS),
    Position::new(BOARD_SIZE - 2, RADIUS),
    Position::new(RADIUS, 1),
    Position::new(RADIUS, BOARD_SIZE - 2),
];

/// The four cells directly behind each gate. Off-board except by crossing
/// through the corresponding gate.
pub const BEHIND_GATES: [Position; 4] = [
    Position::new(0, RADIUS),
    Position::new(BOARD_SIZE - 1, RADIUS),
    Position::new(RADIUS, 0),
    Position::new(RADIUS, BOARD_SIZE - 1),
];

/// True if (row, col) lies within the circular board of radius `RADIUS`
/// centered on `CENTER`. Ported from `PaiShoGame.is_valid`.
pub fn is_valid(row: i32, col: i32) -> bool {
    let dr = row - RADIUS;
    let dc = col - RADIUS;
    dr * dr + dc * dc <= RADIUS * RADIUS
}

/// All playable cells: inside the circle and not one of the four
/// behind-gate cells. Ported from `PaiShoGame.VALID_SPACES`.
pub fn valid_spaces() -> Vec<Position> {
    let mut spaces = Vec::new();
    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            let pos = Position::new(row, col);
            if is_valid(row, col) && !BEHIND_GATES.contains(&pos) {
                spaces.push(pos);
            }
        }
    }
    spaces
}

/// True if (row, col) is a playable cell: inside the circle and not one of
/// the four behind-gate cells. The per-cell version of the predicate
/// `valid_spaces` collects into a `Vec` — use this when checking a single
/// candidate cell so as not to allocate the whole list. Ported from
/// membership in `PaiShoGame._VALID_SPACES_SET`.
pub fn is_valid_space(row: i32, col: i32) -> bool {
    is_valid(row, col) && !BEHIND_GATES.contains(&Position::new(row, col))
}

/// Which color's flowers are barred from a cell. Red/white flowers can't be
/// planted or moved onto the opposing color's garden. `Neutral` cells (both
/// board axes, and everything past the near-center diamond) accept either
/// color. Ported from `PaiShoGame.garden_of` — note the Python source has
/// extra branches past the `abs(dr) + abs(dc) < 7` check that all return
/// 'neutral' too (dead code kept there for symmetry with the four board
/// edges); they're collapsed here since they're behaviorally identical to
/// the final fallthrough.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Garden {
    Neutral,
    Red,
    White,
}

pub fn garden_of(row: i32, col: i32) -> Garden {
    let dr = row - RADIUS;
    let dc = col - RADIUS;
    if dr == 0 || dc == 0 {
        return Garden::Neutral;
    }
    if dr.abs() + dc.abs() < 7 {
        let is_red = (dr < 0) != (dc < 0);
        return if is_red { Garden::Red } else { Garden::White };
    }
    Garden::Neutral
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_size_is_19() {
        assert_eq!(BOARD_SIZE, 19);
    }

    #[test]
    fn center_is_valid() {
        assert!(is_valid(CENTER.row, CENTER.col));
    }

    #[test]
    fn corners_are_invalid() {
        for (r, c) in [(0, 0), (0, 18), (18, 0), (18, 18)] {
            assert!(!is_valid(r, c), "({r}, {c}) should be invalid");
        }
    }

    #[test]
    fn gates_are_valid_spaces() {
        let spaces = valid_spaces();
        for gate in GATES {
            assert!(spaces.contains(&gate), "{gate:?} should be a valid space");
        }
    }

    #[test]
    fn valid_spaces_count_matches_python_oracle() {
        // len(VALID_SPACES) computed by running engine/PythonEngine/PaiShoGame.py directly.
        assert_eq!(valid_spaces().len(), 249);
    }

    #[test]
    fn garden_center_is_neutral() {
        assert_eq!(garden_of(CENTER.row, CENTER.col), Garden::Neutral);
    }

    #[test]
    fn garden_axes_are_neutral() {
        // dr == 0 or dc == 0 is always neutral, regardless of distance from center.
        for (r, c) in [(1, RADIUS), (RADIUS, 1), (17, RADIUS), (RADIUS, 17)] {
            assert_eq!(garden_of(r, c), Garden::Neutral, "({r}, {c}) should be neutral");
        }
    }

    #[test]
    fn garden_colors_match_python_oracle() {
        // Computed by running engine/PythonEngine/PaiShoGame.py's garden_of() directly.
        let red = [(6, 11), (12, 7), (11, 6), (7, 12), (8, 10), (10, 8), (5, 11), (13, 7)];
        let white = [(6, 7), (12, 11), (7, 6), (11, 12), (8, 8), (10, 10), (5, 7), (13, 11)];
        for (r, c) in red {
            assert_eq!(garden_of(r, c), Garden::Red, "({r}, {c}) should be red");
        }
        for (r, c) in white {
            assert_eq!(garden_of(r, c), Garden::White, "({r}, {c}) should be white");
        }
    }

    #[test]
    fn is_valid_space_matches_is_valid_minus_behind_gates() {
        for pos in BEHIND_GATES {
            assert!(is_valid(pos.row, pos.col), "{pos:?} should be inside the circle");
            assert!(!is_valid_space(pos.row, pos.col), "{pos:?} is behind a gate, not a playable space");
        }
        assert!(is_valid_space(CENTER.row, CENTER.col));
    }
}
