//! Whose turn it is / who owns a piece. Ported from `PaiShoGame.py`'s use of
//! raw ints `1` and `2` for `current_player` and `tile['player']`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    One,
    Two,
}

impl Player {
    /// The other player. Ported from the `3 - player` idiom used throughout
    /// `PaiShoGame.py`.
    pub fn other(self) -> Player {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_is_the_opposite_player() {
        assert_eq!(Player::One.other(), Player::Two);
        assert_eq!(Player::Two.other(), Player::One);
    }

    #[test]
    fn other_is_its_own_inverse() {
        assert_eq!(Player::One.other().other(), Player::One);
        assert_eq!(Player::Two.other().other(), Player::Two);
    }
}
