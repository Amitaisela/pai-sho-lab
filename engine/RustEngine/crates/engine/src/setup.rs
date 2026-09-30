//! Pre-game options (rulebook p.8): each player's four accent tiles and the
//! opening basic flower. Mirrors `normalize_accents`/`resolve_opening` in
//! `engine/PythonEngine/PaiShoGame.py` ('random' is resolved by the Python
//! bridge, so the engine only ever sees a concrete flower or none).

use std::collections::HashMap;

use crate::flower::Flower;
use crate::player::Player;
use crate::tile::AccentTile;

pub const ACCENTS_PER_PLAYER: usize = 4;
pub const MAX_PER_ACCENT: usize = 2;
pub const BASIC_FLOWERS_PER_KIND: i32 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Setup {
    /// Each player's chosen accents, sorted in `AccentTile::ALL` order.
    pub accents: HashMap<Player, Vec<AccentTile>>,
    /// The Guest's opening flower (placed for both players), or `None` for an informal start.
    pub opening: Option<Flower>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupError {
    WrongAccentCount(Player),
    TooManyOfOneAccent(Player, AccentTile),
}

fn accent_order(a: &AccentTile) -> usize {
    AccentTile::ALL.iter().position(|x| x == a).expect("every accent is in ALL")
}

impl Setup {
    pub fn new(p1: Vec<AccentTile>, p2: Vec<AccentTile>, opening: Option<Flower>) -> Result<Setup, SetupError> {
        let mut accents = HashMap::new();
        for (player, mut chosen) in [(Player::One, p1), (Player::Two, p2)] {
            if chosen.len() != ACCENTS_PER_PLAYER {
                return Err(SetupError::WrongAccentCount(player));
            }
            for a in AccentTile::ALL {
                if chosen.iter().filter(|&&x| x == a).count() > MAX_PER_ACCENT {
                    return Err(SetupError::TooManyOfOneAccent(player, a));
                }
            }
            chosen.sort_by_key(accent_order);
            accents.insert(player, chosen);
        }
        Ok(Setup { accents, opening })
    }
}

impl Default for Setup {
    /// The rulebook's beginner default: one of each accent, informal start.
    fn default() -> Self {
        Setup::new(AccentTile::ALL.to_vec(), AccentTile::ALL.to_vec(), None).expect("the default setup is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_three_of_one_accent() {
        let bad = vec![AccentTile::Rock, AccentTile::Rock, AccentTile::Rock, AccentTile::Boat];
        assert_eq!(
            Setup::new(bad, AccentTile::ALL.to_vec(), None),
            Err(SetupError::TooManyOfOneAccent(Player::One, AccentTile::Rock))
        );
    }

    #[test]
    fn sorts_accents_in_canonical_order() {
        let s = Setup::new(
            vec![AccentTile::Boat, AccentTile::Rock, AccentTile::Boat, AccentTile::Rock],
            AccentTile::ALL.to_vec(),
            None,
        )
        .unwrap();
        assert_eq!(s.accents[&Player::One], vec![AccentTile::Rock, AccentTile::Rock, AccentTile::Boat, AccentTile::Boat]);
    }
}
