//! A tile occupying a board cell: which kind, who owns it, and whether it's
//! "growing" — planted in a gate and not yet arranged out of it (growing ⇔ on
//! a gate). Growing tiles form no harmonies or clashes and can't be captured
//! or targeted by a Boat.
//! Ported from `PaiShoGame.py`'s per-cell dict:
//! `{'flower': ..., 'player': ..., 'growing': ...}`.

use crate::player::Player;
use crate::tile::Tile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub tile: Tile,
    pub player: Player,
    pub growing: bool,
}
