//! A tile occupying a board cell: which kind, who owns it, and whether it's
//! still "growing" — freshly planted this turn, not yet eligible to be the
//! *source* of an `arrange` move during a bonus turn (see `game::Board`).
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
