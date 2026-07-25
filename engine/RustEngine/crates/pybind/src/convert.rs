//! Name <-> type conversions between Python's string-based tile/flower
//! representation (`'Rose'`, `'Rock'`, `'Orchid'`, ...) and the Rust enums.
//! These strings must match `engine/PythonEngine/PaiShoGame.py`'s `CIRCLE`,
//! `ACCENT_TILES`, and `SPECIAL_TILES` literals exactly.

use pai_sho_engine::flower::Flower;
use pai_sho_engine::player::Player;
use pai_sho_engine::tile::{AccentTile, SpecialTile, Tile};

pub fn flower_name(f: Flower) -> &'static str {
    match f {
        Flower::Rose => "Rose",
        Flower::Chrysanthemum => "Chrysanthemum",
        Flower::Rhododendron => "Rhododendron",
        Flower::Jasmine => "Jasmine",
        Flower::Lily => "Lily",
        Flower::Jade => "Jade",
    }
}

pub fn accent_name(a: AccentTile) -> &'static str {
    match a {
        AccentTile::Rock => "Rock",
        AccentTile::Wheel => "Wheel",
        AccentTile::Knotweed => "Knotweed",
        AccentTile::Boat => "Boat",
    }
}

pub fn special_name(s: SpecialTile) -> &'static str {
    match s {
        SpecialTile::Orchid => "Orchid",
        SpecialTile::WhiteLotus => "WhiteLotus",
    }
}

pub fn tile_name(t: Tile) -> &'static str {
    match t {
        Tile::Flower(f) => flower_name(f),
        Tile::Accent(a) => accent_name(a),
        Tile::Special(s) => special_name(s),
    }
}

pub fn tile_from_name(name: &str) -> Option<Tile> {
    match name {
        "Rose" => Some(Tile::Flower(Flower::Rose)),
        "Chrysanthemum" => Some(Tile::Flower(Flower::Chrysanthemum)),
        "Rhododendron" => Some(Tile::Flower(Flower::Rhododendron)),
        "Jasmine" => Some(Tile::Flower(Flower::Jasmine)),
        "Lily" => Some(Tile::Flower(Flower::Lily)),
        "Jade" => Some(Tile::Flower(Flower::Jade)),
        "Rock" => Some(Tile::Accent(AccentTile::Rock)),
        "Wheel" => Some(Tile::Accent(AccentTile::Wheel)),
        "Knotweed" => Some(Tile::Accent(AccentTile::Knotweed)),
        "Boat" => Some(Tile::Accent(AccentTile::Boat)),
        "Orchid" => Some(Tile::Special(SpecialTile::Orchid)),
        "WhiteLotus" => Some(Tile::Special(SpecialTile::WhiteLotus)),
        _ => None,
    }
}

pub fn flower_from_name(name: &str) -> Option<Flower> {
    match tile_from_name(name) {
        Some(Tile::Flower(f)) => Some(f),
        _ => None,
    }
}

pub fn player_to_int(p: Player) -> i32 {
    match p {
        Player::One => 1,
        Player::Two => 2,
    }
}

pub fn player_from_int(i: i32) -> Option<Player> {
    match i {
        1 => Some(Player::One),
        2 => Some(Player::Two),
        _ => None,
    }
}
