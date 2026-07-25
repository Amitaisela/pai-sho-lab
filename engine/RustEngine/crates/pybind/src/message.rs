//! Synthesizes `PaiShoGame.message`-identical display strings from a
//! `Board`'s post-move state. `crates/engine` deliberately has no
//! `message: String` field (pure rules, no presentation text — see
//! Milestone 5's Global Constraints); this is where that text finally gets
//! built, entirely from public `Board` state, for `pybind`'s Python-facing
//! `.message` property.
//!
//! `Board.winner` only records *who* won (`Outcome::Winner(Player) | Tie`),
//! not *why* (harmony ring vs. the Last Basic Flower tiebreak) — Milestone 5
//! deliberately didn't carry that reason into the core engine either. This
//! module re-derives it: a harmony-ring win leaves the winning ring
//! standing (the game stops mutating once `winner` is set), so
//! `board.check_harmony_ring(winner)` is still `true` when queried
//! afterward; the Last Basic Flower path is only ever reached when neither
//! player already has a ring (`end_turn` checks rings first), so ring-check
//! `false` with `winner.is_some()` unambiguously means the tiebreak fired.

use pai_sho_engine::flower::Flower;
use pai_sho_engine::game::Board;
use pai_sho_engine::moves::Outcome;
use pai_sho_engine::player::Player;
use pai_sho_engine::tile::Tile;

use crate::convert::player_to_int;

pub fn initial_message() -> String {
    "Player 1: Plant a flower onto an open Gate".to_string()
}

/// The player (checked One before Two, matching `PaiShoGame._basic_flowers_exhausted`'s
/// iteration order) whose hand is out of basic flowers, if any.
fn basic_flowers_exhausted(board: &Board) -> Option<Player> {
    for player in [Player::One, Player::Two] {
        let hand = &board.hands[&player];
        let total: i32 = Flower::ALL.iter().map(|&f| hand.get(&Tile::Flower(f)).copied().unwrap_or(0)).sum();
        if total == 0 {
            return Some(player);
        }
    }
    None
}

fn last_basic_flower_message(board: &Board, winner: Option<Outcome>) -> String {
    let exhausted = basic_flowers_exhausted(board).expect(
        "last_basic_flower_message called on a board whose winner wasn't decided by hand exhaustion",
    );
    let c1 = board.count_midline_harmonies(Player::One);
    let c2 = board.count_midline_harmonies(Player::Two);
    let trigger = format!(
        "Player {} planted their last basic flower. Midline-crossing harmonies \u{2014} P1: {c1}, P2: {c2}.",
        player_to_int(exhausted)
    );
    match winner {
        Some(Outcome::Winner(p)) => format!("Player {} wins by Last Basic Flower rule. {trigger}", player_to_int(p)),
        Some(Outcome::Tie) => format!("Tie by Last Basic Flower rule. {trigger}"),
        None => unreachable!("last_basic_flower_message requires a decided winner"),
    }
}

/// The message to show after a move that didn't end the game: either a
/// bonus-turn announcement (if `bonus_turn` just became true) or the plain
/// "your turn" prompt. `bonus_turn_just_granted` distinguishes "just became
/// true this move" from "was already true" — only the former gets the
/// harmony-announcement text, matching `PaiShoGame._end_turn`'s branch that
/// sets the bonus-turn message only at the moment it grants one.
pub fn ongoing_message(board: &Board, bonus_turn_just_granted: bool) -> String {
    let player = player_to_int(board.current_player);
    if bonus_turn_just_granted {
        format!("Player {player}: Harmony! Bonus turn - plant, arrange, or place an accent/special tile.")
    } else {
        format!("Player {player}: Plant in a Gate or Arrange a tile")
    }
}

/// The message after a move that just ended the game (`board.winner` is
/// freshly `Some`).
pub fn terminal_message(board: &Board) -> String {
    let winner = board.winner.expect("terminal_message requires board.winner to be Some");
    match winner {
        Outcome::Winner(p) if board.check_harmony_ring(p) => {
            format!("Player {} wins by Harmony Ring rule.", player_to_int(p))
        }
        _ => last_basic_flower_message(board, Some(winner)),
    }
}
