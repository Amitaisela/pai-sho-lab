//! Display strings for `PaiShoGame.message`, identical to PaiShoGame.py's (the
//! parity fuzzer compares them). Built from public `Board` state only.

use pai_sho_engine::game::Board;
use pai_sho_engine::moves::{EndReason, Outcome};
use pai_sho_engine::player::Player;

use crate::convert::player_to_int;

pub fn initial_message() -> String {
    "Player 1: Plant in a Gate or Arrange a tile".to_string()
}

/// The message after an action: the end message if the game is over, the bonus
/// announcement if this action just granted a bonus, else the turn prompt.
pub fn message_for(board: &Board, bonus_just_granted: bool) -> String {
    if let Some(winner) = board.winner {
        return end_message(board, winner);
    }
    let p = player_to_int(board.current_player);
    if bonus_just_granted {
        format!(
            "Player {p}: Harmony! Bonus: place an accent, plant a special flower, plant a basic flower (if none are growing), or skip."
        )
    } else {
        format!("Player {p}: Plant in a Gate or Arrange a tile")
    }
}

fn end_message(board: &Board, winner: Outcome) -> String {
    let reason = match board.end_reason {
        Some(r) => r,
        None => return "Game over.".to_string(),
    };
    match (reason, winner) {
        (EndReason::Ring, Outcome::Tie) => "Tie: both players formed a Harmony Ring.".to_string(),
        (EndReason::Ring, Outcome::Winner(p)) => format!("Player {} wins by Harmony Ring.", player_to_int(p)),
        (EndReason::Resign, Outcome::Winner(p)) => {
            format!("Player {} resigned. Player {} wins.", player_to_int(p.other()), player_to_int(p))
        }
        (EndReason::Resign, Outcome::Tie) => "Game over.".to_string(),
        (reason, outcome) => {
            let c1 = board.count_midline_harmonies(Player::One);
            let c2 = board.count_midline_harmonies(Player::Two);
            let result = match outcome {
                Outcome::Tie => "Tie".to_string(),
                Outcome::Winner(p) => format!("Player {} wins", player_to_int(p)),
            };
            let who = player_to_int(board.current_player);
            let cause = if reason == EndReason::LastBasicFlower {
                format!("Player {who} planted their last basic flower.")
            } else {
                format!("Player {who} has no legal move.")
            };
            format!("{result} on midline harmonies. {cause} Midline-crossing harmonies \u{2014} P1: {c1}, P2: {c2}.")
        }
    }
}
