//! Everything that mutates a `Board`: planting, arranging, and the turn-
//! ending orchestration that decides wins, ties, and bonus turns. Ported
//! from `PaiShoGame`'s `plant`/`arrange`/`step`/`_end_turn` and their
//! helpers.

use std::collections::HashSet;

use crate::board::Position;
use crate::flower::Flower;
use crate::game::{Action, Board};
use crate::piece::Piece;
use crate::player::Player;
use crate::tile::{AccentTile, Tile};

/// Why the game ended. Ported from `PaiShoGame.winner`'s three observed
/// values (`None` = ongoing, `1`/`2` = that player won, `0` = tie) — `None`
/// is represented by `Board.winner` being `Option::None`, not by a variant
/// here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Winner(Player),
    Tie,
}

/// Why a finished game ended. `None` on `Board.end_reason` while the game is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    Ring,
    LastBasicFlower,
    NoMoves,
    Resign,
}

/// Why a `plant`/`arrange`/`step` call was rejected. Ported from the
/// distinct `ValueError` messages `PaiShoGame`'s mutating methods raise —
/// pruned to `IllegalAction`/`GameOver` now that `step` validates against
/// the full `legal_actions()` set instead of checking each rule inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveError {
    GameOver,
    IllegalAction,
}

fn norm_pair(a: Position, b: Position) -> (Position, Position) {
    if (a.row, a.col) <= (b.row, b.col) {
        (a, b)
    } else {
        (b, a)
    }
}

impl Board {
    /// Applies a legal `action` and finishes the turn; returns whether the game is over.
    /// Anything not in `legal_actions()` is rejected. Ported from `PaiShoGame.step`.
    pub fn step(&mut self, action: Action) -> Result<bool, MoveError> {
        if self.winner.is_some() {
            return Err(MoveError::GameOver);
        }
        if !self.legal_actions().contains(&action) {
            return Err(MoveError::IllegalAction);
        }
        Ok(self.step_prevalidated(action))
    }

    /// `step` without the legality check, for callers that already hold this state's
    /// `legal_actions()` (the Python bridge caches them) and have confirmed `action` is
    /// in it. Anything else leaves the board in an unspecified state. Returns whether
    /// the game is over.
    pub fn step_prevalidated(&mut self, action: Action) -> bool {
        let before = match action {
            Action::Arrange { from, to } => Some(self.pairs_after_relocation(self.current_player, from, to)),
            _ => None,
        };
        self.apply_effects(action);
        self.end_turn(action, before);
        self.winner.is_some()
    }

    pub fn arrange(&mut self, from: Position, to: Position) -> Result<(), MoveError> {
        self.step(Action::Arrange { from, to }).map(|_| ())
    }

    pub fn plant(&mut self, tile: Tile, at: Position, displace: Option<Position>) -> Result<(), MoveError> {
        self.step(Action::Plant { tile, at, displace }).map(|_| ())
    }

    /// Board and hand changes for a legal `action`, no turn bookkeeping. Ported from
    /// `PaiShoGame._apply_effects`.
    pub(crate) fn apply_effects(&mut self, action: Action) {
        let player = self.current_player;
        match action {
            Action::Arrange { from, to } => {
                // Captured tiles leave the game for good (rulebook p.9).
                self.pieces.remove(&to);
                let piece = self.pieces.remove(&from).expect("a legal arrange has a mover");
                self.pieces.insert(to, Piece { growing: false, ..piece });
            }
            Action::Plant { tile: Tile::Accent(accent), at, displace } => self.place_accent(accent, at, displace),
            Action::Plant { tile, at, .. } => {
                self.pieces.insert(at, Piece { tile, player, growing: true });
                *self.hands.get_mut(&player).unwrap().get_mut(&tile).unwrap() -= 1;
            }
            Action::SkipBonus => {}
        }
    }

    fn pairs_after_relocation(&self, player: Player, from: Position, to: Position) -> HashSet<(Position, Position)> {
        let moved = |p: Position| if p == from { to } else { p };
        self.find_harmonies(player).into_iter().map(|(a, b)| norm_pair(moved(a), moved(b))).collect()
    }

    fn end_turn(&mut self, action: Action, before: Option<HashSet<(Position, Position)>>) {
        let actor = self.current_player;
        let (ring_one, ring_two) = (self.check_harmony_ring(Player::One), self.check_harmony_ring(Player::Two));
        if ring_one || ring_two {
            self.winner = Some(match (ring_one, ring_two) {
                (true, true) => Outcome::Tie,
                (true, false) => Outcome::Winner(Player::One),
                _ => Outcome::Winner(Player::Two),
            });
            self.finish(EndReason::Ring);
            return;
        }
        if let Action::Plant { tile: Tile::Flower(_), .. } = action {
            if self.basic_count(actor) == 0 {
                self.score_by_midlines();
                self.finish(EndReason::LastBasicFlower);
                return;
            }
        }
        if let (Action::Arrange { .. }, Some(before)) = (action, before.as_ref()) {
            let now: HashSet<(Position, Position)> =
                self.find_harmonies(actor).into_iter().map(|(a, b)| norm_pair(a, b)).collect();
            if !self.bonus_turn && !now.is_subset(before) {
                self.bonus_turn = true;
                return;
            }
        }
        self.bonus_turn = false;
        self.current_player = actor.other();
        if !self.has_any_legal_action() {
            self.score_by_midlines();
            self.finish(EndReason::NoMoves);
        }
    }

    pub(crate) fn basic_count(&self, player: Player) -> i32 {
        let hand = &self.hands[&player];
        Flower::ALL.iter().map(|&f| hand.get(&Tile::Flower(f)).copied().unwrap_or(0)).sum()
    }

    pub(crate) fn score_by_midlines(&mut self) {
        let (c1, c2) = (self.count_midline_harmonies(Player::One), self.count_midline_harmonies(Player::Two));
        self.winner = Some(match c1.cmp(&c2) {
            std::cmp::Ordering::Greater => Outcome::Winner(Player::One),
            std::cmp::Ordering::Less => Outcome::Winner(Player::Two),
            std::cmp::Ordering::Equal => Outcome::Tie,
        });
    }

    pub(crate) fn finish(&mut self, reason: EndReason) {
        self.end_reason = Some(reason);
        self.bonus_turn = false;
    }

    /// Forfeit (rulebook p.15): `player` resigns and the opponent wins.
    pub fn resign(&mut self, player: Player) -> Result<(), MoveError> {
        if self.winner.is_some() {
            return Err(MoveError::GameOver);
        }
        self.winner = Some(Outcome::Winner(player.other()));
        self.finish(EndReason::Resign);
        Ok(())
    }

    /// Board/hand effects of an already-validated accent placement. Ported from
    /// `PaiShoGame._place_accent`.
    pub(crate) fn place_accent(&mut self, accent: AccentTile, at: Position, displace: Option<Position>) {
        let player = self.current_player;
        match accent {
            AccentTile::Boat => {
                let target = self.pieces.remove(&at).expect("a validated Boat has a target");
                if !matches!(target.tile, Tile::Accent(_)) {
                    self.pieces.insert(displace.expect("a Boat on a flower has a displacement"), target);
                    self.pieces.insert(at, Piece { tile: Tile::Accent(AccentTile::Boat), player, growing: false });
                }
            }
            AccentTile::Wheel => {
                let moves = self.wheel_rotation(at).expect("a validated Wheel has a rotation");
                self.pieces.insert(at, Piece { tile: Tile::Accent(AccentTile::Wheel), player, growing: false });
                let moved: Vec<(Position, Piece)> = moves.iter().map(|&(src, dest)| (dest, self.pieces[&src])).collect();
                for &(src, _) in &moves {
                    self.pieces.remove(&src);
                }
                for (dest, piece) in moved {
                    self.pieces.insert(dest, piece);
                }
            }
            _ => {
                self.pieces.insert(at, Piece { tile: Tile::Accent(accent), player, growing: false });
            }
        }
        *self.hands.get_mut(&player).unwrap().get_mut(&Tile::Accent(accent)).unwrap() -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Position, GATES};
    use crate::piece::Piece;
    use crate::tile::SpecialTile;

    fn piece(tile: Tile, player: Player) -> Piece {
        Piece { tile, player, growing: false }
    }

    #[test]
    fn no_harmony_no_exhaustion_switches_player() {
        // Nothing on the board -> plain turn switch, no winner, no bonus turn.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.end_turn(Action::SkipBonus, None);
        assert_eq!(board.winner, None);
        assert!(!board.bonus_turn);
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn harmony_count_increase_grants_a_bonus_turn_without_switching_player() {
        // Board already has 1 harmony for player 1 (post-move state); `before` (the pairs prior
        // to the move) is empty -> a new pair exists -> bonus turn, current_player unchanged.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Chrysanthemum), Player::One));
        let action = Action::Arrange { from: Position::new(9, 5), to: Position::new(9, 5) };
        board.end_turn(action, Some(HashSet::new()));
        assert_eq!(board.winner, None);
        assert!(board.bonus_turn);
        assert_eq!(board.current_player, Player::One);
    }

    #[test]
    fn current_players_harmony_ring_wins_immediately() {
        // Board already reflects the rectangle-ring shape for player 1 (post-move state).
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(5, 13), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.pieces.insert(Position::new(13, 13), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(13, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.end_turn(Action::SkipBonus, None);
        assert_eq!(board.winner, Some(Outcome::Winner(Player::One)));
        assert_eq!(board.end_reason, Some(EndReason::Ring));
        assert!(!board.bonus_turn);
        assert_eq!(board.current_player, Player::One, "a win does not advance current_player");
    }

    #[test]
    fn opponents_harmony_ring_is_also_checked_and_wins() {
        // Player 2's ring is already on the board when player 1's end_turn runs (checked every
        // end_turn, regardless of whose move triggered it).
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::Two));
        board.pieces.insert(Position::new(5, 13), piece(Tile::Special(SpecialTile::WhiteLotus), Player::Two));
        board.pieces.insert(Position::new(13, 13), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        board.pieces.insert(Position::new(13, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::Two));
        board.end_turn(Action::SkipBonus, None);
        assert_eq!(board.winner, Some(Outcome::Winner(Player::Two)));
        assert_eq!(board.end_reason, Some(EndReason::Ring));
    }

    #[test]
    fn both_rings_at_once_is_a_tie() {
        // Task 8 (official rule): if both players' harmony rings are complete at the same
        // end_turn, the game ties rather than favoring whichever player is "current". Layout
        // taken from tests/fixtures/rules/ending_both_rings.json's post-move board (verified
        // against the Python oracle): player 2's ring is already up, and this is player 1's
        // ring's fourth corner sliding into place.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(3, 6), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(3, 12), piece(Tile::Flower(Flower::Chrysanthemum), Player::One));
        board.pieces.insert(Position::new(15, 12), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(15, 6), piece(Tile::Flower(Flower::Chrysanthemum), Player::One));
        board.pieces.insert(Position::new(5, 2), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        board.pieces.insert(Position::new(5, 16), piece(Tile::Flower(Flower::Lily), Player::Two));
        board.pieces.insert(Position::new(13, 16), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        board.pieces.insert(Position::new(13, 2), piece(Tile::Flower(Flower::Lily), Player::Two));
        board.end_turn(Action::SkipBonus, None);
        assert_eq!(board.winner, Some(Outcome::Tie));
        assert_eq!(board.end_reason, Some(EndReason::Ring));
    }

    #[test]
    fn exhausted_hand_triggers_the_last_basic_flower_tiebreak() {
        // Player 1's hand has zero basic flowers left (post-decrement state); player 1 has 1
        // midline-crossing harmony, player 2 has 0 -> player 1 wins by tiebreak. Task 8 (official
        // rule): this check now only runs for the plant-a-basic-flower action that emptied the
        // hand, so `end_turn` is called with such an action instead of a bare exhaustion check.
        // Task 6 (bug B5): a harmony with a tile ON a midline (row/col 9) no longer counts, so
        // this uses col 5 (off both midlines) instead of col 9.
        let mut board = Board::new();
        board.current_player = Player::One;
        for f in Flower::ALL {
            board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(f), 0);
        }
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(13, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        let action = Action::Plant { tile: Tile::Flower(Flower::Rose), at: Position::new(1, 9), displace: None };
        board.end_turn(action, None);
        assert_eq!(board.winner, Some(Outcome::Winner(Player::One)));
        assert_eq!(board.end_reason, Some(EndReason::LastBasicFlower));
        assert!(!board.bonus_turn);
        assert_eq!(board.current_player, Player::One, "the tiebreak branch does not advance current_player");
    }

    #[test]
    fn exhausted_hand_with_equal_midline_harmonies_ties() {
        let mut board = Board::new();
        board.current_player = Player::One;
        for f in Flower::ALL {
            board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(f), 0);
        }
        // No pieces at all -> both players have 0 midline-crossing harmonies -> tie.
        let action = Action::Plant { tile: Tile::Flower(Flower::Rose), at: Position::new(1, 9), displace: None };
        board.end_turn(action, None);
        assert_eq!(board.winner, Some(Outcome::Tie));
        assert_eq!(board.end_reason, Some(EndReason::LastBasicFlower));
    }

    #[test]
    fn arrange_rejects_a_tile_that_is_not_the_movers() {
        // g.board[(9,9)]={'flower':'Rose','player':2,...}; g.current_player=1
        // g.arrange(9,9,9,10) raises ValueError('Not your tile') — Task 8 collapses every
        // step() rejection reason into IllegalAction (the action just isn't in legal_actions()).
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        assert_eq!(board.arrange(Position::new(9, 9), Position::new(9, 10)), Err(MoveError::IllegalAction));
    }

    #[test]
    fn arrange_rejects_a_destination_outside_valid_destinations() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        // Rose has move range 3; (9, 20) is off the board entirely.
        assert_eq!(board.arrange(Position::new(9, 9), Position::new(9, 20)), Err(MoveError::IllegalAction));
    }

    #[test]
    fn arrange_moves_the_piece_clears_growing_and_switches_player() {
        // g.board[(9,9)]={'flower':'Rose','player':1,'growing':True}; g.current_player=1
        // g.arrange(9,9,9,10) -> board[(9,9)] gone, board[(9,10)]={'flower':'Rose','player':1,'growing':False}
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(
            Position::new(9, 9),
            Piece { tile: Tile::Flower(Flower::Rose), player: Player::One, growing: true },
        );
        assert!(board.arrange(Position::new(9, 9), Position::new(9, 10)).is_ok());
        assert!(!board.pieces.contains_key(&Position::new(9, 9)));
        let moved = board.pieces[&Position::new(9, 10)];
        assert_eq!(moved.tile, Tile::Flower(Flower::Rose));
        assert_eq!(moved.player, Player::One);
        assert!(!moved.growing);
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn arrange_capturing_an_enemy_tile_does_not_credit_its_owners_hand() {
        // g.board[(9,9)]={'flower':'Rose','player':1,...}; g.board[(9,10)]={'flower':'Jasmine','player':2,...}
        // (clash pair, capturable). g.arrange(9,9,9,10): hands[2]['Jasmine'] stays 2 (the Python
        // reference's `+= 0` is a documented no-op — see this task's context note).
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        let before = board.hands[&Player::Two][&Tile::Flower(Flower::Jasmine)];
        assert!(board.arrange(Position::new(9, 9), Position::new(9, 10)).is_ok());
        assert_eq!(board.hands[&Player::Two][&Tile::Flower(Flower::Jasmine)], before);
        assert!(!board.pieces.contains_key(&Position::new(9, 9)));
        assert_eq!(board.pieces[&Position::new(9, 10)].tile, Tile::Flower(Flower::Rose));
    }

    #[test]
    fn arrange_creating_a_new_harmony_grants_a_bonus_turn() {
        // g.board[(9,3)]={'flower':'Chrysanthemum','player':1,...}; g.board[(5,5)]={'flower':'Rose','player':1,...}
        // (not aligned, no pre-existing harmony). g.arrange(9,3,5,3) puts Chrysanthemum on Rose's
        // column, creating a fresh harmony -> bonus turn, current_player unchanged.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 3), piece(Tile::Flower(Flower::Chrysanthemum), Player::One));
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        assert!(board.arrange(Position::new(9, 3), Position::new(5, 3)).is_ok());
        assert!(board.bonus_turn);
        assert_eq!(board.current_player, Player::One);
    }

    #[test]
    fn plant_rejects_a_non_gate_position() {
        // Task 8: step() validates against legal_actions() as a whole, so every rejection
        // reason collapses to IllegalAction.
        let mut board = Board::new();
        board.current_player = Player::One;
        assert_eq!(board.plant(Tile::Flower(Flower::Rose), Position::new(9, 9), None), Err(MoveError::IllegalAction));
    }

    #[test]
    fn plant_rejects_an_occupied_gate() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(GATES[0], piece(Tile::Flower(Flower::Rose), Player::Two));
        assert_eq!(board.plant(Tile::Flower(Flower::Chrysanthemum), GATES[0], None), Err(MoveError::IllegalAction));
    }

    #[test]
    fn plant_rejects_a_tile_with_none_left_in_hand() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(Flower::Rose), 0);
        assert_eq!(board.plant(Tile::Flower(Flower::Rose), GATES[0], None), Err(MoveError::IllegalAction));
    }

    #[test]
    fn plant_a_basic_flower_into_a_gate_decrements_hand_and_switches_player() {
        // g.plant('Rose', 1, 9): board[(1,9)]={'flower':'Rose','player':1,'growing':True},
        // hands[1]['Rose'] 3->2, current_player -> 2.
        let mut board = Board::new();
        board.current_player = Player::One;
        assert!(board.plant(Tile::Flower(Flower::Rose), GATES[0], None).is_ok());
        let planted = board.pieces[&GATES[0]];
        assert_eq!(planted.tile, Tile::Flower(Flower::Rose));
        assert_eq!(planted.player, Player::One);
        assert!(planted.growing);
        assert_eq!(board.hands[&Player::One][&Tile::Flower(Flower::Rose)], 2);
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn plant_a_special_tile_uses_the_same_gate_planting_path_as_a_basic_flower() {
        // Computed from: g.plant('Orchid', 1, 9) -> board[(1,9)]={'flower':'Orchid','player':1,'growing':True},
        // hands[1]['Orchid'] 1->0, current_player -> 2 — identical shape to a basic-flower plant.
        // Task 8 (official rule): a special tile is only a Harmony Bonus option, never a plain-turn one.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        assert!(board.plant(Tile::Special(SpecialTile::Orchid), GATES[0], None).is_ok());
        let planted = board.pieces[&GATES[0]];
        assert_eq!(planted.tile, Tile::Special(SpecialTile::Orchid));
        assert!(planted.growing);
        assert_eq!(board.hands[&Player::One][&Tile::Special(SpecialTile::Orchid)], 0);
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn plant_exhausting_the_last_basic_flower_triggers_the_tiebreak_not_a_plain_switch() {
        // Computed from this task's Last Basic Flower oracle case: hands[1] has only 1 Rose left
        // and 0 of every other basic flower; a pre-existing Jasmine/WhiteLotus pair on column 5
        // (midline-crossing, adjacent quadrants; Task 6's bug B5 fix excludes col 9 itself)
        // already gives player 1 one midline harmony, player 2 zero.
        // g.plant('Rose', 1, 9) -> winner == 1 (Last Basic Flower tiebreak), not a turn switch.
        let mut board = Board::new();
        board.current_player = Player::One;
        for f in Flower::ALL {
            board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(f), 0);
        }
        board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(Flower::Rose), 1);
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(13, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(board.plant(Tile::Flower(Flower::Rose), GATES[0], None).is_ok());
        assert_eq!(board.winner, Some(Outcome::Winner(Player::One)));
        assert_eq!(board.current_player, Player::One, "the tiebreak branch does not advance current_player");
    }

    #[test]
    fn plant_accent_rejects_a_gate_position() {
        // Task 7: accent legality is now checked via `accent_actions` membership, so every
        // rejection (gate, occupied space, illegal Wheel/Boat) surfaces as IllegalAction.
        // Task 8: accents are only offered during a Harmony Bonus, so bonus_turn must be set
        // for this to fail on the gate rule specifically rather than "no accents right now".
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        assert_eq!(board.plant(Tile::Accent(AccentTile::Rock), GATES[0], None), Err(MoveError::IllegalAction));
    }

    #[test]
    fn plant_accent_rejects_an_occupied_space() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        assert_eq!(board.plant(Tile::Accent(AccentTile::Rock), Position::new(9, 9), None), Err(MoveError::IllegalAction));
    }

    #[test]
    fn plant_rock_decrements_hand_and_switches_player() {
        // Task 8 (official rule): accents are a Harmony Bonus option, so this needs bonus_turn.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        assert!(board.plant(Tile::Accent(AccentTile::Rock), Position::new(9, 9), None).is_ok());
        let planted = board.pieces[&Position::new(9, 9)];
        assert_eq!(planted.tile, Tile::Accent(AccentTile::Rock));
        assert!(!planted.growing);
        assert_eq!(board.hands[&Player::One][&Tile::Accent(AccentTile::Rock)], 0);
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn boat_may_target_its_own_players_tile() {
        // Task 7 (official rule): a Boat may target any owner's Blooming Flower or Accent, not
        // only an enemy's. Targeting an own Rock removes both, same as an enemy Rock would.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Accent(AccentTile::Rock), Player::One));
        assert!(board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None).is_ok());
        assert!(board.pieces.is_empty());
    }

    #[test]
    fn boat_rejects_targeting_an_empty_cell() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None),
            Err(MoveError::IllegalAction)
        );
    }

    #[test]
    fn boat_captures_an_enemy_accent_tile_outright_with_no_boat_tile_placed() {
        // Computed from: g.board[(9,9)]={'flower':'Rock','player':2,...}; g.plant('Boat',9,9)
        // -> board == {} (the Rock is removed; unlike the flower/special case below, no Boat
        // tile is placed at (9,9) at all for an accent-tile target).
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Accent(AccentTile::Rock), Player::Two));
        assert!(board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None).is_ok());
        assert!(board.pieces.is_empty());
        assert_eq!(board.hands[&Player::One][&Tile::Accent(AccentTile::Boat)], 0);
    }

    #[test]
    fn boat_displaces_to_an_explicit_landing_cell_when_given_one() {
        // Task 7 (official rule): a Boat only ever moves its target to one of the 8 points
        // immediately surrounding it (`boat_displacements`), not to any legal square on the
        // board — (9, 6) (three columns away) is no longer a legal landing cell, only an actual
        // neighbour like (8, 9) (north of (9, 9), row 9's own garden is neutral) is.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        assert!(board
            .plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), Some(Position::new(8, 9)))
            .is_ok());
        assert_eq!(board.pieces[&Position::new(8, 9)].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.pieces[&Position::new(9, 9)].tile, Tile::Accent(AccentTile::Boat));
    }

    #[test]
    fn boat_rejects_an_illegal_explicit_landing_cell() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        board.pieces.insert(Position::new(10, 10), piece(Tile::Flower(Flower::Jade), Player::One));
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), Some(Position::new(10, 10))),
            Err(MoveError::IllegalAction)
        );
    }

    #[test]
    fn boat_on_a_flower_with_no_legal_displacement_anywhere_is_illegal() {
        // Task 7 (official rule): a Boat on a Blooming Flower always needs a chosen, legal
        // surrounding point (a 6-element action) — there is no more "loses it entirely" fallback
        // when boxed in. Box a Rose in on all 8 sides with Rocks, so `boat_displacements` is
        // empty; no Boat action on that Rose exists at all, with or without a displacement.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
            board
                .pieces
                .insert(Position::new(9 + dr, 9 + dc), piece(Tile::Accent(AccentTile::Rock), Player::One));
        }
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None),
            Err(MoveError::IllegalAction)
        );
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), Some(Position::new(8, 8))),
            Err(MoveError::IllegalAction)
        );
    }

    #[test]
    fn wheel_rotates_two_neighbors_one_step_clockwise() {
        // Computed from: g.board[(8,9)]={'flower':'Rose','player':1,...};
        // g.board[(9,10)]={'flower':'Lily','player':2,...}; g.plant('Wheel',9,9)
        // Surrounds order for (9,9): [(8,9),(8,10),(9,10),(10,10),(10,9),(10,8),(9,8),(8,8)].
        // Rose@(8,9) is index 0 -> rotates to surrounds[1]=(8,10).
        // Lily@(9,10) is index 2 -> rotates to surrounds[3]=(10,10).
        //
        // Deviation from the brief: the brief's original fixture used Jasmine
        // (not Lily) for the second tile, computed by running the Python
        // oracle directly. That data is misleading: Jasmine is Rose's ring
        // clash partner (circle distance 3), and (8,10)/(10,10) end up
        // column-aligned with (9,10) between them post-rotation. Python's
        // `find_clashes(custom_board=...)` happens not to flag it, but only
        // because `_clear_line_between` (called from `find_clashes`) checks
        // `self.board` — the live, not-yet-mutated board, which still shows
        // (9,10) occupied by the not-yet-moved Jasmine — instead of the
        // `custom_board` parameter it was just given. That's a genuine bug in
        // the Python oracle's custom-board clash check, not a rule this port
        // should reproduce: this task's `find_clashes_on` (Task 1) is a pure
        // function of whichever board it's handed, exactly as its own doc
        // comment specifies ("Whole-board clash scan over an arbitrary board
        // snapshot ... not necessarily the live Board"), and correctly sees
        // (9,10) as empty in the simulated post-rotation snapshot. Given
        // Rose+Jasmine, the pure check here correctly finds a clash and
        // cancels the rotation — which would defeat this test's actual
        // purpose (demonstrating an ordinary two-tile rotation that
        // succeeds). Lily is not Rose's clash partner (ring distance 2, and
        // per Python: `is_clash('Rose', 'Lily') == False`), so this pair
        // rotates cleanly under both the buggy Python check and this port's
        // pure one — verified directly against the Python oracle.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(8, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Lily), Player::Two));
        assert!(board.plant(Tile::Accent(AccentTile::Wheel), Position::new(9, 9), None).is_ok());
        assert_eq!(board.pieces[&Position::new(8, 10)].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.pieces[&Position::new(10, 10)].tile, Tile::Flower(Flower::Lily));
        assert!(!board.pieces.contains_key(&Position::new(8, 9)));
        assert!(!board.pieces.contains_key(&Position::new(9, 10)));
    }

    #[test]
    fn wheel_rejected_entirely_when_it_would_create_a_clash() {
        // Computed from: g.board[(8,9)]={'flower':'Rose','player':1,...} (would rotate to
        // (8,10)); g.board[(8,15)]={'flower':'Jasmine','player':2,...} (same row as (8,10),
        // clash pair, clear line). Task 7 (official rule): `_wheel_rotation`/`wheel_rotation`
        // returning None over a would-be clash makes the whole placement illegal — it's no
        // longer a legal-but-cancelled no-op — so plant() rejects it outright and nothing moves.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(8, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(8, 15), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Wheel), Position::new(9, 9), None),
            Err(MoveError::IllegalAction)
        );
        assert_eq!(board.pieces[&Position::new(8, 9)].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.pieces[&Position::new(8, 15)].tile, Tile::Flower(Flower::Jasmine));
        assert!(!board.pieces.contains_key(&Position::new(9, 9)), "the Wheel itself must not be placed either");
    }

    #[test]
    fn wheel_with_no_occupied_neighbors_is_a_pure_placement_no_op() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        assert!(board.plant(Tile::Accent(AccentTile::Wheel), Position::new(9, 9), None).is_ok());
        assert_eq!(board.pieces.len(), 1);
        assert_eq!(board.pieces[&Position::new(9, 9)].tile, Tile::Accent(AccentTile::Wheel));
    }

    #[test]
    fn step_rejects_any_action_once_the_game_has_a_winner() {
        let mut board = Board::new();
        board.winner = Some(Outcome::Winner(Player::One));
        assert_eq!(
            board.step(Action::Plant { tile: Tile::Flower(Flower::Rose), at: GATES[0], displace: None }),
            Err(MoveError::GameOver)
        );
    }

    #[test]
    fn step_dispatches_a_plant_action_and_reports_no_winner_yet() {
        let mut board = Board::new();
        board.current_player = Player::One;
        let result = board.step(Action::Plant { tile: Tile::Flower(Flower::Rose), at: GATES[0], displace: None });
        assert_eq!(result, Ok(false));
        assert_eq!(board.pieces[&GATES[0]].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn step_dispatches_an_arrange_action_and_reports_a_winner() {
        // Reuses Task 3's current_players_harmony_ring_wins_immediately board shape, but reaches
        // the win by arranging the final White Lotus corner into place via step().
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(5, 13), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.pieces.insert(Position::new(13, 13), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(12, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        let result = board.step(Action::Arrange { from: Position::new(12, 5), to: Position::new(13, 5) });
        assert_eq!(result, Ok(true));
        assert_eq!(board.winner, Some(Outcome::Winner(Player::One)));
    }

    #[test]
    fn step_threads_the_actions_displace_field_through_to_plant() {
        // Task 7: the auto-displacement fallback is gone; Action::Plant's `displace` field (the
        // 6-element `('plant','Boat',r,c,dr,dc)` shape) is the only way to move the tile a Boat
        // targets, and step() must pass it through unchanged.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.bonus_turn = true;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        let result = board.step(Action::Plant {
            tile: Tile::Accent(AccentTile::Boat),
            at: Position::new(9, 9),
            displace: Some(Position::new(8, 9)),
        });
        assert_eq!(result, Ok(false));
        assert_eq!(board.pieces[&Position::new(8, 9)].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.pieces[&Position::new(9, 9)].tile, Tile::Accent(AccentTile::Boat));
    }
}
