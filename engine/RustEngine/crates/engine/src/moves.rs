//! Everything that mutates a `Board`: planting, arranging, and the turn-
//! ending orchestration that decides wins, ties, and bonus turns. Ported
//! from `PaiShoGame`'s `plant`/`arrange`/`step`/`_end_turn` and their
//! helpers.

use std::collections::HashMap;

use crate::board::{garden_of, is_valid_space, Garden, Position, GATES};
use crate::flower::{Color, Flower};
use crate::game::{Action, Board};
use crate::harmony::find_clashes_on;
use crate::piece::Piece;
use crate::player::Player;
use crate::tile::{AccentTile, Tile};

/// The 8 neighbor directions Boat tries, in order, when the caller doesn't
/// specify a landing cell for a displaced tile. Ported from the
/// `[(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]`
/// fallback loop in `PaiShoGame._plant_accent`.
const DISPLACEMENT_OFFSETS: [(i32, i32); 8] =
    [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)];

/// Why the game ended. Ported from `PaiShoGame.winner`'s three observed
/// values (`None` = ongoing, `1`/`2` = that player won, `0` = tie) — `None`
/// is represented by `Board.winner` being `Option::None`, not by a variant
/// here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Winner(Player),
    Tie,
}

impl Board {
    /// Decides the game's fate after a move: a harmony ring for the current
    /// player wins immediately; failing that, one for the other player
    /// wins; failing that, either player being out of basic flowers
    /// triggers the Last Basic Flower tiebreak; failing that, an increase
    /// in the current player's harmony count (vs. `pre_move_harmonies`)
    /// grants a bonus turn; otherwise turn passes to the other player.
    /// Ported from `PaiShoGame._end_turn`.
    fn end_turn(&mut self, pre_move_harmonies: Option<&HashMap<Player, i32>>) {
        let current = self.current_player;
        let other = current.other();

        if self.check_harmony_ring(current) {
            self.winner = Some(Outcome::Winner(current));
            self.bonus_turn = false;
            return;
        }
        if self.check_harmony_ring(other) {
            self.winner = Some(Outcome::Winner(other));
            self.bonus_turn = false;
            return;
        }

        if self.basic_flowers_exhausted().is_some() {
            self.finish_by_midline_harmonies();
            return;
        }

        if !self.bonus_turn {
            if let Some(pre) = pre_move_harmonies {
                let current_harmonies = self.find_harmonies(current).len() as i32;
                let before = pre.get(&current).copied().unwrap_or(0);
                if current_harmonies > before {
                    self.bonus_turn = true;
                    return;
                }
            }
        }

        self.bonus_turn = false;
        self.current_player = other;
    }

    /// The first player (checked in order One, then Two) holding zero
    /// basic flowers across all six kinds in their hand, or `None` if both
    /// still have at least one. Ported from
    /// `PaiShoGame._basic_flowers_exhausted`.
    fn basic_flowers_exhausted(&self) -> Option<Player> {
        for player in [Player::One, Player::Two] {
            let hand = &self.hands[&player];
            let total: i32 = Flower::ALL.iter().map(|&f| hand.get(&Tile::Flower(f)).copied().unwrap_or(0)).sum();
            if total == 0 {
                return Some(player);
            }
        }
        None
    }

    /// The "Last Basic Flower" tiebreak: whichever player has more
    /// midline-crossing harmonies wins; equal counts (including zero-zero)
    /// tie. Ported from `PaiShoGame._finish_by_midline_harmonies`.
    fn finish_by_midline_harmonies(&mut self) {
        let p1 = self.count_midline_harmonies(Player::One);
        let p2 = self.count_midline_harmonies(Player::Two);
        self.winner = Some(match p1.cmp(&p2) {
            std::cmp::Ordering::Greater => Outcome::Winner(Player::One),
            std::cmp::Ordering::Less => Outcome::Winner(Player::Two),
            std::cmp::Ordering::Equal => Outcome::Tie,
        });
        self.bonus_turn = false;
    }
}

/// Why a `plant`/`arrange`/`step` call was rejected. Ported from the
/// distinct `ValueError` messages `PaiShoGame`'s mutating methods raise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveError {
    NotYourTile,
    InvalidMove,
    MustPlantInGate,
    GateOccupied,
    NoTilesOfThatType,
    BoatMustTargetEnemy,
    InvalidDisplacement,
    CannotPlaceAccentOnGate,
    InvalidPosition,
    SpaceOccupied,
    GameOver,
}

impl Board {
    /// Moves the piece at `from` to `to`, which must be one of `from`'s
    /// `valid_destinations` — capture eligibility, the post-move clash
    /// veto, and garden restriction are already enforced there. Clears the
    /// moved piece's `growing` flag. Ported from `PaiShoGame.arrange`.
    pub fn arrange(&mut self, from: Position, to: Position) -> Result<(), MoveError> {
        let pre_move_harmonies: HashMap<Player, i32> = [Player::One, Player::Two]
            .into_iter()
            .map(|p| (p, self.find_harmonies(p).len() as i32))
            .collect();

        let piece = match self.pieces.get(&from) {
            Some(p) if p.player == self.current_player => *p,
            _ => return Err(MoveError::NotYourTile),
        };
        if !self.valid_destinations(from).contains(&to) {
            return Err(MoveError::InvalidMove);
        }

        self.pieces.remove(&from);
        self.pieces.insert(to, Piece { tile: piece.tile, player: piece.player, growing: false });

        self.end_turn(Some(&pre_move_harmonies));
        Ok(())
    }

    /// Plants `tile` at `at`. Accent tiles (`Rock`/`Wheel`/`Knotweed`/`Boat`)
    /// delegate to `plant_accent`; every other tile (circle flowers and the
    /// two special tiles) must target an empty gate, decrements the
    /// player's hand, and is planted `growing: true`. `displace` is unused
    /// by this path — see `plant_accent`'s Boat handling (Task 6). Ported
    /// from `PaiShoGame.plant`.
    pub fn plant(&mut self, tile: Tile, at: Position, displace: Option<Position>) -> Result<(), MoveError> {
        let pre_move_harmonies: HashMap<Player, i32> = [Player::One, Player::Two]
            .into_iter()
            .map(|p| (p, self.find_harmonies(p).len() as i32))
            .collect();

        if let Tile::Accent(accent) = tile {
            return self.plant_accent(accent, at, &pre_move_harmonies, displace);
        }

        if !GATES.contains(&at) {
            return Err(MoveError::MustPlantInGate);
        }
        if self.pieces.contains_key(&at) {
            return Err(MoveError::GateOccupied);
        }
        let player = self.current_player;
        let count = self.hands.get(&player).and_then(|h| h.get(&tile)).copied().unwrap_or(0);
        if count <= 0 {
            return Err(MoveError::NoTilesOfThatType);
        }

        self.pieces.insert(at, Piece { tile, player, growing: true });
        *self.hands.get_mut(&player).unwrap().get_mut(&tile).unwrap() -= 1;

        self.end_turn(Some(&pre_move_harmonies));
        Ok(())
    }

    /// Dispatches `action` to `plant` or `arrange`. Refuses once the game
    /// already has a winner. Returns whether the game now has a winner.
    /// Boat's displacement always uses the automatic first-legal-neighbor
    /// fallback (`Action::Plant` carries no displacement field). Ported
    /// from `PaiShoGame.step`.
    pub fn step(&mut self, action: Action) -> Result<bool, MoveError> {
        if self.winner.is_some() {
            return Err(MoveError::GameOver);
        }
        match action {
            Action::Plant { tile, at } => self.plant(tile, at, None)?,
            Action::Arrange { from, to } => self.arrange(from, to)?,
        }
        Ok(self.winner.is_some())
    }

    fn plant_accent(
        &mut self,
        accent: AccentTile,
        at: Position,
        pre_move_harmonies: &HashMap<Player, i32>,
        displace: Option<Position>,
    ) -> Result<(), MoveError> {
        let player = self.current_player;

        if accent == AccentTile::Boat {
            let target = match self.pieces.get(&at) {
                Some(t) if t.player != player => *t,
                _ => return Err(MoveError::BoatMustTargetEnemy),
            };

            if let Tile::Accent(_) = target.tile {
                self.pieces.remove(&at);
            } else {
                let legal_displacement = |pieces: &HashMap<Position, Piece>, pos: Position| -> bool {
                    if !is_valid_space(pos.row, pos.col) || GATES.contains(&pos) {
                        return false;
                    }
                    if pieces.contains_key(&pos) {
                        return false;
                    }
                    if let Some(color) = target.tile.color() {
                        let garden = garden_of(pos.row, pos.col);
                        if (color == Color::Red && garden == Garden::White) || (color == Color::White && garden == Garden::Red) {
                            return false;
                        }
                    }
                    true
                };

                let landing = if let Some(explicit) = displace {
                    if !legal_displacement(&self.pieces, explicit) {
                        return Err(MoveError::InvalidDisplacement);
                    }
                    Some(explicit)
                } else {
                    DISPLACEMENT_OFFSETS
                        .into_iter()
                        .map(|(dr, dc)| Position::new(at.row + dr, at.col + dc))
                        .find(|&pos| legal_displacement(&self.pieces, pos))
                };

                if let Some(landing) = landing {
                    self.pieces.insert(landing, target);
                }
                self.pieces.remove(&at);
                self.pieces.insert(at, Piece { tile: Tile::Accent(AccentTile::Boat), player, growing: false });
            }

            *self.hands.get_mut(&player).unwrap().get_mut(&Tile::Accent(AccentTile::Boat)).unwrap() -= 1;
        } else {
            if GATES.contains(&at) {
                return Err(MoveError::CannotPlaceAccentOnGate);
            }
            if !is_valid_space(at.row, at.col) {
                return Err(MoveError::InvalidPosition);
            }
            if self.pieces.contains_key(&at) {
                return Err(MoveError::SpaceOccupied);
            }

            self.pieces.insert(at, Piece { tile: Tile::Accent(accent), player, growing: false });
            if accent == AccentTile::Wheel {
                self.apply_wheel(at);
            }
            *self.hands.get_mut(&player).unwrap().get_mut(&Tile::Accent(accent)).unwrap() -= 1;
        }

        self.end_turn(Some(pre_move_harmonies));
        Ok(())
    }

    /// Rotates every tile in the 8 cells surrounding `at` one step
    /// clockwise. All-or-nothing: cancelled entirely (board left
    /// unchanged) if any destination is off-board/a gate, violates a
    /// flower's garden-color restriction, or the resulting board would
    /// contain a clash. Ported from `PaiShoGame._apply_wheel` — see this
    /// task's context note on the two dead-code guard checks intentionally
    /// not ported.
    fn apply_wheel(&mut self, at: Position) {
        let surrounds: [Position; 8] = [
            Position::new(at.row - 1, at.col),
            Position::new(at.row - 1, at.col + 1),
            Position::new(at.row, at.col + 1),
            Position::new(at.row + 1, at.col + 1),
            Position::new(at.row + 1, at.col),
            Position::new(at.row + 1, at.col - 1),
            Position::new(at.row, at.col - 1),
            Position::new(at.row - 1, at.col - 1),
        ];

        let occupied: Vec<Position> = surrounds.iter().copied().filter(|p| self.pieces.contains_key(p)).collect();
        if occupied.is_empty() {
            return;
        }

        let index_of: HashMap<Position, usize> = surrounds.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let moves: Vec<(Position, Position)> =
            occupied.iter().map(|&src| (src, surrounds[(index_of[&src] + 1) % 8])).collect();

        for &(_, dest) in &moves {
            if !is_valid_space(dest.row, dest.col) || GATES.contains(&dest) {
                return;
            }
        }
        for &(src, dest) in &moves {
            if let Some(color) = self.pieces[&src].tile.color() {
                let garden = garden_of(dest.row, dest.col);
                if (color == Color::Red && garden == Garden::White) || (color == Color::White && garden == Garden::Red) {
                    return;
                }
            }
        }

        let moved_data: Vec<(Position, Piece)> = moves.iter().map(|&(src, dest)| (dest, self.pieces[&src])).collect();
        let mut simulated = self.pieces.clone();
        for &(src, _) in &moves {
            simulated.remove(&src);
        }
        for &(dest, piece) in &moved_data {
            simulated.insert(dest, piece);
        }
        if !find_clashes_on(&simulated).is_empty() {
            return;
        }

        for &(src, _) in &moves {
            self.pieces.remove(&src);
        }
        for (dest, piece) in moved_data {
            self.pieces.insert(dest, piece);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::piece::Piece;
    use crate::tile::SpecialTile;

    fn piece(tile: Tile, player: Player) -> Piece {
        Piece { tile, player, growing: false }
    }

    #[test]
    fn no_harmony_no_exhaustion_switches_player() {
        // Nothing on the board, no pre_move_harmonies increase possible (both start at 0)
        // -> plain turn switch, no winner, no bonus turn.
        let mut board = Board::new();
        board.current_player = Player::One;
        let pre = HashMap::from([(Player::One, 0), (Player::Two, 0)]);
        board.end_turn(Some(&pre));
        assert_eq!(board.winner, None);
        assert!(!board.bonus_turn);
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn harmony_count_increase_grants_a_bonus_turn_without_switching_player() {
        // Board already has 1 harmony for player 1 (post-move state); pre_move_harmonies says
        // player 1 had 0 before the move -> bonus turn, current_player unchanged.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Chrysanthemum), Player::One));
        let pre = HashMap::from([(Player::One, 0), (Player::Two, 0)]);
        board.end_turn(Some(&pre));
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
        board.end_turn(None);
        assert_eq!(board.winner, Some(Outcome::Winner(Player::One)));
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
        board.end_turn(None);
        assert_eq!(board.winner, Some(Outcome::Winner(Player::Two)));
    }

    #[test]
    fn exhausted_hand_triggers_the_last_basic_flower_tiebreak() {
        // Player 1's hand has zero basic flowers left (post-decrement state); player 1 has 1
        // midline-crossing harmony, player 2 has 0 -> player 1 wins by tiebreak.
        // Computed from: g.hands[1] = {all basic flowers: 0}; g.board[(5,9)]={'flower':'Jasmine','player':1,...};
        // g.board[(13,9)]={'flower':'WhiteLotus','player':1,...}; g._basic_flowers_exhausted() == 1;
        // g._finish_by_midline_harmonies(1) -> winner == 1
        let mut board = Board::new();
        board.current_player = Player::One;
        for f in Flower::ALL {
            board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(f), 0);
        }
        board.pieces.insert(Position::new(5, 9), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(13, 9), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.end_turn(None);
        assert_eq!(board.winner, Some(Outcome::Winner(Player::One)));
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
        board.end_turn(None);
        assert_eq!(board.winner, Some(Outcome::Tie));
    }

    #[test]
    fn exhaustion_takes_priority_over_bonus_turn() {
        // Player 1's hand is exhausted AND their harmony count increased vs. pre_move_harmonies —
        // exhaustion must still win (matches Python's _end_turn checking it before the bonus-turn
        // branch). No midline-crossing harmonies exist here, so this ties rather than granting bonus.
        let mut board = Board::new();
        board.current_player = Player::One;
        for f in Flower::ALL {
            board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(f), 0);
        }
        board.pieces.insert(Position::new(9, 5), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Chrysanthemum), Player::One));
        let pre = HashMap::from([(Player::One, 0), (Player::Two, 0)]);
        board.end_turn(Some(&pre));
        assert_eq!(board.winner, Some(Outcome::Tie));
        assert!(!board.bonus_turn);
    }

    #[test]
    fn arrange_rejects_a_tile_that_is_not_the_movers() {
        // g.board[(9,9)]={'flower':'Rose','player':2,...}; g.current_player=1
        // g.arrange(9,9,9,10) raises ValueError('Not your tile')
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        assert_eq!(board.arrange(Position::new(9, 9), Position::new(9, 10)), Err(MoveError::NotYourTile));
    }

    #[test]
    fn arrange_rejects_a_destination_outside_valid_destinations() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        // Rose has move range 3; (9, 20) is off the board entirely.
        assert_eq!(board.arrange(Position::new(9, 9), Position::new(9, 20)), Err(MoveError::InvalidMove));
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
        let mut board = Board::new();
        board.current_player = Player::One;
        assert_eq!(
            board.plant(Tile::Flower(Flower::Rose), Position::new(9, 9), None),
            Err(MoveError::MustPlantInGate)
        );
    }

    #[test]
    fn plant_rejects_an_occupied_gate() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(GATES[0], piece(Tile::Flower(Flower::Rose), Player::Two));
        assert_eq!(board.plant(Tile::Flower(Flower::Chrysanthemum), GATES[0], None), Err(MoveError::GateOccupied));
    }

    #[test]
    fn plant_rejects_a_tile_with_none_left_in_hand() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(Flower::Rose), 0);
        assert_eq!(board.plant(Tile::Flower(Flower::Rose), GATES[0], None), Err(MoveError::NoTilesOfThatType));
    }

    #[test]
    fn plant_a_basic_flower_into_a_gate_decrements_hand_and_switches_player() {
        // g.plant('Rose', 1, 9): board[(1,9)]={'flower':'Rose','player':1,'growing':True},
        // hands[1]['Rose'] 2->1, current_player -> 2.
        let mut board = Board::new();
        board.current_player = Player::One;
        assert!(board.plant(Tile::Flower(Flower::Rose), GATES[0], None).is_ok());
        let planted = board.pieces[&GATES[0]];
        assert_eq!(planted.tile, Tile::Flower(Flower::Rose));
        assert_eq!(planted.player, Player::One);
        assert!(planted.growing);
        assert_eq!(board.hands[&Player::One][&Tile::Flower(Flower::Rose)], 1);
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn plant_a_special_tile_uses_the_same_gate_planting_path_as_a_basic_flower() {
        // Computed from: g.plant('Orchid', 1, 9) -> board[(1,9)]={'flower':'Orchid','player':1,'growing':True},
        // hands[1]['Orchid'] 1->0, current_player -> 2 — identical shape to a basic-flower plant.
        let mut board = Board::new();
        board.current_player = Player::One;
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
        // and 0 of every other basic flower; a pre-existing Jasmine/WhiteLotus pair on column 9
        // (midline-crossing) already gives player 1 one midline harmony, player 2 zero.
        // g.plant('Rose', 1, 9) -> winner == 1 (Last Basic Flower tiebreak), not a turn switch.
        let mut board = Board::new();
        board.current_player = Player::One;
        for f in Flower::ALL {
            board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(f), 0);
        }
        board.hands.get_mut(&Player::One).unwrap().insert(Tile::Flower(Flower::Rose), 1);
        board.pieces.insert(Position::new(5, 9), piece(Tile::Flower(Flower::Jasmine), Player::One));
        board.pieces.insert(Position::new(13, 9), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(board.plant(Tile::Flower(Flower::Rose), GATES[0], None).is_ok());
        assert_eq!(board.winner, Some(Outcome::Winner(Player::One)));
        assert_eq!(board.current_player, Player::One, "the tiebreak branch does not advance current_player");
    }

    #[test]
    fn plant_accent_rejects_a_gate_position() {
        let mut board = Board::new();
        board.current_player = Player::One;
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Rock), GATES[0], None),
            Err(MoveError::CannotPlaceAccentOnGate)
        );
    }

    #[test]
    fn plant_accent_rejects_an_occupied_space() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        assert_eq!(board.plant(Tile::Accent(AccentTile::Rock), Position::new(9, 9), None), Err(MoveError::SpaceOccupied));
    }

    #[test]
    fn plant_rock_decrements_hand_and_switches_player() {
        let mut board = Board::new();
        board.current_player = Player::One;
        assert!(board.plant(Tile::Accent(AccentTile::Rock), Position::new(9, 9), None).is_ok());
        let planted = board.pieces[&Position::new(9, 9)];
        assert_eq!(planted.tile, Tile::Accent(AccentTile::Rock));
        assert!(!planted.growing);
        assert_eq!(board.hands[&Player::One][&Tile::Accent(AccentTile::Rock)], 0);
        assert_eq!(board.current_player, Player::Two);
    }

    #[test]
    fn boat_rejects_targeting_its_own_players_tile() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Accent(AccentTile::Rock), Player::One));
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None),
            Err(MoveError::BoatMustTargetEnemy)
        );
    }

    #[test]
    fn boat_rejects_targeting_an_empty_cell() {
        let mut board = Board::new();
        board.current_player = Player::One;
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None),
            Err(MoveError::BoatMustTargetEnemy)
        );
    }

    #[test]
    fn boat_captures_an_enemy_accent_tile_outright_with_no_boat_tile_placed() {
        // Computed from: g.board[(9,9)]={'flower':'Rock','player':2,...}; g.plant('Boat',9,9)
        // -> board == {} (the Rock is removed; unlike the flower/special case below, no Boat
        // tile is placed at (9,9) at all for an accent-tile target).
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Accent(AccentTile::Rock), Player::Two));
        assert!(board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None).is_ok());
        assert!(board.pieces.is_empty());
        assert_eq!(board.hands[&Player::One][&Tile::Accent(AccentTile::Boat)], 0);
    }

    #[test]
    fn boat_displaces_an_enemy_flower_to_the_first_legal_neighbor_and_takes_its_place() {
        // Computed from: g.board[(9,9)]={'flower':'Rose','player':2,...}; g.plant('Boat',9,9)
        // -> board[(8,9)]={'flower':'Rose','player':2,'growing':False} (first offset (-1,0)),
        // board[(9,9)]={'flower':'Boat','player':1,'growing':False}.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        assert!(board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None).is_ok());
        let displaced = board.pieces[&Position::new(8, 9)];
        assert_eq!(displaced.tile, Tile::Flower(Flower::Rose));
        assert_eq!(displaced.player, Player::Two);
        assert!(!displaced.growing);
        let boat = board.pieces[&Position::new(9, 9)];
        assert_eq!(boat.tile, Tile::Accent(AccentTile::Boat));
        assert_eq!(boat.player, Player::One);
    }

    #[test]
    fn boat_displaces_to_an_explicit_landing_cell_when_given_one() {
        // (9, 6) is on row 9 (dr == 0 -> neutral garden), so it's a legal landing cell for a Red
        // Rose. NOTE: the task brief's original version of this test used (10, 10) as the landing
        // cell, but (10, 10) is in the White garden (dr=1, dc=1, abs(dr)+abs(dc)=2<7, is_red =
        // (dr<0)!=(dc<0) = false -> 'white'), and a Red Rose can't legally land in a White garden.
        // Confirmed against the Python oracle directly: g.plant('Boat', 9, 9, 10, 10) raises
        // ValueError('Invalid displacement position') for this exact setup, while
        // g.plant('Boat', 9, 9, 9, 6) succeeds. Swapped to (9, 6), which the oracle accepts.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        assert!(board
            .plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), Some(Position::new(9, 6)))
            .is_ok());
        assert_eq!(board.pieces[&Position::new(9, 6)].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.pieces[&Position::new(9, 9)].tile, Tile::Accent(AccentTile::Boat));
    }

    #[test]
    fn boat_rejects_an_illegal_explicit_landing_cell() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        board.pieces.insert(Position::new(10, 10), piece(Tile::Flower(Flower::Jade), Player::One));
        assert_eq!(
            board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), Some(Position::new(10, 10))),
            Err(MoveError::InvalidDisplacement)
        );
    }

    #[test]
    fn boat_displacing_a_tile_with_no_legal_neighbor_at_all_loses_it_entirely() {
        // Computed from the oracle: box a Rose in on all 8 sides with Rock, then Boat-capture it.
        // No legal displacement exists anywhere -> the Rose is gone; only the Boat remains.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        for (dr, dc) in DISPLACEMENT_OFFSETS {
            board
                .pieces
                .insert(Position::new(9 + dr, 9 + dc), piece(Tile::Accent(AccentTile::Rock), Player::One));
        }
        assert!(board.plant(Tile::Accent(AccentTile::Boat), Position::new(9, 9), None).is_ok());
        assert!(!board.pieces.values().any(|p| p.tile == Tile::Flower(Flower::Rose)));
        assert_eq!(board.pieces[&Position::new(9, 9)].tile, Tile::Accent(AccentTile::Boat));
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
        board.pieces.insert(Position::new(8, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Lily), Player::Two));
        assert!(board.plant(Tile::Accent(AccentTile::Wheel), Position::new(9, 9), None).is_ok());
        assert_eq!(board.pieces[&Position::new(8, 10)].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.pieces[&Position::new(10, 10)].tile, Tile::Flower(Flower::Lily));
        assert!(!board.pieces.contains_key(&Position::new(8, 9)));
        assert!(!board.pieces.contains_key(&Position::new(9, 10)));
    }

    #[test]
    fn wheel_rotation_cancelled_entirely_when_it_would_create_a_clash() {
        // Computed from: g.board[(8,9)]={'flower':'Rose','player':1,...} (rotates to (8,10));
        // g.board[(8,15)]={'flower':'Jasmine','player':2,...} (same row as (8,10), clash pair,
        // clear line). g.plant('Wheel',9,9) -> Rose stays at (8,9), rotation cancelled entirely.
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(8, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(8, 15), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        assert!(board.plant(Tile::Accent(AccentTile::Wheel), Position::new(9, 9), None).is_ok());
        assert_eq!(board.pieces[&Position::new(8, 9)].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.pieces[&Position::new(8, 15)].tile, Tile::Flower(Flower::Jasmine));
        assert!(!board.pieces.contains_key(&Position::new(8, 10)));
    }

    #[test]
    fn wheel_with_no_occupied_neighbors_is_a_pure_placement_no_op() {
        let mut board = Board::new();
        board.current_player = Player::One;
        assert!(board.plant(Tile::Accent(AccentTile::Wheel), Position::new(9, 9), None).is_ok());
        assert_eq!(board.pieces.len(), 1);
        assert_eq!(board.pieces[&Position::new(9, 9)].tile, Tile::Accent(AccentTile::Wheel));
    }

    #[test]
    fn step_rejects_any_action_once_the_game_has_a_winner() {
        let mut board = Board::new();
        board.winner = Some(Outcome::Winner(Player::One));
        assert_eq!(
            board.step(Action::Plant { tile: Tile::Flower(Flower::Rose), at: GATES[0] }),
            Err(MoveError::GameOver)
        );
    }

    #[test]
    fn step_dispatches_a_plant_action_and_reports_no_winner_yet() {
        let mut board = Board::new();
        board.current_player = Player::One;
        let result = board.step(Action::Plant { tile: Tile::Flower(Flower::Rose), at: GATES[0] });
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
    fn step_boat_plant_always_uses_the_automatic_displacement_fallback() {
        let mut board = Board::new();
        board.current_player = Player::One;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::Two));
        let result = board.step(Action::Plant { tile: Tile::Accent(AccentTile::Boat), at: Position::new(9, 9) });
        assert_eq!(result, Ok(false));
        assert_eq!(board.pieces[&Position::new(8, 9)].tile, Tile::Flower(Flower::Rose));
        assert_eq!(board.pieces[&Position::new(9, 9)].tile, Tile::Accent(AccentTile::Boat));
    }
}
