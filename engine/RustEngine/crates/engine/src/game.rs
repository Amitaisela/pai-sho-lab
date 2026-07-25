//! The live game board: which pieces are where, each player's remaining
//! hand, and whose turn it is. Ported from `PaiShoGame.py`'s `PaiShoGame`
//! class (the board/hands/turn parts only — win state and move application
//! are later milestones; see this file's module-level scope note in the
//! implementation plan).

use std::collections::{HashMap, VecDeque};

use crate::board::{garden_of, is_valid_space, valid_spaces, Garden, Position, GATES};
use crate::flower::{is_clash, Color, Flower};
use crate::moves::Outcome;
use crate::piece::Piece;
use crate::player::Player;
use crate::tile::{AccentTile, SpecialTile, Tile, ORTHOGONAL_OFFSETS};

/// A full game state: board occupancy, both hands, whose turn it is, and
/// the outcome once the game ends. `Clone` is load-bearing — search agents
/// (minimax, MCTS) clone a `Board` before simulating a candidate move,
/// mirroring `PaiShoGame.clone()`.
#[derive(Debug, Clone)]
pub struct Board {
    pub pieces: HashMap<Position, Piece>,
    pub hands: HashMap<Player, HashMap<Tile, i32>>,
    pub current_player: Player,
    pub bonus_turn: bool,
    pub winner: Option<Outcome>,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// A fresh starting board: no pieces, Player::One to move, no bonus
    /// turn, and each player's hand stocked per `PaiShoGame.reset`'s
    /// `hand = {f: 2 for f in CIRCLE}` plus one each of the four accent and
    /// two special tiles.
    pub fn new() -> Board {
        let mut hand = HashMap::new();
        for f in Flower::ALL {
            hand.insert(Tile::Flower(f), 2);
        }
        for a in AccentTile::ALL {
            hand.insert(Tile::Accent(a), 1);
        }
        for s in SpecialTile::ALL {
            hand.insert(Tile::Special(s), 1);
        }

        let mut hands = HashMap::new();
        hands.insert(Player::One, hand.clone());
        hands.insert(Player::Two, hand);

        Board {
            pieces: HashMap::new(),
            hands,
            current_player: Player::One,
            bonus_turn: false,
            winner: None,
        }
    }

    /// Resets `self` to a fresh starting board in place. Ported from
    /// `PaiShoGame.reset`.
    pub fn reset(&mut self) {
        *self = Board::new();
    }

    /// All cells the piece at `from` could move to, given only its move
    /// range, board occupancy (which blocks passage — capturing an enemy
    /// occupant is added in a later task), and garden-color restriction.
    /// Returns an empty list if there's no piece at `from`, if it's an
    /// accent tile (they don't move via this system), or if it's frozen by
    /// the Orchid trap. Ported from `PaiShoGame.valid_destinations`
    /// (partially — see this method's scope note in the implementation
    /// plan for what's not here yet).
    pub fn valid_destinations(&self, from: Position) -> Vec<Position> {
        let piece = match self.pieces.get(&from) {
            Some(p) => *p,
            None => return Vec::new(),
        };

        if let Tile::Accent(_) = piece.tile {
            return Vec::new();
        }

        // Orchid trap: frozen if adjacent to an enemy blooming Orchid,
        // unless standing on a gate.
        if !GATES.contains(&from) {
            let enemy = piece.player.other();
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let adj = Position::new(from.row + dr, from.col + dc);
                    if let Some(adj_piece) = self.pieces.get(&adj) {
                        if adj_piece.tile == Tile::Special(SpecialTile::Orchid)
                            && adj_piece.player == enemy
                            && !adj_piece.growing
                        {
                            return Vec::new();
                        }
                    }
                }
            }
        }

        let limit = match piece.tile.move_range() {
            Some(l) => l,
            None => return Vec::new(),
        };
        let color = piece.tile.color();

        let mut destinations = Vec::new();
        let mut visited: HashMap<Position, i32> = HashMap::new();
        visited.insert(from, 0);
        let mut queue: VecDeque<(Position, i32)> = VecDeque::new();
        queue.push_back((from, 0));

        while let Some((current, dist)) = queue.pop_front() {
            if dist == limit {
                continue;
            }
            for (dr, dc) in ORTHOGONAL_OFFSETS {
                let next = Position::new(current.row + dr, current.col + dc);
                if !is_valid_space(next.row, next.col) || GATES.contains(&next) {
                    continue;
                }
                let new_dist = dist + 1;
                if visited.get(&next).copied().unwrap_or(i32::MAX) <= new_dist {
                    continue;
                }
                visited.insert(next, new_dist);

                if let Some(occupant) = self.pieces.get(&next) {
                    if occupant.player != piece.player {
                        let can_capture = match piece.tile {
                            Tile::Special(SpecialTile::Orchid) => self.has_blooming_white_lotus(piece.player),
                            Tile::Special(SpecialTile::WhiteLotus) => false,
                            Tile::Flower(mover) => match occupant.tile {
                                Tile::Special(SpecialTile::Orchid) => self.has_blooming_white_lotus(occupant.player),
                                Tile::Special(SpecialTile::WhiteLotus) => false,
                                Tile::Accent(_) => false,
                                Tile::Flower(defender) => is_clash(mover, defender),
                            },
                            Tile::Accent(_) => unreachable!("accent tiles never reach the BFS body"),
                        };
                        if can_capture {
                            let garden = garden_of(next.row, next.col);
                            let garden_blocked = matches!(
                                (color, garden),
                                (Some(Color::Red), Garden::White) | (Some(Color::White), Garden::Red)
                            );
                            if !garden_blocked {
                                let mut simulated = self.pieces.clone();
                                simulated.remove(&from);
                                simulated.insert(next, Piece { tile: piece.tile, player: piece.player, growing: false });
                                if !creates_clash(&simulated, from, next) {
                                    destinations.push(next);
                                }
                            }
                        }
                    }
                    continue;
                }

                let garden = garden_of(next.row, next.col);
                let garden_blocked = matches!(
                    (color, garden),
                    (Some(Color::Red), Garden::White) | (Some(Color::White), Garden::Red)
                );
                if !garden_blocked {
                    let mut simulated = self.pieces.clone();
                    simulated.remove(&from);
                    simulated.insert(next, Piece { tile: piece.tile, player: piece.player, growing: false });
                    if !creates_clash(&simulated, from, next) {
                        destinations.push(next);
                    }
                }
                queue.push_back((next, new_dist));
            }
        }

        destinations
    }

    /// True if `player` has a White Lotus on the board that isn't still
    /// growing and isn't sitting on a gate. Ported from the
    /// `has_blooming_wl` pre-scan in `PaiShoGame.valid_destinations`.
    fn has_blooming_white_lotus(&self, player: Player) -> bool {
        self.pieces.iter().any(|(pos, p)| {
            p.tile == Tile::Special(SpecialTile::WhiteLotus)
                && p.player == player
                && !p.growing
                && !GATES.contains(pos)
        })
    }
}

/// True if every cell strictly between two same-row or same-column
/// positions is both empty and not a gate (gates block line-of-sight, same
/// as an occupied cell). If `a` and `b` share neither a row nor a column,
/// there's nothing to check and this returns `true` — matches Python's
/// `_clear_line_between`, which only has `if`/`elif` branches for the
/// aligned cases and otherwise falls through to its final `return True`.
pub(crate) fn clear_line_between(pieces: &HashMap<Position, Piece>, a: Position, b: Position) -> bool {
    if a.row == b.row {
        let (lo, hi) = (a.col.min(b.col), a.col.max(b.col));
        for c in (lo + 1)..hi {
            let p = Position::new(a.row, c);
            if pieces.contains_key(&p) || GATES.contains(&p) {
                return false;
            }
        }
    } else if a.col == b.col {
        let (lo, hi) = (a.row.min(b.row), a.row.max(b.row));
        for r in (lo + 1)..hi {
            let p = Position::new(r, a.col);
            if pieces.contains_key(&p) || GATES.contains(&p) {
                return false;
            }
        }
    }
    true
}

/// True if, on a board snapshot that already reflects a move from `from` to
/// `to`, some pair of non-growing circle flowers now forms a clash — either
/// a fresh one at `to`, or one unblocked by vacating `from`. Ported from
/// `PaiShoGame._check_clash_after_move`.
fn creates_clash(pieces: &HashMap<Position, Piece>, from: Position, to: Position) -> bool {
    if let Some(moved) = pieces.get(&to) {
        if !moved.growing {
            if let Tile::Flower(mover) = moved.tile {
                for (&pos, p) in pieces.iter() {
                    if pos == to || p.growing {
                        continue;
                    }
                    if let Tile::Flower(other) = p.tile {
                        if (pos.row == to.row || pos.col == to.col)
                            && is_clash(mover, other)
                            && clear_line_between(pieces, to, pos)
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }

    let mut row_cands = Vec::new();
    let mut col_cands = Vec::new();
    for (&pos, p) in pieces.iter() {
        if p.growing {
            continue;
        }
        if let Tile::Flower(f) = p.tile {
            if pos.row == from.row {
                row_cands.push((pos, f));
            }
            if pos.col == from.col {
                col_cands.push((pos, f));
            }
        }
    }
    for cands in [&row_cands, &col_cands] {
        for i in 0..cands.len() {
            for j in (i + 1)..cands.len() {
                let (p1, f1) = cands[i];
                let (p2, f2) = cands[j];
                if is_clash(f1, f2) && clear_line_between(pieces, p1, p2) {
                    return true;
                }
            }
        }
    }
    false
}

/// A single plant-a-tile or move-a-tile action. Ported from
/// `PaiShoGame`'s `('plant', flower, r, c)` / `('arrange', fr, fc, tr, tc)`
/// action tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Plant { tile: Tile, at: Position },
    Arrange { from: Position, to: Position },
}

impl Board {
    /// Every legal action the current player can take: planting a basic
    /// flower into an empty gate; during a bonus turn, also planting an
    /// accent or special tile (Boat targets an enemy-occupied cell instead
    /// of an empty one); and arranging any owned non-accent tile (skipping
    /// still-growing ones during a bonus turn) to any of its
    /// `valid_destinations`. Empty once the game has a winner. Ported from
    /// `PaiShoGame.get_legal_actions` — Milestone 4 deliberately omitted the
    /// `if self.winner is not None: return []` guard because `Board` had no
    /// `winner` field yet at the time; Milestone 5 added the field, and this
    /// wires the guard back in.
    pub fn legal_actions(&self) -> Vec<Action> {
        if self.winner.is_some() {
            return Vec::new();
        }

        let player = self.current_player;
        let mut actions = Vec::new();

        let empty_gates: Vec<Position> = GATES.into_iter().filter(|g| !self.pieces.contains_key(g)).collect();
        if !empty_gates.is_empty() {
            if let Some(hand) = self.hands.get(&player) {
                for flower in Flower::ALL {
                    if hand.get(&Tile::Flower(flower)).copied().unwrap_or(0) > 0 {
                        for &gate in &empty_gates {
                            actions.push(Action::Plant { tile: Tile::Flower(flower), at: gate });
                        }
                    }
                }
            }
        }

        if self.bonus_turn {
            if let Some(hand) = self.hands.get(&player) {
                for accent in AccentTile::ALL {
                    if hand.get(&Tile::Accent(accent)).copied().unwrap_or(0) > 0 {
                        if accent == AccentTile::Boat {
                            for (&pos, occupant) in &self.pieces {
                                if occupant.player != player {
                                    actions.push(Action::Plant { tile: Tile::Accent(accent), at: pos });
                                }
                            }
                        } else {
                            for pos in valid_spaces() {
                                if !GATES.contains(&pos) && !self.pieces.contains_key(&pos) {
                                    actions.push(Action::Plant { tile: Tile::Accent(accent), at: pos });
                                }
                            }
                        }
                    }
                }
                for special in SpecialTile::ALL {
                    if hand.get(&Tile::Special(special)).copied().unwrap_or(0) > 0 {
                        for &gate in &empty_gates {
                            actions.push(Action::Plant { tile: Tile::Special(special), at: gate });
                        }
                    }
                }
            }
        }

        let owned: Vec<(Position, Piece)> = self
            .pieces
            .iter()
            .filter(|(_, p)| p.player == player && !matches!(p.tile, Tile::Accent(_)))
            .map(|(&pos, &p)| (pos, p))
            .collect();
        for (from, occupant) in owned {
            if self.bonus_turn && occupant.growing {
                continue;
            }
            for to in self.valid_destinations(from) {
                actions.push(Action::Arrange { from, to });
            }
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_with_empty_board_and_player_one() {
        let board = Board::new();
        assert!(board.pieces.is_empty());
        assert_eq!(board.current_player, Player::One);
        assert!(!board.bonus_turn);
    }

    #[test]
    fn new_starts_with_no_winner() {
        assert_eq!(Board::new().winner, None);
    }

    #[test]
    fn new_hands_match_python_oracle() {
        // Computed by running engine/PythonEngine/PaiShoGame.py:
        //   PaiShoGame().hands == {'Rose': 2, 'Chrysanthemum': 2, 'Rhododendron': 2,
        //     'Jasmine': 2, 'Lily': 2, 'Jade': 2, 'Rock': 1, 'Wheel': 1, 'Knotweed': 1,
        //     'Boat': 1, 'Orchid': 1, 'WhiteLotus': 1}  (identical for both players)
        let board = Board::new();
        for player in [Player::One, Player::Two] {
            let hand = &board.hands[&player];
            assert_eq!(hand.len(), 12, "{player:?} should have 12 distinct tile kinds");
            for f in Flower::ALL {
                assert_eq!(hand[&Tile::Flower(f)], 2, "{player:?}/{f:?} should start with 2");
            }
            for a in AccentTile::ALL {
                assert_eq!(hand[&Tile::Accent(a)], 1, "{player:?}/{a:?} should start with 1");
            }
            for s in SpecialTile::ALL {
                assert_eq!(hand[&Tile::Special(s)], 1, "{player:?}/{s:?} should start with 1");
            }
        }
    }

    fn piece(tile: Tile, player: Player) -> Piece {
        Piece { tile, player, growing: false }
    }

    #[test]
    fn lone_flower_reaches_all_cells_within_range() {
        // Computed from: g.board[(9,9)] = {'flower':'Rose','player':1,'growing':False};
        // g.valid_destinations(9,9)
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        let mut dests = board.valid_destinations(Position::new(9, 9));
        dests.sort_by_key(|p| (p.row, p.col));
        let expected = [
            (6, 9), (7, 9), (7, 10), (8, 9), (8, 10), (8, 11), (9, 6), (9, 7), (9, 8), (9, 10),
            (9, 11), (9, 12), (10, 7), (10, 8), (10, 9), (11, 8), (11, 9), (12, 9),
        ]
        .map(|(r, c)| Position::new(r, c));
        assert_eq!(dests, expected);
    }

    #[test]
    fn white_flower_cannot_enter_red_gardens() {
        // Computed from: g.board[(8,8)] = {'flower':'Jasmine','player':1,'growing':False};
        // g.valid_destinations(8,8) — every result is white or neutral, none red.
        let mut board = Board::new();
        board.pieces.insert(Position::new(8, 8), piece(Tile::Flower(Flower::Jasmine), Player::One));
        let mut dests = board.valid_destinations(Position::new(8, 8));
        dests.sort_by_key(|p| (p.row, p.col));
        let expected = [
            (5, 8), (6, 7), (6, 8), (6, 9), (7, 6), (7, 7), (7, 8), (7, 9), (8, 5), (8, 6),
            (8, 7), (8, 9), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 9),
        ]
        .map(|(r, c)| Position::new(r, c));
        assert_eq!(dests, expected);
        for pos in &dests {
            assert_ne!(garden_of(pos.row, pos.col), Garden::Red, "{pos:?} should not be a red garden cell");
        }
    }

    #[test]
    fn enemy_occupied_cell_blocks_passage_without_capture_logic() {
        // Computed from: g.board[(9,9)] = {'flower':'Rose','player':1,'growing':False};
        // g.board[(9,10)] = {'flower':'Chrysanthemum','player':2,'growing':False};
        // g.valid_destinations(9,9) — Chrysanthemum is a harmony partner of Rose, not a
        // clash partner, so even the real (capture-aware) Python engine can't capture it;
        // it just blocks passage. This task's Rust code has no capture logic at all yet,
        // so any occupied cell blocking passage is already the right behavior here.
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Chrysanthemum), Player::Two));
        let mut dests = board.valid_destinations(Position::new(9, 9));
        dests.sort_by_key(|p| (p.row, p.col));
        let expected = [
            (6, 9), (7, 9), (7, 10), (8, 9), (8, 10), (8, 11), (9, 6), (9, 7), (9, 8), (10, 7),
            (10, 8), (10, 9), (11, 8), (11, 9), (12, 9),
        ]
        .map(|(r, c)| Position::new(r, c));
        assert_eq!(dests, expected);
    }

    #[test]
    fn own_occupied_cell_blocks_passage() {
        // Computed from: g.board[(9,9)] = {'flower':'Rose','player':1,'growing':False};
        // g.board[(9,10)] = {'flower':'Jade','player':1,'growing':False};
        // g.valid_destinations(9,9) — same resulting set as the enemy-blocks case above,
        // since an occupied cell blocks passage regardless of owner.
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Jade), Player::One));
        let mut dests = board.valid_destinations(Position::new(9, 9));
        dests.sort_by_key(|p| (p.row, p.col));
        let expected = [
            (6, 9), (7, 9), (7, 10), (8, 9), (8, 10), (8, 11), (9, 6), (9, 7), (9, 8), (10, 7),
            (10, 8), (10, 9), (11, 8), (11, 9), (12, 9),
        ]
        .map(|(r, c)| Position::new(r, c));
        assert_eq!(dests, expected);
    }

    #[test]
    fn accent_tile_never_moves() {
        // Computed from: g.board[(9,9)] = {'flower':'Rock','player':1,'growing':False};
        // g.valid_destinations(9,9) == []
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Accent(AccentTile::Rock), Player::One));
        assert_eq!(board.valid_destinations(Position::new(9, 9)), Vec::new());
    }

    #[test]
    fn orchid_trap_freezes_a_tile_adjacent_to_an_enemy_blooming_orchid() {
        // Computed from: g.board[(9,9)] = {'flower':'Rose','player':1,'growing':False};
        // g.board[(9,10)] = {'flower':'Orchid','player':2,'growing':False};
        // g.valid_destinations(9,9) == []  (Rose is adjacent to the enemy Orchid)
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Special(SpecialTile::Orchid), Player::Two));
        assert_eq!(board.valid_destinations(Position::new(9, 9)), Vec::new());
    }

    #[test]
    fn clash_pair_enemy_is_capturable() {
        // g.board[(9,9)]={'flower':'Rose','player':1,...}; g.board[(9,10)]={'flower':'Jasmine','player':2,...}
        // Rose/Jasmine are a clash pair (circle distance 3). g.valid_destinations(9,9) includes (9,10).
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        assert!(board.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));
    }

    #[test]
    fn harmony_pair_enemy_is_still_not_capturable() {
        // Same board as Task 2's enemy-blocks test — capture eligibility must not make a
        // harmony (non-clash) enemy suddenly capturable. Regression guard on Task 2's list.
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Chrysanthemum), Player::Two));
        assert!(!board.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));
    }

    #[test]
    fn white_lotus_is_never_capturable_by_a_clash_flower() {
        // g.board[(9,9)]={'flower':'Rose',...}; g.board[(9,10)]={'flower':'WhiteLotus','player':2,...}
        // g.valid_destinations(9,9) excludes (9,10) even though WhiteLotus isn't a clash concept at all.
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Special(SpecialTile::WhiteLotus), Player::Two));
        assert!(!board.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));
    }

    #[test]
    fn basic_flower_cannot_capture_an_accent_tile() {
        // g.board[(9,9)]={'flower':'Rose',...}; g.board[(9,10)]={'flower':'Rock','player':2,...}
        // g.valid_destinations(9,9) excludes (9,10).
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Accent(AccentTile::Rock), Player::Two));
        assert!(!board.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));
    }

    #[test]
    fn orchid_mover_capture_requires_its_own_owners_blooming_white_lotus() {
        // g.board[(9,9)]={'flower':'Orchid','player':1,...}; g.board[(9,10)]={'flower':'Rose','player':2,...}
        // Without a blooming WL for player 1: (9,10) not capturable.
        // With g.board[(5,5)]={'flower':'WhiteLotus','player':1,...}: (9,10) becomes capturable.
        let mut without_wl = Board::new();
        without_wl.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::Orchid), Player::One));
        without_wl.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Rose), Player::Two));
        assert!(!without_wl.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));

        let mut with_wl = Board::new();
        with_wl.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::Orchid), Player::One));
        with_wl.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Rose), Player::Two));
        with_wl.pieces.insert(Position::new(5, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(with_wl.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));
    }

    #[test]
    fn capturing_an_enemy_orchid_requires_the_defenders_own_blooming_white_lotus() {
        // g.board[(9,9)]={'flower':'Rose','player':1,...}; g.board[(9,11)]={'flower':'Orchid','player':2,...}
        // (kept 2 cells apart so the Orchid TRAP doesn't also freeze the mover — verified separately).
        // No blooming WL at all: not capturable. Defender's (player 2's) own WL: capturable.
        // Attacker's (player 1's) WL instead: still NOT capturable (irrelevant to this rule).
        let mut none = Board::new();
        none.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        none.pieces.insert(Position::new(9, 11), piece(Tile::Special(SpecialTile::Orchid), Player::Two));
        assert!(!none.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 11)));

        let mut defender_wl = Board::new();
        defender_wl.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        defender_wl.pieces.insert(Position::new(9, 11), piece(Tile::Special(SpecialTile::Orchid), Player::Two));
        defender_wl.pieces.insert(Position::new(5, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::Two));
        assert!(defender_wl.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 11)));

        let mut attacker_wl = Board::new();
        attacker_wl.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        attacker_wl.pieces.insert(Position::new(9, 11), piece(Tile::Special(SpecialTile::Orchid), Player::Two));
        attacker_wl.pieces.insert(Position::new(5, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(!attacker_wl.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 11)));
    }

    #[test]
    fn empowered_orchid_can_capture_any_enemy_tile_kind() {
        // An Orchid mover with its own blooming WL can capture an enemy accent tile,
        // an enemy WhiteLotus, and even an enemy Orchid — all otherwise-uncapturable
        // tile kinds for a basic flower. Each enemy kept 2 cells away to dodge the trap.
        let mut vs_accent = Board::new();
        vs_accent.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::Orchid), Player::One));
        vs_accent.pieces.insert(Position::new(9, 10), piece(Tile::Accent(AccentTile::Rock), Player::Two));
        vs_accent.pieces.insert(Position::new(5, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(vs_accent.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));

        let mut vs_white_lotus = Board::new();
        vs_white_lotus.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::Orchid), Player::One));
        vs_white_lotus.pieces.insert(Position::new(9, 10), piece(Tile::Special(SpecialTile::WhiteLotus), Player::Two));
        vs_white_lotus.pieces.insert(Position::new(5, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(vs_white_lotus.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));

        let mut vs_orchid = Board::new();
        vs_orchid.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::Orchid), Player::One));
        vs_orchid.pieces.insert(Position::new(9, 11), piece(Tile::Special(SpecialTile::Orchid), Player::Two));
        vs_orchid.pieces.insert(Position::new(5, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(vs_orchid.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 11)));
    }

    #[test]
    fn white_lotus_mover_never_captures_anything() {
        // g.board[(9,9)]={'flower':'WhiteLotus','player':1,...}; g.board[(9,10)]={'flower':'Rose','player':2,...}
        // g.board[(5,5)]={'flower':'WhiteLotus','player':1,...} (own blooming WL elsewhere — irrelevant)
        // g.valid_destinations(9,9) excludes (9,10).
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Rose), Player::Two));
        board.pieces.insert(Position::new(5, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(!board.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));
    }

    #[test]
    fn clear_line_between_same_row_with_nothing_in_the_way() {
        // g._clear_line_between(9,5,9,9) == True
        let pieces = HashMap::new();
        assert!(clear_line_between(&pieces, Position::new(9, 5), Position::new(9, 9)));
    }

    #[test]
    fn clear_line_between_same_row_blocked_by_a_piece() {
        // g.board[(9,7)]={'flower':'Rose',...}; g._clear_line_between(9,5,9,9) == False
        let mut pieces = HashMap::new();
        pieces.insert(Position::new(9, 7), piece(Tile::Flower(Flower::Rose), Player::One));
        assert!(!clear_line_between(&pieces, Position::new(9, 5), Position::new(9, 9)));
    }

    #[test]
    fn clear_line_between_same_column_blocked_by_a_gate() {
        // g._clear_line_between(0,9,2,9) == False — gate (1,9) sits strictly between.
        let pieces = HashMap::new();
        assert!(!clear_line_between(&pieces, Position::new(0, 9), Position::new(2, 9)));
    }

    #[test]
    fn clear_line_between_unaligned_positions_is_vacuously_true() {
        // g._clear_line_between(5,5,6,6) == True — neither same row nor same column, so
        // there's nothing to check; matches Python's fallthrough behavior exactly.
        let pieces = HashMap::new();
        assert!(clear_line_between(&pieces, Position::new(5, 5), Position::new(6, 6)));
    }

    #[test]
    fn creates_clash_detects_a_fresh_clash_at_the_destination() {
        // g.board[(9,10)]={'flower':'Rose','player':1,'growing':False};
        // g.board[(9,15)]={'flower':'Jasmine','player':2,'growing':False};
        // g._check_clash_after_move(9,9,9,10) == True
        let mut pieces = HashMap::new();
        pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Rose), Player::One));
        pieces.insert(Position::new(9, 15), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        assert!(creates_clash(&pieces, Position::new(9, 9), Position::new(9, 10)));
    }

    #[test]
    fn creates_clash_is_false_with_no_other_flowers() {
        // g.board[(9,10)]={'flower':'Rose','player':1,'growing':False};
        // g._check_clash_after_move(9,9,9,10) == False
        let mut pieces = HashMap::new();
        pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Rose), Player::One));
        assert!(!creates_clash(&pieces, Position::new(9, 9), Position::new(9, 10)));
    }

    #[test]
    fn creates_clash_detects_a_clash_unblocked_by_vacating_the_source() {
        // g.board[(9,3)]={'flower':'Rose','player':1,'growing':False};
        // g.board[(9,8)]={'flower':'Jasmine','player':2,'growing':False};
        // g.board[(2,2)]={'flower':'Lily','player':1,'growing':False};  (the mover, now at 2,2)
        // g._check_clash_after_move(9,5,2,2) == True — Rose@(9,3) and Jasmine@(9,8) can now
        // see each other on row 9 now that whatever was at the vacated (9,5) is gone.
        let mut pieces = HashMap::new();
        pieces.insert(Position::new(9, 3), piece(Tile::Flower(Flower::Rose), Player::One));
        pieces.insert(Position::new(9, 8), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        pieces.insert(Position::new(2, 2), piece(Tile::Flower(Flower::Lily), Player::One));
        assert!(creates_clash(&pieces, Position::new(9, 5), Position::new(2, 2)));
    }

    #[test]
    fn post_move_clash_blocks_an_otherwise_legal_empty_cell_move() {
        // g.board[(9,9)]={'flower':'Rose','player':1,...}; g.board[(9,15)]={'flower':'Jasmine','player':2,...}
        // g.valid_destinations(9,9) excludes (9,10), (9,11), (9,12) — all present without the
        // Jasmine (see Task 2's lone_flower_reaches_all_cells_within_range) — because landing on
        // any of them would put Rose in clash-line with the Jasmine at (9,15) on row 9.
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 15), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        let dests = board.valid_destinations(Position::new(9, 9));
        for blocked in [(9, 10), (9, 11), (9, 12)] {
            assert!(!dests.contains(&Position::new(blocked.0, blocked.1)), "{blocked:?} should be blocked by post-move clash");
        }
    }

    #[test]
    fn post_move_clash_blocks_an_otherwise_legal_capture() {
        // g.board[(9,9)]={'flower':'Rose','player':1,...}; g.board[(9,10)]={'flower':'Jasmine','player':2,...}
        // g.board[(15,10)]={'flower':'Jasmine','player':2,...}
        // Without the extra Jasmine at (15,10) (see Task 3's clash_pair_enemy_is_capturable),
        // (9,10) is capturable. With it, capturing would put Rose in clash-line (column 10)
        // with the Jasmine at (15,10), so it's blocked.
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        board.pieces.insert(Position::new(15, 10), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        assert!(!board.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));
    }

    #[test]
    fn legal_actions_is_empty_once_the_game_has_a_winner() {
        // Found via cross-engine fuzzing against the Python reference (running many random
        // self-play games through both engines in lockstep): a board with pieces and gates
        // that would otherwise produce legal actions, but winner is Some, must report none.
        let mut board = Board::new();
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.winner = Some(crate::moves::Outcome::Winner(Player::One));
        assert_eq!(board.legal_actions(), Vec::new());
    }

    #[test]
    fn initial_board_has_24_basic_flower_plants() {
        // PaiShoGame().get_legal_actions() has 24 actions, all ('plant', <one of the 6
        // circle flowers>, <one of the 4 gates>) — 6 * 4 = 24.
        let board = Board::new();
        let actions = board.legal_actions();
        assert_eq!(actions.len(), 24);
        for action in &actions {
            match action {
                Action::Plant { tile: Tile::Flower(_), at } => assert!(GATES.contains(at)),
                other => panic!("expected a basic-flower gate plant, got {other:?}"),
            }
        }
    }

    #[test]
    fn occupying_a_gate_removes_its_plant_actions_but_adds_arrange_actions() {
        // g.board[(1,9)]={'flower':'Rose','player':1,'growing':True}
        // g.get_legal_actions(): 18 plant actions (6 flowers * remaining 3 empty gates)
        // + 15 arrange actions (Rose's own valid_destinations from (1,9)) = 33 total.
        let mut board = Board::new();
        board.pieces.insert(
            Position::new(1, 9),
            Piece { tile: Tile::Flower(Flower::Rose), player: Player::One, growing: true },
        );
        let actions = board.legal_actions();
        let plants = actions.iter().filter(|a| matches!(a, Action::Plant { .. })).count();
        let arranges = actions.iter().filter(|a| matches!(a, Action::Arrange { .. })).count();
        assert_eq!(plants, 18);
        assert_eq!(arranges, 15);
        assert_eq!(actions.len(), 33);
    }

    #[test]
    fn bonus_turn_enumerates_accent_boat_and_special_plants() {
        // g.board[(9,9)]={'flower':'Rose','player':1,'growing':False}
        // g.board[(9,10)]={'flower':'Jasmine','player':2,'growing':False}, g.bonus_turn=True
        // get_legal_actions() breaks down as: 24 basic-flower gate plants, 729 accent plants
        // (3 kinds [Rock/Wheel/Knotweed] * 243 empty non-gate cells), 1 Boat plant (only the
        // enemy-occupied cell (9,10)), 8 special plants (2 kinds * 4 empty gates), and 11
        // arrange actions from Rose (not growing, so bonus turn doesn't exclude it) = 773 total.
        let mut board = Board::new();
        board.bonus_turn = true;
        board.pieces.insert(Position::new(9, 9), piece(Tile::Flower(Flower::Rose), Player::One));
        board.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Jasmine), Player::Two));
        let actions = board.legal_actions();

        let basic_plants = actions
            .iter()
            .filter(|a| matches!(a, Action::Plant { tile: Tile::Flower(_), .. }))
            .count();
        let accent_plants = actions
            .iter()
            .filter(|a| matches!(a, Action::Plant { tile: Tile::Accent(a), .. } if *a != AccentTile::Boat))
            .count();
        let boat_plants: Vec<Position> = actions
            .iter()
            .filter_map(|a| match a {
                Action::Plant { tile: Tile::Accent(AccentTile::Boat), at } => Some(*at),
                _ => None,
            })
            .collect();
        let special_plants = actions
            .iter()
            .filter(|a| matches!(a, Action::Plant { tile: Tile::Special(_), .. }))
            .count();
        let arranges = actions.iter().filter(|a| matches!(a, Action::Arrange { .. })).count();

        assert_eq!(basic_plants, 24);
        assert_eq!(accent_plants, 729);
        assert_eq!(boat_plants, vec![Position::new(9, 10)]);
        assert_eq!(special_plants, 8);
        assert_eq!(arranges, 11);
        assert_eq!(actions.len(), 773);
    }

    #[test]
    fn bonus_turn_excludes_arranging_a_still_growing_tile() {
        // g.board[(1,9)]={'flower':'Rose','player':1,'growing':True}, g.bonus_turn=True
        // get_legal_actions() has 0 arrange actions FROM (1,9) — growing tiles can't be the
        // source of an arrange during a bonus turn (they still can outside a bonus turn; see
        // occupying_a_gate_removes_its_plant_actions_but_adds_arrange_actions's 15 arranges
        // from the same growing Rose when bonus_turn is false).
        let mut board = Board::new();
        board.bonus_turn = true;
        board.pieces.insert(
            Position::new(1, 9),
            Piece { tile: Tile::Flower(Flower::Rose), player: Player::One, growing: true },
        );
        let arranges_from_1_9 = board
            .legal_actions()
            .into_iter()
            .filter(|a| matches!(a, Action::Arrange { from, .. } if *from == Position::new(1, 9)))
            .count();
        assert_eq!(arranges_from_1_9, 0);
    }
}
