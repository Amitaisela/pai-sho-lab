//! The game state (`Board`: pieces, hands, whose turn, bonus, setup, outcome)
//! and the rules that read it: arrange destinations (movement, gates, gardens,
//! captures, the Orchid trap, no clash), legal-action generation with the
//! no-stranding filter, and the Wheel/Boat accent rules. Applying actions and
//! ending turns lives in `moves`; harmonies, clashes and rings in `harmony`.
//! Ported from `PaiShoGame.py`'s `PaiShoGame` class.

use std::collections::{HashMap, VecDeque};

use crate::board::{garden_of, is_valid_space, valid_spaces, Garden, Position, GATES, GUEST_GATE, HOST_GATE};
use crate::flower::{is_clash, Color, Flower};
use crate::harmony::find_clashes_on;
use crate::moves::{EndReason, Outcome};
use crate::piece::Piece;
use crate::player::Player;
use crate::setup::{Setup, BASIC_FLOWERS_PER_KIND};
use crate::tile::{AccentTile, SpecialTile, Tile, NEIGHBOURS_8, ORTHOGONAL_OFFSETS};

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
    pub setup: Setup,
    pub end_reason: Option<EndReason>,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// A fresh board with the default setup (one of each accent, informal start).
    pub fn new() -> Board {
        Board::with_setup(Setup::default())
    }

    /// A fresh board for `setup`: 3 of each basic flower, the chosen accents, one of
    /// each special flower; with an opening flower, both players start with one
    /// growing in their gate. Ported from `PaiShoGame.reset`.
    pub fn with_setup(setup: Setup) -> Board {
        let mut hands = HashMap::new();
        for player in [Player::One, Player::Two] {
            let mut hand = HashMap::new();
            for f in Flower::ALL {
                hand.insert(Tile::Flower(f), BASIC_FLOWERS_PER_KIND);
            }
            for a in AccentTile::ALL {
                hand.insert(Tile::Accent(a), 0);
            }
            for &a in &setup.accents[&player] {
                *hand.get_mut(&Tile::Accent(a)).expect("every accent key was inserted") += 1;
            }
            for s in SpecialTile::ALL {
                hand.insert(Tile::Special(s), 1);
            }
            hands.insert(player, hand);
        }
        let mut board = Board {
            pieces: HashMap::new(),
            hands,
            current_player: Player::One,
            bonus_turn: false,
            winner: None,
            setup,
            end_reason: None,
        };
        if let Some(flower) = board.setup.opening {
            for (player, gate) in [(Player::One, GUEST_GATE), (Player::Two, HOST_GATE)] {
                board.pieces.insert(gate, Piece { tile: Tile::Flower(flower), player, growing: true });
                *board.hands.get_mut(&player).unwrap().get_mut(&Tile::Flower(flower)).unwrap() -= 1;
            }
        }
        board
    }

    /// Resets `self` to a fresh board with the same setup. Ported from `PaiShoGame.reset`.
    pub fn reset(&mut self) {
        *self = Board::with_setup(self.setup.clone());
    }

    /// All cells the piece at `from` could arrange to: orthogonal steps up to its
    /// move range, passing through open gates but never stopping on one, blocked by
    /// occupied cells (except for a legal capture on the final step), obeying the
    /// garden rule for basic flowers, and never producing a clash (including one
    /// uncovered by vacating `from`). Empty if there's no piece at `from`, it's an
    /// accent tile, or it's frozen by the Orchid trap. Ported from
    /// `PaiShoGame.valid_destinations`.
    pub fn valid_destinations(&self, from: Position) -> Vec<Position> {
        self.destinations(from, false)
    }

    /// Whether the piece at `from` has at least one destination; stops at the first.
    /// Same answer as `!valid_destinations(from).is_empty()`.
    pub fn has_any_destination(&self, from: Position) -> bool {
        !self.destinations(from, true).is_empty()
    }

    fn destinations(&self, from: Position, first_only: bool) -> Vec<Position> {
        let piece = match self.pieces.get(&from) {
            Some(p) => *p,
            None => return Vec::new(),
        };
        if matches!(piece.tile, Tile::Accent(_)) || self.is_trapped(from, piece) {
            return Vec::new();
        }
        let limit = match piece.tile.move_range() {
            Some(l) => l,
            None => return Vec::new(),
        };
        let wild = (self.has_blooming_white_lotus(Player::One), self.has_blooming_white_lotus(Player::Two));
        let moved = Piece { tile: piece.tile, player: piece.player, growing: false };
        // The mover is lifted off a scratch copy once; each candidate landing is tried on
        // it and undone (mirrors the Python engine's pop-and-restore).
        let mut scratch = self.pieces.clone();
        scratch.remove(&from);

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
                if !is_valid_space(next.row, next.col) {
                    continue;
                }
                let new_dist = dist + 1;
                if visited.get(&next).copied().unwrap_or(i32::MAX) <= new_dist {
                    continue;
                }
                visited.insert(next, new_dist);
                let occupant = self.pieces.get(&next).copied();
                if GATES.contains(&next) {
                    // A tile may pass through an open Gate but never stop on one (rulebook p.9).
                    if occupant.is_none() {
                        queue.push_back((next, new_dist));
                    }
                    continue;
                }
                if let Some(target) = occupant {
                    if can_capture(piece, target, wild)
                        && !garden_blocks(piece.tile, next)
                        && landing_is_clash_free(&mut scratch, from, next, moved)
                    {
                        destinations.push(next);
                        if first_only {
                            return destinations;
                        }
                    }
                    continue;
                }
                if !garden_blocks(piece.tile, next) && landing_is_clash_free(&mut scratch, from, next, moved) {
                    destinations.push(next);
                    if first_only {
                        return destinations;
                    }
                }
                queue.push_back((next, new_dist));
            }
        }
        destinations
    }

    /// Orchid trap: an opponent's blooming flower tile next to a blooming Orchid can't arrange.
    fn is_trapped(&self, at: Position, piece: Piece) -> bool {
        if piece.growing {
            return false;
        }
        let enemy = piece.player.other();
        NEIGHBOURS_8.iter().any(|&(dr, dc)| {
            matches!(
                self.pieces.get(&Position::new(at.row + dr, at.col + dc)),
                Some(p) if p.tile == Tile::Special(SpecialTile::Orchid) && p.player == enemy && !p.growing
            )
        })
    }

    /// True if `player` has a blooming White Lotus that isn't on a gate, which
    /// makes their Orchid wild (rulebook p.12). Ported from `PaiShoGame._is_wild`.
    fn has_blooming_white_lotus(&self, player: Player) -> bool {
        self.pieces.iter().any(|(pos, p)| {
            p.tile == Tile::Special(SpecialTile::WhiteLotus)
                && p.player == player
                && !p.growing
                && !GATES.contains(pos)
        })
    }
}

/// True if `tile` may not end a move on `pos`: basic flowers can't stop in the
/// opposite-coloured garden (rulebook p.9). Ported from `_garden_blocks`.
pub(crate) fn garden_blocks(tile: Tile, pos: Position) -> bool {
    matches!(
        (tile.color(), garden_of(pos.row, pos.col)),
        (Some(Color::Red), Garden::White) | (Some(Color::White), Garden::Red)
    )
}

/// With the mover already lifted off `from` in `scratch`, would it landing on `to`
/// (capturing any occupant) leave no clash? `scratch` is restored before returning.
/// Ported from `PaiShoGame._landing_is_clash_free`.
fn landing_is_clash_free(scratch: &mut HashMap<Position, Piece>, from: Position, to: Position, moved: Piece) -> bool {
    let previous = scratch.insert(to, moved);
    let clash_free = !creates_clash(scratch, from, to);
    match previous {
        Some(p) => {
            scratch.insert(to, p);
        }
        None => {
            scratch.remove(&to);
        }
    }
    clash_free
}

/// Ported from `PaiShoGame._can_capture`. `wild` = (Player One's Orchid is wild, Player Two's).
fn can_capture(mover: Piece, target: Piece, wild: (bool, bool)) -> bool {
    let is_wild = |p: Player| if p == Player::One { wild.0 } else { wild.1 };
    if target.player == mover.player || target.growing || matches!(target.tile, Tile::Accent(_)) {
        return false;
    }
    if mover.tile == Tile::Special(SpecialTile::Orchid) && is_wild(mover.player) {
        return true;
    }
    if target.tile == Tile::Special(SpecialTile::Orchid) && is_wild(target.player) {
        return true;
    }
    match (mover.tile, target.tile) {
        (Tile::Flower(a), Tile::Flower(b)) => is_clash(a, b),
        _ => false,
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
    /// Plant a flower or place an accent. `displace` is only for a Boat played on a
    /// flower: where that flower moves (`('plant','Boat',r,c,dr,dc)` in Python).
    Plant { tile: Tile, at: Position, displace: Option<Position> },
    Arrange { from: Position, to: Position },
    SkipBonus,
}

impl Board {
    /// Every legal action for the side to move (rulebook p.8): on a normal turn, plant a
    /// basic flower or arrange; during a bonus, an accent, a special flower, a basic
    /// flower if none of yours are growing, or skip. Ported from `PaiShoGame.get_legal_actions`.
    pub fn legal_actions(&self) -> Vec<Action> {
        if self.winner.is_some() {
            return Vec::new();
        }
        let mut actions = Vec::new();
        self.visit_candidates(&mut |a| {
            if self.opponent_can_move_after(&a) {
                actions.push(a);
            }
            false
        });
        if self.bonus_turn {
            actions.push(Action::SkipBonus);
        }
        actions
    }

    /// No stranding: after `action` (as if the turn ended), the opponent can still Plant
    /// or Arrange. Ported from `PaiShoGame._opponent_can_move_after`.
    fn opponent_can_move_after(&self, action: &Action) -> bool {
        let opponent = self.current_player.other();
        if self.basic_count(opponent) > 0 {
            let mut open: Vec<Position> = GATES.into_iter().filter(|g| !self.pieces.contains_key(g)).collect();
            match *action {
                Action::Plant { at, .. } if GATES.contains(&at) => open.retain(|&g| g != at),
                Action::Arrange { from, .. } if GATES.contains(&from) => open.push(from),
                _ => {}
            }
            if !open.is_empty() {
                return true;
            }
        }
        let mut sim = self.clone();
        sim.apply_effects(*action);
        sim.can_move(opponent)
    }

    /// Whether `player` would have any Plant or Arrange on a normal turn. Ported from
    /// `PaiShoGame._can_move`.
    pub(crate) fn can_move(&self, player: Player) -> bool {
        if self.basic_count(player) > 0 && GATES.iter().any(|g| !self.pieces.contains_key(g)) {
            return true;
        }
        self.pieces
            .iter()
            .any(|(&pos, p)| p.player == player && !matches!(p.tile, Tile::Accent(_)) && self.has_any_destination(pos))
    }

    /// Same answer as `!legal_actions().is_empty()` while `bonus_turn` is false (the only
    /// way `end_turn` calls it), but stops at the first legal candidate.
    pub(crate) fn has_any_legal_action(&self) -> bool {
        self.winner.is_none() && (self.bonus_turn || self.visit_candidates(&mut |a| self.opponent_can_move_after(&a)))
    }

    /// Calls `visit` on every candidate action for the side to move (rulebook p.8),
    /// before the no-stranding filter, and stops as soon as `visit` returns true;
    /// returns whether it stopped early. Ported from `PaiShoGame._iter_candidate_actions`.
    fn visit_candidates(&self, visit: &mut dyn FnMut(Action) -> bool) -> bool {
        let player = self.current_player;
        let hand = &self.hands[&player];
        let count = |t: Tile| hand.get(&t).copied().unwrap_or(0);
        let empty_gates: Vec<Position> = GATES.into_iter().filter(|g| !self.pieces.contains_key(g)).collect();
        // A bonus offers basic flowers only while none of the player's tiles are growing.
        if !self.bonus_turn || !self.has_growing(player) {
            for f in Flower::ALL {
                if count(Tile::Flower(f)) > 0 {
                    for &gate in &empty_gates {
                        if visit(Action::Plant { tile: Tile::Flower(f), at: gate, displace: None }) {
                            return true;
                        }
                    }
                }
            }
        }
        if self.bonus_turn {
            for s in SpecialTile::ALL {
                if count(Tile::Special(s)) > 0 {
                    for &gate in &empty_gates {
                        if visit(Action::Plant { tile: Tile::Special(s), at: gate, displace: None }) {
                            return true;
                        }
                    }
                }
            }
            return self.accent_actions(player).into_iter().any(visit);
        }
        for (&from, p) in &self.pieces {
            if p.player == player && !matches!(p.tile, Tile::Accent(_)) {
                for to in self.valid_destinations(from) {
                    if visit(Action::Arrange { from, to }) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn has_growing(&self, player: Player) -> bool {
        self.pieces.values().any(|p| p.player == player && p.growing)
    }

    /// The (src, dest) moves a Wheel at `at` would make, or `None` if a Wheel may not be
    /// placed there. Ported from `PaiShoGame._wheel_rotation`.
    pub fn wheel_rotation(&self, at: Position) -> Option<Vec<(Position, Position)>> {
        if !is_valid_space(at.row, at.col) || GATES.contains(&at) || self.pieces.contains_key(&at) {
            return None;
        }
        let ring: Vec<Position> = NEIGHBOURS_8.iter().map(|&(dr, dc)| Position::new(at.row + dr, at.col + dc)).collect();
        let mut moves = Vec::new();
        for i in 0..8 {
            let src = ring[i];
            let piece = match self.pieces.get(&src) {
                Some(p) => *p,
                None => continue,
            };
            let dest = ring[(i + 1) % 8];
            if piece.tile == Tile::Accent(AccentTile::Rock) {
                return None;
            }
            if GATES.contains(&src) || GATES.contains(&dest) || !is_valid_space(dest.row, dest.col) {
                return None;
            }
            if garden_blocks(piece.tile, dest) {
                return None;
            }
            moves.push((src, dest));
        }
        if moves.is_empty() {
            return Some(moves);
        }
        let mut simulated = self.pieces.clone();
        for &(src, _) in &moves {
            simulated.remove(&src);
        }
        for &(src, dest) in &moves {
            simulated.insert(dest, self.pieces[&src]);
        }
        simulated.insert(at, Piece { tile: Tile::Accent(AccentTile::Wheel), player: self.current_player, growing: false });
        if !find_clashes_on(&simulated).is_empty() {
            return None;
        }
        Some(moves)
    }

    /// Empty surrounding points a Boat played on the flower at `at` may move it to.
    /// Ported from `PaiShoGame._boat_displacements`.
    pub fn boat_displacements(&self, at: Position) -> Vec<Position> {
        let target = match self.pieces.get(&at) {
            Some(p) => *p,
            None => return Vec::new(),
        };
        let mut out = Vec::new();
        for &(dr, dc) in NEIGHBOURS_8.iter() {
            let dest = Position::new(at.row + dr, at.col + dc);
            if !is_valid_space(dest.row, dest.col) || GATES.contains(&dest) || self.pieces.contains_key(&dest) {
                continue;
            }
            if garden_blocks(target.tile, dest) {
                continue;
            }
            let mut simulated = self.pieces.clone();
            simulated.insert(dest, target);
            simulated.insert(at, Piece { tile: Tile::Accent(AccentTile::Boat), player: self.current_player, growing: false });
            if find_clashes_on(&simulated).is_empty() {
                out.push(dest);
            }
        }
        out
    }

    /// Every legal accent placement for `player`. Ported from `PaiShoGame._accent_actions`.
    pub fn accent_actions(&self, player: Player) -> Vec<Action> {
        let hand = &self.hands[&player];
        let has = |a: AccentTile| hand.get(&Tile::Accent(a)).copied().unwrap_or(0) > 0;
        let empty: Vec<Position> =
            valid_spaces().into_iter().filter(|p| !GATES.contains(p) && !self.pieces.contains_key(p)).collect();
        let mut out = Vec::new();
        for accent in [AccentTile::Rock, AccentTile::Knotweed] {
            if has(accent) {
                out.extend(empty.iter().map(|&at| Action::Plant { tile: Tile::Accent(accent), at, displace: None }));
            }
        }
        if has(AccentTile::Wheel) {
            for &at in &empty {
                if self.wheel_rotation(at).is_some() {
                    out.push(Action::Plant { tile: Tile::Accent(AccentTile::Wheel), at, displace: None });
                }
            }
        }
        if has(AccentTile::Boat) {
            let boat = Tile::Accent(AccentTile::Boat);
            for (&at, target) in &self.pieces {
                if matches!(target.tile, Tile::Accent(_)) {
                    // Both tiles leave the game; emptying the point must not uncover a clash.
                    let mut simulated = self.pieces.clone();
                    simulated.remove(&at);
                    if find_clashes_on(&simulated).is_empty() {
                        out.push(Action::Plant { tile: boat, at, displace: None });
                    }
                } else if !target.growing {
                    for to in self.boat_displacements(at) {
                        out.push(Action::Plant { tile: boat, at, displace: Some(to) });
                    }
                }
            }
        }
        out
    }

    /// Order-independent hash of the tiles on the board. Mirrors
    /// `PaiShoGame.board_sig()` in meaning (not value — the two engines'
    /// sigs are never comparable to each other).
    pub fn board_sig(&self) -> u64 {
        let mut h = 0u64;
        for (pos, p) in &self.pieces {
            let key = ((pos.row as u64 & 0xFF) << 40)
                ^ ((pos.col as u64 & 0xFF) << 32)
                ^ (tile_sig_index(p.tile) << 8)
                ^ (player_sig_index(p.player) << 1)
                ^ (p.growing as u64);
            h ^= splitmix64(key);
        }
        h
    }

    /// `board_sig` plus side to move, bonus flag and both hands: the
    /// transposition key. Mirrors `PaiShoGame.state_sig()`.
    pub fn state_sig(&self) -> u64 {
        let mut h = self.board_sig();
        h ^= splitmix64(0xA000_0000_0000 ^ (player_sig_index(self.current_player) << 1) ^ (self.bonus_turn as u64));
        for (&player, hand) in &self.hands {
            for (&tile, &count) in hand {
                h ^= splitmix64(
                    0xB000_0000_0000 ^ (player_sig_index(player) << 24) ^ (tile_sig_index(tile) << 12) ^ (count as u64),
                );
            }
        }
        h
    }
}

/// A fast, fixed, deterministic mixing function (splitmix64) used to turn
/// small per-piece/per-hand-entry keys into well-distributed bits before
/// XOR-combining them in `Board::board_sig`/`state_sig`.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Small stable integer for a `Player`, used only to build sig keys.
fn player_sig_index(player: Player) -> u64 {
    match player {
        Player::One => 0,
        Player::Two => 1,
    }
}

/// Small stable integer for a `Tile`, used only to build sig keys. The three
/// tile families (flower/accent/special) are offset into disjoint ranges so
/// the combined index is unique across all tile kinds.
fn tile_sig_index(tile: Tile) -> u64 {
    match tile {
        Tile::Flower(f) => f.circle_index() as u64,
        Tile::Accent(a) => 10 + match a {
            AccentTile::Rock => 0,
            AccentTile::Wheel => 1,
            AccentTile::Knotweed => 2,
            AccentTile::Boat => 3,
        },
        Tile::Special(s) => 20 + match s {
            SpecialTile::Orchid => 0,
            SpecialTile::WhiteLotus => 1,
        },
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
        // Computed by running engine/PythonEngine/PaiShoGame.py (post Task 4):
        //   PaiShoGame(opening=None).hands == {'Rose': 3, 'Chrysanthemum': 3, 'Rhododendron': 3,
        //     'Jasmine': 3, 'Lily': 3, 'Jade': 3, 'Rock': 1, 'Wheel': 1, 'Knotweed': 1,
        //     'Boat': 1, 'Orchid': 1, 'WhiteLotus': 1}  (identical for both players; default
        //     setup is one of each accent, informal start)
        let board = Board::new();
        for player in [Player::One, Player::Two] {
            let hand = &board.hands[&player];
            assert_eq!(hand.len(), 12, "{player:?} should have 12 distinct tile kinds");
            for f in Flower::ALL {
                assert_eq!(hand[&Tile::Flower(f)], 3, "{player:?}/{f:?} should start with 3");
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
    fn state_sig_is_order_independent_across_the_same_two_placements() {
        let mut a = Board::new();
        a.pieces.insert(Position::new(9, 8), piece(Tile::Flower(Flower::Rose), Player::One));
        a.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Jade), Player::Two));

        let mut b = Board::new();
        b.pieces.insert(Position::new(9, 10), piece(Tile::Flower(Flower::Jade), Player::Two));
        b.pieces.insert(Position::new(9, 8), piece(Tile::Flower(Flower::Rose), Player::One));

        assert_eq!(a.board_sig(), b.board_sig());
        assert_eq!(a.state_sig(), b.state_sig());
    }

    #[test]
    fn state_sig_changes_when_current_player_flips_but_board_sig_does_not() {
        let mut a = Board::new();
        a.pieces.insert(Position::new(9, 8), piece(Tile::Flower(Flower::Rose), Player::One));
        let mut b = a.clone();
        b.current_player = Player::Two;

        assert_eq!(a.board_sig(), b.board_sig());
        assert_ne!(a.state_sig(), b.state_sig());
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
    fn empowered_orchid_cannot_capture_an_accent_tile() {
        // Task 5 / rulebook p.12: a raging Orchid captures Flower Tiles — accent tiles are
        // never captured by anything, wild Orchid included. Was
        // `empowered_orchid_can_capture_any_enemy_tile_kind`'s vs_accent case before this
        // rule was tightened; see fixture capture_raging_orchid_vs_accent.
        let mut vs_accent = Board::new();
        vs_accent.pieces.insert(Position::new(9, 9), piece(Tile::Special(SpecialTile::Orchid), Player::One));
        vs_accent.pieces.insert(Position::new(9, 10), piece(Tile::Accent(AccentTile::Rock), Player::Two));
        vs_accent.pieces.insert(Position::new(5, 5), piece(Tile::Special(SpecialTile::WhiteLotus), Player::One));
        assert!(!vs_accent.valid_destinations(Position::new(9, 9)).contains(&Position::new(9, 10)));
    }

    #[test]
    fn empowered_orchid_can_capture_any_enemy_flower_tile_kind() {
        // An Orchid mover with its own blooming WL can capture an enemy WhiteLotus, and even
        // an enemy Orchid — both otherwise-uncapturable tile kinds for a basic flower. Each
        // enemy kept 2 cells away to dodge the trap.
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
                Action::Plant { tile: Tile::Flower(_), at, .. } => assert!(GATES.contains(at)),
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
        // Task 8 (official rule): a Harmony Bonus offers exactly one action — accent, special
        // flower, basic flower (if none of the actor's are growing), or skip — and never an
        // arrange, so the 11 arrange actions Task 7 counted here are gone and `skip_bonus` is
        // added instead. Everything else recomputed against the Python oracle directly for
        // this exact setup: 24 basic-flower gate plants (Rose@(9,9) isn't growing, so the bonus
        // doesn't suppress them); 721 Rock/Wheel/Knotweed plants (fewer than the flat 3*243=729,
        // because a Wheel next to Rose/Jasmine is now illegal wherever rotating them would foul
        // a gate/garden/clash — see `wheel_rotation`); 7 Boat plants — 4 choices of displacement
        // for the Rose at (9,9) (Boat may target the mover's own tile) and 3 for the enemy
        // Jasmine at (9,10); 8 special plants (2 kinds * 4 empty gates); plus 1 for `skip_bonus`
        // = 761 total.
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
        let boat_plants: Vec<(Position, Option<Position>)> = actions
            .iter()
            .filter_map(|a| match a {
                Action::Plant { tile: Tile::Accent(AccentTile::Boat), at, displace } => Some((*at, *displace)),
                _ => None,
            })
            .collect();
        let special_plants = actions
            .iter()
            .filter(|a| matches!(a, Action::Plant { tile: Tile::Special(_), .. }))
            .count();
        let arranges = actions.iter().filter(|a| matches!(a, Action::Arrange { .. })).count();

        assert_eq!(basic_plants, 24);
        assert_eq!(accent_plants, 721);
        assert_eq!(boat_plants.iter().filter(|(at, _)| *at == Position::new(9, 9)).count(), 4);
        assert_eq!(boat_plants.iter().filter(|(at, _)| *at == Position::new(9, 10)).count(), 3);
        assert!(boat_plants.iter().all(|(_, displace)| displace.is_some()), "every Boat-on-a-flower plant carries a displacement");
        assert_eq!(boat_plants.len(), 7);
        assert_eq!(special_plants, 8);
        assert_eq!(arranges, 0, "a Harmony Bonus never offers arrange, regardless of growing state");
        assert!(actions.contains(&Action::SkipBonus));
        assert_eq!(actions.len(), 761);
    }

    #[test]
    fn bonus_turn_never_offers_an_arrange_growing_or_not() {
        // g.board[(1,9)]={'flower':'Rose','player':1,'growing':True}, g.bonus_turn=True
        // Task 8 (official rule): arranging is not one of the Harmony Bonus options at all —
        // not just for a still-growing source tile (see
        // occupying_a_gate_removes_its_plant_actions_but_adds_arrange_actions's 15 arranges from
        // the very same growing Rose when bonus_turn is false).
        let mut board = Board::new();
        board.bonus_turn = true;
        board.pieces.insert(
            Position::new(1, 9),
            Piece { tile: Tile::Flower(Flower::Rose), player: Player::One, growing: true },
        );
        let arranges = board.legal_actions().into_iter().filter(|a| matches!(a, Action::Arrange { .. })).count();
        assert_eq!(arranges, 0);
    }
}
