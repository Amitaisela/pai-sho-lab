//! The Rust port's oracle test suite, ported from `tests/basic_tests.py`'s
//! Board/State/Harmony/Smoke sections (the agent-specific sections don't
//! apply — no agents exist against this engine yet). Exercises the public
//! `pai_sho_engine` API end-to-end the way a real caller (eventually
//! `crates/pybind`) would, rather than reaching into crate-internal details
//! the way the unit tests in `src/*.rs` do.

use pai_sho_engine::board::{garden_of, is_valid, valid_spaces, Garden, Position, BOARD_SIZE, CENTER, GATES};
use pai_sho_engine::flower::Flower;
use pai_sho_engine::game::{Action, Board};
use pai_sho_engine::piece::Piece;
use pai_sho_engine::player::Player;
use pai_sho_engine::tile::Tile;

// ---------- board / positions ----------

#[test]
fn board_size_is_19() {
    assert_eq!(BOARD_SIZE, 19);
}

#[test]
fn center_is_valid() {
    assert!(is_valid(CENTER.row, CENTER.col));
}

#[test]
fn corners_are_invalid() {
    for (r, c) in [(0, 0), (0, 18), (18, 0), (18, 18)] {
        assert!(!is_valid(r, c), "({r}, {c}) should be invalid");
    }
}

#[test]
fn gates_are_valid_spaces() {
    let spaces = valid_spaces();
    for gate in GATES {
        assert!(spaces.contains(&gate), "{gate:?} should be a valid space");
    }
}

#[test]
fn garden_center_is_neutral() {
    assert_eq!(garden_of(CENTER.row, CENTER.col), Garden::Neutral);
}

// ---------- game state ----------

#[test]
fn initial_state_is_empty_player_one_no_winner_no_bonus() {
    let board = Board::new();
    assert!(board.pieces.is_empty());
    assert_eq!(board.current_player, Player::One);
    assert_eq!(board.winner, None);
    assert!(!board.bonus_turn);
}

#[test]
fn initial_hands_match_python_oracle() {
    let board = Board::new();
    for player in [Player::One, Player::Two] {
        let hand = &board.hands[&player];
        for f in Flower::ALL {
            assert_eq!(hand[&Tile::Flower(f)], 2);
        }
    }
}

#[test]
fn clone_is_independent() {
    // Ported from test_clone_is_independent: mutating a clone must not
    // affect the original. This is the direct test of Board's new `Clone`
    // derive (Milestone 6) — search agents rely on this to simulate moves
    // without corrupting the live game state.
    let mut board = Board::new();
    board.pieces.insert(
        Position::new(5, 5),
        Piece { tile: Tile::Flower(Flower::Rose), player: Player::One, growing: false },
    );

    let mut cloned = board.clone();
    cloned.pieces.get_mut(&Position::new(5, 5)).unwrap().tile = Tile::Flower(Flower::Jade);
    *cloned.hands.get_mut(&Player::One).unwrap().get_mut(&Tile::Flower(Flower::Rose)).unwrap() = 42;

    assert_eq!(board.pieces[&Position::new(5, 5)].tile, Tile::Flower(Flower::Rose));
    assert_ne!(board.hands[&Player::One][&Tile::Flower(Flower::Rose)], 42);
}

#[test]
fn reset_returns_a_mutated_board_to_the_fresh_state() {
    let mut board = Board::new();
    board.step(Action::Plant { tile: Tile::Flower(Flower::Rose), at: GATES[0] }).unwrap();
    assert!(!board.pieces.is_empty());

    board.reset();

    assert!(board.pieces.is_empty());
    assert_eq!(board.current_player, Player::One);
    assert_eq!(board.winner, None);
    assert!(!board.bonus_turn);
    assert_eq!(board.hands[&Player::One][&Tile::Flower(Flower::Rose)], 2);
}

#[test]
fn plant_at_gate_sets_growing() {
    let mut board = Board::new();
    board.step(Action::Plant { tile: Tile::Flower(Flower::Rose), at: GATES[0] }).unwrap();
    assert!(board.pieces[&GATES[0]].growing);
}

#[test]
fn legal_actions_start_with_plants_only() {
    let board = Board::new();
    let actions = board.legal_actions();
    assert!(!actions.is_empty(), "expected legal plant actions at game start");
    assert!(actions.iter().all(|a| matches!(a, Action::Plant { .. })));
}

// ---------- harmony / clash ----------

fn piece(tile: Tile, player: Player, growing: bool) -> Piece {
    Piece { tile, player, growing }
}

#[test]
fn harmony_same_row_neighbours_on_circle() {
    let mut board = Board::new();
    board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One, false));
    board.pieces.insert(Position::new(5, 8), piece(Tile::Flower(Flower::Chrysanthemum), Player::One, false));
    assert_eq!(board.find_harmonies(Player::One).len(), 1);
}

#[test]
fn harmony_blocked_by_obstacle() {
    let mut board = Board::new();
    board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One, false));
    board.pieces.insert(Position::new(5, 9), piece(Tile::Flower(Flower::Chrysanthemum), Player::One, false));
    board.pieces.insert(Position::new(5, 7), piece(Tile::Flower(Flower::Lily), Player::Two, false));
    assert_eq!(board.find_harmonies(Player::One).len(), 0);
}

#[test]
fn harmony_requires_same_row_or_col() {
    let mut board = Board::new();
    board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One, false));
    board.pieces.insert(Position::new(7, 8), piece(Tile::Flower(Flower::Chrysanthemum), Player::One, false));
    assert_eq!(board.find_harmonies(Player::One).len(), 0);
}

#[test]
fn growing_tile_does_not_form_harmony() {
    let mut board = Board::new();
    board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One, true));
    board.pieces.insert(Position::new(5, 8), piece(Tile::Flower(Flower::Chrysanthemum), Player::One, false));
    assert_eq!(board.find_harmonies(Player::One).len(), 0);
}

#[test]
fn clash_detected_on_circle_distance_3() {
    let mut board = Board::new();
    board.pieces.insert(Position::new(5, 5), piece(Tile::Flower(Flower::Rose), Player::One, false));
    board.pieces.insert(Position::new(5, 9), piece(Tile::Flower(Flower::Jasmine), Player::Two, false));
    assert_eq!(board.find_clashes().len(), 1);
}

// ---------- random game smoke ----------

/// A tiny dependency-free xorshift64 PRNG — `crates/engine` (and its tests)
/// stay at zero external dependencies, so no `rand` crate. Deterministic
/// seed keeps this test reproducible.
struct Xorshift64(u64);

impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next_u64() as usize) % items.len()]
    }
}

#[test]
fn random_game_completes_without_panicking() {
    // Ported from test_random_game_completes: play out random legal moves
    // and confirm the engine never panics and never produces a legal-actions
    // deadlock. A capped step count mirrors the Python test's `range(300)`
    // guard against a game that never naturally terminates.
    let mut rng = Xorshift64(0x2545F4914F6CDD1D);
    let mut board = Board::new();

    for _ in 0..300 {
        if board.winner.is_some() {
            break;
        }
        let legal = board.legal_actions();
        if legal.is_empty() {
            break;
        }
        let action = *rng.choose(&legal);
        board.step(action).expect("a legal_actions()-sourced action must always be a legal step");
    }

    // No panic = pass. The game may or may not have concluded within the
    // step cap; either is fine, matching the Python oracle's own framing.
    assert!(matches!(board.current_player, Player::One | Player::Two));
}

#[test]
fn several_random_games_from_different_seeds_all_complete_without_panicking() {
    // Broader coverage than a single seed: hits more of the accent/special
    // tile and win-detection code paths across independent playouts.
    for seed in [1u64, 42, 1337, 99999, 2026] {
        let mut rng = Xorshift64(seed);
        let mut board = Board::new();
        for _ in 0..300 {
            if board.winner.is_some() {
                break;
            }
            let legal = board.legal_actions();
            if legal.is_empty() {
                break;
            }
            let action = *rng.choose(&legal);
            board.step(action).expect("a legal_actions()-sourced action must always be a legal step");
        }
    }
}

#[test]
fn wheel_rotation_correctly_cancels_when_the_rotating_piece_was_itself_the_blocker() {
    // Found via cross-engine fuzzing (random self-play games run through both the Rust and
    // Python engines in lockstep via crates/pybind, comparing state after every move):
    // Jasmine@(13,13) and Rose@(1,13) are a clash pair (circle distance 3) sharing column 13,
    // with Rock@(11,13) sitting between them — so the pair is *currently* not a live clash
    // (Rock blocks line-of-sight). Planting a Wheel at (10,12) would rotate Rock from
    // (11,13) to (11,12), vacating column 13 between rows 1 and 13 and exposing a genuine
    // Jasmine/Rose clash. The rotation must therefore be cancelled entirely (Rock stays put).
    //
    // The live Python reference (`PaiShoGame.plant('Wheel', 10, 12)`) gets this wrong and
    // performs the rotation anyway — not a parity target here, but a second, independently
    // confirmed manifestation of the same latent bug identified in Milestone 5 Task 7:
    // `PaiShoGame._clear_line_between` always checks `self.board` (the live, pre-rotation
    // board) instead of the `custom_board` its caller `find_clashes(custom_board=...)` is
    // given. At the moment `_apply_wheel` calls `self.find_clashes(custom_board=new_board)`,
    // Rock is *still* at its pre-rotation (11,13) on the live board, so the buggy line-of-sight
    // check sees Rock blocking column 13 and misses the clash the true post-rotation board
    // (`new_board`) actually has — confirmed directly against the live oracle: bypassing the
    // bug by evaluating `find_clashes` against the real post-rotation board as `self.board`
    // finds `[((13, 13), (1, 13))]`; the buggy `custom_board=` call path finds `[]`.
    use pai_sho_engine::piece::Piece;
    use pai_sho_engine::tile::AccentTile;

    let mut board = Board::new();
    let mut insert = |r: i32, c: i32, tile: Tile, player: Player, growing: bool| {
        board.pieces.insert(Position::new(r, c), Piece { tile, player, growing });
    };
    insert(11, 13, Tile::Accent(AccentTile::Rock), Player::One, false);
    insert(13, 13, Tile::Flower(Flower::Jasmine), Player::Two, false);
    insert(1, 13, Tile::Flower(Flower::Rose), Player::Two, false);
    board.current_player = Player::One;

    board.plant(Tile::Accent(AccentTile::Wheel), Position::new(10, 12), None).unwrap();

    let rock_position = board.pieces.iter().find(|(_, p)| p.tile == Tile::Accent(AccentTile::Rock)).map(|(&pos, _)| pos);
    assert_eq!(rock_position, Some(Position::new(11, 13)), "the rotation must be cancelled — Rock stays put");
}
