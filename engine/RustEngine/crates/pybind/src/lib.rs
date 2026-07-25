//! PyO3 bridge: exposes `pai_sho_engine::game::Board` as a `PaiShoGame`
//! Python class matching `engine/PythonEngine/PaiShoGame.py`'s public
//! surface (per `docs/superpowers/specs/2026-07-24-rust-engine-design.md`
//! §4), so existing agent code can use it as a duck-typed drop-in
//! replacement. This is the only crate in the workspace that knows about
//! Python — `crates/engine` stays a pure, dependency-free rules crate.
//!
//! `current_state_web()` is intentionally not ported: it's a manual dev
//! helper that POSTs to a locally-running Flask server for debugging, has
//! no callers anywhere in the codebase (agents, server.py, simulator.py,
//! tests), and pulls in an HTTP client purely for that. Add it here only if
//! something starts actually calling it on a Rust-backed game.

mod convert;
mod message;

use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use pai_sho_engine::board::Position;
use pai_sho_engine::game::{Action, Board};
use pai_sho_engine::moves::{MoveError, Outcome};
use pai_sho_engine::piece::Piece;
use pai_sho_engine::player::Player;
use pai_sho_engine::tile::Tile;

use convert::{flower_from_name, player_from_int, player_to_int, tile_from_name, tile_name};

fn move_error_to_py(e: MoveError) -> PyErr {
    let msg = match e {
        MoveError::NotYourTile => "Not your tile",
        MoveError::InvalidMove => "Invalid move",
        MoveError::MustPlantInGate => "Must plant in a Gate",
        MoveError::GateOccupied => "Gate occupied",
        MoveError::NoTilesOfThatType => "No tiles of that type left",
        MoveError::BoatMustTargetEnemy => "Boat must target an enemy tile",
        MoveError::InvalidDisplacement => "Invalid displacement position",
        MoveError::CannotPlaceAccentOnGate => "Cannot place accent tile on a gate",
        MoveError::InvalidPosition => "Invalid position",
        MoveError::SpaceOccupied => "Space occupied",
        MoveError::GameOver => "Game is already over.",
    };
    PyValueError::new_err(msg)
}

fn parse_player(player: i32) -> PyResult<Player> {
    player_from_int(player).ok_or_else(|| PyValueError::new_err("'current_player' must be 1 or 2"))
}

fn parse_tile(name: &str) -> PyResult<Tile> {
    tile_from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown tile name: {name:?}")))
}

fn position_tuple(py: Python<'_>, pos: Position) -> Py<PyTuple> {
    PyTuple::new(py, [pos.row, pos.col]).expect("2-tuple construction cannot fail").unbind()
}

fn piece_dict<'py>(py: Python<'py>, piece: Piece) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("flower", tile_name(piece.tile))?;
    dict.set_item("player", player_to_int(piece.player))?;
    dict.set_item("growing", piece.growing)?;
    Ok(dict)
}

fn hand_dict<'py>(py: Python<'py>, hand: &HashMap<Tile, i32>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (&tile, &count) in hand {
        dict.set_item(tile_name(tile), count)?;
    }
    Ok(dict)
}

fn action_to_tuple(py: Python<'_>, action: Action) -> PyResult<Py<PyTuple>> {
    let tuple = match action {
        Action::Plant { tile, at } => PyTuple::new(py, [
            "plant".into_pyobject(py)?.into_any().unbind(),
            tile_name(tile).into_pyobject(py)?.into_any().unbind(),
            at.row.into_pyobject(py)?.into_any().unbind(),
            at.col.into_pyobject(py)?.into_any().unbind(),
        ])?,
        Action::Arrange { from, to } => PyTuple::new(py, [
            "arrange".into_pyobject(py)?.into_any().unbind(),
            from.row.into_pyobject(py)?.into_any().unbind(),
            from.col.into_pyobject(py)?.into_any().unbind(),
            to.row.into_pyobject(py)?.into_any().unbind(),
            to.col.into_pyobject(py)?.into_any().unbind(),
        ])?,
    };
    Ok(tuple.unbind())
}

fn action_to_list<'py>(py: Python<'py>, action: Action) -> PyResult<Bound<'py, PyList>> {
    match action {
        Action::Plant { tile, at } => PyList::new(py, [
            "plant".into_pyobject(py)?.into_any().unbind(),
            tile_name(tile).into_pyobject(py)?.into_any().unbind(),
            at.row.into_pyobject(py)?.into_any().unbind(),
            at.col.into_pyobject(py)?.into_any().unbind(),
        ]),
        Action::Arrange { from, to } => PyList::new(py, [
            "arrange".into_pyobject(py)?.into_any().unbind(),
            from.row.into_pyobject(py)?.into_any().unbind(),
            from.col.into_pyobject(py)?.into_any().unbind(),
            to.row.into_pyobject(py)?.into_any().unbind(),
            to.col.into_pyobject(py)?.into_any().unbind(),
        ]),
    }
}

/// Parses a Python action tuple — `('plant', flower, r, c)` or
/// `('arrange', fr, fc, tr, tc)` — into an `Action`. Matches
/// `PaiShoGame.step`'s own unpacking, including its `Unknown action type`
/// error for anything else.
fn parse_action(action: &Bound<'_, PyAny>) -> PyResult<Action> {
    let tuple: &Bound<PyTuple> = action.downcast().map_err(|_| PyTypeError::new_err("action must be a tuple"))?;
    let kind: String = tuple.get_item(0)?.extract()?;
    match kind.as_str() {
        "plant" => {
            let flower: String = tuple.get_item(1)?.extract()?;
            let r: i32 = tuple.get_item(2)?.extract()?;
            let c: i32 = tuple.get_item(3)?.extract()?;
            Ok(Action::Plant { tile: parse_tile(&flower)?, at: Position::new(r, c) })
        }
        "arrange" => {
            let fr: i32 = tuple.get_item(1)?.extract()?;
            let fc: i32 = tuple.get_item(2)?.extract()?;
            let tr: i32 = tuple.get_item(3)?.extract()?;
            let tc: i32 = tuple.get_item(4)?.extract()?;
            Ok(Action::Arrange { from: Position::new(fr, fc), to: Position::new(tr, tc) })
        }
        other => Err(PyValueError::new_err(format!("Unknown action type: {other}"))),
    }
}

fn winner_to_py(winner: Option<Outcome>) -> Option<i32> {
    match winner {
        None => None,
        Some(Outcome::Tie) => Some(0),
        Some(Outcome::Winner(p)) => Some(player_to_int(p)),
    }
}

fn winner_from_py(winner: Option<i32>) -> PyResult<Option<Outcome>> {
    match winner {
        None => Ok(None),
        Some(0) => Ok(Some(Outcome::Tie)),
        Some(1) => Ok(Some(Outcome::Winner(Player::One))),
        Some(2) => Ok(Some(Outcome::Winner(Player::Two))),
        Some(other) => Err(PyValueError::new_err(format!("'winner' must be None, 0, 1, or 2, got {other}"))),
    }
}

#[pyclass(name = "PaiShoGame", module = "RustEngine")]
pub struct PyPaiShoGame {
    board: Board,
    history: Vec<Action>,
    message: String,
}

impl PyPaiShoGame {
    fn from_board(board: Board) -> Self {
        let message = if board.winner.is_some() {
            message::terminal_message(&board)
        } else {
            message::ongoing_message(&board, board.bonus_turn)
        };
        Self { board, history: Vec::new(), message }
    }

    fn refresh_message(&mut self, was_bonus_turn_before: bool) {
        self.message = if self.board.winner.is_some() {
            message::terminal_message(&self.board)
        } else {
            message::ongoing_message(&self.board, !was_bonus_turn_before && self.board.bonus_turn)
        };
    }
}

#[pymethods]
impl PyPaiShoGame {
    #[new]
    fn new() -> Self {
        Self { board: Board::new(), history: Vec::new(), message: message::initial_message() }
    }

    fn reset(&mut self) {
        self.board.reset();
        self.history.clear();
        self.message = message::initial_message();
    }

    /// Deep copy — search agents (minimax, MCTS) clone before simulating a
    /// candidate move. Named `clone` (not `__deepcopy__`) to match
    /// `PaiShoGame.clone()`'s call sites throughout `Agents/`.
    fn clone(&self) -> Self {
        Self { board: self.board.clone(), history: self.history.clone(), message: self.message.clone() }
    }

    #[getter]
    fn board(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (&pos, &piece) in &self.board.pieces {
            dict.set_item(position_tuple(py, pos), piece_dict(py, piece)?)?;
        }
        Ok(dict.unbind())
    }

    #[setter(board)]
    fn set_board(&mut self, board: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut pieces = HashMap::new();
        for (key, value) in board.iter() {
            let (r, c): (i32, i32) = key.extract()?;
            let dict: &Bound<PyDict> = value.downcast().map_err(|_| PyTypeError::new_err("board values must be dicts"))?;
            let flower: String = dict.get_item("flower")?.ok_or_else(|| PyValueError::new_err("tile missing 'flower'"))?.extract()?;
            let player: i32 = dict.get_item("player")?.ok_or_else(|| PyValueError::new_err("tile missing 'player'"))?.extract()?;
            let growing: bool = dict.get_item("growing")?.ok_or_else(|| PyValueError::new_err("tile missing 'growing'"))?.extract()?;
            pieces.insert(
                Position::new(r, c),
                Piece { tile: parse_tile(&flower)?, player: parse_player(player)?, growing },
            );
        }
        self.board.pieces = pieces;
        Ok(())
    }

    #[getter]
    fn hands(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item(1, hand_dict(py, &self.board.hands[&Player::One])?)?;
        dict.set_item(2, hand_dict(py, &self.board.hands[&Player::Two])?)?;
        Ok(dict.unbind())
    }

    #[getter]
    fn current_player(&self) -> i32 {
        player_to_int(self.board.current_player)
    }

    #[setter(current_player)]
    fn set_current_player(&mut self, value: i32) -> PyResult<()> {
        self.board.current_player = parse_player(value)?;
        Ok(())
    }

    #[getter]
    fn winner(&self) -> Option<i32> {
        winner_to_py(self.board.winner)
    }

    #[setter(winner)]
    fn set_winner(&mut self, value: Option<i32>) -> PyResult<()> {
        self.board.winner = winner_from_py(value)?;
        Ok(())
    }

    #[getter]
    fn bonus_turn(&self) -> bool {
        self.board.bonus_turn
    }

    #[setter(bonus_turn)]
    fn set_bonus_turn(&mut self, value: bool) {
        self.board.bonus_turn = value;
    }

    #[getter]
    fn message(&self) -> String {
        self.message.clone()
    }

    #[setter(message)]
    fn set_message(&mut self, value: String) {
        self.message = value;
    }

    #[getter]
    fn history(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for &action in &self.history {
            list.append(action_to_list(py, action)?)?;
        }
        Ok(list.unbind())
    }

    #[setter(history)]
    fn set_history(&mut self, history: &Bound<'_, PyList>) -> PyResult<()> {
        let mut parsed = Vec::new();
        for entry in history.iter() {
            let tuple_form = PyTuple::new(entry.py(), entry.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
            parsed.push(parse_action(tuple_form.as_any())?);
        }
        self.history = parsed;
        Ok(())
    }

    fn is_harmonious(&self, f1: &str, f2: &str) -> bool {
        match (flower_from_name(f1), flower_from_name(f2)) {
            (Some(a), Some(b)) => pai_sho_engine::flower::is_harmonious(a, b),
            _ => false,
        }
    }

    fn is_clash(&self, f1: &str, f2: &str) -> bool {
        match (flower_from_name(f1), flower_from_name(f2)) {
            (Some(a), Some(b)) => pai_sho_engine::flower::is_clash(a, b),
            _ => false,
        }
    }

    fn find_harmonies(&self, py: Python<'_>, player: i32) -> PyResult<Py<PyList>> {
        let player = parse_player(player)?;
        let list = PyList::empty(py);
        for (a, b) in self.board.find_harmonies(player) {
            list.append(PyTuple::new(py, [position_tuple(py, a), position_tuple(py, b)])?)?;
        }
        Ok(list.unbind())
    }

    fn find_clashes(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for (a, b) in self.board.find_clashes() {
            list.append(PyTuple::new(py, [position_tuple(py, a), position_tuple(py, b)])?)?;
        }
        Ok(list.unbind())
    }

    fn check_harmony_ring(&self, player: i32) -> PyResult<bool> {
        Ok(self.board.check_harmony_ring(parse_player(player)?))
    }

    fn valid_destinations(&self, py: Python<'_>, fr: i32, fc: i32) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for pos in self.board.valid_destinations(Position::new(fr, fc)) {
            list.append(position_tuple(py, pos))?;
        }
        Ok(list.unbind())
    }

    fn get_legal_actions(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for action in self.board.legal_actions() {
            list.append(action_to_tuple(py, action)?)?;
        }
        Ok(list.unbind())
    }

    fn step(&mut self, action: &Bound<'_, PyAny>) -> PyResult<bool> {
        let action = parse_action(action)?;
        let was_bonus_turn_before = self.board.bonus_turn;
        let has_winner = self.board.step(action).map_err(move_error_to_py)?;
        self.history.push(action);
        self.refresh_message(was_bonus_turn_before);
        Ok(has_winner)
    }

    #[pyo3(signature = (flower, r, c, displace_r=None, displace_c=None))]
    fn plant(&mut self, flower: &str, r: i32, c: i32, displace_r: Option<i32>, displace_c: Option<i32>) -> PyResult<()> {
        let tile = parse_tile(flower)?;
        let displace = match (displace_r, displace_c) {
            (Some(dr), Some(dc)) => Some(Position::new(dr, dc)),
            _ => None,
        };
        let was_bonus_turn_before = self.board.bonus_turn;
        self.board.plant(tile, Position::new(r, c), displace).map_err(move_error_to_py)?;
        self.history.push(Action::Plant { tile, at: Position::new(r, c) });
        self.refresh_message(was_bonus_turn_before);
        Ok(())
    }

    fn arrange(&mut self, fr: i32, fc: i32, tr: i32, tc: i32) -> PyResult<()> {
        let from = Position::new(fr, fc);
        let to = Position::new(tr, tc);
        let was_bonus_turn_before = self.board.bonus_turn;
        self.board.arrange(from, to).map_err(move_error_to_py)?;
        self.history.push(Action::Arrange { from, to });
        self.refresh_message(was_bonus_turn_before);
        Ok(())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, pyo3::types::PyType>, d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let board_obj = d.get_item("board")?.ok_or_else(|| PyTypeError::new_err("'board' must be an object mapping 'r,c' -> tile"))?;
        let board_dict: &Bound<PyDict> =
            board_obj.downcast().map_err(|_| PyTypeError::new_err("'board' must be an object mapping 'r,c' -> tile"))?;

        let mut pieces = HashMap::new();
        for (key, value) in board_dict.iter() {
            let key_str: String = key.extract().map_err(|_| PyValueError::new_err("malformed board key"))?;
            let mut parts = key_str.split(',');
            let (r, c) = match (parts.next(), parts.next(), parts.next()) {
                (Some(r), Some(c), None) => (
                    r.trim().parse::<i32>().map_err(|e| PyValueError::new_err(format!("malformed board key: {e}")))?,
                    c.trim().parse::<i32>().map_err(|e| PyValueError::new_err(format!("malformed board key: {e}")))?,
                ),
                _ => return Err(PyValueError::new_err(format!("malformed board key: {key_str:?}"))),
            };
            let tile_dict: &Bound<PyDict> =
                value.downcast().map_err(|_| PyTypeError::new_err("board tile values must be dicts"))?;
            let flower: String = tile_dict
                .get_item("flower")?
                .ok_or_else(|| PyValueError::new_err("tile missing 'flower'"))?
                .extract()?;
            let player: i32 = tile_dict
                .get_item("player")?
                .ok_or_else(|| PyValueError::new_err("tile missing 'player'"))?
                .extract()?;
            let growing: bool =
                tile_dict.get_item("growing")?.ok_or_else(|| PyValueError::new_err("tile missing 'growing'"))?.extract()?;
            pieces.insert(
                Position::new(r, c),
                Piece { tile: parse_tile(&flower)?, player: parse_player(player)?, growing },
            );
        }

        let hands_obj = d.get_item("hands")?.ok_or_else(|| PyTypeError::new_err("'hands' must be an object"))?;
        let hands_dict: &Bound<PyDict> = hands_obj.downcast().map_err(|_| PyTypeError::new_err("'hands' must be an object"))?;
        let mut hands = HashMap::new();
        for player in [Player::One, Player::Two] {
            let key_int = player_to_int(player);
            let raw = hands_dict
                .get_item(key_int.to_string())?
                .or(hands_dict.get_item(key_int)?)
                .unwrap_or_else(|| PyDict::new(hands_dict.py()).into_any());
            let mut hand = HashMap::new();
            if let Ok(raw_dict) = raw.downcast::<PyDict>() {
                for (k, v) in raw_dict.iter() {
                    let name: String = k.extract()?;
                    let count: i32 = v.extract()?;
                    if let Some(tile) = tile_from_name(&name) {
                        hand.insert(tile, count);
                    }
                }
            }
            hands.insert(player, hand);
        }

        let current_player_raw: i32 = d
            .get_item("current_player")?
            .ok_or_else(|| PyValueError::new_err("'current_player' must be 1 or 2"))?
            .extract()
            .map_err(|_| PyValueError::new_err("'current_player' must be 1 or 2"))?;
        let current_player = parse_player(current_player_raw)?;

        let winner_raw: Option<i32> = match d.get_item("winner")? {
            Some(v) if !v.is_none() => Some(v.extract()?),
            _ => None,
        };
        let winner = winner_from_py(winner_raw)?;

        let bonus_turn = d.get_item("bonus_turn")?.map(|v| v.extract()).transpose()?.unwrap_or(false);

        let board = Board { pieces, hands, current_player, bonus_turn, winner };
        Ok(Self::from_board(board))
    }

    #[classmethod]
    fn from_save_dict(cls: &Bound<'_, pyo3::types::PyType>, data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let state = data.get_item("state")?.ok_or_else(|| PyValueError::new_err("save data missing 'state'"))?;
        let state_dict: &Bound<PyDict> = state.downcast().map_err(|_| PyTypeError::new_err("'state' must be an object"))?;
        let mut game = Self::from_dict(cls, state_dict)?;
        if let Some(history) = data.get_item("history")? {
            if let Ok(history_list) = history.downcast::<PyList>() {
                game.set_history(history_list)?;
            }
        }
        Ok(game)
    }

    #[pyo3(signature = (p1_name="Player 1".to_string(), p2_name="Player 2".to_string()))]
    fn to_save_dict(&self, py: Python<'_>, p1_name: String, p2_name: String) -> PyResult<Py<PyDict>> {
        let timestamp: String = py
            .import("time")?
            .call_method1("strftime", ("%Y-%m-%d %H:%M:%S",))?
            .extract()?;

        let board_dict = PyDict::new(py);
        for (&pos, &piece) in &self.board.pieces {
            let key = format!("{},{}", pos.row, pos.col);
            board_dict.set_item(key, piece_dict(py, piece)?)?;
        }

        let hands_dict = PyDict::new(py);
        hands_dict.set_item("1", hand_dict(py, &self.board.hands[&Player::One])?)?;
        hands_dict.set_item("2", hand_dict(py, &self.board.hands[&Player::Two])?)?;

        let state = PyDict::new(py);
        state.set_item("board", board_dict)?;
        state.set_item("hands", hands_dict)?;
        state.set_item("current_player", player_to_int(self.board.current_player))?;
        state.set_item("winner", winner_to_py(self.board.winner))?;
        state.set_item("message", &self.message)?;
        state.set_item("bonus_turn", self.board.bonus_turn)?;

        let history_list = PyList::empty(py);
        for &action in &self.history {
            history_list.append(action_to_list(py, action)?)?;
        }

        let out = PyDict::new(py);
        out.set_item("version", 1)?;
        out.set_item("timestamp", timestamp)?;
        out.set_item("p1", p1_name)?;
        out.set_item("p2", p2_name)?;
        out.set_item("history", history_list)?;
        out.set_item("state", state)?;
        Ok(out.unbind())
    }
}

#[pymodule(name = "RustEngine")]
fn pai_sho_engine_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPaiShoGame>()?;
    Ok(())
}
