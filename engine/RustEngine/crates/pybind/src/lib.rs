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
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyTuple};

use pai_sho_engine::board::Position;
use pai_sho_engine::flower::Flower;
use pai_sho_engine::game::{Action, Board};
use pai_sho_engine::moves::{MoveError, Outcome};
use pai_sho_engine::piece::Piece;
use pai_sho_engine::player::Player;
use pai_sho_engine::setup::{Setup, SetupError};
use pai_sho_engine::tile::{AccentTile, Tile};

use convert::{
    accent_name, end_reason_from_name, end_reason_name, flower_from_name, flower_name, player_from_int,
    player_to_int, tile_from_name, tile_name,
};

fn move_error_to_py(e: MoveError) -> PyErr {
    let msg = match e {
        MoveError::IllegalAction => "Illegal action",
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

fn action_items(py: Python<'_>, action: Action) -> PyResult<Vec<PyObject>> {
    let mut items: Vec<PyObject> = Vec::new();
    match action {
        Action::Plant { tile, at, displace } => {
            items.push("plant".into_pyobject(py)?.into_any().unbind());
            items.push(tile_name(tile).into_pyobject(py)?.into_any().unbind());
            items.push(at.row.into_pyobject(py)?.into_any().unbind());
            items.push(at.col.into_pyobject(py)?.into_any().unbind());
            if let Some(to) = displace {
                items.push(to.row.into_pyobject(py)?.into_any().unbind());
                items.push(to.col.into_pyobject(py)?.into_any().unbind());
            }
        }
        Action::Arrange { from, to } => {
            items.push("arrange".into_pyobject(py)?.into_any().unbind());
            for v in [from.row, from.col, to.row, to.col] {
                items.push(v.into_pyobject(py)?.into_any().unbind());
            }
        }
        Action::SkipBonus => items.push("skip_bonus".into_pyobject(py)?.into_any().unbind()),
    }
    Ok(items)
}

fn action_to_tuple(py: Python<'_>, action: Action) -> PyResult<Py<PyTuple>> {
    Ok(PyTuple::new(py, action_items(py, action)?)?.unbind())
}

fn action_to_list<'py>(py: Python<'py>, action: Action) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, action_items(py, action)?)
}

/// Parses `('plant', tile, r, c)`, `('plant', 'Boat', r, c, dr, dc)` or
/// `('arrange', fr, fc, tr, tc)`. Wrong arity raises ValueError (never silently truncates).
fn parse_action(action: &Bound<'_, PyAny>) -> PyResult<Action> {
    let tuple: &Bound<PyTuple> = action.downcast().map_err(|_| PyTypeError::new_err("action must be a tuple"))?;
    let kind: String = tuple.get_item(0)?.extract()?;
    let int = |i: usize| -> PyResult<i32> { tuple.get_item(i)?.extract() };
    match (kind.as_str(), tuple.len()) {
        ("plant", 4) => Ok(Action::Plant {
            tile: parse_tile(&tuple.get_item(1)?.extract::<String>()?)?,
            at: Position::new(int(2)?, int(3)?),
            displace: None,
        }),
        ("plant", 6) => Ok(Action::Plant {
            tile: parse_tile(&tuple.get_item(1)?.extract::<String>()?)?,
            at: Position::new(int(2)?, int(3)?),
            displace: Some(Position::new(int(4)?, int(5)?)),
        }),
        ("arrange", 5) => Ok(Action::Arrange {
            from: Position::new(int(1)?, int(2)?),
            to: Position::new(int(3)?, int(4)?),
        }),
        ("skip_bonus", 1) => Ok(Action::SkipBonus),
        ("plant", n) | ("arrange", n) => Err(PyValueError::new_err(format!("Malformed {kind} action with {n} elements"))),
        (other, _) => Err(PyValueError::new_err(format!("Unknown action type: {other}"))),
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

fn setup_error_to_py(e: SetupError) -> PyErr {
    PyValueError::new_err(match e {
        SetupError::WrongAccentCount(p) => format!("accents for player {} must list exactly 4 tiles", player_to_int(p)),
        SetupError::TooManyOfOneAccent(p, a) => format!(
            "at most 2 of each accent tile per player (player {}: {})",
            player_to_int(p),
            accent_name(a)
        ),
    })
}

fn accent_list(accents: &Bound<'_, PyDict>, player: i32) -> PyResult<Vec<AccentTile>> {
    let raw = match accents.get_item(player)? {
        Some(v) => v,
        None => accents
            .get_item(player.to_string())?
            .ok_or_else(|| PyValueError::new_err(format!("accents for player {player} must list exactly 4 tiles")))?,
    };
    let names: Vec<String> = raw.extract()?;
    names
        .iter()
        .map(|n| match tile_from_name(n) {
            Some(Tile::Accent(a)) => Ok(a),
            _ => Err(PyValueError::new_err(format!("unknown accent tile: '{n}'"))),
        })
        .collect()
}

fn accents_from_py(accents: Option<&Bound<'_, PyDict>>) -> PyResult<(Vec<AccentTile>, Vec<AccentTile>)> {
    match accents {
        None => Ok((AccentTile::ALL.to_vec(), AccentTile::ALL.to_vec())),
        Some(d) => Ok((accent_list(d, 1)?, accent_list(d, 2)?)),
    }
}

/// Builds a `Setup` from the Python constructor's `accents`/`opening` arguments.
/// 'random' is drawn with Python's `random.choice` over the six basic flowers
/// (same list, same order as the Python engine), so a seeded `random` gives both
/// engines the same opening.
fn setup_from_py(py: Python<'_>, accents: Option<&Bound<'_, PyDict>>, opening: Option<&str>) -> PyResult<Setup> {
    let (p1, p2) = accents_from_py(accents)?;
    let opening = match opening {
        None => None,
        Some("random") => {
            let names: Vec<&str> = Flower::ALL.iter().map(|&f| flower_name(f)).collect();
            let pick: String = py.import("random")?.call_method1("choice", (names,))?.extract()?;
            flower_from_name(&pick)
        }
        Some(name) => Some(
            flower_from_name(name)
                .ok_or_else(|| PyValueError::new_err("opening must be a basic flower name, 'random', or None"))?,
        ),
    };
    Setup::new(p1, p2, opening).map_err(setup_error_to_py)
}

/// Builds a `Setup` from a saved `setup` dict (`from_dict`/`from_save_dict`). Unlike
/// `setup_from_py`, `opening` here must already be a concrete flower name or `None` —
/// a save is expected to store the opening that was actually resolved, never the
/// literal 'random' convenience the constructor accepts. Rejecting it here keeps
/// `from_dict` deterministic (it never touches the global `random` stream) and
/// matches the Python engine's identical rejection in `PaiShoGame.from_dict`.
fn setup_from_saved(accents: Option<&Bound<'_, PyDict>>, opening: Option<&str>) -> PyResult<Setup> {
    let (p1, p2) = accents_from_py(accents)?;
    let opening = match opening {
        None => None,
        Some(name) => Some(
            flower_from_name(name).ok_or_else(|| PyValueError::new_err("opening must be a basic flower name or None"))?,
        ),
    };
    Setup::new(p1, p2, opening).map_err(setup_error_to_py)
}

/// Accepts only an int `version` equal to 2, with the Python engine's exact messages:
/// an older int "predates the official rules"; anything else (a newer int, a bool, a
/// string, a missing key) is an "unsupported save version", shown by its Python repr.
fn check_save_version(version: Option<Bound<'_, PyAny>>) -> PyResult<()> {
    let unsupported = |repr: String| PyValueError::new_err(format!("unsupported save version {repr}; this build reads v2"));
    let v = match version {
        None => return Err(unsupported("None".to_string())),
        Some(v) => v,
    };
    if v.is_instance_of::<PyBool>() || !v.is_instance_of::<PyInt>() || v.gt(2)? {
        return Err(unsupported(v.repr()?.to_string()));
    }
    if v.lt(2)? {
        return Err(PyValueError::new_err(format!(
            "save format v{} predates the official rules; only v2 saves can be loaded",
            v.str()?
        )));
    }
    Ok(())
}

#[pyclass(name = "PaiShoGame", module = "RustEngine")]
pub struct PyPaiShoGame {
    board: Board,
    history: Vec<Action>,
    history_players: Vec<i32>,
    message: String,
    /// `board.legal_actions()` for the current state, filled lazily by
    /// `get_legal_actions`/`step` and cleared by every mutation of `board`, so a ply
    /// that lists the actions and then steps generates them once, not twice.
    legal: Option<Vec<Action>>,
}

impl PyPaiShoGame {
    fn from_board(board: Board) -> Self {
        let message = message::message_for(&board, board.bonus_turn);
        Self { board, history: Vec::new(), history_players: Vec::new(), message, legal: None }
    }

    fn legal_actions(&mut self) -> &[Action] {
        let board = &self.board;
        self.legal.get_or_insert_with(|| board.legal_actions())
    }

    fn refresh_message(&mut self, was_bonus_turn_before: bool) {
        self.message = message::message_for(&self.board, !was_bonus_turn_before && self.board.bonus_turn);
    }
}

#[pymethods]
impl PyPaiShoGame {
    #[new]
    #[pyo3(signature = (accents=None, opening=Some(String::from("random"))))]
    fn new(py: Python<'_>, accents: Option<&Bound<'_, PyDict>>, opening: Option<String>) -> PyResult<Self> {
        let setup = setup_from_py(py, accents, opening.as_deref())?;
        Ok(Self {
            board: Board::with_setup(setup),
            history: Vec::new(),
            history_players: Vec::new(),
            message: message::initial_message(),
            legal: None,
        })
    }

    #[pyo3(signature = (accents=None, opening=Some(String::from("random"))))]
    fn reset(&mut self, py: Python<'_>, accents: Option<&Bound<'_, PyDict>>, opening: Option<String>) -> PyResult<()> {
        let setup = setup_from_py(py, accents, opening.as_deref())?;
        self.board = Board::with_setup(setup);
        self.legal = None;
        self.history.clear();
        self.history_players.clear();
        self.message = message::initial_message();
        Ok(())
    }

    #[getter]
    fn setup(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let accents = PyDict::new(py);
        for player in [Player::One, Player::Two] {
            let names: Vec<&str> = self.board.setup.accents[&player].iter().map(|&a| accent_name(a)).collect();
            accents.set_item(player_to_int(player), names)?;
        }
        let out = PyDict::new(py);
        out.set_item("accents", accents)?;
        out.set_item("opening", self.board.setup.opening.map(flower_name))?;
        Ok(out.unbind())
    }

    #[getter]
    fn end_reason(&self) -> Option<&'static str> {
        self.board.end_reason.map(end_reason_name)
    }

    /// Deep copy — search agents (minimax, MCTS) clone before simulating a
    /// candidate move. Named `clone` (not `__deepcopy__`) to match
    /// `PaiShoGame.clone()`'s call sites throughout `Agents/`.
    fn clone(&self) -> Self {
        Self {
            board: self.board.clone(),
            history: self.history.clone(),
            history_players: self.history_players.clone(),
            message: self.message.clone(),
            legal: self.legal.clone(),
        }
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
        self.legal = None;
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
        self.legal = None;
        Ok(())
    }

    #[getter]
    fn winner(&self) -> Option<i32> {
        winner_to_py(self.board.winner)
    }

    #[setter(winner)]
    fn set_winner(&mut self, value: Option<i32>) -> PyResult<()> {
        self.board.winner = winner_from_py(value)?;
        self.legal = None;
        Ok(())
    }

    #[getter]
    fn bonus_turn(&self) -> bool {
        self.board.bonus_turn
    }

    #[setter(bonus_turn)]
    fn set_bonus_turn(&mut self, value: bool) {
        self.board.bonus_turn = value;
        self.legal = None;
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
        // An externally replaced history has unknown movers.
        self.history_players.clear();
        Ok(())
    }

    #[getter]
    fn history_players(&self) -> Vec<i32> {
        self.history_players.clone()
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

    /// Order-independent hash of the tiles on the board. Mirrors
    /// `PaiShoGame.board_sig()` in meaning (not value); only comparable
    /// within this engine/process.
    fn board_sig(&self) -> u64 {
        self.board.board_sig()
    }

    /// `board_sig` plus side to move, bonus flag and both hands: the
    /// transposition key. Mirrors `PaiShoGame.state_sig()`.
    fn state_sig(&self) -> u64 {
        self.board.state_sig()
    }

    fn check_harmony_ring(&self, player: i32) -> PyResult<bool> {
        Ok(self.board.check_harmony_ring(parse_player(player)?))
    }

    fn count_midline_harmonies(&self, player: i32) -> PyResult<i32> {
        Ok(self.board.count_midline_harmonies(parse_player(player)?))
    }

    fn valid_destinations(&self, py: Python<'_>, fr: i32, fc: i32) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for pos in self.board.valid_destinations(Position::new(fr, fc)) {
            list.append(position_tuple(py, pos))?;
        }
        Ok(list.unbind())
    }

    fn get_legal_actions(&mut self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for &action in self.legal_actions() {
            list.append(action_to_tuple(py, action)?)?;
        }
        Ok(list.unbind())
    }

    fn step(&mut self, action: &Bound<'_, PyAny>) -> PyResult<bool> {
        let items: Vec<Bound<'_, PyAny>> = action.try_iter()?.collect::<PyResult<_>>()?;
        let action = parse_action(PyTuple::new(action.py(), items)?.as_any())?;
        if self.board.winner.is_some() {
            return Err(move_error_to_py(MoveError::GameOver));
        }
        if !self.legal_actions().contains(&action) {
            return Err(move_error_to_py(MoveError::IllegalAction));
        }
        let was_bonus_turn_before = self.board.bonus_turn;
        let actor = player_to_int(self.board.current_player);
        self.legal = None;
        let over = self.board.step_prevalidated(action);
        self.history.push(action);
        self.history_players.push(actor);
        self.refresh_message(was_bonus_turn_before);
        Ok(over)
    }

    fn skip_bonus(&mut self, py: Python<'_>) -> PyResult<()> {
        let action = PyTuple::new(py, ["skip_bonus"])?;
        self.step(action.as_any()).map(|_| ())
    }

    fn resign(&mut self, player: i32) -> PyResult<()> {
        self.board.resign(parse_player(player)?).map_err(move_error_to_py)?;
        self.legal = None;
        self.message = message::message_for(&self.board, false);
        Ok(())
    }

    #[pyo3(signature = (flower, r, c, displace_r=None, displace_c=None))]
    fn plant(&mut self, py: Python<'_>, flower: &str, r: i32, c: i32, displace_r: Option<i32>, displace_c: Option<i32>) -> PyResult<()> {
        let mut items: Vec<PyObject> = vec![
            "plant".into_pyobject(py)?.into_any().unbind(),
            flower.into_pyobject(py)?.into_any().unbind(),
            r.into_pyobject(py)?.into_any().unbind(),
            c.into_pyobject(py)?.into_any().unbind(),
        ];
        if let (Some(dr), Some(dc)) = (displace_r, displace_c) {
            items.push(dr.into_pyobject(py)?.into_any().unbind());
            items.push(dc.into_pyobject(py)?.into_any().unbind());
        }
        self.step(PyTuple::new(py, items)?.as_any()).map(|_| ())
    }

    fn arrange(&mut self, py: Python<'_>, fr: i32, fc: i32, tr: i32, tc: i32) -> PyResult<()> {
        let mut items: Vec<PyObject> = vec!["arrange".into_pyobject(py)?.into_any().unbind()];
        for v in [fr, fc, tr, tc] {
            items.push(v.into_pyobject(py)?.into_any().unbind());
        }
        self.step(PyTuple::new(py, items)?.as_any()).map(|_| ())
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

        let setup = match d.get_item("setup")? {
            Some(s) if !s.is_none() => {
                let sd: &Bound<PyDict> = s.downcast().map_err(|_| PyTypeError::new_err("'setup' must be an object"))?;
                let accents = match sd.get_item("accents")? {
                    Some(a) if !a.is_none() => Some(a.downcast_into::<PyDict>().map_err(|_| PyTypeError::new_err("'accents' must be an object"))?),
                    _ => None,
                };
                let opening: Option<String> = match sd.get_item("opening")? {
                    Some(o) if !o.is_none() => Some(o.extract()?),
                    _ => None,
                };
                setup_from_saved(accents.as_ref(), opening.as_deref())?
            }
            _ => Setup::default(),
        };
        let end_reason = match d.get_item("end_reason")? {
            Some(v) if !v.is_none() => {
                let name: String = v.extract()?;
                Some(end_reason_from_name(&name).ok_or_else(|| PyValueError::new_err(format!("unknown end_reason: {name:?}")))?)
            }
            _ => None,
        };
        let board = Board { pieces, hands, current_player, bonus_turn, winner, setup, end_reason };
        let mut game = Self::from_board(board);
        if let Some(h) = d.get_item("history")? {
            if !h.is_none() {
                let list: &Bound<PyList> = h.downcast().map_err(|_| PyTypeError::new_err("'history' must be a list"))?;
                game.set_history(list)?;
            }
        }
        if let Some(p) = d.get_item("history_players")? {
            if !p.is_none() {
                let players: Vec<i32> = p.extract()?;
                if players.len() == game.history.len() {
                    game.history_players = players;
                }
            }
        }
        if let Some(m) = d.get_item("message")? {
            if let Ok(text) = m.extract::<String>() {
                game.message = text;
            }
        }
        Ok(game)
    }

    #[classmethod]
    fn from_save_dict(cls: &Bound<'_, pyo3::types::PyType>, data: &Bound<'_, PyDict>) -> PyResult<Self> {
        check_save_version(data.get_item("version")?)?;
        let state = data.get_item("state")?.ok_or_else(|| PyValueError::new_err("save data missing 'state'"))?;
        let state_dict: &Bound<PyDict> = state.downcast().map_err(|_| PyTypeError::new_err("'state' must be an object"))?;
        let mut game = Self::from_dict(cls, state_dict)?;
        if let Some(history) = data.get_item("history")? {
            if let Ok(history_list) = history.downcast::<PyList>() {
                game.set_history(history_list)?;
            }
        }
        if let Some(p) = data.get_item("history_players")? {
            if !p.is_none() {
                let players: Vec<i32> = p.extract()?;
                if players.len() == game.history.len() {
                    game.history_players = players;
                }
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
        state.set_item("end_reason", self.board.end_reason.map(end_reason_name))?;

        let accents = PyDict::new(py);
        for player in [Player::One, Player::Two] {
            let names: Vec<&str> = self.board.setup.accents[&player].iter().map(|&a| accent_name(a)).collect();
            accents.set_item(player_to_int(player).to_string(), names)?;
        }
        let setup = PyDict::new(py);
        setup.set_item("accents", accents)?;
        setup.set_item("opening", self.board.setup.opening.map(flower_name))?;
        state.set_item("setup", setup)?;

        let history_list = PyList::empty(py);
        for &action in &self.history {
            history_list.append(action_to_list(py, action)?)?;
        }

        let out = PyDict::new(py);
        out.set_item("version", 2)?;
        out.set_item("timestamp", timestamp)?;
        out.set_item("p1", p1_name)?;
        out.set_item("p2", p2_name)?;
        out.set_item("history", history_list)?;
        out.set_item("history_players", self.history_players.clone())?;
        out.set_item("state", state)?;
        Ok(out.unbind())
    }
}

#[pymodule(name = "RustEngine")]
fn pai_sho_engine_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPaiShoGame>()?;
    Ok(())
}
