/* Pai Sho Lab — the Learn page's 7-step interactive tutorial (`/learn`).
 *
 * Each step asks the server to build a tiny, real engine position
 * (`POST /api/tutorial/<n>`, a `pass`-mode registry game built from
 * frontend/static/tutorial/steps.json through `from_dict`) and renders it
 * with the same board.js module the game page uses — no forked rendering.
 * The goal check mirrors tests/test_integration.py's `_goal_met` exactly,
 * but works from the plain state objects the move endpoints already return
 * (harmony counts, board size, winner, end_reason) rather than the engine.
 */
import * as ui from '/static/js/ui.js';
import { createBoard } from '/static/js/board.js';

const PROGRESS_KEY = 'pai_tutorial_done';
const ACCENTS = ['Rock', 'Wheel', 'Knotweed', 'Boat'];

const dotsEl = document.getElementById('tutorial-dots');
const introEl = document.getElementById('tutorial-intro');
const startBtn = document.getElementById('tutorial-start-btn');
const stepEl = document.getElementById('tutorial-step');
const titleEl = document.getElementById('tutorial-step-title');
const textEl = document.getElementById('tutorial-step-text');
const boardMount = document.getElementById('tutorial-board');
const feedbackEl = document.getElementById('tutorial-feedback');
const resetBtn = document.getElementById('tutorial-reset-btn');
const nextBtn = document.getElementById('tutorial-next-btn');
const doneEl = document.getElementById('tutorial-done');
const restartBtn = document.getElementById('tutorial-restart-btn');

let steps = [];
let stepIndex = 0;        // 0-based into `steps`
let gid = null;
let state = null;         // current game state (ui.server's serialize() shape)
let beforeState = null;   // state as loaded, before the learner's move
let mover = null;         // state.current_player at step start
let solved = false;
let selection = null;
let hints = [];
let board = null;

function readProgress() {
  try {
    const raw = localStorage.getItem(PROGRESS_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function writeProgress(done) {
  try { localStorage.setItem(PROGRESS_KEY, JSON.stringify(done)); } catch { /* private mode etc. */ }
}

function markDone(id) {
  const done = readProgress();
  if (!done.includes(id)) {
    done.push(id);
    writeProgress(done);
  }
}

function renderDots() {
  const done = readProgress();
  dotsEl.innerHTML = steps.map((s, i) => {
    const cls = i === stepIndex && stepEl && !stepEl.hidden ? 'is-current'
      : done.includes(s.id) ? 'is-done' : '';
    return `<li class="${cls}" aria-label="${escapeHtml(s.title)}"></li>`;
  }).join('');
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function ensureBoard() {
  if (!board) {
    board = createBoard(boardMount, { flip: false, onCellClick, onTileClick, onHandClick });
  }
  return board;
}

function clearSelection() {
  selection = null;
  hints = [];
}

function render() {
  board.render(state, { selected: selection, hints, myHandPlayer: solved ? null : state.current_player });
}

function setFeedback(kind, text) {
  feedbackEl.textContent = text;
  feedbackEl.classList.toggle('is-success', kind === 'success');
  feedbackEl.classList.toggle('is-error', kind === 'error');
}

// ---------------------------------------------------------------- goal check

function goalMet(goal, actionKind, actionFlower) {
  switch (goal.type) {
    case 'action_kind':
      return actionKind === goal.kind;
    case 'harmony_created': {
      const before = ((beforeState.harmonies || {})[String(mover)] || []).length;
      const after = ((state.harmonies || {})[String(mover)] || []).length;
      return after > before;
    }
    case 'capture': {
      const beforeN = Object.keys(beforeState.board || {}).length;
      const afterN = Object.keys(state.board || {}).length;
      return actionKind === 'arrange' && afterN < beforeN;
    }
    case 'bonus_used':
      return (state.history || []).length >= 1 && !!beforeState.bonus_turn && actionKind === 'plant';
    case 'accent_placed':
      return actionKind === 'plant' && ACCENTS.includes(actionFlower);
    case 'ring':
      return state.winner === mover && state.end_reason === 'ring';
    default:
      return false;
  }
}

function afterMove(actionKind, actionFlower, goal) {
  if (goalMet(goal, actionKind, actionFlower)) {
    solved = true;
    clearSelection();
    render();
    setFeedback('success', '✓ Nice');
    nextBtn.hidden = false;
    markDone(steps[stepIndex].id);
    renderDots();
  } else {
    setFeedback('error', 'Not quite — try again');
    window.setTimeout(() => startStep(stepIndex), 700);
  }
}

// --------------------------------------------------------------- board move

async function onHandClick(player, flower) {
  if (solved || player !== state.current_player) return;
  if (selection && selection.type === 'hand' && selection.player === player && selection.flower === flower) {
    clearSelection();
    render();
    return;
  }
  selection = { type: 'hand', player, flower };
  hints = [];
  render();
  try {
    const r = await ui.api(`/api/valid_plant_moves/${gid}`, { method: 'POST', body: { tile: flower } });
    hints = r.moves || [];
  } catch {
    hints = [];
  }
  render();
}

async function onTileClick(r, c) {
  if (solved) return;
  if (selection && (selection.type === 'hand' || selection.type === 'boat_target')) {
    return onCellClick(r, c);
  }
  const tile = state.board[`${r},${c}`];
  if (tile && tile.player !== state.current_player) {
    if (selection) return onCellClick(r, c);
    return;
  }
  if (selection && selection.type === 'board' && selection.row === r && selection.col === c) {
    clearSelection();
    render();
    return;
  }
  selection = { type: 'board', row: r, col: c };
  hints = [];
  render();
  try {
    const resp = await ui.api(`/api/valid_moves/${gid}`, { method: 'POST', body: { row: r, col: c } });
    hints = resp.moves || [];
  } catch {
    hints = [];
  }
  render();
}

async function onCellClick(r, c) {
  if (solved || !selection) return;
  const isHint = hints.some(([hr, hc]) => hr === r && hc === c);
  if (!isHint) {
    clearSelection();
    render();
    return;
  }

  if (selection.type === 'boat_target') {
    await doMove('plant', {
      flower: 'Boat', row: selection.targetRow, col: selection.targetCol,
      displace_row: r, displace_col: c,
    });
    return;
  }

  if (selection.type === 'hand') {
    if (selection.flower === 'Boat') {
      let dr;
      try {
        dr = await ui.api(`/api/valid_boat_displacement/${gid}`, { method: 'POST', body: { target_row: r, target_col: c } });
      } catch {
        dr = { moves: [], direct: false };
      }
      if (!dr.direct) {
        if (!dr.moves || dr.moves.length === 0) {
          setFeedback('error', 'No legal Boat move there');
          return;
        }
        selection = { type: 'boat_target', targetRow: r, targetCol: c };
        hints = dr.moves;
        render();
        return;
      }
    }
    await doMove('plant', { flower: selection.flower, row: r, col: c });
    return;
  }

  await doMove('arrange', { from_row: selection.row, from_col: selection.col, to_row: r, to_col: c });
}

async function doMove(kind, body) {
  const path = kind === 'plant' ? `/api/plant/${gid}` : `/api/arrange/${gid}`;
  const goal = steps[stepIndex].goal;
  const flower = kind === 'plant' ? body.flower : null;
  try {
    const r = await ui.api(path, { method: 'POST', body });
    state = r.state;
    clearSelection();
    render();
    afterMove(kind, flower, goal);
  } catch (e) {
    setFeedback('error', (e && e.error) || 'that move is not allowed');
  }
}

// -------------------------------------------------------------- step flow

// Next/Reset/Start are disabled for the duration of a step load, set
// synchronously before the first `await` - this is what closes the bug
// report's race: a fast double-click used to fire two overlapping
// `startStep()` calls, each minting its own tutorial game, which could run
// a learner straight into the per-IP live-game cap (fixed server-side too -
// see play.py's create_tutorial_game) and always left a stray game behind.
function setStepControlsBusy(busy) {
  startBtn.disabled = busy;
  resetBtn.disabled = busy;
  nextBtn.disabled = busy;
}

async function startStep(index) {
  stepIndex = index;
  solved = false;
  clearSelection();
  const step = steps[stepIndex];
  titleEl.textContent = step.title;
  textEl.textContent = step.text;
  setFeedback('', '');
  nextBtn.hidden = true;
  introEl.hidden = true;
  doneEl.hidden = true;
  stepEl.hidden = false;
  renderDots();
  setStepControlsBusy(true);

  try {
    const created = await ui.api(`/api/tutorial/${stepIndex + 1}`, { method: 'POST', body: {} });
    gid = created.game_id;
    const data = await ui.api(`/api/game/${gid}`);
    state = data.state;
    beforeState = data.state;
    mover = state.current_player;
    ensureBoard();
    render();
  } catch (e) {
    // Show the server's actual reason (e.g. "the server is full, try again
    // in a few minutes") rather than a generic message that reads like the
    // site broke even when the backend is healthy and just said no.
    setFeedback('error', (e && e.error) || 'Could not load this step — try again in a moment.');
  } finally {
    setStepControlsBusy(false);
  }
}

function showDone() {
  stepEl.hidden = true;
  introEl.hidden = true;
  doneEl.hidden = false;
  renderDots();
}

function firstIncompleteIndex() {
  const done = readProgress();
  const i = steps.findIndex((s) => !done.includes(s.id));
  return i === -1 ? 0 : i;
}

startBtn.addEventListener('click', () => startStep(firstIncompleteIndex()));
resetBtn.addEventListener('click', () => startStep(stepIndex));
nextBtn.addEventListener('click', () => {
  const next = stepIndex + 1;
  if (next >= steps.length) {
    showDone();
  } else {
    startStep(next);
  }
});
restartBtn.addEventListener('click', () => {
  writeProgress([]);
  startStep(0);
});

async function init() {
  try {
    const r = await fetch('/static/tutorial/steps.json');
    steps = await r.json();
  } catch {
    steps = [];
  }
  renderDots();
}

init();
