/* Pai Sho Lab — the game page (`/game/<id>`).
 *
 * Owns everything board.js doesn't: fetching and polling game state, who
 * holds which seat (from localStorage via ui.js), move/hint network calls,
 * the action row (skip bonus, resign, undo/redo, share), the join/resign/
 * result sheets, and the move list. The circular board and both hands are
 * rendered by board.js; this file only decides *what* to render.
 */
import * as ui from '/static/js/ui.js';
import * as sound from '/static/js/sound.js';
import { createBoard, psnMoveLines, psnTurns } from '/static/js/board.js';

const root = document.getElementById('game-root');
const gid = root.dataset.gid;

let state = null;
let mode = null;
let seats = null;
let over = false;
let ctx = { watching: true, mySeat: null, token: null, openSeat: null };

let selection = null;
let hints = [];
let canRedo = false;
let resultShown = false;
let joinOffered = false;
let botInFlight = false;
let pollTimer = null;
let wsLive = false;      // true once this game's state has arrived over the socket
let lastPly = -1;       // history length of the newest state shown; older pushes are ignored
let hintResetTimer = null;
let clockState = null;     // state.clock as last received: {time_control, remaining_ms, running}, or null
let clockReceivedAt = 0;   // performance.now() when clockState arrived
let clockTimer = null;
let flagFetchAt = 0;       // performance.now() of the last "my display hit zero" re-fetch
let pending = null;        // a move sent over the socket and not yet answered: {snapshot, timer}
let viewPly = null;        // null: the live position; else the ply being reviewed in the move list
const positions = new Map();   // ply -> the position after it, fetched for review (cleared on undo)
let flipOverride = false;  // the player flipped the board (the Flip button or `f`)

function clearSelection() {
  selection = null;
  hints = [];
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ------------------------------------------------------------- seat context

function computeSeatContext() {
  if (mode === 'pass') {
    return { watching: false, mySeat: null, token: null, openSeat: null };
  }
  const tok = ui.seatToken(gid);
  if (tok && seats[tok.seat]) {
    return { watching: false, mySeat: tok.seat, token: tok.token, openSeat: null };
  }
  if (mode === 'human') {
    const openSeat = [1, 2].find((p) => seats[p].kind === 'human' && seats[p].name === null);
    if (openSeat) return { watching: false, mySeat: null, token: null, openSeat };
  }
  return { watching: true, mySeat: null, token: null, openSeat: null };
}

function isMyTurnNow() {
  if (!state || over || pending || viewPly !== null) return false;
  if (mode === 'pass') return true;
  return ctx.mySeat !== null && ctx.mySeat === state.current_player;
}

function handPlayerFor() {
  if (!state || over) return null;
  if (mode === 'pass') return state.current_player;
  return ctx.mySeat;
}

function orientation() {
  const natural = mode === 'pass' ? state.current_player === 2 : ctx.mySeat === 2;
  const flip = natural !== flipOverride;
  return { flip, topPlayer: flip ? 1 : 2, bottomPlayer: flip ? 2 : 1 };
}

// -------------------------------------------------------------------- board

const boardEl = document.getElementById('game-board');
const board = createBoard(boardEl, {
  flip: false, onCellClick, onTileClick, onHandClick, onDragStart, onDragEnd,
  handEls: { top: document.getElementById('game-hand-top'), bottom: document.getElementById('game-hand-bottom') },
});

// Wide windows get a drag handle on the board's bottom-right corner to make it smaller than the
// full-height default. The factor (0.5-1 of the largest square that fits) is remembered here.
const FIT_KEY = 'pai_board_fit';
function readFit() {
  try {
    const v = parseFloat(localStorage.getItem(FIT_KEY));
    return v >= 0.5 && v <= 1 ? v : 1;
  } catch { return 1; }
}
function applyFit(v) { root.style.setProperty('--board-fit', String(v)); }
applyFit(readFit());
(function mountResizeHandle() {
  const wrap = boardEl.querySelector('.pb-wrap');
  if (!wrap) return;
  const handle = document.createElement('div');
  handle.id = 'board-resize';
  handle.title = 'Drag to resize the board';
  wrap.appendChild(handle);
  let drag = null;
  handle.addEventListener('pointerdown', (e) => {
    handle.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, fit: readFit(), cell: Math.min(boardEl.clientWidth, boardEl.clientHeight) };
    e.preventDefault();
  });
  handle.addEventListener('pointermove', (e) => {
    if (!drag) return;
    // The board is centred, so a size change of S moves its corner by S/2 on each axis.
    const d = ((e.clientX - drag.x) + (e.clientY - drag.y)) / 2;
    applyFit(Math.min(1, Math.max(0.5, drag.fit + (2 * d) / drag.cell)));
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    try { localStorage.setItem(FIT_KEY, root.style.getPropertyValue('--board-fit')); } catch { /* private mode */ }
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
})();

// Legal moves arrive inside the state (`state.legal`, see serialize.legal_payload), so a
// selection shows its hints in the same frame as the click: no request, no waiting.
function legalBlock() {
  return (state && state.legal) || { plants: {}, boats: {}, arranges: {} };
}

function handHints(flower) {
  const legal = legalBlock();
  if (flower === 'Boat') {
    // a Boat targets an accent (removing both tiles) or a flower (which then moves)
    return [...(legal.plants.Boat || []), ...Object.keys(legal.boats).map((k) => k.split(',').map(Number))];
  }
  return legal.plants[flower] || [];
}

function pieceHints(r, c) {
  return legalBlock().arranges[`${r},${c}`] || [];
}

function boatDisplacement(r, c) {
  const legal = legalBlock();
  return {
    direct: (legal.plants.Boat || []).some(([pr, pc]) => pr === r && pc === c),
    moves: legal.boats[`${r},${c}`] || [],
  };
}

// The hints for the current selection, always read from the state on screen, so a state that
// arrives while something is selected can never leave stale targets behind.
function currentHints() {
  if (!selection || !state) return [];
  if (selection.type === 'hand') return handHints(selection.flower);
  if (selection.type === 'board') return pieceHints(selection.row, selection.col);
  if (selection.type === 'boat_target') return boatDisplacement(selection.targetRow, selection.targetCol).moves;
  return [];
}

// A pointer that has moved off one of my pieces or hand tiles: select it as a click would, and
// say whether there is anywhere to drag it. The drop arrives as onCellClick (see board.js).
function onDragStart(source) {
  if (!isMyTurnNow()) return false;
  if (source.type === 'hand') {
    if (source.player !== state.current_player) return false;
    selection = { type: 'hand', player: source.player, flower: source.flower };
  } else {
    const tile = state.board[`${source.row},${source.col}`];
    if (!tile || tile.player !== state.current_player) return false;
    selection = { type: 'board', row: source.row, col: source.col };
  }
  if (currentHints().length === 0) {
    clearSelection();
    return false;
  }
  sound.buzz(10);
  renderAll();
  return true;
}

function onDragEnd() {
  clearSelection();
  renderAll();
}

function onHandClick(player, flower) {
  if (!isMyTurnNow() || player !== state.current_player) return;
  if (selection && selection.type === 'hand' && selection.player === player && selection.flower === flower) {
    clearSelection();
    renderAll();
    return;
  }
  selection = { type: 'hand', player, flower };
  hints = handHints(flower);
  sound.buzz(10);
  renderAll();
}

async function onTileClick(r, c) {
  if (!isMyTurnNow()) return;
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
    renderAll();
    return;
  }
  selection = { type: 'board', row: r, col: c };
  hints = pieceHints(r, c);
  sound.buzz(10);
  renderAll();
}

async function onCellClick(r, c, info = {}) {
  if (!selection || !isMyTurnNow()) return;
  const isHint = hints.some(([hr, hc]) => hr === r && hc === c);
  if (!isHint) {
    // A finger that missed every legal point keeps the selection (the points are ~20px wide, and
    // starting over after each miss is the worst part of playing on a phone); a mouse click on
    // an empty point, or a drop on one, deselects.
    if (info.pointerType === 'touch' && !info.dropped) return;
    clearSelection();
    renderAll();
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
      const dr = boatDisplacement(r, c);
      if (!dr.direct) {
        if (!dr.moves || dr.moves.length === 0) {
          flashHint('No legal Boat move there');
          return;
        }
        selection = { type: 'boat_target', targetRow: r, targetCol: c };
        hints = dr.moves;
        renderAll();
        return;
      }
    }
    await doMove('plant', { flower: selection.flower, row: r, col: c });
    return;
  }

  await doMove('arrange', { from_row: selection.row, from_col: selection.col, to_row: r, to_col: c });
}

// ------------------------------------------------------- optimistic moves
//
// A move goes over the socket (`move`, with the ply it applies to, so a resend is harmless) and
// is shown at once. The page predicts only what it can predict without the rules engine: an
// Arrange, or a Plant of a flower. Accents (Wheel, Boat, ...) change other tiles, so they wait
// for the server. The server's state push replaces the prediction; an `error` frame, or silence
// for PENDING_MS, restores the exact state from before the click. HTTP stays as the fallback
// whenever the socket is not open.

const PENDING_MS = 4000;
const ACCENTS = ['Rock', 'Wheel', 'Knotweed', 'Boat'];

function actionFor(kind, body) {
  if (kind === 'arrange') return ['arrange', body.from_row, body.from_col, body.to_row, body.to_col];
  if (body.displace_row !== undefined) {
    return ['plant', body.flower, body.row, body.col, body.displace_row, body.displace_col];
  }
  return ['plant', body.flower, body.row, body.col];
}

/** A copy of `s` with `action` applied as far as it can be known on the page, or null. */
function predictMove(s, action) {
  const p = s.current_player;
  const board = { ...s.board };
  let hands = s.hands;
  if (action[0] === 'arrange') {
    const [, fr, fc, tr, tc] = action;
    const tile = board[`${fr},${fc}`];
    if (!tile) return null;
    delete board[`${fr},${fc}`];
    board[`${tr},${tc}`] = { ...tile, growing: false };      // a capture is simply overwritten
  } else if (action[0] === 'plant' && action.length === 4 && !ACCENTS.includes(action[1])) {
    const [, flower, r, c] = action;
    board[`${r},${c}`] = { flower, player: p, growing: true };
    hands = { ...s.hands, [p]: { ...s.hands[p], [flower]: ((s.hands[p] || {})[flower] || 0) - 1 } };
  } else {
    return null;
  }
  // Harmony lines that touch a point the move changed are no longer true: drop them until the
  // server's state (with the real harmonies) arrives.
  const touched = new Set(action[0] === 'arrange'
    ? [`${action[1]},${action[2]}`, `${action[3]},${action[4]}`] : [`${action[2]},${action[3]}`]);
  const harmonies = {};
  for (const k of ['1', '2']) {
    harmonies[k] = ((s.harmonies || {})[k] || []).filter(
      ([a, b]) => !touched.has(`${a[0]},${a[1]}`) && !touched.has(`${b[0]},${b[1]}`));
  }
  return {
    ...s, board, hands, harmonies,
    history: [...(s.history || []), action],
    history_players: [...(s.history_players || []), p],
  };
}

// ------------------------------------------------------ sound and haptics
//
// One sound per move, whoever made it: a thud for a capture, a tick otherwise. `feedbackPly` is
// the history length already sounded, so the optimistic prediction and the server's push for the
// same move make one sound, a first load is silent, and an undo or rollback just resets it.

let feedbackPly = -1;
const lowTimeSounded = { 1: false, 2: false };      // per seat: one warning when a clock drops under 10 s

function giveMoveFeedback(prev, next) {
  const n = (next.history || []).length;
  if (feedbackPly === -1 || n < feedbackPly) { feedbackPly = n; return; }
  if (n === feedbackPly) return;
  feedbackPly = n;
  const last = next.history[n - 1];
  const target = Array.isArray(last) && last[0] === 'arrange' ? `${last[3]},${last[4]}` : null;
  const capture = !!(target && prev && prev.board && prev.board[target]);
  sound.play(capture ? 'capture' : 'move');
  if (ctx.mySeat !== null && next.history_players && next.history_players[n - 1] === ctx.mySeat) sound.buzz(20);
}

function clearPending() {
  if (!pending) return false;
  clearTimeout(pending.timer);
  pending = null;
  return true;
}

function rollBack(message) {
  if (!pending) return;
  const { snapshot } = pending;
  clearPending();
  state = snapshot;
  clearSelection();
  noteClock();
  renderAll();
  if (message) flashHint(message);
}

window.addEventListener('mushi:error', (ev) => {
  if (!pending) return;
  const e = ev.detail || {};
  rollBack(e.message || 'that move is not allowed');
  if (e.code === 'ply_gap' || e.code === 'not_found') refetchGame();     // we were out of sync
});

function sendOverSocket(action) {
  if (!wsLive || !window.MushiLive || !window.MushiLive.isOpen()) return false;
  if (mode !== 'pass' && !ctx.token) return false;
  const snapshot = state;
  const ok = window.MushiLive.move(gid, (state.history || []).length, action, ctx.token || undefined);
  if (!ok) return false;
  pending = { snapshot, timer: setTimeout(() => { if (clearPending()) refetchGame(); }, PENDING_MS) };
  const predicted = predictMove(snapshot, action);
  if (predicted) {
    state = predicted;
    giveMoveFeedback(snapshot, predicted);
  }
  return true;
}

async function doMove(kind, body) {
  if (sendOverSocket(actionFor(kind, body))) {
    canRedo = false;
    clearSelection();
    renderAll();
    return;
  }
  const path = kind === 'plant' ? `/api/plant/${gid}` : `/api/arrange/${gid}`;
  const payload = ctx.token ? { ...body, seat_token: ctx.token } : body;
  try {
    const r = await ui.api(path, { method: 'POST', body: payload });
    applyFreshState(r.state);
    canRedo = false;
    clearSelection();
    renderAll();
    maybeShowResultSheet();
    maybeBotMove();
  } catch (e) {
    flashHint(e && e.error ? e.error : 'that move is not allowed');
  }
}

function applyFreshState(newState) {
  clearPending();
  giveMoveFeedback(state, newState);
  state = newState;
  lastPly = (state.history || []).length;   // undo/redo may legitimately lower this
  over = state.winner !== null && state.winner !== undefined;
  noteClock();
  publishDebugHook();
}

function publishDebugHook() {
  // A read-only hook for Playwright checks, published only with ?debug=1. It never carries
  // the seat token: any script on the page could read it from window.
  if (!new URLSearchParams(location.search).has('debug')) return;
  const { token, ...safeCtx } = ctx;
  window.__pai = { state, mode, over, wsLive, ctx: safeCtx };
}

// ------------------------------------------------------------------ render

// ------------------------------------------------- connection, wake lock, visibility
//
// A "Reconnecting..." pill shows once the live connection has been down for a moment (no flicker
// for a blip), or at once when the browser reports it is offline. Coming back to the tab asks for
// a fresh snapshot, because a phone may have frozen the page and its socket meanwhile. A timed
// game keeps the screen awake for the player: a phone dimming in the middle of a blitz game
// costs real games.

let connTimer = null;

function setConnected(up) {
  const pill = document.getElementById('game-conn');
  if (!pill) return;
  clearTimeout(connTimer);
  if (up && navigator.onLine !== false) {
    pill.hidden = true;
  } else if (navigator.onLine === false) {
    pill.hidden = false;
  } else {
    connTimer = setTimeout(() => { pill.hidden = false; }, 1200);
  }
}

window.addEventListener('offline', () => setConnected(false));
window.addEventListener('online', () => {
  if (window.MushiLive) {
    window.MushiLive.connect();
    ensureSubscribed();
  }
  setConnected(!!(window.MushiLive && window.MushiLive.isOpen()));
  refetchGame();
});

let wakeLock = null;

async function syncWakeLock() {
  if (!('wakeLock' in navigator)) return;
  const want = !!(state && state.clock && !over && !ctx.watching && document.visibilityState === 'visible');
  try {
    if (want && !wakeLock) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
    } else if (!want && wakeLock) {
      const lock = wakeLock;
      wakeLock = null;
      await lock.release();
    }
  } catch (e) {
    wakeLock = null;              // refused (low battery, no permission): the game goes on without
  }
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible' || !state) return;
  syncWakeLock();
  if (window.MushiLive && !window.MushiLive.isOpen()) window.MushiLive.connect();
  ensureSubscribed();             // a fresh snapshot, whatever the socket did while the tab was hidden
  if (!over) refetchGame();
});

function renderAll() {
  if (viewPly !== null && viewPly >= (state.history || []).length) viewPly = null;   // caught up with the live game
  hints = viewPly === null ? currentHints() : [];
  renderBadge();
  renderRated();
  renderStrips();
  renderHint();
  renderActions();
  renderMoves();
  const { flip } = orientation();
  board.setFlip(flip);
  // Reviewing an earlier move shows that position, read-only; the live state is untouched.
  const shown = viewPly !== null && positions.get(viewPly) ? positions.get(viewPly) : state;
  board.render(shown, {
    selected: viewPly === null ? selection : null, hints, myHandPlayer: viewPly === null ? handPlayerFor() : null,
  });
  syncWakeLock();
}

// The board's size is pure CSS (pai.css, ".game": a grid whose board cell is a size container),
// so nothing here measures the window.
function renderBadge() {
  const el = document.getElementById('game-badge');
  if (el) el.hidden = !ctx.watching;
  root.classList.toggle('game--watching', !!ctx.watching);
}

// The name as the strip shows it. A signed-in seat links to the profile (only where the site
// has accounts); a guest's name is only typed text, so it is italic with a "guest" tag; a bot
// carries a BOT tag. This keeps a guest called "MushiBot" from passing as the real thing.
function seatNameHtml(p) {
  const seat = seats[p];
  const label = escapeHtml(seatLabel(p));
  if (!seat || seat.name == null) return label;
  if (seat.kind === 'bot') return `${label} <span class="strip__tag">BOT</span>`;
  if (seat.account && seat.username) {
    return document.body.hasAttribute('data-accounts')
      ? `<a class="strip__user" href="/@/${encodeURIComponent(seat.username)}">${label}</a>`
      : label;
  }
  if (mode === 'pass') return label;
  return `<span class="strip__guest">${label}</span> <span class="strip__tag">guest</span>`;
}

function seatLabel(p) {
  const seat = seats[p];
  if (!seat) return `Player ${p}`;
  if (seat.kind === 'bot') return seat.name || `MushiBot level ${seat.level}`;
  return seat.name || 'Open seat';
}

// ------------------------------------------------------------------ ratings
//
// Display only: everything comes from state.rated, which the server computes. It is null for an
// unrated game; `players` appears once both seats are filled; `result` once the game was rated;
// `unrated_reason` when a rated game turned out not to count.

function ratedBlock() {
  return state && state.rated ? state.rated : null;
}

function ratedPlayer(p) {
  const block = ratedBlock();
  return block && block.players ? block.players[String(p)] || null : null;
}

function ratingText(info) {
  return `${info.rating}${info.provisional ? '?' : ''}`;
}

function signed(n) {
  const v = Math.round(Number(n) || 0);
  return v > 0 ? `+${v}` : String(v);
}

function changeText(change) {
  if (!change) return '';
  return `${change.before} → ${change.after} (${signed(change.after - change.before)})`;
}

// The line under the badge: what a win or a loss is worth before the first move, the rating
// change afterwards, or why the game does not count.
function ratedDetailText() {
  const block = ratedBlock();
  if (!block) return '';
  if (block.unrated_reason) return `Unrated: ${block.unrated_reason}`;
  const result = block.result;
  if (over && result && result.changes) {
    const mine = ctx.mySeat !== null ? result.changes[String(ctx.mySeat)] : null;
    if (mine) return changeText(mine);
    return [1, 2].filter((p) => result.changes[String(p)]).map((p) => {
      const info = ratedPlayer(p);
      return `${info ? info.username : seatLabel(p)} ${changeText(result.changes[String(p)])}`;
    }).join(' · ');
  }
  if (over) return '';
  const hist = state.history || [];
  const mineInfo = ctx.mySeat !== null ? ratedPlayer(ctx.mySeat) : null;
  if (hist.length === 0 && mineInfo) return `${signed(mineInfo.win)} / ${signed(mineInfo.loss)}`;
  return '';
}

function renderRated() {
  const el = document.getElementById('game-rated');
  if (!el) return;
  const block = ratedBlock();
  if (!block) {
    el.hidden = true;
    el.innerHTML = '';
    return;
  }
  el.hidden = false;
  const detail = ratedDetailText();
  el.innerHTML = `<span class="rated-badge">Rated${block.category ? ` · ${escapeHtml(block.category)}` : ''}</span>`
    + `${detail ? ` <span class="rated-detail" id="rated-detail">${escapeHtml(detail)}</span>` : ''}`;
  // A result sheet that opened before the rating arrived picks the change up here.
  const line = document.getElementById('result-rating');
  if (line) line.textContent = detail;
}

function fillStrip(el, p) {
  if (!el) return;
  // The clock spans live in a hidden stash until a strip claims them; park them there before
  // the strip's markup is rebuilt, so rebuilding it never destroys them.
  const stash = document.getElementById('game-clock-stash');
  if (stash) el.querySelectorAll('[data-clock]').forEach((n) => stash.appendChild(n));
  const isTurn = !over && state.current_player === p;
  el.classList.toggle('is-turn', isTurn);
  const you = ctx.mySeat === p ? ' <span class="muted">(you)</span>' : '';
  const info = ratedPlayer(p);
  const rating = info ? ` <span class="strip__rating" data-rating-seat="${p}">${escapeHtml(ratingText(info))}</span>` : '';
  el.innerHTML = `<div class="strip__name"><span class="strip__turn-dot"></span>${seatNameHtml(p)}${rating}${you}</div>`;
  const clockEl = document.querySelector(`[data-clock="${p}"]`);
  if (clockEl) el.appendChild(clockEl);
  renderClocks();
}

// ------------------------------------------------------------------- clocks
//
// Display only. The server's numbers arrive with every state (HTTP or socket) and are
// re-applied each time; between arrivals the running side's clock is extrapolated from
// performance.now(), so a wrong or tampered browser clock never decides anything. When
// the local count hits zero the page re-fetches the game rather than deciding a result.

function noteClock() {
  clockState = state && state.clock ? state.clock : null;
  clockReceivedAt = performance.now();
  if (clockState && !over) {
    if (!clockTimer) clockTimer = setInterval(tickClocks, 100);
  } else if (clockTimer) {
    clearInterval(clockTimer);
    clockTimer = null;
  }
  renderClocks();
}

function clockLeft(seat) {
  const base = Number(clockState.remaining_ms[String(seat)]) || 0;
  const elapsed = !over && clockState.running === seat ? performance.now() - clockReceivedAt : 0;
  return Math.max(0, base - elapsed);
}

function renderClocks() {
  document.querySelectorAll('[data-clock]').forEach((el) => {
    if (!clockState) {
      el.hidden = true;
      return;
    }
    const seat = Number(el.dataset.clock);
    const left = clockLeft(seat);
    const running = !over && clockState.running === seat;
    el.hidden = false;
    el.textContent = running && left <= 0 ? 'Time…' : ui.formatClock(left);
    el.classList.toggle('clock--running', running);
    el.classList.toggle('clock--low', left < 10000);
    if (running && left < 10000 && left > 0 && ctx.mySeat === seat && !lowTimeSounded[seat]) {
      lowTimeSounded[seat] = true;
      sound.play('lowtime');
      sound.buzz([60, 80, 60]);
    }
    if (left >= 10000) lowTimeSounded[seat] = false;
    el.classList.toggle('clock--warn', left >= 10000 && left < 30000);
  });
}

function tickClocks() {
  if (!clockState || over) {
    noteClock();
    return;
  }
  renderClocks();
  const runner = clockState.running;
  if (runner && clockLeft(runner) <= 0 && performance.now() - flagFetchAt > 1500) {
    flagFetchAt = performance.now();
    refetchGame();
  }
}

// One fetch that lets the server settle (and push) the result; nothing is decided here.
async function refetchGame() {
  try {
    onGameData(await ui.api(`/api/game/${gid}`, { headers: seatHeaders(ctx.token) }));
  } catch (e) {
    if (e && e.status === 404) showExpired();
  }
}

function renderStrips() {
  const { topPlayer, bottomPlayer } = orientation();
  fillStrip(document.getElementById('game-top-strip'), topPlayer);
  fillStrip(document.getElementById('game-bottom-strip'), bottomPlayer);
}

function turnAction() {
  return state.bonus_turn
    ? 'place an accent or special tile, plant a flower if none of yours are growing, or skip'
    : 'plant a flower in a gate, or move one of yours';
}

function renderHint() {
  const el = document.getElementById('game-hint');
  if (!el) return;
  if (viewPly !== null) {
    el.textContent = `Move ${viewPly} of ${(state.history || []).length} - Live (End) returns to the game`;
    return;
  }
  if (over) {
    el.textContent = state.message || '';
    return;
  }
  if (ctx.watching) {
    el.textContent = state.message || '';
    return;
  }
  if (mode === 'pass') {
    // Same names as the top/bottom strips (seatLabel() reads them from the
    // same seats data) rather than a separate Host/Guest vocabulary.
    el.textContent = `${seatLabel(state.current_player)} to move — ${turnAction()}`;
    return;
  }
  const mine = ctx.mySeat === state.current_player;
  if (!mine) {
    el.textContent = `${seatLabel(state.current_player)} is thinking…`;
    return;
  }
  el.textContent = state.bonus_turn
    ? `Harmony! Bonus move — ${turnAction()}`
    : `Your turn — ${turnAction()}`;
}

function flashHint(msg) {
  const el = document.getElementById('game-hint');
  if (!el) return;
  clearTimeout(hintResetTimer);
  el.textContent = msg;
  hintResetTimer = setTimeout(renderHint, 1800);
}

function actionButton(id, label, cls, onClick) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.id = id;
  btn.className = `btn ${cls}`;
  btn.textContent = label;
  btn.addEventListener('click', onClick);
  return btn;
}

function canResignNow() {
  if (over) return false;
  if (mode === 'pass') return true;
  return ctx.mySeat !== null;
}

function renderActions() {
  const el = document.getElementById('game-actions');
  if (!el) return;
  el.innerHTML = '';
  const flipBtn = () => actionButton('flip-btn', 'Flip', 'btn-quiet', toggleFlip);
  if (over || ctx.watching) {
    el.appendChild(flipBtn());
    return;
  }

  if (state.can_skip_bonus && isMyTurnNow()) {
    el.appendChild(actionButton('skip-btn', 'Skip bonus', 'btn-secondary', skipBonus));
  }
  if (canResignNow()) {
    el.appendChild(actionButton('resign-btn', 'Resign', 'btn-quiet', confirmResign));
  }
  if (mode !== 'human' && !state.clock && (state.history || []).length > 0) {
    el.appendChild(actionButton('undo-btn', 'Undo', 'btn-quiet', doUndo));
    if (canRedo) el.appendChild(actionButton('redo-btn', 'Redo', 'btn-quiet', doRedo));
  }
  el.appendChild(flipBtn());
  // The seat is still open: the one useful link invites an opponent, and a watcher who opened
  // it would be offered the seat, so only "Invite to play" is shown. Once both seats are
  // filled the link can only be watched.
  if (mode === 'human' && hasOpenSeat()) {
    el.appendChild(actionButton('invite-btn', 'Invite to play', 'btn-secondary', () => shareGameLink('invite')));
  } else {
    el.appendChild(actionButton('share-btn', 'Share to watch', 'btn-quiet', () => shareGameLink('watch')));
  }
}

// ----------------------------------------------------------------- move list
//
// PSN, as the engine writes it (psnTurns in board.js): a table of turns (Guest | Host) on a wide
// window, a horizontal strip on a phone. Tapping a move, or the arrow keys, shows the position
// after it, read-only (fetched once from /api/game/{gid}/position/{ply} and cached); "Live"
// (End) returns. New moves keep arriving underneath without pulling the board away.

function realPlies() {
  // ply numbers worth stopping on: 0, then the history length after every move that is not a skip
  const out = [0];
  (state.history || []).forEach((m, i) => { if (!Array.isArray(m) || m[0] !== 'skip_bonus') out.push(i + 1); });
  return out;
}

let viewReq = 0;          // the newest review request; an older one that finishes late is dropped

async function showPly(ply) {
  const n = (state.history || []).length;
  const plies = realPlies();
  if (ply >= plies[plies.length - 1]) { goLive(); return; }
  ply = Math.max(0, Math.min(ply, n));
  const req = ++viewReq;
  if (!positions.has(ply)) {
    try {
      const r = await ui.api(`/api/game/${gid}/position/${ply}`, { headers: seatHeaders(ctx.token) });
      positions.set(ply, r.state);
    } catch (e) {
      if (req === viewReq) goLive();
      return;
    }
  }
  if (req !== viewReq) return;
  // Review starts only once the position is in hand, so the live board is never shown under a
  // "reviewing" label.
  viewPly = ply;
  clearSelection();
  renderAll();
}

function goLive() {
  viewReq++;
  viewPly = null;
  renderAll();
}

function stepPly(direction) {
  if (!state) return;
  const plies = realPlies();
  const cur = viewPly === null ? plies[plies.length - 1] : viewPly;
  const target = direction < 0
    ? [...plies].reverse().find((p) => p < cur)
    : plies.find((p) => p > cur);
  if (target !== undefined) showPly(target);
}

function turnCell(t) {
  const shownPly = viewPly === null ? Infinity : viewPly;
  const cur = (ply) => (viewPly === null ? ply === realPlies().slice(-1)[0] : ply === shownPly);
  let html = `<span class="mv-turn is-p${t.player || 1}"><button type="button" class="mv-cell" data-ply="${t.mainPly}"`
    + `${cur(t.mainPly) ? ' aria-current="true"' : ''}>${escapeHtml(t.main)}</button>`;
  if (t.bonus) {
    html += `<button type="button" class="mv-cell mv-bonus" data-ply="${t.bonusPly}"`
      + `${cur(t.bonusPly) ? ' aria-current="true"' : ''}>+${escapeHtml(t.bonus)}</button>`;
  }
  return `${html}</span>`;
}

let psnUrl = null;

function psnText() {
  const accents = (state.setup && state.setup.accents) || {};
  const result = { 1: '1-0', 2: '0-1', 0: '1/2-1/2' }[state.winner] || '*';
  const turns = psnTurns(state.history || [], state.history_players || []);
  const stamp = new Date().toISOString().slice(0, 19).replace('T', ' ');
  const tag = (k, v) => `[${k} "${String(v).replace(/"/g, "'")}"]`;
  const lines = [
    tag('Event', 'Pai Sho Lab Game'), tag('Date', stamp),
    tag('Player1', seatLabel(1)), tag('Player2', seatLabel(2)),
    tag('GuestAccents', (accents['1'] || []).join(',')), tag('HostAccents', (accents['2'] || []).join(',')),
    tag('Opening', (state.setup && state.setup.opening) || 'Informal'), tag('Result', result),
  ];
  if (state.end_reason) lines.push(tag('Termination', state.end_reason));
  lines.push(tag('Turns', turns.length), '');
  lines.push(...psnMoveLines(state.history || [], state.history_players || [], !!state.bonus_turn));
  return `${lines.join('\n')}\n`;
}

function renderMoves() {
  const list = document.getElementById('game-move-list');
  const liveBtn = document.getElementById('mv-live');
  if (!list) return;
  const turns = psnTurns(state.history || [], state.history_players || []);
  list.innerHTML = turns.map((t, i) => (i % 2 === 0
    ? `<li class="mv-row"><span class="mv-no">${i / 2 + 1}.</span>${turnCell(t)}`
    : `${turnCell(t)}</li>`)).join('') + (turns.length % 2 === 1 ? '</li>' : '');
  if (viewPly === null) {
    list.scrollTop = list.scrollHeight;
    list.scrollLeft = list.scrollWidth;
  } else {
    const cur = list.querySelector('[aria-current="true"]');
    if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  if (liveBtn) liveBtn.hidden = viewPly === null;
  const dl = document.getElementById('mv-download');
  if (dl) {
    dl.hidden = !over || turns.length === 0;
    if (!dl.hidden) {
      if (psnUrl) URL.revokeObjectURL(psnUrl);
      psnUrl = URL.createObjectURL(new Blob([psnText()], { type: 'text/plain' }));
      dl.href = psnUrl;
    }
  }
  document.querySelectorAll('#mv-nav [data-nav]').forEach((btn) => {
    const plies = realPlies();
    const cur = viewPly === null ? plies[plies.length - 1] : viewPly;
    const atStart = cur === 0;
    const atEnd = viewPly === null;
    btn.disabled = ((btn.dataset.nav === 'first' || btn.dataset.nav === 'prev') && atStart)
      || ((btn.dataset.nav === 'next' || btn.dataset.nav === 'last') && atEnd);
  });
}

document.getElementById('game-moves').addEventListener('click', (e) => {
  const cell = e.target.closest('.mv-cell');
  if (cell) { showPly(Number(cell.dataset.ply)); return; }
  const nav = e.target.closest('[data-nav]');
  if (nav) {
    if (nav.dataset.nav === 'first') showPly(0);
    else if (nav.dataset.nav === 'prev') stepPly(-1);
    else if (nav.dataset.nav === 'next') stepPly(1);
    else goLive();
    return;
  }
  if (e.target.closest('#mv-live')) goLive();
});

function toggleFlip() {
  flipOverride = !flipOverride;
  renderAll();
}

// Keyboard: arrows and Home/End walk the move list, `f` flips, `s` skips a bonus, Esc deselects
// (or leaves review), `?` lists them. Ignored while typing, with a modifier held, or while the
// focus is on the board, where the arrows move between points (board.js).
function showShortcuts() {
  ui.sheet(`
    <h2>Keyboard</h2>
    <ul class="shortcut-list">
      <li><kbd>&larr;</kbd> <kbd>&rarr;</kbd> previous / next move</li>
      <li><kbd>Home</kbd> <kbd>End</kbd> first move / back to the live game</li>
      <li><kbd>f</kbd> flip the board</li>
      <li><kbd>s</kbd> skip a Harmony Bonus</li>
      <li><kbd>Esc</kbd> deselect, or leave review</li>
      <li>On the board: arrows move between points, <kbd>Enter</kbd> selects or plays</li>
    </ul>`);
}

window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || !state) return;
  const t = e.target;
  if (t && (t.closest('input, textarea, select, [contenteditable]') || t.closest('.sheet'))) return;
  const onBoard = t && t.closest && t.closest('.pb-grid');
  if (!onBoard && e.key === 'ArrowLeft') { e.preventDefault(); stepPly(-1); }
  else if (!onBoard && e.key === 'ArrowRight') { e.preventDefault(); stepPly(1); }
  else if (e.key === 'Home') { e.preventDefault(); showPly(0); }
  else if (e.key === 'End') { e.preventDefault(); goLive(); }
  else if (e.key === 'Escape') {
    if (viewPly !== null) goLive();
    else if (selection) { clearSelection(); renderAll(); }
  } else if (e.key === 'f') toggleFlip();
  else if (e.key === 's' && state.can_skip_bonus && isMyTurnNow()) skipBonus();
  else if (e.key === '?') showShortcuts();
});

// ------------------------------------------------------------------ action handlers

async function skipBonus() {
  try {
    const body = ctx.token ? { seat_token: ctx.token } : {};
    const r = await ui.api(`/api/skip_bonus/${gid}`, { method: 'POST', body });
    applyFreshState(r.state);
    canRedo = false;
    clearSelection();
    renderAll();
    maybeShowResultSheet();
    maybeBotMove();
  } catch (e) {
    flashHint(e && e.error ? e.error : 'could not skip the bonus');
  }
}

function resignPlayer() {
  return mode === 'pass' ? state.current_player : ctx.mySeat;
}

function confirmResign() {
  const player = resignPlayer();
  const close = ui.sheet(`
    <h2>Resign?</h2>
    <p class="muted">Your opponent will win the game.</p>
    <div class="game__sheet-actions">
      <button type="button" class="btn btn-primary" id="confirm-resign-btn">Resign</button>
      <button type="button" class="btn btn-quiet" id="cancel-resign-btn">Cancel</button>
    </div>
  `);
  document.getElementById('cancel-resign-btn').addEventListener('click', close);
  document.getElementById('confirm-resign-btn').addEventListener('click', async () => {
    close();
    try {
      const body = { player };
      if (ctx.token) body.seat_token = ctx.token;
      const r = await ui.api(`/api/resign/${gid}`, { method: 'POST', body });
      applyFreshState(r.state);
      renderAll();
      maybeShowResultSheet();
    } catch (e) {
      flashHint(e && e.error ? e.error : 'could not resign');
    }
  });
}

async function doUndo() {
  try {
    const body = mode === 'bot' && ctx.token ? { seat_token: ctx.token } : {};
    const r = await ui.api(`/api/undo/${gid}`, { method: 'POST', body });
    applyFreshState(r.state);
    canRedo = !!r.can_redo;
    clearSelection();
    renderAll();
  } catch (e) {
    flashHint(e && e.error ? e.error : 'nothing to undo');
  }
}

async function doRedo() {
  try {
    const body = mode === 'bot' && ctx.token ? { seat_token: ctx.token } : {};
    const r = await ui.api(`/api/redo/${gid}`, { method: 'POST', body });
    applyFreshState(r.state);
    canRedo = !!r.can_redo;
    clearSelection();
    renderAll();
  } catch (e) {
    flashHint(e && e.error ? e.error : 'nothing to redo');
  }
}

async function shareGameLink(kind) {
  const url = new URL(`/game/${gid}`, location.origin).href;
  const invite = kind === 'invite';
  const result = await ui.shareLink({
    url, title: invite ? 'Play me at Skud Pai Sho' : 'Watch this Skud Pai Sho game',
  });
  if (result === 'copied') flashHint(invite ? 'Invite link copied' : 'Watch link copied');
  else if (result === 'failed') flashHint('Could not copy the link');
}

// -------------------------------------------------------------- bot driver

async function maybeBotMove() {
  if (mode !== 'bot' || over || botInFlight) return;
  if (ctx.mySeat === null || !ctx.token) return;
  const botSeat = 3 - ctx.mySeat;
  if (state.current_player !== botSeat) return;
  botInFlight = true;
  try {
    const r = await ui.api(`/api/bot_move/${gid}`, { method: 'POST', body: { seat_token: ctx.token } });
    applyFreshState(r.state);
    canRedo = false;
    renderAll();
    maybeShowResultSheet();
  } catch (e) {
    if (!e || e.status !== 409) flashHint('The bot could not move');
  } finally {
    botInFlight = false;
  }
}

// ------------------------------------------------------------- join / result

function maybeOfferJoin() {
  if (over || mode !== 'human' || ctx.mySeat !== null || ctx.openSeat === null || joinOffered) return;
  joinOffered = true;
  const close = ui.sheet(`
    <h2>Join this game</h2>
    <p class="muted">Take the open seat and start playing.</p>
    ${ui.inAppBrowserNote('Your seat is kept in this app\'s own storage and is lost when you switch.')}
    <label class="game__field">Your name
      <input id="join-name" type="text" maxlength="24" placeholder="${escapeHtml(ui.nickname())}"
             autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false" enterkeyhint="done">
    </label>
    <div class="game__sheet-actions">
      <button type="button" class="btn btn-primary" id="join-btn">Take the seat</button>
      <button type="button" class="btn btn-quiet" id="join-cancel-btn">Just watch</button>
    </div>
  `);
  document.getElementById('join-cancel-btn').addEventListener('click', close);
  document.getElementById('join-btn').addEventListener('click', async () => {
    const input = document.getElementById('join-name');
    const name = (input.value || '').trim() || ui.nickname();
    try {
      const r = await ui.api(`/api/join/${gid}`, { method: 'POST', body: { nickname: name } });
      ui.setSeatToken(gid, r.seat, r.seat_token);
      close();
      const data = await ui.api(`/api/game/${gid}`, { headers: seatHeaders(r.seat_token) });
      onGameData(data);
      ensureSubscribed();   // re-subscribe with the new seat token
    } catch (e) {
      flashHint(e && e.error ? e.error : 'could not join');
    }
  });
}

const END_REASON_TEXT = {
  ring: 'Harmony Ring',
  last_basic_flower: 'Last basic flower — midline harmonies decide',
  no_moves: 'No legal moves',
  resign: 'Resignation',
  timeout: 'Time out',
};

function resultDetail() {
  if (state.winner === 0) return 'Tie';
  if (state.end_reason === 'timeout' && (state.winner === 1 || state.winner === 2)) {
    return `${seatLabel(3 - state.winner)} ran out of time`;
  }
  return END_REASON_TEXT[state.end_reason] || state.message || 'Game over';
}

function resultHeadline() {
  if (ctx.mySeat != null) {
    if (state.winner === 0) return 'Tie';
    if (state.winner === ctx.mySeat) return 'You win';
    if (state.winner) return 'You lose';
  }
  return resultDetail();
}

// The finished game's own time control as a preset key, so a rematch plays like the game just
// played. An untimed game rematches untimed; a non-preset (test-only) control falls back to the
// saved choice.
function rematchTimeControl() {
  if (!state.clock) return 'untimed';
  const tc = state.clock.time_control;
  const key = tc ? `${tc.base_s / 60}+${tc.inc_s}` : '';
  return ui.isTimeControl(key) ? key : ui.timeControl();
}

async function doRematch() {
  try {
    if (mode === 'bot') {
      const mySeatNow = ctx.mySeat || 1;
      const botSeat = seats[3 - mySeatNow];
      const level = botSeat ? botSeat.level : 1;
      const r = await ui.api('/api/bot_game', {
        method: 'POST', body: { level, seat: 3 - mySeatNow, nickname: ui.nickname(), time_control: rematchTimeControl() },
      });
      ui.setSeatToken(r.game_id, r.seat, r.seat_token);
      location.href = `/game/${r.game_id}`;
    } else if (mode === 'human') {
      // Rated again only if this game was and I am still signed in with a username; the opponent
      // must be too, or the server refuses their join.
      const acct = ui.getAccountState();
      const rated = !!state.rated && acct.signedIn && !!acct.username && !!state.clock;
      const r = await ui.api('/api/challenge', {
        method: 'POST',
        body: { nickname: ui.nickname(), seat: 'random', time_control: rematchTimeControl(), rated },
      });
      ui.setSeatToken(r.game_id, r.seat, r.seat_token);
      showShareSheet(r.join_url);
    } else {
      const r = await ui.api('/api/pass_game', { method: 'POST', body: {} });
      location.href = `/game/${r.game_id}`;
    }
  } catch (e) {
    flashHint('Could not start a rematch');
  }
}

function showShareSheet(joinUrl) {
  const url = new URL(joinUrl, location.origin).href;
  ui.sheet(`
    <h2>Challenge created</h2>
    <p class="muted">Send this link to your opponent.</p>
    <label class="game__field">Join link
      <input readonly value="${escapeHtml(url)}" onclick="this.select()">
    </label>
  `);
}

function maybeShowResultSheet() {
  if (!over || resultShown) return;
  resultShown = true;
  sound.play('end');
  const close = ui.sheet(`
    <h2 id="result-title">${escapeHtml(resultHeadline())}</h2>
    <p class="muted">${escapeHtml(resultDetail())}</p>
    <p class="rated-detail" id="result-rating">${escapeHtml(ratedDetailText())}</p>
    <div class="game__sheet-actions">
      <button type="button" class="btn btn-primary" id="rematch-btn">Rematch</button>
      <a class="btn btn-secondary" href="/">New opponent</a>
      <a class="btn btn-quiet" href="/">Home</a>
    </div>
  `);
  document.getElementById('rematch-btn').addEventListener('click', () => {
    close();
    doRematch();
  });
}

// ----------------------------------------------------------------- lifecycle

function showExpired() {
  stopPolling();
  root.innerHTML = `
    <div class="game__expired">
      <p>This game has ended or expired.</p>
      <div class="game__sheet-actions">
        <a class="btn btn-primary" href="/">New game</a>
        <a class="btn btn-secondary" href="/">Home</a>
      </div>
    </div>`;
}

function onGameData(data) {
  clearPending();
  giveMoveFeedback(state, data.state);
  state = data.state;
  lastPly = (state.history || []).length;
  mode = data.mode;
  seats = data.seats;
  over = data.over;
  noteClock();
  ctx = computeSeatContext();
  publishDebugHook();
  renderAll();
  maybeOfferJoin();
  maybeShowResultSheet();
  // The FastAPI site plays the bot's reply inside every move request, so this is a
  // no-op there (it isn't the bot's turn any more); the legacy Flask server does not,
  // and relies on this driver. It also revives a bot-to-move position left by an undo.
  maybeBotMove();
  syncPolling();
}

// ------------------------------------------------------------ live updates
//
// State arrives over the shared WebSocket (live.js). Polling survives only as a
// fallback: while the socket is down or was never available (the legacy Flask server
// has no /ws), and, at a slower seats-only pace, while a human seat is still open,
// because taking a seat is not pushed over the socket.

function hasOpenSeat() {
  return mode === 'human' && !!seats && [1, 2].some((p) => seats[p].kind === 'human' && seats[p].name === null);
}

function needsPolling() {
  return !over && (!wsLive || hasOpenSeat());
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function syncPolling() {
  if (needsPolling()) {
    if (!pollTimer) pollTimer = setInterval(pollOnce, 1000);
  } else {
    stopPolling();
  }
}

// Spectators must never keep a game's idle timer alive just by polling it
// (as the old Flask player API did) - only a seated player's own poll does,
// by sending its seat_token.
function seatHeaders(token) {
  return token ? { 'X-Seat-Token': token } : undefined;
}

async function pollOnce() {
  if (over) {
    syncPolling();
    return;
  }
  try {
    const data = await ui.api(`/api/game/${gid}`, { headers: seatHeaders(ctx.token) });
    const prevOpen = hasOpenSeat();
    if (wsLive) {
      // The socket owns the board; this poll only learns who sits where.
      seats = data.seats;
      ctx = computeSeatContext();
      renderStrips();
      renderBadge();
      maybeOfferJoin();
      if (prevOpen && !hasOpenSeat()) {
        // Someone joined. Joining starts a challenge's clock but is not pushed over the
        // socket, so take this response as the authoritative snapshot (it carries the clock).
        ensureSubscribed();   // refresh the socket snapshot too
        onGameData(data);
        return;
      }
      syncPolling();
      return;
    }
    const newLen = (data.state.history || []).length;
    const oldLen = (state.history || []).length;
    const changed = newLen !== oldLen || data.state.winner !== state.winner || data.over !== over;
    // Seats (a name filling an open seat) don't move history/winner, so they're
    // refreshed every tick regardless. The heavier path (board re-render, re-running
    // the join/result/bot side effects) is reserved for an actual game change.
    seats = data.seats;
    if (changed) {
      onGameData(data);
    } else {
      ctx = computeSeatContext();
      renderStrips();
      renderBadge();
      maybeOfferJoin();
      maybeBotMove();
      syncPolling();
    }
  } catch (e) {
    if (e && e.status === 404) showExpired();
  }
}

function sameShown(a, b) {
  return a.winner === b.winner && a.current_player === b.current_player
    && a.bonus_turn === b.bonus_turn && a.end_reason === b.end_reason
    && (a.history || []).length === (b.history || []).length && a.message === b.message
    && JSON.stringify(a.board) === JSON.stringify(b.board)
    && JSON.stringify(a.hands) === JSON.stringify(b.hands)
    && JSON.stringify(a.rated) === JSON.stringify(b.rated);
}

function onSocketState(msg) {
  if (!state) return;                 // the first HTTP load hasn't landed yet
  wsLive = true;
  publishDebugHook();
  const answered = clearPending();    // the server's word replaces any prediction
  // `ply` is just len(history), not a version: undo/redo (from any tab) legitimately lowers
  // it, and pushes arrive in order on one socket, so every push is accepted; exact
  // duplicates are dropped by sameShown below.
  lastPly = msg.ply;
  if (!sameShown(state, msg.state)) {
    const moved = (state.history || []).length !== (msg.state.history || []).length;
    giveMoveFeedback(state, msg.state);
    state = msg.state;
    over = !!msg.over || (state.winner !== null && state.winner !== undefined);
    noteClock();
    ctx = computeSeatContext();
    if (moved) clearSelection();
    publishDebugHook();
    renderAll();
    maybeShowResultSheet();
    // No maybeBotMove() here: over the socket the server owns the bot reply, and a push
    // lands between the human move and the bot's, so asking for a bot move now would race it.
  } else if (JSON.stringify(state.clock) !== JSON.stringify(msg.state.clock)) {
    state.clock = msg.state.clock;      // same position, new clock reading (e.g. the clock started)
    noteClock();
    if (answered) renderAll();
  } else if (answered) {
    renderAll();
  }
  syncPolling();
}

function ensureSubscribed() {
  if (!window.MushiLive) return;
  const token = ctx.token || null;
  window.MushiLive.subscribeGame(gid, token, onSocketState, {
    // A (re)connect delivers a fresh snapshot, which is authoritative even if its ply
    // is lower than what we last showed (e.g. an undo done from another tab).
    onUp: () => { lastPly = -1; setConnected(true); },
    onDown: () => { wsLive = false; setConnected(false); syncPolling(); },
  });
}

async function init() {
  try {
    // ctx isn't computed yet (that needs the mode/seats this very call
    // returns), but a seat token already in localStorage from a previous
    // visit is known up front - send it so even the very first load, a
    // reload, counts as the seated player's own poll.
    const stored = ui.seatToken(gid);
    const data = await ui.api(`/api/game/${gid}`, { headers: seatHeaders(stored && stored.token) });
    onGameData(data);
    ensureSubscribed();
  } catch (e) {
    showExpired();
  }
}

init();
