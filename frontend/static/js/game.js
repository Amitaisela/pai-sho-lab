/* Pai Sho Lab — the game page (`/game/<id>`).
 *
 * Owns everything board.js doesn't: fetching and polling game state, who
 * holds which seat (from localStorage via ui.js), move/hint network calls,
 * the action row (skip bonus, resign, undo/redo, share), the join/resign/
 * result sheets, and the move list. The circular board and both hands are
 * rendered by board.js; this file only decides *what* to render.
 */
import * as ui from '/static/js/ui.js';
import { createBoard, formatMove } from '/static/js/board.js';

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
let hintResetTimer = null;

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
  if (!state || over) return false;
  if (mode === 'pass') return true;
  return ctx.mySeat !== null && ctx.mySeat === state.current_player;
}

function handPlayerFor() {
  if (!state || over) return null;
  if (mode === 'pass') return state.current_player;
  return ctx.mySeat;
}

function orientation() {
  const flip = mode === 'pass' ? state.current_player === 2 : ctx.mySeat === 2;
  return { flip, topPlayer: flip ? 1 : 2, bottomPlayer: flip ? 2 : 1 };
}

// -------------------------------------------------------------------- board

const boardEl = document.getElementById('game-board');
const board = createBoard(boardEl, { flip: false, onCellClick, onTileClick, onHandClick });

async function onHandClick(player, flower) {
  if (!isMyTurnNow() || player !== state.current_player) return;
  if (selection && selection.type === 'hand' && selection.player === player && selection.flower === flower) {
    clearSelection();
    renderAll();
    return;
  }
  selection = { type: 'hand', player, flower };
  hints = [];
  renderAll();
  try {
    const r = await ui.api(`/api/valid_plant_moves/${gid}`, { method: 'POST', body: { tile: flower } });
    hints = r.moves || [];
  } catch (e) {
    hints = [];
  }
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
  hints = [];
  renderAll();
  try {
    const resp = await ui.api(`/api/valid_moves/${gid}`, { method: 'POST', body: { row: r, col: c } });
    hints = resp.moves || [];
  } catch (e) {
    hints = [];
  }
  renderAll();
}

async function onCellClick(r, c) {
  if (!selection || !isMyTurnNow()) return;
  const isHint = hints.some(([hr, hc]) => hr === r && hc === c);
  if (!isHint) {
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
      let dr;
      try {
        dr = await ui.api(`/api/valid_boat_displacement/${gid}`, {
          method: 'POST', body: { target_row: r, target_col: c },
        });
      } catch (e) {
        dr = { moves: [], direct: false };
      }
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

async function doMove(kind, body) {
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
  state = newState;
  over = state.winner !== null && state.winner !== undefined;
  publishDebugHook();
}

function publishDebugHook() {
  // A read-only hook for the Playwright checks in tests/e2e_play.py — not
  // used by the page itself. Harmless to leave in production: it just
  // mirrors module state that's otherwise private to this closure.
  window.__pai = { state, mode, over, ctx };
}

// ------------------------------------------------------------------ render

function renderAll() {
  renderBadge();
  renderStrips();
  renderHint();
  renderActions();
  renderMoves();
  const { flip } = orientation();
  board.setFlip(flip);
  board.render(state, { selected: selection, hints, myHandPlayer: handPlayerFor() });
  sizeBoard();
}

// The board (board.js's own CSS) is naturally responsive by width alone —
// `width: 100%; max-width: 820px` — which is right on a tall phone screen,
// but on a short wide one (a laptop at 1280x900, say) that width-only sizing
// lets a tall circular board overflow the viewport, pushing the player's own
// hand and the action row below the fold. This measures the vertical space
// everything *around* the board actually takes (the top bar, strips, hint
// line, action row, move list, and — below the pai.css ≥900px breakpoint,
// where the hands still stack above/below the board instead of sitting
// beside it — the board's own two hand rows) and constrains the board to
// whichever is smaller: the width it would have claimed anyway, or the
// height left over — clamped to a 320px floor and a ceiling the design
// calls for (820px stacked, 900px side-by-side, since the hands no longer
// take vertical space there). On a tall phone the height budget is never
// the tighter constraint, so the board still ends up full width there.
let sizeRaf = null;
function scheduleSizeBoard() {
  if (sizeRaf) return;
  sizeRaf = requestAnimationFrame(() => {
    sizeRaf = null;
    sizeBoard();
  });
}
window.addEventListener('resize', scheduleSizeBoard);

const DESKTOP_HANDS_QUERY = '(min-width: 900px)'; // matches pai.css's .pb-root breakpoint

function sizeBoard() {
  const wrap = boardEl.querySelector('.pb-wrap');
  if (!wrap) return;
  const sideBySide = window.matchMedia(DESKTOP_HANDS_QUERY).matches;
  const topHand = boardEl.querySelector('.pb-hand[data-slot="top"]');
  const bottomHand = boardEl.querySelector('.pb-hand[data-slot="bottom"]');
  const chromeEls = [
    document.querySelector('.topbar'),
    document.getElementById('game-badge'),
    document.getElementById('game-top-strip'),
    document.getElementById('game-bottom-strip'),
    document.getElementById('game-hint'),
    document.getElementById('game-actions'),
    document.getElementById('game-moves'),
    ...(sideBySide ? [] : [topHand, bottomHand]),
  ];
  let chromeHeight = 0;
  for (const el of chromeEls) {
    if (el && !el.hidden) chromeHeight += el.getBoundingClientRect().height;
  }
  const GAPS_RESERVE = 72; // the page's/flex containers' own gaps and padding
  const availableHeight = window.innerHeight - chromeHeight - GAPS_RESERVE;
  // Clear any max-width this function set on a previous call before
  // measuring: that's what makes the width reading accurate once the hands
  // sit beside the board (≥900px) — the grid column .pb-wrap occupies there
  // is narrower than its own parent's full width, and a stale max-width
  // from an earlier, wider layout would otherwise be read back as if it
  // were the column's real size.
  wrap.style.maxWidth = 'none';
  const availableWidth = wrap.getBoundingClientRect().width || 720;
  const maxSize = sideBySide ? 900 : 820;
  const size = Math.max(320, Math.min(maxSize, Math.min(availableWidth, availableHeight)));
  wrap.style.maxWidth = `${Math.floor(size)}px`;
}

function renderBadge() {
  const el = document.getElementById('game-badge');
  if (el) el.hidden = !ctx.watching;
}

function seatLabel(p) {
  const seat = seats[p];
  if (!seat) return `Player ${p}`;
  if (seat.kind === 'bot') return seat.name || `MushiBot level ${seat.level}`;
  return seat.name || 'Open seat';
}

function fillStrip(el, p) {
  if (!el) return;
  const isTurn = !over && state.current_player === p;
  el.classList.toggle('is-turn', isTurn);
  const you = ctx.mySeat === p ? ' <span class="muted">(you)</span>' : '';
  el.innerHTML = `<div class="strip__name"><span class="strip__turn-dot"></span>${escapeHtml(seatLabel(p))}${you}</div>`;
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
  if (over || ctx.watching) return;

  if (state.can_skip_bonus && isMyTurnNow()) {
    el.appendChild(actionButton('skip-btn', 'Skip bonus', 'btn-secondary', skipBonus));
  }
  if (canResignNow()) {
    el.appendChild(actionButton('resign-btn', 'Resign', 'btn-quiet', confirmResign));
  }
  if (mode !== 'human' && (state.history || []).length > 0) {
    el.appendChild(actionButton('undo-btn', 'Undo', 'btn-quiet', doUndo));
    if (canRedo) el.appendChild(actionButton('redo-btn', 'Redo', 'btn-quiet', doRedo));
  }
  el.appendChild(actionButton('share-btn', 'Share', 'btn-quiet', shareLink));
}

function renderMoves() {
  const list = document.getElementById('game-move-list');
  const details = document.getElementById('game-moves');
  if (!list || !details) return;
  const history = state.history || [];
  const players = state.history_players || [];
  list.innerHTML = history.map((move, i) => {
    const p = players.length === history.length ? players[i] : (i % 2 === 0 ? 1 : 2);
    return `<li class="is-p${p}">${i + 1}. ${escapeHtml(formatMove(move))}</li>`;
  }).join('');
  const summary = details.querySelector('summary');
  if (summary) summary.textContent = `Moves (${history.length})`;
}

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

async function shareLink() {
  try {
    await navigator.clipboard.writeText(location.href);
    flashHint('Link copied');
  } catch (e) {
    flashHint('Could not copy the link');
  }
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
    <label class="game__field">Your name
      <input id="join-name" type="text" maxlength="24" placeholder="${escapeHtml(ui.nickname())}">
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
      const data = await ui.api(`/api/game/${gid}`);
      onGameData(data);
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
};

function resultDetail() {
  if (state.winner === 0) return 'Tie';
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

async function doRematch() {
  try {
    if (mode === 'bot') {
      const mySeatNow = ctx.mySeat || 1;
      const botSeat = seats[3 - mySeatNow];
      const level = botSeat ? botSeat.level : 1;
      const r = await ui.api('/api/bot_game', {
        method: 'POST', body: { level, seat: 3 - mySeatNow, nickname: ui.nickname() },
      });
      ui.setSeatToken(r.game_id, r.seat, r.seat_token);
      location.href = `/game/${r.game_id}`;
    } else if (mode === 'human') {
      const r = await ui.api('/api/challenge', { method: 'POST', body: { nickname: ui.nickname(), seat: 'random' } });
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
  const close = ui.sheet(`
    <h2 id="result-title">${escapeHtml(resultHeadline())}</h2>
    <p class="muted">${escapeHtml(resultDetail())}</p>
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
  if (pollTimer) clearInterval(pollTimer);
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
  state = data.state;
  mode = data.mode;
  seats = data.seats;
  over = data.over;
  ctx = computeSeatContext();
  publishDebugHook();
  renderAll();
  maybeOfferJoin();
  maybeShowResultSheet();
  maybeBotMove();
}

// Spectators must never keep a game's idle timer alive just by polling it
// (see backend/ui/play.py's api_game) - only a seated player's own poll does,
// by sending its seat_token.
function gamePollUrl() {
  return ctx.token ? `/api/game/${gid}?seat_token=${encodeURIComponent(ctx.token)}` : `/api/game/${gid}`;
}

function startPolling() {
  pollTimer = setInterval(async () => {
    if (over) {
      clearInterval(pollTimer);
      return;
    }
    try {
      const data = await ui.api(gamePollUrl());
      const newLen = (data.state.history || []).length;
      const oldLen = (state.history || []).length;
      const changed = newLen !== oldLen || data.state.winner !== state.winner || data.over !== over;
      // Seats (a name filling an open seat) don't move history/winner, so they're
      // refreshed every tick regardless — cheap, and it's how the other browser
      // finds out someone joined. The heavier path (board re-render, re-running
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
      }
    } catch (e) {
      if (e && e.status === 404) {
        clearInterval(pollTimer);
        showExpired();
      }
    }
  }, 1000);
}

async function init() {
  try {
    // ctx isn't computed yet (that needs the mode/seats this very call
    // returns), but a seat token already in localStorage from a previous
    // visit is known up front - send it so even the very first load, a
    // reload, counts as the seated player's own poll.
    const stored = ui.seatToken(gid);
    const url = stored ? `/api/game/${gid}?seat_token=${encodeURIComponent(stored.token)}` : `/api/game/${gid}`;
    const data = await ui.api(url);
    onGameData(data);
    startPolling();
  } catch (e) {
    showExpired();
  }
}

init();
