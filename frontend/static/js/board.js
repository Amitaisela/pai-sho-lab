/* Pai Sho Lab — the board module.
 *
 * A pure rendering component: it turns a serialized game state (the shape
 * `ui.server`'s `serialize()` returns) plus a selection/hints/whose-hand-is-
 * yours description into DOM, and reports clicks back through callbacks. It
 * never calls the network — `/game`'s game.js (and later the Learn tutorial)
 * own all the fetching and decide what counts as a legal selection.
 *
 * Ported from the rendering logic in `frontend/templates/lab/board.html`
 * (today's full-control Lab board), restructured for a single mobile-first
 * column (hand / board / hand) instead of side panels, and for a
 * percentage-based 19x19 grid instead of a resize-observed pixel cell size.
 */

const BOARD_SIZE = 19;

const CIRCLE = ['Rose', 'Chrysanthemum', 'Rhododendron', 'Jasmine', 'Lily', 'Jade'];

const FLOWER_META = {
  Rose: { p1_img: 'HR3.png', p2_img: 'GR3.png' },
  Chrysanthemum: { p1_img: 'HR4.png', p2_img: 'GR4.png' },
  Rhododendron: { p1_img: 'HR5.png', p2_img: 'GR5.png' },
  Jasmine: { p1_img: 'HW3.png', p2_img: 'GW3.png' },
  Lily: { p1_img: 'HW4.png', p2_img: 'GW4.png' },
  Jade: { p1_img: 'HW5.png', p2_img: 'GW5.png' },
};

const ACCENT_META = {
  Rock: { p1_img: 'HR.png', p2_img: 'GR.png' },
  Wheel: { p1_img: 'HW.png', p2_img: 'GW.png' },
  Knotweed: { p1_img: 'HK.png', p2_img: 'GK.png' },
  Boat: { p1_img: 'HB.png', p2_img: 'GB.png' },
};

const SPECIAL_META = {
  Orchid: { p1_img: 'HO.png', p2_img: 'GO.png' },
  WhiteLotus: { p1_img: 'HL.png', p2_img: 'GL.png' },
};

// Two hand rows: the six basic flowers in Circle-of-Harmony order (CIRCLE
// is already that order — see PaiShoGame.py's _HARMONY_PAIRS/_CLASH_PAIRS,
// built from the same list — so neighbours in this array harmonise and a
// clashing pair always has exactly two tiles between them), then the four
// accents plus the two specials.
const HAND_ROW_BASIC = CIRCLE;
const HAND_ROW_OTHER = [...Object.keys(ACCENT_META), ...Object.keys(SPECIAL_META)];

const LABELS = { WhiteLotus: 'White Lotus' };

function tileMeta(flower) {
  return FLOWER_META[flower] || ACCENT_META[flower] || SPECIAL_META[flower] || null;
}

/** The two flowers that harmonise with `flower` on the Circle of Harmony
 * (its neighbours), or `[]` for a non-basic flower. */
function harmonyPartners(flower) {
  const i = CIRCLE.indexOf(flower);
  return i === -1 ? [] : [CIRCLE[(i + 5) % 6], CIRCLE[(i + 1) % 6]];
}

/** The one flower directly opposite `flower` on the Circle (same number,
 * opposite colour — they clash), or `null` for a non-basic flower. */
function clashOf(flower) {
  const i = CIRCLE.indexOf(flower);
  return i === -1 ? null : CIRCLE[(i + 3) % 6];
}

/** The basic flower the current selection represents — a selected hand tile,
 * or the flower of a selected board tile — so the harmony/clash indicator
 * can be driven by either. `null` for no selection, a non-basic selection,
 * or a selected empty/invalid board point. */
function selectedBasicFlower(state, selected) {
  if (!selected) return null;
  if (selected.type === 'hand') return CIRCLE.includes(selected.flower) ? selected.flower : null;
  if (selected.type === 'board') {
    const tile = state.board[key(selected.row, selected.col)];
    return tile && CIRCLE.includes(tile.flower) ? tile.flower : null;
  }
  return null;
}

/** A short, plain-words label for a flower/accent/special name. */
export function tileLabel(flower) {
  return LABELS[flower] || flower;
}

/** Formats one history move (the array shape `serialize()` returns) as one line of text. */
export function formatMove(move) {
  if (!Array.isArray(move)) return String(move);
  if (move[0] === 'plant' && move.length === 6) {
    return `Boat on (${move[2]},${move[3]}) moves a flower to (${move[4]},${move[5]})`;
  }
  if (move[0] === 'plant') {
    return `Plant ${tileLabel(move[1])} at (${move[2]},${move[3]})`;
  }
  if (move[0] === 'arrange') {
    return `Move (${move[1]},${move[2]}) to (${move[3]},${move[4]})`;
  }
  if (move[0] === 'skip_bonus') return 'Skip bonus';
  return move.join(' ');
}

function key(r, c) {
  return `${r},${c}`;
}

/** One history action as a PSN v2 token, the same text engine/PythonEngine/notation.py writes:
 * `Rose@9,1` (plant), `Boat@r,c>dr,dc` (Boat on a flower), `fr,fc-tr,tc` (arrange). */
export function psnToken(m) {
  if (!Array.isArray(m)) return String(m);
  if (m[0] === 'plant') return `${m[1]}@${m[2]},${m[3]}${m.length === 6 ? `>${m[4]},${m[5]}` : ''}`;
  if (m[0] === 'arrange') return `${m[1]},${m[2]}-${m[3]},${m[4]}`;
  return m.join(' ');
}

/** The numbered PSN lines for a game, `"1. a  b"`: two turns (Guest, Host) a line, a bonus joined
 * to its main action with `+`. `bonusPending` (the engine's `bonus_turn`) marks a bonus that was
 * earned but neither played nor skipped when the record ends with a lone trailing `+`, exactly as
 * notation.py writes it. */
export function psnMoveLines(history, players, bonusPending = false) {
  const turns = psnTurns(history, players);
  const tokens = turns.map((t) => (t.bonus ? `${t.main}+${t.bonus}` : t.main));
  if (bonusPending && tokens.length) tokens[tokens.length - 1] += '+';
  const lines = [];
  for (let i = 0; i < tokens.length; i += 2) lines.push(`${i / 2 + 1}. ${tokens.slice(i, i + 2).join('  ')}`);
  return lines;
}

/** The history grouped into turns, as PSN writes them: a turn is one player's main action plus
 * the Harmony Bonus they played with it. A skipped bonus is no move, and two consecutive moves by
 * one player can only be a main action and its bonus (turns alternate). Each token carries `ply`,
 * the history length once it has been played, so the list can show the position after it:
 * `[{player, main, mainPly, bonus, bonusPly}]`. `players` is `history_players`. */
export function psnTurns(history, players) {
  const turns = [];
  const known = Array.isArray(players) && players.length === history.length;
  let lastIdx = -2;
  history.forEach((m, i) => {
    if (!Array.isArray(m) || m[0] === 'skip_bonus') return;
    const player = known ? players[i] : null;
    const prev = turns[turns.length - 1];
    if (prev && !prev.bonus && known && prev.player === player && lastIdx === i - 1) {
      prev.bonus = psnToken(m);
      prev.bonusPly = i + 1;
    } else {
      turns.push({ player, main: psnToken(m), mainPly: i + 1, bonus: null, bonusPly: null });
    }
    lastIdx = i;
  });
  return turns;
}

/** The newest move's points from the history: `{ from, to }` as "r,c" keys (`from` is null for
 * a plant). A skipped bonus is not a move, so it is looked past. `{}` before the first move. */
function lastMoveCells(state) {
  const history = state.history || [];
  for (let i = history.length - 1; i >= 0; i--) {
    const m = history[i];
    if (!Array.isArray(m) || m[0] === 'skip_bonus') continue;
    if (m[0] === 'arrange') return { from: key(m[1], m[2]), to: key(m[3], m[4]) };
    if (m[0] === 'plant' && m.length === 6) return { from: key(m[2], m[3]), to: key(m[4], m[5]) };   // Boat
    if (m[0] === 'plant') return { from: null, to: key(m[2], m[3]) };
  }
  return {};
}

function isGameOver(state) {
  return state.winner !== null && state.winner !== undefined;
}

/**
 * Mounts a board widget into `el`: a top hand, the circular board, a bottom
 * hand. `onCellClick(r, c)` fires for any board point (hinted or not — the
 * caller decides what a click on an unhinted point means, e.g. clearing the
 * selection); `onTileClick(r, c)` fires for a click on an occupied point's
 * piece specifically (it stops the click from also reaching onCellClick);
 * `onHandClick(player, flower)` fires for a hand tile button, but only when
 * that button isn't `disabled` (native DOM behaviour — board.js disables a
 * hand tile whenever `render()`'s `myHandPlayer` says it isn't yours to play).
 *
 * Drag and drop: `onDragStart(source)` is asked when a pointer has moved a few pixels from one of
 * the player's pieces (`{type:'board', row, col}`) or hand tiles (`{type:'hand', player, flower}`);
 * it should select it exactly as a click would and return true if there is anything to drag. A
 * drop on a point then reports as `onCellClick(r, c, {dropped: true})`, so it takes the very
 * same path as click-click; a drop outside the board calls `onDragEnd()`. Taps with a finger snap
 * to the nearest hinted point within 1.2 points (the 19 columns are ~20px wide on a phone), and
 * `onCellClick`'s third argument says whether the tap was a touch and whether it snapped.
 *
 * `handEls` ({ top, bottom }) lets the page own the hands' placement: the board then renders
 * only the circle, and fills those two elements instead of its own (the game page lays out
 * hands, clocks and actions around the board itself).
 *
 * Returns `{ render(state, { selected, hints, myHandPlayer }), setFlip(bool) }`.
 */
export function createBoard(el, {
  flip = false, onCellClick, onTileClick, onHandClick, onDragStart, onDragEnd, handEls = null,
} = {}) {
  let _flip = !!flip;
  let lastState = null;
  let lastOpts = {};
  let builtValidSpacesKey = null;
  let prevTileKeys = new Set(); // which points held a tile as of the last renderCells() call
  let prevTiles = new Map();    // "r,c" -> the tile that was there, as of the last renderCells() call
  let prevPly = -1;             // history length as of the last renderCells() call (-1: never drawn)
  let dragKey = null;           // "r,c" of the board piece being dragged, to dim it while it is away
  const cells = new Map(); // "r,c" -> cell element
  const pieceSlots = new Map(); // "r,c" -> the child element a piece renders into

  el.innerHTML = `
    <div class="pb-root">
      ${handEls ? '' : '<div class="pb-hand" data-slot="top"></div>'}
      <div class="pb-wrap">
        <div class="pb-visuals">
          <div class="pb-diamond"></div>
          <div class="pb-gate pb-gate--n"></div>
          <div class="pb-gate pb-gate--s"></div>
          <div class="pb-gate pb-gate--w"></div>
          <div class="pb-gate pb-gate--e"></div>
          <svg class="pb-lines" viewBox="0 0 19 19" preserveAspectRatio="none" aria-hidden="true"></svg>
        </div>
        <div class="pb-grid"></div>
        <svg class="pb-harmony" viewBox="0 0 19 19" preserveAspectRatio="none"></svg>
      </div>
      ${handEls ? '' : '<div class="pb-hand" data-slot="bottom"></div>'}
    </div>`;

  const gridEl = el.querySelector('.pb-grid');
  // Board lines through every point (cell centres, c + 0.5): Pai Sho tiles sit on
  // intersections. The centre lines are slightly stronger, like a real board.
  (function drawLines(svg) {
    const NS = 'http://www.w3.org/2000/svg';
    for (let i = 0; i < BOARD_SIZE; i++) {
      const at = i + 0.5;
      for (const [x1, y1, x2, y2] of [[at, 0, at, BOARD_SIZE], [0, at, BOARD_SIZE, at]]) {
        const ln = document.createElementNS(NS, 'line');
        ln.setAttribute('x1', String(x1)); ln.setAttribute('y1', String(y1));
        ln.setAttribute('x2', String(x2)); ln.setAttribute('y2', String(y2));
        if (i === (BOARD_SIZE - 1) / 2) ln.setAttribute('class', 'pb-line--axis');
        svg.appendChild(ln);
      }
    }
  })(el.querySelector('.pb-lines'));
  const harmonyEl = el.querySelector('.pb-harmony');
  const topHandEl = handEls ? handEls.top : el.querySelector('.pb-hand[data-slot="top"]');
  const bottomHandEl = handEls ? handEls.bottom : el.querySelector('.pb-hand[data-slot="bottom"]');
  if (handEls) {
    for (const [slot, handEl] of [['top', topHandEl], ['bottom', bottomHandEl]]) {
      handEl.classList.add('pb-hand');
      handEl.dataset.slot = slot;
    }
  }

  function disp(r, c) {
    return _flip ? [BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c] : [r, c];
  }

  // Keyboard: the grid is one tab stop (a roving tabindex); the arrow keys move between points in
  // the direction they are drawn (so they follow a flipped board), Enter or Space acts on the
  // focused point exactly as a click or tap would.
  let focusKey = null;
  gridEl.setAttribute('role', 'grid');
  gridEl.setAttribute('aria-label', 'Board');

  function focusCell(k) {
    const next = cells.get(k);
    if (!next) return;
    if (focusKey && cells.get(focusKey)) cells.get(focusKey).tabIndex = -1;
    focusKey = k;
    next.tabIndex = 0;
    next.focus();
  }

  gridEl.addEventListener('keydown', (e) => {
    const cellEl = e.target.closest && e.target.closest('.pb-cell');
    if (!cellEl) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const piece = cellEl.querySelector('.pb-piece:not(.pb-piece--ghost)');
      (piece || cellEl).click();
      return;
    }
    const step = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
    if (!step) return;
    e.preventDefault();
    const [dr, dc] = disp(Number(cellEl.dataset.r), Number(cellEl.dataset.c));
    for (let n = 1; n < BOARD_SIZE; n++) {
      const [gr, gc] = disp(dr + step[0] * n, dc + step[1] * n);        // disp() is its own inverse
      if (cells.has(key(gr, gc))) { focusCell(key(gr, gc)); return; }
    }
  });

  function buildGrid(validSpaces) {
    gridEl.innerHTML = '';
    cells.clear();
    pieceSlots.clear();
    for (const [r, c] of validSpaces) {
      const cellEl = document.createElement('div');
      cellEl.className = 'pb-cell';
      cellEl.dataset.r = String(r);
      cellEl.dataset.c = String(c);
      cellEl.setAttribute('role', 'gridcell');
      cellEl.tabIndex = r === 9 && c === 9 ? 0 : -1;
      if (r === 9 && c === 9) focusKey = key(r, c);
      // A permanent, low-contrast marker for the intersection itself — present
      // on every valid space regardless of occupancy, so the 249 playable
      // points are visible (not just inferable from the garden/gate art) and
      // so "how many points does the board render" is a stable, state-
      // independent count for tests. The piece (if any) renders on top of it
      // into its own sibling slot, which is the only part renderCells() clears.
      const point = document.createElement('span');
      point.className = 'pb-point';
      cellEl.appendChild(point);
      const slot = document.createElement('div');
      slot.className = 'pb-piece-slot';
      cellEl.appendChild(slot);
      cellEl.addEventListener('click', (e) => {
        if (!onCellClick) return;
        const touch = e.pointerType === 'touch';
        const snap = touch ? nearestHint(e.clientX, e.clientY) : null;
        if (snap) onCellClick(snap[0], snap[1], { pointerType: e.pointerType, snapped: true });
        else onCellClick(r, c, { pointerType: e.pointerType || '', snapped: false });
      });
      gridEl.appendChild(cellEl);
      cells.set(key(r, c), cellEl);
      pieceSlots.set(key(r, c), slot);
    }
  }

  // The hinted point nearest to a finger, in game coordinates, if one is within 1.2 points. A
  // point is ~20px on a phone, so a tap a little off a legal target still lands on it.
  function nearestHint(clientX, clientY) {
    const hints = lastOpts.hints || [];
    if (!hints.length) return null;
    const rect = gridEl.getBoundingClientRect();
    const cell = rect.width / BOARD_SIZE;
    const dc = (clientX - rect.left) / cell - 0.5;          // display coordinates; a point's centre is an integer
    const dr = (clientY - rect.top) / cell - 0.5;
    const gr = _flip ? BOARD_SIZE - 1 - dr : dr;
    const gc = _flip ? BOARD_SIZE - 1 - dc : dc;
    let best = null;
    let bestDist = 1.2;
    for (const [hr, hc] of hints) {
      const d = Math.hypot(hr - gr, hc - gc);
      if (d <= bestDist) { best = [hr, hc]; bestDist = d; }
    }
    return best;
  }

  function positionGrid() {
    for (const [k, cellEl] of cells) {
      const [r, c] = k.split(',').map(Number);
      const [dr, dc] = disp(r, c);
      cellEl.style.gridRow = String(dr + 1);
      cellEl.style.gridColumn = String(dc + 1);
    }
  }

  function renderCells(state, hints, selected, relations) {
    const hintSet = new Set((hints || []).map(([r, c]) => key(r, c)));
    const selKey = selected && selected.type === 'board' ? key(selected.row, selected.col) : null;
    const { harmonySet, clashFlower } = relations;
    const nextTileKeys = new Set();
    // The newest move is highlighted, and when this render is the first to show it (the history
    // grew since the last draw: never on a redraw for a new selection, an undo or a rollback)
    // its piece slides in from where it came from and a captured piece fades out.
    const last = lastMoveCells(state);
    const ply = (state.history || []).length;
    const isNewMove = prevPly !== -1 && ply > prevPly;
    for (const [k, cellEl] of cells) {
      const [r, c] = k.split(',').map(Number);
      const tile = state.board[k] || null;
      cellEl.classList.toggle('pb-cell--hint', hintSet.has(k));
      cellEl.classList.toggle('pb-cell--last-from', k === last.from);
      cellEl.classList.toggle('pb-cell--last-to', k === last.to);
      cellEl.setAttribute('aria-label', `${r},${c}${tile ? `: ${tile.player === 1 ? 'Guest' : 'Host'} ${tileLabel(tile.flower)}${tile.growing ? ', growing' : ''}` : ''}${hintSet.has(k) ? ', legal move' : ''}`);
      const slot = pieceSlots.get(k);
      slot.innerHTML = '';
      if (!tile) continue;
      nextTileKeys.add(k);
      const meta = tileMeta(tile.flower);
      const piece = document.createElement('div');
      piece.className = `pb-piece pb-piece--p${tile.player}`;
      // Only a point that didn't hold a tile last render gets the landing
      // animation - a tile already there (redrawn because the selection or
      // hints changed, or a poll tick found nothing new) never replays it.
      const was = prevTiles.get(last.from);
      if (isNewMove && k === last.to && last.from && was && was.flower === tile.flower && was.player === tile.player) {
        const [fr, fc] = last.from.split(',').map(Number);
        const [dfr, dfc] = disp(fr, fc);
        const [dtr, dtc] = disp(r, c);
        piece.classList.add('pb-piece--slide');
        piece.style.setProperty('--dx', String(dfc - dtc));
        piece.style.setProperty('--dy', String(dfr - dtr));
      } else if (!prevTileKeys.has(k)) {
        piece.classList.add('pb-piece--enter');
      }
      if (tile.growing) piece.classList.add('pb-piece--growing');
      if (k === selKey) piece.classList.add('pb-piece--selected');
      if (k === dragKey) piece.classList.add('pb-piece--dragging');
      if (harmonySet.has(tile.flower)) piece.classList.add('pb-piece--harmony-hint');
      if (tile.flower === clashFlower) piece.classList.add('pb-piece--clash-hint');
      if (meta) {
        const img = tile.player === 1 ? meta.p1_img : meta.p2_img;
        piece.innerHTML = `<img class="pb-piece__img" src="/static/tiles/${img}" alt="${tileLabel(tile.flower)}">`;
      } else {
        piece.textContent = tile.flower.slice(0, 2);
      }
      piece.title = `Player ${tile.player} ${tileLabel(tile.flower)}${tile.growing ? ' (growing)' : ''}`;
      piece.addEventListener('click', (e) => {
        e.stopPropagation();
        onTileClick && onTileClick(r, c);
      });
      slot.appendChild(piece);
    }
    // A capture: the piece that stood on the landing point fades out under the new one.
    const lost = isNewMove && last.from ? prevTiles.get(last.to) : null;
    if (lost && state.board[last.to] && (lost.player !== state.board[last.to].player || lost.flower !== state.board[last.to].flower)) {
      const meta = tileMeta(lost.flower);
      const ghost = document.createElement('div');
      ghost.className = `pb-piece pb-piece--p${lost.player} pb-piece--ghost`;
      if (meta) {
        ghost.innerHTML = `<img class="pb-piece__img" src="/static/tiles/${lost.player === 1 ? meta.p1_img : meta.p2_img}" alt="">`;
      }
      pieceSlots.get(last.to).prepend(ghost);
    }
    prevTileKeys = nextTileKeys;
    prevTiles = new Map(Object.entries(state.board));
    prevPly = ply;
  }

  function renderHarmonies(state) {
    harmonyEl.innerHTML = '';
    if (!state.harmonies) return;
    const over = isGameOver(state);
    for (const [pkey, cls] of [['1', 'pb-harmony-line--p1'], ['2', 'pb-harmony-line--p2']]) {
      const isWinner = over && String(state.winner) === pkey;
      for (const [[r1, c1], [r2, c2]] of state.harmonies[pkey] || []) {
        const [dr1, dc1] = disp(r1, c1);
        const [dr2, dc2] = disp(r2, c2);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', String(dc1 + 0.5));
        line.setAttribute('y1', String(dr1 + 0.5));
        line.setAttribute('x2', String(dc2 + 0.5));
        line.setAttribute('y2', String(dr2 + 0.5));
        line.setAttribute('class', isWinner ? `${cls} pb-harmony-line--win` : cls);
        harmonyEl.appendChild(line);
      }
    }
  }

  function handTileAllowed(state, player, flower, myHandPlayer) {
    if (player !== myHandPlayer) return false;
    if (player !== state.current_player) return false;
    if (isGameOver(state)) return false;
    const count = (state.hands[String(player)] || {})[flower] || 0;
    if (count <= 0) return false;
    const isAccentOrSpecial = flower in ACCENT_META || flower in SPECIAL_META;
    if (isAccentOrSpecial) return !!state.bonus_turn;
    if (state.bonus_turn) {
      const growing = Object.values(state.board).some((t) => t.player === player && t.growing);
      if (growing) return false;
    }
    return true;
  }

  function renderHandRow(rowEl, tiles, player, state, selected, myHandPlayer, relations) {
    rowEl.innerHTML = '';
    const { harmonySet, clashFlower } = relations;
    for (const flower of tiles) {
      const meta = tileMeta(flower);
      const count = (state.hands[String(player)] || {})[flower] || 0;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pb-tile';
      btn.dataset.player = String(player);
      btn.dataset.flower = flower;
      btn.disabled = !handTileAllowed(state, player, flower, myHandPlayer);
      const isSel = !!(selected && selected.type === 'hand' && selected.player === player && selected.flower === flower);
      btn.classList.toggle('pb-tile--selected', isSel);
      btn.classList.toggle('pb-tile--empty', count === 0);
      btn.classList.toggle('pb-tile--harmony-hint', harmonySet.has(flower));
      btn.classList.toggle('pb-tile--clash-hint', flower === clashFlower);
      const img = player === 1 ? meta.p1_img : meta.p2_img;
      btn.innerHTML = `<img class="pb-tile__img" src="/static/tiles/${img}" alt="${tileLabel(flower)}">`
        + `<span class="pb-tile__count">${count}</span>`;
      btn.title = `${tileLabel(flower)} — ${count} left`;
      btn.addEventListener('click', () => onHandClick && onHandClick(player, flower));
      rowEl.appendChild(btn);
    }
  }

  // A hand is two rows: the six basic flowers in Circle-of-Harmony order,
  // then the accents and specials — see HAND_ROW_BASIC/HAND_ROW_OTHER.
  function renderHand(container, player, state, selected, myHandPlayer, relations) {
    container.dataset.player = String(player);
    // Markers the page's CSS can key on: whose turn it is, and "this is my bonus turn" (the
    // accent/special row is only playable then, so a compact phone layout shows it just then).
    container.toggleAttribute('data-turn', !isGameOver(state) && player === state.current_player);
    container.toggleAttribute('data-bonus', !isGameOver(state) && !!state.bonus_turn
      && player === state.current_player && player === myHandPlayer);
    if (!container.dataset.built) {
      container.innerHTML = '<div class="pb-hand-row pb-hand-row--basic"></div>'
        + '<div class="pb-hand-row pb-hand-row--other"></div>';
      container.dataset.built = '1';
    }
    const [basicRow, otherRow] = container.children;
    renderHandRow(basicRow, HAND_ROW_BASIC, player, state, selected, myHandPlayer, relations);
    renderHandRow(otherRow, HAND_ROW_OTHER, player, state, selected, myHandPlayer, relations);
  }

  function render(state, opts = {}) {
    lastState = state;
    lastOpts = opts;
    const { selected = null, hints = [], myHandPlayer = null } = opts;

    const vsKey = (state.valid_spaces || []).map(([r, c]) => key(r, c)).join('|');
    if (vsKey !== builtValidSpacesKey) {
      buildGrid(state.valid_spaces || []);
      builtValidSpacesKey = vsKey;
      positionGrid();
    }

    // The harmony/clash indicator: driven by the selected hand or board
    // tile when it's a basic flower, shown on every matching flower in both
    // hands and on the board regardless of owner (harmony/clash is about
    // flower identity, not whose tile it is).
    const relFlower = selectedBasicFlower(state, selected);
    const relations = {
      harmonySet: new Set(relFlower ? harmonyPartners(relFlower) : []),
      clashFlower: relFlower ? clashOf(relFlower) : null,
    };

    renderCells(state, hints, selected, relations);
    renderHarmonies(state);

    const topPlayer = _flip ? 1 : 2;
    const bottomPlayer = _flip ? 2 : 1;
    renderHand(topHandEl, topPlayer, state, selected, myHandPlayer, relations);
    renderHand(bottomHandEl, bottomPlayer, state, selected, myHandPlayer, relations);
  }

  // ------------------------------------------------------------ drag and drop
  // A press that stays put is a plain click (nothing here runs). Once the pointer has moved
  // DRAG_PX the page is asked to select the source; if it agrees a ghost follows the pointer and
  // the point under it lights up when it is a legal target. Listeners live on `window`, because
  // starting a drag re-renders the board and replaces the very element that was pressed.
  const DRAG_PX = 6;
  let drag = null;               // { source, x, y, active, ghost, over, size }
  let swallowClickUntil = 0;

  function cellUnder(x, y) {
    const hit = document.elementFromPoint(x, y);
    return hit ? hit.closest('.pb-cell') : null;
  }

  function setOver(cellEl) {
    const next = cellEl && cellEl.classList.contains('pb-cell--hint') ? cellEl : null;
    if (drag.over === next) return;
    if (drag.over) drag.over.classList.remove('pb-cell--drop');
    drag.over = next;
    if (next) next.classList.add('pb-cell--drop');
  }

  function endDrag() {
    if (!drag) return false;
    if (drag.ghost) drag.ghost.remove();
    if (drag.over) drag.over.classList.remove('pb-cell--drop');
    window.removeEventListener('pointermove', onDragMove);
    window.removeEventListener('pointerup', onDragUp);
    window.removeEventListener('pointercancel', onDragCancel);
    const wasActive = drag.active;
    drag = null;
    if (dragKey !== null) {
      dragKey = null;
      if (lastState) render(lastState, lastOpts);
    }
    return wasActive;
  }

  function onDragMove(e) {
    if (!drag) return;
    if (!drag.active) {
      if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < DRAG_PX) return;
      if (!(onDragStart && onDragStart(drag.source))) { endDrag(); return; }
      drag.active = true;
      dragKey = drag.source.type === 'board' ? key(drag.source.row, drag.source.col) : null;
      const size = (gridEl.getBoundingClientRect().width / BOARD_SIZE) * 0.94;
      const ghost = document.createElement('div');
      ghost.className = 'pb-ghost';
      ghost.style.width = `${size}px`;
      ghost.style.height = `${size}px`;
      ghost.innerHTML = `<img src="${drag.source.img}" alt="">`;
      document.body.appendChild(ghost);
      drag.ghost = ghost;
      drag.size = size;
      if (dragKey !== null && lastState) render(lastState, lastOpts);
    }
    drag.ghost.style.left = `${e.clientX - drag.size / 2}px`;
    drag.ghost.style.top = `${e.clientY - drag.size / 2}px`;
    setOver(cellUnder(e.clientX, e.clientY));
  }

  function onDragUp(e) {
    if (!drag) return;
    const wasActive = drag.active;
    const target = wasActive ? cellUnder(e.clientX, e.clientY) : null;
    endDrag();
    if (!wasActive) return;
    swallowClickUntil = performance.now() + 150;       // the click a drop may still produce is not a second tap
    if (target) {
      if (onCellClick) {
        onCellClick(Number(target.dataset.r), Number(target.dataset.c),
          { pointerType: e.pointerType || '', snapped: false, dropped: true });
      }
    } else if (onDragEnd) {
      onDragEnd();
    }
  }

  function onDragCancel() {
    if (endDrag() && onDragEnd) onDragEnd();
  }

  function onPress(e) {
    if (!onDragStart || drag || e.button !== 0) return;
    const pieceEl = e.target.closest('.pb-piece');
    const tileEl = e.target.closest('.pb-tile');
    let source = null;
    if (pieceEl && !pieceEl.classList.contains('pb-piece--ghost')) {
      const cellEl = pieceEl.closest('.pb-cell');
      const img = pieceEl.querySelector('img');
      if (cellEl && img) {
        source = { type: 'board', row: Number(cellEl.dataset.r), col: Number(cellEl.dataset.c), img: img.src };
      }
    } else if (tileEl && !tileEl.disabled) {
      const img = tileEl.querySelector('img');
      if (img) {
        source = { type: 'hand', player: Number(tileEl.dataset.player), flower: tileEl.dataset.flower, img: img.src };
      }
    }
    if (!source) return;
    drag = { source, x: e.clientX, y: e.clientY, active: false, ghost: null, over: null, size: 0 };
    window.addEventListener('pointermove', onDragMove);
    window.addEventListener('pointerup', onDragUp);
    window.addEventListener('pointercancel', onDragCancel);
  }

  function swallowClick(e) {
    if (performance.now() < swallowClickUntil) {
      e.stopPropagation();
      e.preventDefault();
    }
  }

  for (const target of new Set([el, topHandEl, bottomHandEl])) {
    target.addEventListener('pointerdown', onPress);
    target.addEventListener('click', swallowClick, true);
  }

  function setFlip(next) {
    const changed = _flip !== !!next;
    _flip = !!next;
    if (changed) positionGrid();
    if (lastState) render(lastState, lastOpts);
  }

  return { render, setFlip };
}
