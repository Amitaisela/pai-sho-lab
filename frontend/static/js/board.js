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
 * Returns `{ render(state, { selected, hints, myHandPlayer }), setFlip(bool) }`.
 */
export function createBoard(el, { flip = false, onCellClick, onTileClick, onHandClick } = {}) {
  let _flip = !!flip;
  let lastState = null;
  let lastOpts = {};
  let builtValidSpacesKey = null;
  let prevTileKeys = new Set(); // which points held a tile as of the last renderCells() call
  const cells = new Map(); // "r,c" -> cell element
  const pieceSlots = new Map(); // "r,c" -> the child element a piece renders into

  el.innerHTML = `
    <div class="pb-root">
      <div class="pb-hand" data-slot="top"></div>
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
      <div class="pb-hand" data-slot="bottom"></div>
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
  const topHandEl = el.querySelector('.pb-hand[data-slot="top"]');
  const bottomHandEl = el.querySelector('.pb-hand[data-slot="bottom"]');

  function disp(r, c) {
    return _flip ? [BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c] : [r, c];
  }

  function buildGrid(validSpaces) {
    gridEl.innerHTML = '';
    cells.clear();
    pieceSlots.clear();
    for (const [r, c] of validSpaces) {
      const cellEl = document.createElement('div');
      cellEl.className = 'pb-cell';
      cellEl.dataset.r = String(r);
      cellEl.dataset.c = String(c);
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
      cellEl.addEventListener('click', () => onCellClick && onCellClick(r, c));
      gridEl.appendChild(cellEl);
      cells.set(key(r, c), cellEl);
      pieceSlots.set(key(r, c), slot);
    }
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
    for (const [k, cellEl] of cells) {
      const [r, c] = k.split(',').map(Number);
      const tile = state.board[k] || null;
      cellEl.classList.toggle('pb-cell--hint', hintSet.has(k));
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
      if (!prevTileKeys.has(k)) piece.classList.add('pb-piece--enter');
      if (tile.growing) piece.classList.add('pb-piece--growing');
      if (k === selKey) piece.classList.add('pb-piece--selected');
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
    prevTileKeys = nextTileKeys;
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

  function setFlip(next) {
    const changed = _flip !== !!next;
    _flip = !!next;
    if (changed) positionGrid();
    if (lastState) render(lastState, lastOpts);
  }

  return { render, setFlip };
}
