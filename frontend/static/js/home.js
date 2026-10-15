/* Pai Sho Lab — the home page (`/`), a lobby in the Lichess style.
 *
 * A grid of time-control tiles that each start a quick match in one tap, the open games waiting
 * for an opponent, three other ways to play (a friend, MushiBot, pass and play), and the live
 * games as mini boards. Every mutating call goes through ui.js's
 * api()/clientId()/nickname()/setSeatToken(); this file owns only what's specific to the home
 * page.
 */
import * as ui from '/static/js/ui.js';

// Phase 0b calibrated ratings (approximate), shown as a tooltip on each level
// chip. Not fetched from the server — the brief keeps this as a small static
// map rather than wiring a new field through /api/house_bots. They are on the bots' own
// scale (Bradley-Terry against each other), not the 1500-centred Glicko-2 scale that people are
// rated on, so the UI labels them as such rather than letting them read as site ratings.
const LEVEL_RATINGS = { 1: 225, 2: 300, 3: 311, 4: 385, 5: 397, 6: 476 };
const ACCENTS = ['Rock', 'Wheel', 'Knotweed', 'Boat'];

let selectedLevel = 1;
let selectedSeat = 'random';   // the colour for a MushiBot game: 'random', 1 (Guest, moves first) or 2 (Host)
let currentSetup = null; // {opening, accents} | null — applied to the next game
let qmPollTimer = null;
let qmSeekId = null;

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function setupBody() {
  return currentSetup ? { setup: currentSetup } : {};
}

// ------------------------------------------------------------ rated / casual
//
// Rated games need a signed-in user with a username and a time control, so the toggle is only
// shown then; with it hidden the choice is Casual whatever was stored.

const RATED_KEY = 'pai_rated';

function storedRated() {
  try { return localStorage.getItem(RATED_KEY) === 'rated'; } catch (e) { return false; }
}

function ratedAllowed() {
  const acct = ui.getAccountState();
  return acct.signedIn && !!acct.username && ui.timeControl() !== 'untimed';
}

function ratedChoice() {
  return ratedAllowed() && storedRated();
}

// Why "Rated" is unavailable right now, or '' when it is. A guest still sees the toggle (so rated
// play is discoverable) but the Rated chip says what it needs.
function ratedBlockedReason() {
  const acct = ui.getAccountState();
  if (!acct.signedIn) return 'Sign in to play rated games';
  if (!acct.username) return 'Choose a username to play rated games';
  if (ui.timeControl() === 'untimed') return 'Rated games need a time control';
  return '';
}

function syncRatedToggle() {
  const el = document.getElementById('rated-toggle');
  if (!el) return;
  // Where there are no accounts (the Lab, the public mirror) there is nothing to toggle.
  el.hidden = !document.body.hasAttribute('data-accounts');
  const rated = ratedChoice();
  const reason = ratedBlockedReason();
  el.querySelectorAll('[data-rated]').forEach((b) => {
    const on = (b.dataset.rated === 'rated') === rated;
    b.classList.toggle('is-selected', on);
    b.setAttribute('aria-pressed', String(on));
    if (b.dataset.rated === 'rated') {
      b.classList.toggle('is-blocked', !!reason);
      b.title = reason;
    }
  });
}

function wireRatedToggle() {
  const el = document.getElementById('rated-toggle');
  if (!el) return;
  el.querySelectorAll('[data-rated]').forEach((b) => {
    b.addEventListener('click', () => {
      const reason = b.dataset.rated === 'rated' ? ratedBlockedReason() : '';
      if (reason) {
        ui.sheet(`<h2>${escapeHtml(reason)}</h2><p class="muted">Rated games are between signed-in players and change both ratings.</p>`);
        return;
      }
      try { localStorage.setItem(RATED_KEY, b.dataset.rated); } catch (e) { /* private mode */ }
      syncRatedToggle();
    });
  });
  // The time-control chips save their own choice first (their handler is on the chip itself).
  document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest('[data-time-control]')) syncRatedToggle();
  });
  ui.accountReady().then(syncRatedToggle);
  syncRatedToggle();
}

// --------------------------------------------------------------- quick match

function stopQmPoll() {
  if (qmPollTimer) {
    clearInterval(qmPollTimer);
    qmPollTimer = null;
  }
  qmSeekId = null;
}

function goToGame(gid, seat, token) {
  if (seat != null && token != null) ui.setSeatToken(gid, seat, token);
  location.href = `/game/${gid}`;
}

// Fire-and-forget cancel, used from every path that drops a seek: the Cancel
// button, Escape, a backdrop click (all three via sheet()'s onClose), and the
// page unloading outright (pagehide). `keepalive: true` lets the pagehide
// case's request outlive the page it was issued from.
function cancelSeekBeacon(sid) {
  if (!sid) return;
  try {
    fetch(`/api/seek/${sid}?client_id=${encodeURIComponent(ui.clientId())}`,
      { method: 'DELETE', keepalive: true, headers: ui.csrfHeaders() }).catch(() => {});
  } catch (e) { /* best effort */ }
}

window.addEventListener('pagehide', () => {
  cancelSeekBeacon(qmSeekId);
});

async function startQuickMatch(seekId, seekTimeControl, seekRated) {
  let data;
  try {
    data = await ui.api('/api/seek', {
      method: 'POST',
      body: {
        nickname: ui.nickname(), client_id: ui.clientId(),
        // Taking a listed seek means taking the time control and rated-ness that seek asked for.
        time_control: seekId && seekTimeControl ? seekTimeControl : ui.timeControl(),
        rated: seekId ? !!seekRated : ratedChoice(),
        ...(seekId ? { seek_id: seekId } : {}), ...setupBody(),
      },
    });
  } catch (e) {
    if (seekId && e && e.status === 401) {
      // A guest tapped a rated seek in the list: say why nothing happened.
      ui.sheet('<h2>Sign in to join rated games</h2><p class="muted">Rated games are for signed-in players with a username.</p>');
    } else if (seekId && e && e.status === 403 && e.error) {
      ui.sheet(`<h2>${escapeHtml(e.error)}</h2>`);
    }
    return;
  }
  if (data.game_id) {
    goToGame(data.game_id, data.seat, data.seat_token);
    return;
  }
  qmSeekId = data.seek_id;
  const close = ui.sheet(`
    <h2>Looking for an opponent&hellip;</h2>
    <p class="muted" id="qm-sheet-status">We'll take you to the game as soon as someone joins.</p>
    <button type="button" class="btn btn-secondary" id="qm-cancel-btn">Cancel</button>
  `, {
    onClose: () => {
      const sid = qmSeekId;
      stopQmPoll();
      cancelSeekBeacon(sid);
    },
  });
  document.getElementById('qm-cancel-btn').addEventListener('click', close);
  qmPollTimer = setInterval(async () => {
    if (!qmSeekId) return;
    let r;
    try {
      r = await ui.api(`/api/seek/${qmSeekId}?client_id=${encodeURIComponent(ui.clientId())}`);
    } catch (e) {
      stopQmPoll();
      close();
      return;
    }
    if (r.status === 'paired') {
      stopQmPoll();
      close();
      goToGame(r.game_id, r.seat, r.seat_token);
    }
  }, 1000);
}

async function refreshSeeksList() {
  let rows;
  try {
    rows = await ui.api('/api/seeks');
  } catch (e) {
    return;
  }
  renderSeeks(rows);
}

// Pushed seek rows carry `age_s` as of the moment they were published and are not
// re-pushed as it ticks, so the age shown is computed here: the age at receipt plus
// the time since receipt (per seek, so a re-push of the same seek keeps counting).
const seekFirstSeen = new Map();   // seek_id -> performance.now() - age_s*1000 (its creation, locally)

function seekAgeText(seekId) {
  const t0 = seekFirstSeen.get(seekId);
  if (t0 === undefined) return '';
  const s = Math.max(0, Math.floor((performance.now() - t0) / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m`;
}

function tickSeekAges() {
  document.querySelectorAll('#quick-match-seeks .qm-seek-age').forEach((span) => {
    span.textContent = seekAgeText(span.dataset.seekId);
  });
}

function renderSeeks(rows) {
  const el = document.getElementById('quick-match-seeks');
  const empty = document.getElementById('quick-match-empty');
  if (!el) return;
  if (empty) empty.hidden = rows.length > 0;
  const now = performance.now();
  const live = new Set(rows.map((r) => r.seek_id));
  for (const id of [...seekFirstSeen.keys()]) if (!live.has(id)) seekFirstSeen.delete(id);
  for (const r of rows) {
    if (!seekFirstSeen.has(r.seek_id)) seekFirstSeen.set(r.seek_id, now - (Number(r.age_s) || 0) * 1000);
  }
  if (!rows.length) {
    el.hidden = true;
    el.innerHTML = '';
    return;
  }
  el.hidden = false;
  el.innerHTML = rows.map((r) => `
    <li><button type="button" class="qm-seek-item" data-seek-id="${escapeHtml(r.seek_id)}"
      data-time-control-of-seek="${escapeHtml(r.time_control || 'untimed')}"
      data-rated-seek="${r.rated ? 'true' : 'false'}">
      <span class="qm-who">${escapeHtml(r.nickname)}${r.rating != null ? ` <span class="muted">(${escapeHtml(r.rating)})</span>` : ''}</span>
      <span class="qm-tc">${escapeHtml(r.time_control || 'untimed')}</span>
      ${r.rated ? '<span class="rated-tag">Rated</span>' : '<span class="qm-mode muted">Casual</span>'}
      <span class="muted qm-seek-age" data-seek-id="${escapeHtml(r.seek_id)}">${seekAgeText(r.seek_id)}</span>
    </button></li>
  `).join('');
  el.querySelectorAll('.qm-seek-item').forEach((btn) => {
    btn.addEventListener('click', () => startQuickMatch(
      btn.dataset.seekId, btn.dataset.timeControlOfSeek, btn.dataset.ratedSeek === 'true'));
  });
}

// ----------------------------------------------------------- challenge / pass

function qrSvgFor(url) {
  if (typeof window.qrcode !== 'function') return '';
  try {
    const qr = window.qrcode(0, 'M');
    qr.addData(url);
    qr.make();
    return qr.createSvgTag({ cellSize: 4, margin: 2 });
  } catch (e) {
    return '';
  }
}

function openFriendSheet() {
  const close = ui.sheet(`
    <h2>Play with a friend</h2>
    <p class="muted">Pick a time control, then send the link. They take the other seat.</p>
    <div class="time-control"><div data-time-control-mount></div></div>
    <p id="friend-rated-note" class="muted" hidden>Rated game: both players need an account.</p>
    <button type="button" class="btn btn-primary" id="friend-create-btn">Create link</button>
  `);
  document.querySelectorAll('.sheet [data-time-control-mount]').forEach((el) => ui.mountTimeControlPicker(el));
  const note = document.getElementById('friend-rated-note');
  const syncNote = () => { note.hidden = !ratedChoice(); };
  syncNote();
  document.querySelector('.sheet').addEventListener('click', syncNote);
  document.getElementById('friend-create-btn').addEventListener('click', () => {
    close();
    startChallenge();
  });
}

async function startChallenge() {
  let data;
  const wantRated = ratedChoice();
  try {
    data = await ui.api('/api/challenge', {
      method: 'POST',
      body: {
        nickname: ui.nickname(), seat: 'random', time_control: ui.timeControl(),
        rated: wantRated, ...setupBody(),
      },
    });
  } catch (e) {
    return;
  }
  ui.setSeatToken(data.game_id, data.seat, data.seat_token);
  const url = new URL(data.join_url, location.origin).href;
  const close = ui.sheet(`
    <h2>Challenge created</h2>
    ${wantRated ?'<p class="muted">Rated game: both players need an account.</p>' : ''}
    <p class="muted">Send this link, or share the QR code.</p>
    <label class="game__field">Join link
      <input id="challenge-link-input" readonly value="${escapeHtml(url)}" onclick="this.select()">
    </label>
    <div class="game__sheet-actions">
      <button type="button" class="btn btn-secondary" id="challenge-copy-btn">Copy link</button>
      <button type="button" class="btn btn-secondary" id="challenge-share-btn" hidden>Share</button>
    </div>
    <div id="challenge-qr" class="qr-wrap">${qrSvgFor(url)}</div>
    <button type="button" class="btn btn-primary" id="challenge-open-btn">Open game</button>
  `);
  document.getElementById('challenge-copy-btn').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(url);
    } catch (e) {
      document.getElementById('challenge-link-input').select();
    }
  });
  if (navigator.share) {
    const shareBtn = document.getElementById('challenge-share-btn');
    shareBtn.hidden = false;
    shareBtn.addEventListener('click', () => {
      navigator.share({ title: 'Play me at Skud Pai Sho', url }).catch(() => {});
    });
  }
  document.getElementById('challenge-open-btn').addEventListener('click', () => {
    close();
    goToGame(data.game_id, null, null);
  });
}

async function startPassAndPlay() {
  let data;
  try {
    data = await ui.api('/api/pass_game', { method: 'POST', body: { ...setupBody() } });
  } catch (e) {
    return;
  }
  goToGame(data.game_id, null, null);
}

// -------------------------------------------------------------------- bots

function updateLevelLine() {
  const line = document.getElementById('bot-level-line');
  if (!line) return;
  line.textContent = `MushiBot level ${selectedLevel} \u00b7 bot-scale strength \u2248 ${LEVEL_RATINGS[selectedLevel] ?? '?'}`
    + ' (not comparable to the ratings people have)';
}

function renderLevelChips(bots) {
  const row = document.getElementById('bot-level-row');
  if (!row) return;
  row.innerHTML = bots.map((b) => `
    <button type="button" class="chip chip--level${b.level === selectedLevel ? ' is-selected' : ''}"
      data-level="${b.level}" title="Level ${b.level}: bot-scale strength &asymp; ${LEVEL_RATINGS[b.level] ?? '?'}">${b.level}</button>
  `).join('');
  row.querySelectorAll('.chip--level').forEach((chip) => {
    chip.addEventListener('click', () => {
      selectedLevel = Number(chip.dataset.level);
      row.querySelectorAll('.chip--level').forEach((c) => {
        c.classList.toggle('is-selected', Number(c.dataset.level) === selectedLevel);
      });
      updateLevelLine();
    });
  });
  updateLevelLine();
}

async function loadHouseBots() {
  let bots;
  try {
    bots = await ui.api('/api/house_bots');
  } catch (e) {
    bots = [1, 2, 3, 4, 5, 6].map((level) => ({ level, label: `MushiBot level ${level}` }));
  }
  renderLevelChips(bots);
}

function openBotSheet() {
  const close = ui.sheet(`
    <h2>Play MushiBot</h2>
    <p class="muted time-control__label">Level</p>
    <div id="bot-level-row" class="chip-row"></div>
    <p id="bot-level-line" class="muted bot-level-line"></p>
    <p class="muted time-control__label">You play</p>
    <div class="chip-row" id="bot-seat-row" role="group" aria-label="Colour">
      <button type="button" class="chip" data-seat="1">Guest (first)</button>
      <button type="button" class="chip" data-seat="random">Random</button>
      <button type="button" class="chip" data-seat="2">Host</button>
    </div>
    <p class="muted time-control__label">Time control</p>
    <div class="time-control"><div data-time-control-mount></div></div>
    <button type="button" class="btn btn-primary" id="bot-play-btn">Play</button>
  `);
  document.querySelectorAll('.sheet [data-time-control-mount]').forEach((el) => ui.mountTimeControlPicker(el));
  const seatRow = document.getElementById('bot-seat-row');
  const syncSeat = () => seatRow.querySelectorAll('[data-seat]').forEach((c) => {
    const on = c.dataset.seat === String(selectedSeat);
    c.classList.toggle('is-selected', on);
    c.setAttribute('aria-pressed', String(on));
  });
  seatRow.querySelectorAll('[data-seat]').forEach((c) => c.addEventListener('click', () => {
    selectedSeat = c.dataset.seat === 'random' ? 'random' : Number(c.dataset.seat);
    syncSeat();
  }));
  syncSeat();
  loadHouseBots();
  document.getElementById('bot-play-btn').addEventListener('click', () => {
    close();
    playBot();
  });
}

async function playBot() {
  let data;
  try {
    data = await ui.api('/api/bot_game', {
      method: 'POST',
      body: { level: selectedLevel, seat: selectedSeat, nickname: ui.nickname(), time_control: ui.timeControl() },
    });
  } catch (e) {
    return;
  }
  goToGame(data.game_id, data.seat, data.seat_token);
}

// ---------------------------------------------------------------- live now

async function refreshLive() {
  let rows;
  try {
    rows = await ui.api('/api/live');
  } catch (e) {
    return;
  }
  renderLive(rows);
}

// A mini board for the lobby: the circle, the diamond, and one dot per tile, blue for the Guest
// and orange for the Host (a dashed ring while growing). `board` is {"row,col": [player, growing]}.
function miniBoardSvg(board) {
  const dots = Object.entries(board || {}).map(([k, v]) => {
    const [r, c] = k.split(',').map(Number);
    return `<circle cx="${c + 0.5}" cy="${r + 0.5}" r="0.46" class="mini-p${v[0]}${v[1] ? ' mini-growing' : ''}"/>`;
  }).join('');
  return '<svg class="mini-board" viewBox="0 0 19 19" aria-hidden="true">'
    + '<circle cx="9.5" cy="9.5" r="9.4" class="mini-bg"/>'
    + '<path d="M9.5 2.5 L16.5 9.5 L9.5 16.5 L2.5 9.5 Z" class="mini-diamond"/>'
    + `${dots}</svg>`;
}

function renderLive(rows) {
  const el = document.getElementById('live-list');
  if (!el) return;
  if (!rows.length) {
    el.innerHTML = '<p class="muted">No games right now &mdash; start one!</p>';
    return;
  }
  el.innerHTML = rows.map((r) => `
    <a class="live-card" href="/game/${escapeHtml(r.game_id)}">
      ${miniBoardSvg(r.board)}
      <span class="live-card__names">${escapeHtml(r.names[0] || 'Player 1')} <span class="muted">vs</span> ${escapeHtml(r.names[1] || 'Player 2')}</span>
      <span class="muted live-card__moves">${r.moves} moves</span>
    </a>
  `).join('');
}

// ------------------------------------------------------------- custom game

function accentFieldsHtml(player) {
  return ACCENTS.map((a) => `
    <label class="custom-accent-field">${a}
      <input type="number" min="0" max="2" value="1" data-player="${player}" data-accent="${a}">
    </label>
  `).join('');
}

function readAccents(player) {
  const out = [];
  document.querySelectorAll(`#custom-game-sheet [data-player="${player}"]`).forEach((input) => {
    const n = Math.max(0, Math.min(2, parseInt(input.value || '0', 10)));
    for (let i = 0; i < n; i++) out.push(input.dataset.accent);
  });
  return out;
}

function openCustomGameSheet() {
  const close = ui.sheet(`
    <div id="custom-game-sheet">
      <h2>Custom game</h2>
      <label class="game__field">Opening
        <select id="custom-opening">
          <option value="random">Random opening</option>
          <option value="">Informal (no forced flower)</option>
        </select>
      </label>
      <div class="custom-accents">
        <div>
          <b>Player 1 accents</b> <span class="muted">(2 of each at most, 4 total)</span>
          ${accentFieldsHtml(1)}
        </div>
        <div>
          <b>Player 2 accents</b> <span class="muted">(2 of each at most, 4 total)</span>
          ${accentFieldsHtml(2)}
        </div>
      </div>
      <p id="custom-game-error" class="muted" hidden></p>
      <button type="button" class="btn btn-primary" id="custom-game-apply">Use for the next game</button>
    </div>
  `);
  if (currentSetup) {
    document.getElementById('custom-opening').value = currentSetup.opening || '';
  }
  document.getElementById('custom-game-apply').addEventListener('click', () => {
    const p1 = readAccents(1);
    const p2 = readAccents(2);
    const errEl = document.getElementById('custom-game-error');
    if (p1.length !== 4 || p2.length !== 4) {
      errEl.hidden = false;
      errEl.textContent = 'Each player needs exactly 4 accents (at most 2 of each).';
      return;
    }
    const openingRaw = document.getElementById('custom-opening').value;
    currentSetup = { opening: openingRaw === '' ? null : openingRaw, accents: { 1: p1, 2: p2 } };
    close();
  });
}

// ------------------------------------------------------------------- init

function wire() {
  // A time-control tile sets the saved time control and starts a quick match with it. Wrapped in
  // a lambda: addEventListener hands the click Event as the first argument, which would
  // otherwise land in startQuickMatch's seekId parameter.
  document.querySelectorAll('#pairing-grid [data-pair]').forEach((tile) => {
    tile.addEventListener('click', () => {
      ui.setTimeControl(tile.dataset.pair);
      syncRatedToggle();
      startQuickMatch();
    });
  });
  document.getElementById('challenge-tile').addEventListener('click', openFriendSheet);
  document.getElementById('bot-open-btn').addEventListener('click', openBotSheet);
  document.getElementById('pass-tile').addEventListener('click', startPassAndPlay);
  document.getElementById('custom-game-link').addEventListener('click', openCustomGameSheet);

  wireRatedToggle();
  refreshSeeksList();
  refreshLive();
  setInterval(tickSeekAges, 1000);   // local clock only, no network
  startLobbyFeed();
}

// The lobby lists arrive over the shared WebSocket (live.js), which sends a snapshot
// on every (re)connect. The old polls come back only while the socket is down or was
// never available (the legacy Flask server has no /ws). The waiting player's own
// one-second seek poll (startQuickMatch) is separate and stays as is.
let lobbyFallbackTimers = [];

function startLobbyFallback() {
  if (lobbyFallbackTimers.length) return;
  lobbyFallbackTimers = [setInterval(refreshSeeksList, 3000), setInterval(refreshLive, 5000)];
}

function stopLobbyFallback() {
  lobbyFallbackTimers.forEach(clearInterval);
  lobbyFallbackTimers = [];
}

function startLobbyFeed() {
  if (!window.MushiLive) {
    startLobbyFallback();
    return;
  }
  window.MushiLive.subscribeLobby(
    { onLive: renderLive, onSeeks: renderSeeks },
    {
      onUp: stopLobbyFallback,
      onDown: () => { startLobbyFallback(); },
    },
  );
}

wire();
