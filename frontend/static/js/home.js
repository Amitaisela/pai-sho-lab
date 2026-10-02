/* Pai Sho Lab — the home page (`/`).
 *
 * Three ways to play a person (quick match, challenge a friend, pass and
 * play), the MushiBot level row, and the live-games list. Every mutating
 * call goes through ui.js's api()/clientId()/nickname()/setSeatToken(); this
 * file owns only what's specific to the home page.
 */
import * as ui from '/static/js/ui.js';

// Phase 0b calibrated ratings (approximate), shown as a tooltip on each level
// chip. Not fetched from the server — the brief keeps this as a small static
// map rather than wiring a new field through /api/house_bots.
const LEVEL_RATINGS = { 1: 225, 2: 300, 3: 311, 4: 385, 5: 397, 6: 476 };
const ACCENTS = ['Rock', 'Wheel', 'Knotweed', 'Boat'];

let selectedLevel = 1;
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
      { method: 'DELETE', keepalive: true }).catch(() => {});
  } catch (e) { /* best effort */ }
}

window.addEventListener('pagehide', () => {
  cancelSeekBeacon(qmSeekId);
});

async function startQuickMatch(seekId) {
  let data;
  try {
    data = await ui.api('/api/seek', {
      method: 'POST',
      body: {
        nickname: ui.nickname(), client_id: ui.clientId(),
        ...(seekId ? { seek_id: seekId } : {}), ...setupBody(),
      },
    });
  } catch (e) {
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
  const el = document.getElementById('quick-match-seeks');
  if (!el) return;
  let rows;
  try {
    rows = await ui.api('/api/seeks');
  } catch (e) {
    return;
  }
  if (!rows.length) {
    el.hidden = true;
    el.innerHTML = '';
    return;
  }
  el.hidden = false;
  el.innerHTML = rows.map((r) => `
    <li><button type="button" class="qm-seek-item" data-seek-id="${escapeHtml(r.seek_id)}">
      ${escapeHtml(r.nickname)} is waiting &mdash; tap to play
    </button></li>
  `).join('');
  el.querySelectorAll('.qm-seek-item').forEach((btn) => {
    btn.addEventListener('click', () => startQuickMatch(btn.dataset.seekId));
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

async function startChallenge() {
  let data;
  try {
    data = await ui.api('/api/challenge', {
      method: 'POST', body: { nickname: ui.nickname(), seat: 'random', ...setupBody() },
    });
  } catch (e) {
    return;
  }
  ui.setSeatToken(data.game_id, data.seat, data.seat_token);
  const url = new URL(data.join_url, location.origin).href;
  const close = ui.sheet(`
    <h2>Challenge created</h2>
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
      navigator.share({ title: 'Pai Sho Lab', url }).catch(() => {});
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
  line.textContent = `MushiBot level ${selectedLevel} · ≈ ${LEVEL_RATINGS[selectedLevel] ?? '?'}`;
}

function renderLevelChips(bots) {
  const row = document.getElementById('bot-level-row');
  if (!row) return;
  row.innerHTML = bots.map((b) => `
    <button type="button" class="chip chip--level${b.level === selectedLevel ? ' is-selected' : ''}"
      data-level="${b.level}" title="&asymp; ${LEVEL_RATINGS[b.level] ?? '?'} rating">${b.level}</button>
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

async function playBot() {
  let data;
  try {
    data = await ui.api('/api/bot_game', {
      method: 'POST', body: { level: selectedLevel, seat: 'random', nickname: ui.nickname() },
    });
  } catch (e) {
    return;
  }
  goToGame(data.game_id, data.seat, data.seat_token);
}

// ---------------------------------------------------------------- live now

async function refreshLive() {
  const el = document.getElementById('live-list');
  if (!el) return;
  let rows;
  try {
    rows = await ui.api('/api/live');
  } catch (e) {
    return;
  }
  if (!rows.length) {
    el.innerHTML = '<p class="muted">No games right now &mdash; start one!</p>';
    return;
  }
  el.innerHTML = rows.map((r) => `
    <a class="live-card" href="/game/${escapeHtml(r.game_id)}">
      ${escapeHtml(r.names[0] || 'Player 1')} vs ${escapeHtml(r.names[1] || 'Player 2')} &middot; ${r.moves} moves
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
  // Wrapped in a lambda, not passed directly: addEventListener hands the
  // click Event as the first arg, which would otherwise land in
  // startQuickMatch's seekId parameter.
  document.getElementById('quick-match-tile').addEventListener('click', () => startQuickMatch());
  document.getElementById('challenge-tile').addEventListener('click', startChallenge);
  document.getElementById('pass-tile').addEventListener('click', startPassAndPlay);
  document.getElementById('custom-game-link').addEventListener('click', openCustomGameSheet);
  document.getElementById('bot-play-btn').addEventListener('click', playBot);

  loadHouseBots();
  refreshSeeksList();
  refreshLive();
  setInterval(refreshSeeksList, 3000);
  setInterval(refreshLive, 5000);
}

wire();
