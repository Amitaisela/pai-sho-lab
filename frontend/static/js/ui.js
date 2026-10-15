/* Pai Sho Lab — shared chrome and small utilities.
 *
 * Imported as an ES module by base.html (and, later, by the game/learn pages)
 * for the identity bits that are the same everywhere: the nickname a player
 * shows opponents, the per-browser client id used to pair quick-match seeks,
 * the seat token each game stores so a refresh doesn't lose your seat, the
 * Lab link's visibility, a tiny fetch wrapper, and the bottom-sheet/dialog
 * used for the menu and (later) game setup.
 */
import * as sounds from '/static/js/sound.js';

const NICK_KEY = 'pai_nickname';
const CLIENT_KEY = 'pai_client_id';
const THEME_KEY = 'pai_theme';

function randomDigits(n) {
  let s = '';
  for (let i = 0; i < n; i++) s += Math.floor(Math.random() * 10);
  return s;
}

function uuidFallback() {
  // crypto.randomUUID() needs a secure context; this covers plain http on
  // the tailnet (no secure-context guarantee) without pulling in a library.
  // Still draws from crypto.getRandomValues when it's available (it doesn't
  // need a secure context) rather than Math.random, which is not a
  // cryptographically random source and collides client ids more than a
  // random identifier should.
  const bytes = new Uint8Array(16);
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0'));
  return `${hex.slice(0, 4).join('')}-${hex.slice(4, 6).join('')}-${hex.slice(6, 8).join('')}-`
    + `${hex.slice(8, 10).join('')}-${hex.slice(10, 16).join('')}`;
}

function readLS(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeLS(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode etc. */ }
}

export function nickname() {
  let n = readLS(NICK_KEY);
  if (!n) {
    n = `Guest-${randomDigits(4)}`;
    writeLS(NICK_KEY, n);
  }
  return n;
}

export function setNickname(s) {
  const clean = String(s || '').trim().slice(0, 24) || nickname();
  writeLS(NICK_KEY, clean);
  return clean;
}

export function clientId() {
  let id = readLS(CLIENT_KEY);
  if (!id) {
    id = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : uuidFallback();
    writeLS(CLIENT_KEY, id);
  }
  return id;
}

export function seatToken(gid) {
  const raw = readLS(`pai_seat_${gid}`);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

export function setSeatToken(gid, seat, token) {
  writeLS(`pai_seat_${gid}`, JSON.stringify({ seat, token }));
}

// ---------------------------------------------------------- time control

const TIME_CONTROL_KEY = 'pai_time_control';
const TIME_CONTROLS = ['untimed', '3+2', '5+3', '10+5', '15+10'];

/** The saved time-control choice (a preset key), 'untimed' by default. Anything stored that
 * is not one of the five presets reads as untimed. */
export function timeControl() {
  const v = readLS(TIME_CONTROL_KEY);
  return TIME_CONTROLS.includes(v) ? v : 'untimed';
}

export function isTimeControl(key) {
  return TIME_CONTROLS.includes(key);
}

export function setTimeControl(key) {
  if (TIME_CONTROLS.includes(key)) writeLS(TIME_CONTROL_KEY, key);
  return timeControl();
}

/** A chip row with the five options; the saved choice is pre-selected. Pair with
 * mountTimeControlPicker(), which also wires the clicks. */
export function timeControlPickerHtml() {
  const current = timeControl();
  const chips = TIME_CONTROLS.map((k) => `
    <button type="button" class="chip${k === current ? ' is-selected' : ''}"
      data-time-control="${k}" aria-pressed="${k === current}">${k === 'untimed' ? 'Untimed' : k}</button>
  `).join('');
  return `<div class="chip-row" role="group" aria-label="Time control">${chips}</div>`;
}

/** Fills `el` with the picker and keeps every mounted picker on the page showing the saved choice. */
export function mountTimeControlPicker(el) {
  if (!el) return;
  el.innerHTML = timeControlPickerHtml();
  el.querySelectorAll('[data-time-control]').forEach((chip) => {
    chip.addEventListener('click', () => {
      const chosen = setTimeControl(chip.dataset.timeControl);
      document.querySelectorAll('[data-time-control]').forEach((c) => {
        const on = c.dataset.timeControl === chosen;
        c.classList.toggle('is-selected', on);
        c.setAttribute('aria-pressed', String(on));
      });
    });
  });
}

/** Milliseconds as a clock face: 'm:ss', or 'm:ss.t' (tenths) below ten seconds. */
export function formatClock(ms) {
  const left = Math.max(0, Number(ms) || 0);
  if (left < 10000) {
    const tenths = Math.floor(left / 100);
    return `0:0${Math.floor(tenths / 10)}.${tenths % 10}`;
  }
  const total = Math.ceil(left / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

const LAB_TOKEN_KEY = 'mushibot_api_token';

/** Full result of `/api/lab/check`: `ok` (this browser is in), and
 * `tokenRequired` (whether the server has MUSHIBOT_API_TOKEN configured at
 * all — false means local dev, where /lab stays open with no gate). */
export async function labCheck() {
  try {
    const token = readLS(LAB_TOKEN_KEY) || '';
    const headers = token ? { 'X-API-Token': token } : {};
    const r = await fetch('/api/lab/check', { headers });
    let data = null;
    try { data = await r.json(); } catch { /* no body */ }
    // The server answers 200 even when ok is false (no token configured at
    // all, so there's nothing to 401 against) - read `ok`/`token_required`
    // from the body itself rather than the HTTP status.
    return { ok: !!(data && data.ok), tokenRequired: data ? !!data.token_required : true };
  } catch {
    return { ok: false, tokenRequired: true };
  }
}

export async function labAllowed() {
  return (await labCheck()).ok;
}

export function setLabToken(token) {
  writeLS(LAB_TOKEN_KEY, String(token || '').trim());
}

// Pages that have accounts load account_chrome.js, which registers these hooks so every
// mutating api() call waits for the signed-in state and carries its CSRF header. Pages
// without accounts (the Lab) never set them and behave exactly as before.
const accountHooks = {
  ready: Promise.resolve(), csrf: () => ({}), state: () => ({ user: null }),
};

export function setAccountHooks(h) {
  if (h && h.ready) accountHooks.ready = h.ready;
  if (h && h.csrf) accountHooks.csrf = h.csrf;
  if (h && h.state) accountHooks.state = h.state;
}

/** Resolves once the signed-in state is known (immediately on pages without accounts). */
export function accountReady() {
  return accountHooks.ready;
}

/** {signedIn, username} as of now: signed out until the account hooks say otherwise. */
export function getAccountState() {
  let s = null;
  try { s = accountHooks.state(); } catch { s = null; }
  const user = s && s.user ? s.user : null;
  return { signedIn: !!user, username: user && user.username ? user.username : null };
}

/** Headers a cookie-authenticated mutation must carry (empty for signed-out visitors). */
export function csrfHeaders() {
  return accountHooks.csrf();
}

export async function api(path, { method = 'GET', body, headers } = {}) {
  const opts = { method, headers: { ...(headers || {}) } };
  if (method !== 'GET') {
    await accountHooks.ready;
    Object.assign(opts.headers, accountHooks.csrf());
  }
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch { /* empty body, e.g. 204 */ }
  if (!r.ok) {
    throw { status: r.status, error: (data && data.error) || r.statusText };
  }
  return data;
}

/** Opens the shared bottom sheet / dialog with `html` as its body content.
 * Returns a close() function. Clicking the backdrop or pressing Escape closes
 * it too - `onClose`, if given, fires exactly once no matter which of those
 * three ways (the returned function, Escape, or the backdrop click) actually
 * closed it, so a caller with cleanup to do (e.g. cancelling a quick-match
 * seek) only has to register it once instead of wiring every close path. */
export function sheet(html, { onClose } = {}) {
  const backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  const panel = document.createElement('div');
  panel.className = 'sheet';
  panel.innerHTML = html;
  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    backdrop.classList.remove('is-open');
    document.removeEventListener('keydown', onKey);
    setTimeout(() => backdrop.remove(), 220);
    if (onClose) onClose();
  };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
  document.addEventListener('keydown', onKey);

  requestAnimationFrame(() => backdrop.classList.add('is-open'));
  return close;
}

/** The name of the in-app browser (Instagram, Facebook, WhatsApp...) the page is open in, or null.
 * Challenge links are often opened from a chat; those browsers keep their own storage (a seat
 * taken there is lost in the real browser) and some (Google sign-in) refuse to run at all. */
export function inAppBrowser() {
  const ua = navigator.userAgent || '';
  const m = ua.match(/(Instagram|FBAN|FBAV|FB_IAB|Messenger|WhatsApp|Snapchat|Telegram|MicroMessenger|Line\/|TikTok|Twitter)/i);
  if (!m) return null;
  return m[1].replace(/^FB.*/i, 'Facebook').replace(/^Line.*/i, 'Line').replace(/^MicroMessenger/i, 'WeChat');
}

/** HTML for the note shown when the page is in an in-app browser, or '' otherwise. */
export function inAppBrowserNote(what) {
  const app = inAppBrowser();
  if (!app) return '';
  return `<p class="inapp-note" role="note">You are in the ${app} browser. ${what} Open this page in Chrome or Safari`
    + ' (the menu in the corner usually has "Open in browser").</p>';
}

/** Shares `url` with the system share sheet where there is one (phones), else copies it.
 * Resolves 'shared', 'copied', 'cancelled' or 'failed'. */
export async function shareLink({ url, title }) {
  if (navigator.share) {
    try {
      await navigator.share({ url, title });
      return 'shared';
    } catch (e) {
      if (e && e.name === 'AbortError') return 'cancelled';
    }
  }
  try {
    await navigator.clipboard.writeText(url);
    return 'copied';
  } catch {
    return 'failed';
  }
}

function applyStoredTheme() {
  const t = readLS(THEME_KEY);
  if (t === 'dark' || t === 'light') {
    document.documentElement.setAttribute('data-theme', t);
  }
}

function currentTheme() {
  const stored = readLS(THEME_KEY);
  if (stored === 'dark' || stored === 'light') return stored;
  return (typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: light)').matches)
    ? 'light' : 'dark';
}

function toggleTheme() {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  writeLS(THEME_KEY, next);
  document.documentElement.setAttribute('data-theme', next);
  return next;
}

// Lab pages set <body data-chrome="lab"> and get the Lab's own menu: the player-facing
// pages (/learn, /bots, /developers) live on the site, not on the Lab server.
function menuHtml() {
  if (document.body.dataset.chrome === 'lab') {
    return `
    <div class="menu-list">
      <a href="/lab/board">Board</a>
      <a href="/lab/train">Train</a>
      <a href="/lab/simulate">Simulate</a>
      <a href="/lab/leaderboard">Leaderboard</a>
      <a href="/lab/rules">Rules</a>
      <a href="/lab/guide">Guide</a>
      <div class="menu-sep"></div>
      <button type="button" id="menu-theme-btn"></button>
    </div>
  `;
  }
  return `
    <div class="menu-list">
      <a href="/">Play</a>
      <a href="/learn">Learn</a>
      <a href="/bots">Bots</a>
      <a href="/developers">Developers</a>
      <a id="menu-lab-link" href="/lab" hidden>Lab</a>
      <button type="button" id="menu-signin-btn" hidden>Sign in</button>
      <button type="button" id="menu-signout-btn" hidden>Sign out</button>
      <div class="menu-sep"></div>
      <button type="button" id="menu-theme-btn"></button>
      <button type="button" id="menu-sound-btn"></button>
      <button type="button" id="menu-haptics-btn" hidden></button>
    </div>
  `;
}

function labelForTheme(t) {
  return t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
}

/** Wires up the chrome every page's base.html ships: the nickname chip, the
 * hamburger menu (and its theme toggle), and the Lab link, which stays
 * hidden until `labAllowed()` resolves true. Safe to call once per page. */
export function initChrome() {
  applyStoredTheme();
  // The top bar's own navigation marks the page you are on.
  document.querySelectorAll('.topbar__nav a').forEach((a) => {
    const here = a.getAttribute('href') === '/'
      ? location.pathname === '/' || location.pathname.startsWith('/game/')
      : location.pathname === a.getAttribute('href') || location.pathname.startsWith(`${a.getAttribute('href')}/`);
    if (here) a.setAttribute('aria-current', 'page');
  });
  let labAllowedCache = false;

  const chip = document.querySelector('[data-nickname-chip]');
  if (chip) {
    chip.textContent = nickname();
    chip.addEventListener('click', () => {
      // A sheet, not window.prompt: prompt() looks different in every browser and is blocked
      // or mangled in the in-app browsers that open shared challenge links.
      const close = sheet(`
        <h2>Your name</h2>
        <p class="muted">Shown to opponents when you play as a guest.</p>
        <label class="game__field">Name
          <input id="nick-input" type="text" maxlength="24" autocomplete="off" autocapitalize="off"
                 autocorrect="off" spellcheck="false" enterkeyhint="done">
        </label>
        <div class="game__sheet-actions">
          <button type="button" class="btn btn-primary" id="nick-save">Save</button>
          <button type="button" class="btn btn-quiet" id="nick-cancel">Cancel</button>
        </div>`);
      const input = document.getElementById('nick-input');
      input.value = nickname();
      input.focus();
      input.select();
      const save = () => { chip.textContent = setNickname(input.value); close(); };
      document.getElementById('nick-save').addEventListener('click', save);
      document.getElementById('nick-cancel').addEventListener('click', close);
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') save(); });
    });
  }

  const menuBtn = document.querySelector('[data-menu-btn]');
  if (menuBtn) {
    menuBtn.addEventListener('click', () => {
      const close = sheet(menuHtml());
      const panel = document.querySelector('.sheet-backdrop.is-open, .sheet-backdrop');
      const root = panel ? panel.querySelector('.sheet') : document;
      const labLink = root.querySelector('#menu-lab-link');
      if (labLink) labLink.hidden = !labAllowedCache;
      const themeBtn = root.querySelector('#menu-theme-btn');
      if (themeBtn) {
        themeBtn.textContent = labelForTheme(currentTheme());
        themeBtn.addEventListener('click', () => {
          themeBtn.textContent = labelForTheme(toggleTheme());
        });
      }
      const soundBtn = root.querySelector('#menu-sound-btn');
      if (soundBtn) {
        const label = () => { soundBtn.textContent = `Sound: ${sounds.soundOn() ? 'on' : 'off'}`; };
        label();
        soundBtn.addEventListener('click', () => {
          sounds.setSound(!sounds.soundOn());
          label();
          sounds.play('move');         // so turning it on is audible
        });
      }
      const hapticsBtn = root.querySelector('#menu-haptics-btn');
      if (hapticsBtn && sounds.canVibrate()) {
        hapticsBtn.hidden = false;
        const label = () => { hapticsBtn.textContent = `Vibration: ${sounds.hapticsOn() ? 'on' : 'off'}`; };
        label();
        hapticsBtn.addEventListener('click', () => {
          sounds.setHaptics(!sounds.hapticsOn());
          label();
          sounds.buzz(20);
        });
      }
      root.querySelectorAll('a').forEach((a) => a.addEventListener('click', close));
    });
  }

  const labLink = document.getElementById('lab-link');
  labAllowed().then((ok) => {
    labAllowedCache = ok;
    if (ok && labLink) labLink.hidden = false;
  });

  if (document.body.hasAttribute('data-accounts')) {
    // Set synchronously, before the module has loaded: any mutating api() call made on page load
    // waits for the module AND its signed-in state, so it cannot leave without the CSRF header.
    const loaded = import('/static/js/account_chrome.js');
    accountHooks.ready = loaded.then((m) => m.ready).catch(() => { /* signed out: never hang */ });
    loaded.then((m) => m.init()).catch(() => { /* signed-out chrome */ });
  }
}
