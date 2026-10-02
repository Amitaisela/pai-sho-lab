/* Pai Sho Lab — the Bots page (`/bots`).
 *
 * Six MushiBot levels, each a calibrated build of a different agent (see
 * Agents/registry.py's `house_bots()`). Level order is play-strength order —
 * level 6 is the strongest, level 1 the gentlest. The personalities and
 * ratings below are a small static map (same convention as home.js's
 * LEVEL_RATINGS), written in plain words rather than the registry's
 * architecture jargon ("alpha-beta", "NNUE", ...) — that detail belongs on
 * the Lab's agent pages, not here.
 */
import * as ui from '/static/js/ui.js';

const RATINGS = { 1: 225, 2: 300, 3: 311, 4: 385, 5: 397, 6: 476 };

const PERSONALITIES = {
  1: 'Plays by feel, picked up from thousands of practice games — quick, and a little unpredictable.',
  2: 'A fast, even-tempered judge of a position.',
  3: 'Shaped by generations of trial and error — comfortable taking a chance.',
  4: 'A careful planner who checks a few moves ahead before committing.',
  5: 'Weighs out many possible futures before choosing — patient and thorough.',
  6: 'Sticks to a steady set of rules of thumb. Simple, but no pushover.',
};

function meterHtml(level) {
  return Array.from({ length: 6 }, (_, i) => (
    `<span class="bot-meter__seg${i < level ? ' is-filled' : ''}"></span>`
  )).join('');
}

function cardHtml(level) {
  const rating = RATINGS[level] ?? '?';
  return `
    <article class="bot-card" data-level="${level}">
      <div class="bot-card__head">
        <span class="bot-card__level">${level}</span>
        <h3>MushiBot level ${level}</h3>
      </div>
      <p class="bot-card__personality muted">${PERSONALITIES[level] ?? ''}</p>
      <div class="bot-meter" aria-hidden="true">${meterHtml(level)}</div>
      <div class="bot-card__foot">
        <span class="bot-card__rating muted">&asymp; ${rating} rating</span>
        <button type="button" class="btn btn-primary" data-play-level="${level}">Play</button>
      </div>
    </article>
  `;
}

async function playLevel(level) {
  let data;
  try {
    data = await ui.api('/api/bot_game', {
      method: 'POST', body: { level, seat: 'random', nickname: ui.nickname() },
    });
  } catch (e) {
    return;
  }
  if (data.seat != null && data.seat_token != null) ui.setSeatToken(data.game_id, data.seat, data.seat_token);
  location.href = `/game/${data.game_id}`;
}

async function init() {
  const grid = document.getElementById('bots-grid');
  if (!grid) return;
  let levels = [1, 2, 3, 4, 5, 6];
  try {
    const bots = await ui.api('/api/house_bots');
    if (Array.isArray(bots) && bots.length) levels = bots.map((b) => b.level).sort((a, b) => a - b);
  } catch (e) {
    // Fall back to the static 1..6 ladder below.
  }
  grid.innerHTML = levels.map(cardHtml).join('');
  grid.querySelectorAll('[data-play-level]').forEach((btn) => {
    btn.addEventListener('click', () => playLevel(Number(btn.dataset.playLevel)));
  });
}

init();
