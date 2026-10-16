import * as ui from '/static/js/ui.js';

const list = document.getElementById('bot-list');
const msg = document.getElementById('bot-msg');

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

function button(label, onClick, cls = 'btn btn-quiet') {
  const b = el('button', label, cls);
  b.type = 'button';
  b.addEventListener('click', onClick);
  return b;
}

async function mint(botId) {
  try {
    const t = await ui.api(`/api/account/bots/${botId}/tokens`, { method: 'POST', body: { name: 'token' } });
    msg.textContent = '';
    const box = el('p', null, 'account-msg');
    box.appendChild(el('strong', 'Copy this token now, it will not be shown again: '));
    box.appendChild(el('code', t.token));
    list.prepend(box);
    await load(false);
  } catch (err) {
    msg.textContent = err.error || 'Could not create a token.';
  }
}

async function revoke(tokenId) {
  try {
    await ui.api(`/api/account/tokens/${tokenId}`, { method: 'DELETE' });
    await load(false);
  } catch (err) {
    msg.textContent = err.error || 'Could not revoke that token.';
  }
}

async function load(clear = true) {
  const data = await ui.api('/api/account/bots');
  const keep = clear ? [] : [...list.querySelectorAll('p.account-msg')];
  list.replaceChildren(...keep);
  data.bots.forEach((b) => {
    const card = el('div', null, 'bots-card');
    card.appendChild(el('h2', `${b.username} BOT${b.banned ? ' (banned)' : ''}`));
    b.tokens.forEach((t) => {
      const row = el('p');
      row.appendChild(el('span', `${t.name} — ${t.prefix}… `));
      row.appendChild(button('Revoke', () => revoke(t.id)));
      card.appendChild(row);
    });
    card.appendChild(button('New token', () => mint(b.id), 'btn btn-secondary'));
    list.appendChild(card);
  });
  if (!data.bots.length) list.appendChild(el('p', `You have no bots yet (limit ${data.limit}).`, 'muted'));
}

document.getElementById('bot-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  msg.textContent = '';
  try {
    await ui.api('/api/account/bots', { method: 'POST', body: { username: document.getElementById('bot-name').value.trim() } });
    document.getElementById('bot-name').value = '';
    await load();
  } catch (err) {
    msg.textContent = err.error || 'Could not create that bot.';
  }
});

load().catch(() => { msg.textContent = 'Could not load your bots.'; });
