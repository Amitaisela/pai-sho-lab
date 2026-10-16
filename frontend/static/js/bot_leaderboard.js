const body = document.querySelector('#bot-lb tbody');
const empty = document.getElementById('bot-lb-empty');

function cell(text, link) {
  const td = document.createElement('td');
  if (link) {
    const a = document.createElement('a');
    a.href = link;
    a.textContent = text;
    td.appendChild(a);
  } else {
    td.textContent = text;
  }
  return td;
}

const res = await fetch('/api/bots/leaderboard');
const { bots } = await res.json();
empty.hidden = bots.length > 0;
bots.forEach((b, i) => {
  const tr = document.createElement('tr');
  tr.appendChild(cell(String(i + 1)));
  tr.appendChild(cell(`${b.username} BOT`, `/@/${encodeURIComponent(b.username)}`));
  tr.appendChild(cell(b.owner || '', b.owner ? `/@/${encodeURIComponent(b.owner)}` : null));
  tr.appendChild(cell(`${b.rating}${b.provisional ? '?' : ''}`));
  tr.appendChild(cell(String(b.games)));
  body.appendChild(tr);
});
