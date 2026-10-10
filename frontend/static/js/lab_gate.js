/* Pai Sho Lab — the Lab index gate (`/lab`).
 *
 * Every mutating Lab API is protected server-side by MUSHIBOT_API_TOKEN
 * (backend/ui/server.py), so this gate exists only for clarity: a visitor
 * without the token sees a clear reason instead of a page full of tools that
 * will 401 the moment they're used. When the server has no token configured
 * at all (local dev), there is nothing to gate against, so the Lab stays open.
 */
import * as ui from '/static/js/ui.js';

async function render() {
  const gate = document.getElementById('lab-gate');
  const tools = document.getElementById('lab-tools');
  if (!gate || !tools) return;
  const { ok, tokenRequired } = await ui.labCheck();
  if (ok || !tokenRequired) {
    gate.hidden = true;
    tools.hidden = false;
  } else {
    gate.hidden = false;
    tools.hidden = true;
  }
}

function wireSubmit() {
  const input = document.getElementById('lab-token-input');
  const error = document.getElementById('lab-token-error');
  const submit = document.getElementById('lab-token-submit');
  if (!input || !error || !submit) return;
  const attempt = async () => {
    ui.setLabToken(input.value);
    const { ok } = await ui.labCheck();
    if (ok) {
      error.hidden = true;
      await render();
    } else {
      error.hidden = false;
    }
  };
  submit.addEventListener('click', attempt);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') attempt(); });
}

wireSubmit();
render();
