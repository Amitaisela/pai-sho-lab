// One WebSocket per tab. Reconnects with backoff and re-subscribes, so a dropped
// connection costs one fresh snapshot, not a stuck page.
//
// This script is also served by the legacy Flask play server, which has no /ws.
// If the socket has never opened after a few tries we stop retrying; pages see
// that through the onDown hooks (and MushiLive.isOpen()) and keep their old polling.
(function () {
  const MAX_FIRST_FAILURES = 3;
  let ws = null, retry = 0, firstFails = 0, timer = null, everOpened = false, gaveUp = false;
  const games = new Map();      // gid -> {seatToken, onState}
  let lobby = null;             // {onLive, onSeeks}
  const hooks = new Map();      // key (gid or 'lobby') -> {onUp, onDown}; re-subscribing replaces, never accumulates

  function url() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${location.host}/ws`;
  }

  function isOpen() { return !!ws && ws.readyState === 1; }

  function raw(obj) {
    if (!isOpen()) return false;
    ws.send(JSON.stringify(obj));
    return true;
  }

  function sendSub(channel, seatToken) {
    const msg = { type: 'subscribe', channel };
    if (seatToken) msg.seat_token = seatToken;
    raw(msg);
  }

  function fire(name) {
    for (const h of hooks.values()) {
      try { if (h[name]) h[name](); } catch (e) { /* a hook must not break the socket */ }
    }
  }

  function connect() {
    if (gaveUp || (ws && (ws.readyState === 0 || ws.readyState === 1))) return;
    let sock;
    try { sock = new WebSocket(url()); } catch (e) { gaveUp = true; fire('onDown'); return; }
    ws = sock;
    sock.onopen = () => {
      everOpened = true;
      retry = 0;
      fire('onUp');
      if (lobby) sendSub('lobby');
      for (const [gid, rec] of games) sendSub(gid, rec.seatToken);
    };
    sock.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type === 'state') {
        const rec = games.get(msg.gid);
        if (rec) rec.onState(msg);
      } else if (msg.type === 'live_update') {
        if (lobby && lobby.onLive) lobby.onLive(msg.rows);
      } else if (msg.type === 'seek_update') {
        if (lobby && lobby.onSeeks) lobby.onSeeks(msg.rows);
      } else if (msg.type === 'error') {
        window.dispatchEvent(new CustomEvent('mushi:error', { detail: msg }));
      }
    };
    sock.onclose = () => {
      if (ws !== sock) return;
      ws = null;
      fire('onDown');
      if (!everOpened && ++firstFails >= MAX_FIRST_FAILURES) { gaveUp = true; return; }
      const delay = Math.min(10000, 300 * 2 ** retry++);
      timer = setTimeout(connect, delay);
    };
  }

  window.MushiLive = {
    connect,
    isOpen,
    // Re-calling for the same gid (e.g. after taking a seat) replaces the token and
    // makes the server send a fresh snapshot.
    subscribeGame(gid, seatToken, onState, h) {
      games.set(gid, { seatToken, onState });
      if (h) hooks.set(gid, h);
      connect();
      sendSub(gid, seatToken);
    },
    subscribeLobby(handlers, h) {
      lobby = handlers;
      if (h) hooks.set('lobby', h);
      connect();
      sendSub('lobby');
    },
    // Not used by the pages yet (moves stay on HTTP POST); returns false if not connected.
    move(gid, ply, action, seatToken) {
      return raw({ type: 'move', gid, ply, action, seat_token: seatToken });
    },
    close() {
      gaveUp = true;
      if (timer) clearTimeout(timer);
      if (ws) ws.close();
    },
  };
})();
