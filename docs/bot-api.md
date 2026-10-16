# Bot API

This is the protocol the online connector (`paisho_online`, `scripts/play_online.py`) speaks. You can use it from any language. It is plain HTTPS plus streams of newline-delimited JSON (NDJSON), in the style of the Lichess bot API.

Below, `<site-url>` stands for the site's address. The address can change, so take it from the site itself, and create your bot and mint its token on the site's **My bots** page (`/account/bots`). Bot accounts are approved by the site owner: request approval on that page first, and once you are approved you can create bots and tokens there.

## Authentication

Every endpoint except `GET /api/bot/meta` and `GET /api/bots/leaderboard` needs a bearer token:

```
Authorization: Bearer mbt_...
```

The token is shown once when you create it. Keep it in an environment variable, never in the repository, and never log it. A wrong or revoked token gets `401`, and so does every token of an owner whose approval was withdrawn. Requests are rate limited per token (about 10 per second with a burst of 30). Over the limit you get `429` with code `rate_limited` and a `Retry-After: 1` header; wait and retry.

Errors are always JSON: `{"error": "<message>", "code": "<code>"}`, sometimes with extra fields.

## Handshake: `GET /api/bot/meta`

No token needed.

```json
{"api_version": 1, "rules_version": 2,
 "time_controls": ["1+1", "3+2", "5+3"],
 "limits": {"max_concurrent_games": 1, "max_games_per_day": 100, "max_illegal_moves": 3}}
```

`rules_version` is the version of the game rules the site runs. Your engine has its own (`Agents/elo.py`, `RULES_VERSION`). If the two differ, your bot would compute different legal moves than the site, so do not play: tell the developer to update their clone. The event stream's `hello` repeats `rules_version`, and each `gameFull` does too. `python scripts/play_online.py --check` does this comparison for you.

`GET /api/bot/me` (with a token) returns `{"id": ..., "username": "..."}`, which is a cheap way to test a token.

## Event stream: `GET /api/stream/event`

One long-lived response per bot, one JSON object per line. A blank line is a keepalive, sent about every 6 seconds when nothing else happens; ignore it. If you open a second event stream, the older one is closed. When the stream drops, reconnect with a short backoff: on every connect the site replays your running games (`gameStart`) and pending challenges.

Events:

- `hello`: `{"type": "hello", "api_version": 1, "rules_version": 2, "username": "my_bot"}`. Always the first line.
- `gameStart`: `{"type": "gameStart", "game": {"id", "seat", "opponent": {"name", "bot", "house"}, "time_control", "rated"}}`. A game you are in has begun (or is running). `seat` is 1 or 2. Open the game stream for it.
- `challenge`: another bot challenged you: `{"type": "challenge", "challenge": {"id", "challenger": {"id", "name"}, "destUser": {"id", "name"}, "time_control", "rated", "created"}}`. Answer with accept or decline. Challenges expire after about 2 minutes.
- `challengeDeclined` and `challengeCanceled`: a challenge you sent was declined (with a `reason`), or one sent to you was withdrawn.
- `gameFinish`: `{"type": "gameFinish", "game": {"id", "winner", "end_reason", "rated", "rating"}}`.

In events and in `gameFull`, `time_control` is an object (`{"base_s": 180, "inc_s": 2}`). In requests it is the name string (`"3+2"`).

## Game stream: `GET /api/bot/game/stream/<id>`

Opens the stream for a game you sit in (any other game id gives `404 not_found`). You may have a few open at once (up to 3; more gives `429 too_many_streams`). It ends when the game ends.

The first line is `gameFull`:

```json
{"type": "gameFull", "id": "...", "seat": 1, "rules_version": 2, "engine": "...", "initial": "...",
 "players": {"1": {"name": "...", "bot": true, "house": false}, "2": {"name": "...", "bot": true, "house": true}},
 "time_control": {"base_s": 180, "inc_s": 2}, "rated": false, "state": {"type": "gameState", "...": "..."}}
```

Every later line is a `gameState`, sent after every change:

```json
{"type": "gameState", "ply": 0, "actions": [], "current_player": 1, "bonus_turn": false,
 "winner": null, "end_reason": null, "clock": {},
 "legal_actions": [["plant", "Rose", 9, 1], ["plant", "Rose", 9, 17]]}
```

- `actions` is the whole move list so far, on every message, so you can rebuild the position from any message by replaying it with the engine.
- `legal_actions` is present only when it is your move and the game is not over. It is the full list of legal actions. Pick one and send it back unchanged. (The example above is shortened.)
- `bonus_turn` is true when the player to move has a second move in a row.
- `winner` is 1 or 2 when the game is over, otherwise null; `end_reason` says why.
- `ply` is the number of actions played so far. It is also the ply number to send with your next move.

An action is a short JSON list such as `["plant", "Rose", 9, 1]`: words and small integers, as the engine produces them.

## Making a move: `POST /api/bot/game/<id>/move`

```json
{"action": ["plant", "Rose", 9, 1], "ply": 0}
```

`ply` is the `ply` of the `gameState` you are answering. It makes retries safe: if the network drops and you send the same move again, the site sees the ply is already played and answers `{"ok": true, "duplicate": true}` instead of playing twice. Leave `ply` out only if you do not retry.

Success: `{"ok": true, "ply": 1}`. Errors:

| Status | `code` | Meaning |
|---|---|---|
| 400 | `bad_request` | Malformed body or action shape. No strike. |
| 400 | `illegal` | Not a legal action. Extra fields `strikes` and `forfeited`. After 3 illegal moves in a game you forfeit it. Only send a member of `legal_actions`. |
| 403 | `forbidden` | It is not your turn, or you may not move here. |
| 404 | `not_found` | No such game, or you are not in it. |
| 409 | `game_over` | The game is finished (including a loss on time). |
| 409 | `ply_gap` | You sent a ply ahead of the game; the reply carries the real `ply`. Wait for the next `gameState`. |
| 429 | `rate_limited` | Too many requests; honour `Retry-After`. |

`POST /api/bot/game/<id>/resign` resigns (`{"ok": true}`, or `409 game_over`).

## Starting games

All of these take a JSON body, and `time_control` is the name string from `meta.time_controls` (for example `"3+2"`), not an object.

- `POST /api/challenge/<bot-name>` with `{"time_control": "3+2", "rated": false}` challenges another bot. Returns `201 {"id": "<challenge id>"}`. The other bot sees a `challenge` event; if it accepts, both of you get `gameStart`. Rated challenges between two bots of the same owner are refused (`400 same_owner`). You can have at most 5 challenges pending (`429 pending_challenges`).
- `DELETE /api/challenge/<id>` cancels a challenge you sent.
- `POST /api/challenge/<id>/accept` accepts a challenge sent to you. Returns `{"game_id": "..."}`.
- `POST /api/challenge/<id>/decline` declines it. Optional body `{"reason": "..."}`.
- `POST /api/bot/seek` with `{"time_control": "3+2", "rated": true}` joins the bot pool; the site pairs you with another seeking bot, and both get `gameStart`. A seek lasts about 90 seconds (`{"ok": true, "expires_in": 90}`), so repeat it while you want a game. `DELETE /api/bot/seek` withdraws it.
- `POST /api/bot/vs-house` with `{"level": 1, "time_control": "3+2", "color": "random"}` starts an unrated game against a house bot. `level` is 1 to 6; `color` is `1`, `2` or `"random"` (default). Returns `201 {"game_id": "...", "seat": 1}`.

`GET /api/bots/leaderboard` (no token) lists bot ratings.

## Free-tier limits

- 1 game at a time per bot, and 100 games a day. Starting more gives `429` with a `code` saying which limit. Limit errors carry no `Retry-After`; do not retry in a loop.
- Time controls: 1+1, 3+2, 5+3 (the list in `meta`).
- Rated games are bot against bot only, and not between two bots of the same owner. Games against house bots are unrated.
- Up to 3 open game streams, 5 pending challenges, and 3 illegal moves per game.

The values come from `GET /api/bot/meta`; treat that as the source of truth.

## Example with curl

Watch your events:

```bash
curl -N -H "Authorization: Bearer $PAISHO_TOKEN" $PAISHO_URL/api/stream/event
```

Start a practice game against house level 1:

```bash
curl -H "Authorization: Bearer $PAISHO_TOKEN" -H "Content-Type: application/json" \
     -d '{"level":1,"time_control":"3+2"}' $PAISHO_URL/api/bot/vs-house
```

Follow the game it returns, then play a move taken from `legal_actions`. The opening move is a planting, for example:

```bash
curl -N -H "Authorization: Bearer $PAISHO_TOKEN" $PAISHO_URL/api/bot/game/stream/<id>
curl -H "Authorization: Bearer $PAISHO_TOKEN" -H "Content-Type: application/json" \
     -d '{"action":["plant","Rose",9,1],"ply":0}' $PAISHO_URL/api/bot/game/<id>/move
```

Only play an action that appears in the `legal_actions` you were just sent. If you are seat 2, wait for your turn.
