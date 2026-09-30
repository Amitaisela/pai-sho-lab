# Developer reference: engine API, PSN, agents

## Contents
1. Importing the engine and choosing Python or Rust
2. PaiShoGame API
3. Save format v2 and PSN v2 notation
4. Writing an agent: the interface
5. The registry (`Agents/registry.py`)
6. Adding an agent, step by step
7. Training scripts: the contract
8. Running and benchmarking agents
9. Tests
10. Public mirror implications

## 1. Importing the engine and choosing Python or Rust
`pyproject.toml` puts `engine/` and `backend/` on the package path, so after `pip install -e .`:

```python
from PythonEngine.PaiShoGame import PaiShoGame, GATES, VALID_SPACES, FLOWER, CIRCLE, garden_of
from engine_select import game_class            # engine/engine_select.py
Game = game_class("python")                      # or "rust" (needs the maturin-built RustEngine wheel)
game = Game(accents=None, opening='random')
```

- `game_class("rust")` raises a clear `ImportError` explaining how to build the Rust engine if it isn't built.
- Both classes are duck-typed drop-ins with no shared base class, so agent code must not `isinstance`-check `PaiShoGame`.
- `engine_name_of(game)` tells you which engine a game instance came from.

## 2. PaiShoGame API

`PaiShoGame(accents=None, opening='random')`:
- `accents`: `None` (default one of each accent per player) or `{1: [4 names], 2: [4 names]}` — at most 2 of any one accent type. Validated by `normalize_accents`; raises `ValueError` on a bad choice.
- `opening`: `'random'` (default, draws a basic flower), a basic flower name, or `None` (Informal Start, no tile pre-placed).

| Member | Meaning |
|---|---|
| `board` | `{(r,c): {'flower': str, 'player': 1\|2, 'growing': bool}}` |
| `hands` | `{1: {tile_name: count}, 2: {...}}` |
| `current_player` | 1 or 2 (1 is the Guest, who moves first) |
| `bonus_turn` | `True` while the current player is taking a Harmony Bonus move |
| `winner` | `None` (game in progress), 1, 2, or 0 (tie) |
| `end_reason` | `None \| 'ring' \| 'last_basic_flower' \| 'no_moves' \| 'resign'` |
| `setup` | `{'accents': {1: [...], 2: [...]}, 'opening': name \| None}` — the options this game was built with |
| `message` | Human-readable status string, shown in the UI |
| `history` | List of `['plant', f, r, c]` / `['plant', 'Boat', r, c, dr, dc]` / `['arrange', fr, fc, tr, tc]` / `['skip_bonus']` |
| `history_players` | Parallel list of which player (1 or 2) took each `history` entry |
| `reset()` | Restore the initial position (same `accents`/`opening` unless passed again) |
| `clone()` | Deep copy. **Always clone before simulating**, because the game is mutable |
| `get_legal_actions()` | List of action tuples (cached). Can be `[]` |
| `step(action)` | Apply the action; returns `True` if the game is now over. Raises `ValueError` on an illegal action or a finished game |
| `plant(flower, r, c, displace_r=None, displace_c=None)` | Lower-level plant or accent placement. Only this call can choose a Boat displacement; `step()` always uses the automatic first-legal-neighbour |
| `arrange(fr, fc, tr, tc)` | Lower-level move |
| `skip_bonus()` | Decline a pending Harmony Bonus |
| `resign(player)` | Forfeit; sets `winner = 3 - player`, `end_reason = 'resign'`. A method, never an action |
| `valid_destinations(r, c)` | Legal landing squares for the tile at `(r,c)` |
| `find_harmonies(player)` | List of `(pos, pos)` harmony pairs for `player` |
| `find_clashes()` | List of clashing `(pos, pos)` pairs (normally empty) |
| `count_midline_harmonies(player)` | Used by the last-basic-flower / no-moves tiebreak; a harmony with a tile on a midline doesn't count |
| `check_harmony_ring(player)` | `bool` |
| `is_harmonious(f1, f2)` / `is_clash(f1, f2)` | Circle-of-Harmony lookups |
| `from_dict(d)` (classmethod) | Build from `{'board': {"r,c": tile}, 'hands': {'1':…,'2':…}, 'current_player', 'winner', 'end_reason', 'message', 'bonus_turn', 'setup', 'history', 'history_players'}` |
| `to_save_dict(p1, p2)` / `from_save_dict(d)` | JSON save format, **version 2** — see §3 |

**Gotcha:** `find_harmonies` and `get_legal_actions` are cached by the incremental Zobrist hash (`_zhash`), which only `plant`/`arrange`/`from_dict` maintain. If you mutate `game.board` directly on a game that has already computed harmonies, you'll get stale cached results. Build test positions with `PaiShoGame.from_dict({'board': {"r,c": tile, ...}, 'hands': ..., 'current_player': 1, ...})`, or edit only a fresh instance before its first query.

**Action tuples** — never unpack by a fixed index; use `Agents/actions.py`:
- `action_kind(action)` → `'plant'`, `'arrange'`, or `'skip_bonus'`.
- `plant_parts(action)` → `(tile, r, c, displace)`, where `displace` is `(dr, dc)` or `None`.
- `arrange_parts(action)` → `(fr, fc, tr, tc)`.

Shapes:
- `('plant', tile, r, c)`: a basic flower, special flower, Rock/Wheel/Knotweed, or a Boat placed on an accent.
- `('plant', 'Boat', r, c, dr, dc)`: a Boat placed on a flower at `(r,c)`, moving it to `(dr, dc)`.
- `('arrange', fr, fc, tr, tc)`.
- `('skip_bonus',)`.

Bonus moves use the same tuples; the only signal that a move is a bonus is `game.bonus_turn`.

## 3. Save format v2 and PSN v2 notation
**JSON saves** (`to_save_dict`), version 2 — **version 1 saves are rejected** with a `ValueError` (they predate the official rules):
```
{'version': 2, 'timestamp', 'p1', 'p2', 'history', 'history_players',
 'state': {'board': {"r,c": tile}, 'hands': {'1','2'}, 'current_player', 'winner',
           'end_reason', 'message', 'bonus_turn', 'setup': {'accents': {'1','2'}, 'opening'}}}
```
They restore from the state snapshot, not by replaying the moves.

**PSN v2** is Pai Sho Notation, implemented in `engine/PythonEngine/notation.py`:
- Header tags: `[Event]`, `[Date "YYYY-MM-DD HH:MM:SS"]`, `[Player1]` (the Guest), `[Player2]` (the Host), `[GuestAccents "Rock,Wheel,Knotweed,Boat"]`, `[HostAccents "..."]`, `[Opening "Rose"]` (or `"Informal"`), `[Result "1-0"|"0-1"|"1/2-1/2"|"*"]`, `[Termination "resign"]` (only present when `end_reason is not None`), `[Turns "N"]`.
- One token per turn, numbered in Guest/Host pairs: `main+bonus`, where `+bonus` is present only if a Harmony Bonus was played that turn. A bare token means any earned bonus was skipped, **except** a token ending in a lone trailing `+` with no bonus after it — that marks a bonus earned but left pending (neither played nor skipped) when the record ends, which can only happen on the game's last turn.
- Plant: `Flower@r,c` (e.g. `Rose@9,1`, `WhiteLotus@1,9`). Boat on a flower: `Boat@r,c>dr,dc`. Arrange: `fr,fc-tr,tc` (e.g. `9,1-10,2`).

Real output from `game_to_psn(game, "Alice", "Bob")`, 4 turns into a fresh game:
```
[Event "Pai Sho Lab Game"]
[Date "2026-09-29 07:21:10"]
[Player1 "Alice"]
[Player2 "Bob"]
[GuestAccents "Rock,Wheel,Knotweed,Boat"]
[HostAccents "Rock,Wheel,Knotweed,Boat"]
[Opening "Chrysanthemum"]
[Result "*"]
[Turns "4"]

1. Rose@9,1  Rose@9,17
2. 17,9-16,9  Rose@17,9
```
A `main+bonus` token (e.g. `16,9-12,9+Rock@5,5`) or a pending trailing `+` (e.g. `2,9-2,5+`) would appear the same way once a Harmony Bonus is played or left pending.

Functions:
- `action_to_psn(action)` / `psn_to_action(s)`. The latter returns a *list*, not a tuple.
- `game_to_psn(game, p1_name, p2_name, event=…)` replays `game.history` from `game.setup` to build the token stream; raises `ValueError` if the history doesn't replay from the setup or doesn't reach the current board (so a `from_dict`-built mid-game position without full history can't be exported).
- `parse_psn(text)` returns `(tags, actions)`, flattened (skipped/pending bonuses aren't listed).
- `parse_psn_turns(text)` returns `(tags, turns)`, one `[main]` / `[main, bonus]` / `[main, None]` (pending bonus) entry per turn.
- `psn_to_game(text, game_cls)` replays through `step()` (so a Boat's displacement is preserved, unlike v1), and calls `resign()` for the implied loser if `[Termination]` says `"resign"` and the replay didn't already finish.

## 4. Writing an agent: the interface
- A **class** agent needs `choose_action(self, game, verbose=False)`, with exactly that parameter order: `agent_loader.act()` passes `verbose` positionally, and `validate_registry()` enforces the order.
  - It returns one action from `game.get_legal_actions()`, or `None` if that list is empty.
  - Constructor kwargs come from the registry's `play_kwargs`, overridden by `play_params` values the UI sends.
  - If the entry has `needs_player: True`, `agent.player` is re-synced to `game.current_player` before every move.
- A **function** agent (for example `minimax` → `get_best_action(game, **function_kwargs)`) is a plain callable.
- An **inline** agent (only `random`) is handled inside `agent_loader`.
- Trainable agents conventionally expose `save_model(path=None)` / `load_model(path=None)` and load weights when `load=True`. With no file on disk they fall back silently to random initialization; a checkpoint whose shape no longer fits (e.g. after a feature-vector change) is reinitialised with a warning rather than loaded half-way.
- Weights go under `data/params/<PascalDir>/`. Use `Agents.utils.DATA_DIR` rather than relative paths.
- Search agents must `game.clone()` before calling `step()`. Use `Agents/actions.py`'s `action_kind()`/`plant_parts()`/`arrange_parts()` instead of unpacking actions by a fixed length — action tuples can be 4, 5, or 6 elements long. For evaluation ideas, reuse `Agents/utils.py`'s ring/harmony-distance helpers instead of re-deriving ring geometry.

## 5. The registry (`Agents/registry.py`)
The `AGENTS` list is the single source of truth: the Play and Simulate dropdowns, the Train forms, simulator dispatch and training-manager wiring are all generated from it. Current keys: `random`, `minimax`, `mcts`, `neat`, `basic_minimax`, `cnn_basic`, `nneu_basic`. Every key except `random` carries a `house_level` (1–6, unique, contiguous); `house_bots()` returns them sorted with a `"MushiBot level N"` label.

| Field | Meaning |
|---|---|
| `key` | Unique snake_case id, used in URLs, CLI specs and Elo |
| `display_name`, `description`, `architecture` | UI text: the description is one line; the architecture is a paragraph for the model card |
| `kind` | `"inline"` \| `"function"` \| `"class"` |
| `module` + `class_name` / `function_name` | Import target |
| `function_kwargs` | Default kwargs (function kind) |
| `play_kwargs` | Constructor kwargs (class kind), e.g. `{"player": 1, "load": True}` |
| `needs_player` | Re-sync `agent.player` to the side to move |
| `play_params` | UI knobs, built with `num_param`/`text_param`/`checkbox_param`/`select_param` |
| `model_path` | Checkpoint path; the UI shows "trained ✓" if it exists. `None` for weightless agents |
| `training_script` | Path to the training script, or `None` |
| `training_params` | Train-page form fields. **Each must carry a `cli_flag`**. Add `engine_param()` to get the Python/Rust dropdown (`--engine`) |
| `total_episodes_key` | The form field that drives the progress-bar total |
| `log_parser` | Legacy regex fallback for non-EVENT stdout: `neat`, `basic_minimax`, or `None`. Structured `EVENT:` lines always take priority. **New agents should use `None` and emit EVENTs** via `Agents/logging_utils.py`'s `log_event(...)` — the regex parsers are a legacy fallback only |
| `config_file` | Optional free-form config file that can be edited on the Train page |

Helpers: `get_agent(key)`, `all_agents()`, `trainable_agents()`, and `validate_registry()` (run it with `python -m Agents.registry`). `opponent`/`p2` training params have their tooltips auto-filled with every other key.

## 6. Adding an agent, step by step
1. **Scaffold** the agent:
   ```bash
   python scripts/new_agent.py my_agent --template minimax   # weightless → Agents/classical/my_agent.py
   python scripts/new_agent.py my_agent --template cnn       # trainable  → Agents/rl/my_agent.py + Agents/training/my_agent_training.py
   ```
   The scaffolder appends a registry entry with TODOs. Both templates' docstrings mention `('skip_bonus',)`, the 6-element Boat action, and `Agents/actions.py`.
2. **Implement** `choose_action`. For a trainable agent, also implement the model, `save_model`/`load_model`, and the training loop.
3. **Fill in** `description` and `architecture` in the registry entry. For a trainable agent, add `engine_param()` to `training_params` and accept `--engine` in the script, then build games with `engine_select.game_class(args.engine)`. The scaffolded script doesn't do this yet.
4. **Validate**: `python -m Agents.registry`.
5. **Smoke test**: `python backend/simulator.py --mode local --p1 my_agent --p2 random --n 1`.
6. **Train** from the Train page, or run the script directly.
7. **Benchmark** on the Simulate page with "Count toward ELO" ticked (the simulator CLI itself doesn't record Elo), then check the Leaderboard. Elo ratings reset whenever the weights change.
8. **Public mirror** (optional): to ship the agent in pai-sho-lab, add its key to `KEEP_AGENTS` and its files to `KEEP_FILES` / `KEEP_DATA_FILES` in `scripts/distill.py`.

Reference templates to copy: `Agents/classical/basic_minimax.py` (weightless) and `Agents/rl/cnn_basic.py` + `Agents/training/cnn_basic_training.py` (trainable).

## 7. Training scripts: the contract
- Use `argparse` flags that match each `training_params[*].cli_flag`, and always support `--resume` and `--engine {python,rust}`.
- Emit progress with `from Agents.logging_utils import get_logger, log_event`, then `log_event(log, "episode", episode=i, total=N, outcome=..., steps=...)`. This prints `EVENT:{json}` to stdout, and `backend/ui/training_manager.py` parses those lines for the progress bar. Recognized events:
  - `episode` (uses `episode`/`total`);
  - `epoch` (training phase);
  - `gen_game` (generation phase);
  - `eval` (win rates).
- Save to the registry's `model_path` periodically and at the end.
- For opponents, use `Agents/training/opponent_utils.load_opponent(name)`. `'self'` returns `None`, meaning self-play; `'random'` and any registry key return a `choose_action(game)` callable.
- Enforce your own max-steps cap; the engine has no move limit of its own beyond the no-stranding/no-legal-move endings.
- Ties (`winner == 0`) must be scored as a neutral outcome, not a loss — check with `is None`/`is not None`/`in (1, 2)` rather than truthiness.

## 8. Running and benchmarking agents
```bash
python backend/ui/server.py                                   # web UI at http://localhost:5000
python backend/simulator.py --mode local --p1 minimax:time_budget=5 --p2 mcts:time_budget=3,c=0.3 --n 10 --engine rust
```
- Agent specs have the form `key` or `key:k=v,k2=v2`. Values are coerced to int, float or bool, and the keys override `play_params`.
- `--mode` defaults to **`flask`**, which drives a running server over REST. Use `--mode local` for in-process games.
- Other flags: `--save_results`, `--save_games`, `--save_psn`, `--max_steps` (default 1000), `--v`.

**Local arena — measuring a bot before it goes online:**
```bash
python scripts/eval_agents.py my_agent --vs basic_minimax --games 20 --engine python
python scripts/round_robin.py random basic_minimax cnn_basic my_agent --games 10 --engine python
```
- `eval_agents.py` plays one agent against a fixed opponent with alternating seats and reports win/loss/tie/timeout counts; `--vs` defaults to `random`.
- `round_robin.py` plays every unordered pair of the given specs `--games` times, fits Bradley-Terry Elo ratings, and reports each with a 95% bootstrap confidence interval and a pair-matrix.
- Both accept `--engine {python,rust}`, `--allow-untrained` (skip the check that an agent's weights actually loaded), `--seed`, and `--json` to dump results.

## 9. Tests
```bash
python tests/test.py                 # engine unit tests; writes tests/test_report.txt          [private repo only]
python tests/test_rules_contract.py  # JSON rule fixtures (tests/fixtures/rules/)                [survives distillation]
python tests/parity_fuzz.py          # Python vs Rust lockstep fuzzer                            [survives distillation]
python tests/test_notation.py        # PSN v2 round-trips                                        [survives distillation]
python tests/test_integration.py     # end-to-end                                                [private repo only]
python tests/basic_tests.py          # the minimal suite that ships to the public mirror         [survives distillation]
python -m Agents.registry            # registry validation
cd engine/RustEngine && cargo test --workspace
```
These use a custom pass/fail runner, not pytest. When you test captures, assert that the captured player's hand count is **unchanged** (captures don't return to a reserve).

`tests/test.py` and `tests/test_integration.py` are **private-repo only**: `scripts/distill.py` deletes any test file not listed in its `KEEP_TESTS` set, and those two aren't in it. `tests/basic_tests.py`, `tests/test_rules_contract.py`, `tests/parity_fuzz.py`, and `tests/test_notation.py` are exactly `KEEP_TESTS` and survive distillation into `pai-sho-lab`.

## 10. Public mirror implications
- `scripts/distill.py` publishes a pruned copy to `github.com/Amitaisela/pai-sho-lab`, containing:
  - the agents `random`, `basic_minimax` and `cnn_basic`;
  - `Agents/actions.py` and the other files in `KEEP_FILES`;
  - the tests in `KEEP_TESTS`: `tests/basic_tests.py`, `tests/test_rules_contract.py`, `tests/parity_fuzz.py`, `tests/test_notation.py` (see §9);
  - `README.public.md`, renamed to `README.md`.
- This skill ships to the mirror too. Keep it free of anything that only makes sense in the private repo, or make those parts clearly optional.
