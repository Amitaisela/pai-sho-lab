# Pai Sho Lab

[![Tests](https://github.com/Amitaisela/pai-sho-lab/actions/workflows/test.yml/badge.svg)](https://github.com/Amitaisela/pai-sho-lab/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A workbench for building, training, and playing AI agents for **Skud Pai Sho**, a two-player strategy board game on a circular board. You can play the game in your browser, pit agents against each other, and add your own agent by writing one Python file and one registry entry.

> The game is [Skud Pai Sho](https://skudpaisho.com/), created by @SkudPaiSho and The Garden Gate community. This project is an independent AI lab built around it.

**Jump to:** [Quickstart](#quickstart) · [Play](#i-want-to-play) · [Benchmark agents](#i-want-to-benchmark-agents) · [Add an agent](#i-want-to-add-an-agent) · [Testing](#testing)

---

## Quickstart

You need Python 3.11+.

```bash
git clone https://github.com/Amitaisela/pai-sho-lab.git
cd pai-sho-lab
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e . && pip install -r requirements.txt
python backend/ui/server.py       # then open http://localhost:5000
```

Everything else happens in the browser.

## The game in 60 seconds

- **Goal:** make a **Harmony Ring**, a closed loop of your flowers in harmony that surrounds the center of the board. Both players completing one on the same move is a tie.
- **Setup:** each player has 3 of each basic flower, 4 chosen accents (up to 2 of any one type — the default is one of each), 1 White Lotus and 1 Orchid. The Guest (Player 1) moves first.
- **On your turn** you either **Plant** a flower into one of the 4 **Gates** on the edge of the board, or **Arrange**: move one of your flowers up to its number of steps. Moves go up, down, left or right, can turn, can't jump, and can pass through an open gate but never stop on one.
- **Flowers** are red (Rose 3, Chrysanthemum 4, Rhododendron 5) or white (Jasmine 3, Lily 4, Jade 5). The number is how far the flower moves. Red flowers can't stop in the white garden, and white flowers can't stop in the red garden.
- **Harmony:** two of your blooming flowers that are neighbours on the circle R3→R4→R5→W3→W4→W5→R3, lined up in a row or column with nothing (no tile, no gate) between them.
- **Clash:** opposites (R3/W3, R4/W4, R5/W5). A move may never line them up, even if uncovering one. You capture an enemy basic flower that clashes with yours by landing on it; captured tiles leave the game for good.
- **Bonus:** an Arrange that creates a new harmony earns one optional bonus — place an accent tile (Rock, Wheel, Knotweed, Boat), plant a special flower (White Lotus, Orchid), or (if none of yours are growing) plant a basic flower — or skip it.

Full rules are on the **Rules** page in the app. Developers who use Claude Code also get a rules and engine reference skill at [.claude/skills/skud-pai-sho/](.claude/skills/skud-pai-sho/SKILL.md). The engine now matches the official rulebook closely; the skill lists the few remaining intentional differences.

## I want to play

Open http://localhost:5000. The **Play** page is the home page:

1. For each side, choose **Human** or an agent. You can play vs. an AI, human vs. human, or watch AI vs. AI.
2. Press **Start Game**, then click a tile and one of its highlighted destinations to move.
3. Turn **Rated** on to have the result count toward the Elo leaderboard.

**Engine: Python/Rust** picks which rules implementation runs the game. Leave it on Python unless you've built the Rust engine (see [Engines](#engines)).

## I want to benchmark agents

- **Simulate page:** pick two agents and a number of games, then press Run. Tick **Count toward ELO** to update the **Leaderboard**. An agent's rating resets whenever its trained weights change, so the leaderboard always reflects the current model.
- **Command line** (no browser needed):

  ```bash
  python backend/simulator.py --mode local --p1 basic_minimax:time_budget=2 --p2 random --n 10
  ```

  An agent is given as `name` or `name:key=value,key=value`, where the keys are that agent's play settings. `--mode local` runs games in-process; the default `--mode flask` drives a running web UI instead.

### Agents that ship here

| Agent | Kind | What it does |
|---|---|---|
| `random` | baseline | Picks a random legal move. It's the lower bound for everything else. |
| `basic_minimax` | weightless (no training) | Depth-limited alpha-beta search with a fixed hand-written evaluation. It's the **reference template for an agent that needs no training**. |
| `cnn_basic` | trainable | A two-layer CNN value network over an 8-channel 19×19 board encoding. It plays one-ply greedy: it tries every legal move and keeps the best-scoring result. Ships with trained weights at [data/params/CNNBasic/cnn_basic.pt](data/params/CNNBasic/cnn_basic.pt). It's the **reference template for a trainable agent**. |

"Human" in the Play page's dropdown means you; it isn't an agent.

## Local arena

Before you trust an agent, measure it: play it against a fixed opponent, or run an all-pairs round robin that fits Elo ratings.

```bash
python scripts/eval_agents.py my_agent --vs basic_minimax --games 20 --engine python
python scripts/round_robin.py random basic_minimax cnn_basic my_agent --games 10 --engine python
```

Use `--engine rust` instead once you've built the Rust engine; it's about 10x faster.

## I want to add an agent

An agent is a Python class with one method, `choose_action(game, verbose=False)`, which returns a move, plus **one entry** in [Agents/registry.py](Agents/registry.py). The registry is the single source of truth: the Play and Simulate dropdowns, the Train page's form, and the simulator are all generated from it, so **you never touch UI code**.

### Step 1: scaffold

```bash
python scripts/new_agent.py my_agent --template minimax   # no training → Agents/classical/my_agent.py
python scripts/new_agent.py my_agent --template cnn       # trainable   → Agents/rl/my_agent.py + Agents/training/my_agent_training.py
```

This writes the files and appends a registry entry with TODOs. The in-app **Guide** page walks through the same steps.

### Step 2: write `choose_action`

```python
class MyAgentAgent:
    def __init__(self, player=1, time_budget=1.0):
        self.player = player              # re-synced to the side to move before every call
        self.time_budget = time_budget    # any play-time knob you expose in the registry

    def choose_action(self, game, verbose=False):   # keep this exact signature
        legal = game.get_legal_actions()
        if not legal:
            return None
        best, best_score = None, float("-inf")
        for action in legal:
            g = game.clone()              # the game object is mutable, so always clone before trying a move
            g.step(action)
            score = len(g.find_harmonies(self.player))
            if score > best_score:
                best, best_score = action, score
        return best
```

The things you'll use:
- **Moves** are `('plant', flower, row, col)`, `('plant', 'Boat', row, col, dr, dc)` for a Boat displacing a flower, `('arrange', from_row, from_col, to_row, to_col)`, or `('skip_bonus',)` to decline a Harmony Bonus. Use `Agents/actions.py`'s `action_kind()` / `plant_parts()` / `arrange_parts()` instead of unpacking actions by a fixed length.
- **The board** is `game.board`, a dict `{(row, col): {'flower', 'player', 'growing'}}`.
- **State and results:** `game.step(action)` (raises `ValueError` on an illegal action), `game.winner` (`None`/1/2/0, 0 = tie), `game.end_reason`, `game.find_harmonies(player)`, `game.bonus_turn`.

### Step 3: trainable agents only

- The agent class gets a `load=True` constructor argument plus `save_model()` / `load_model()`. `load_model()` should fall back silently to random initialization when there's no checkpoint yet.
- The training script must:
  - accept each hyperparameter as an `argparse` flag that matches its registry `cli_flag`, plus `--resume` and `--engine {python,rust}`;
  - save to the registry's `model_path`;
  - report progress with `log_event(log, "episode", episode=i, total=N, ...)` from [Agents/logging_utils.py](Agents/logging_utils.py). That prints `EVENT:{json}` lines, which drive the Train page's progress bar.

  Copy [Agents/training/cnn_basic_training.py](Agents/training/cnn_basic_training.py) for a full working example.

### Step 4: fill in the registry entry

The scaffolder writes this for you. Here is what the fields mean:

```python
{
    "key": "my_agent",                       # id used in URLs, CLI specs, Elo
    "display_name": "My Agent",
    "description": "One line shown in the UI",
    "architecture": "A paragraph shown on the model card",
    "kind": "class",                         # "class" | "function" | "inline"
    "module": "Agents.rl.my_agent",
    "class_name": "MyAgentAgent",
    "play_kwargs": {"player": 1, "load": True},
    "needs_player": True,
    "play_params": [                         # play-time knobs → form fields on Play/Simulate
        num_param("epsilon", "Exploration", 0.0, min=0.0, max=1.0, step=0.01),
    ],
    "model_path": "data/params/MyAgent/my_agent.pt",   # None if weightless; UI shows "trained ✓" when it exists
    "training_script": "Agents/training/my_agent_training.py",   # None if weightless
    "training_params": [                     # Train-page form; every field needs its CLI flag
        num_param("episodes", "Episodes", 1000, min=1, step=100, cli_flag="--n"),
        num_param("lr", "Learning Rate", 1e-3, min=1e-5, max=1, step=1e-4, cli_flag="--lr"),
        checkbox_param("resume", "Resume from checkpoint", False, cli_flag="--resume"),
        engine_param(),                      # Python/Rust dropdown → --engine
    ],
    "total_episodes_key": "episodes",        # which field is the progress-bar total
    "log_parser": None,                      # None = use EVENT lines (recommended)
    "config_file": None,
}
```

### Step 5: validate, then play it

```bash
python -m Agents.registry                                                  # checks imports, signature, CLI flags, paths
python backend/simulator.py --mode local --p1 my_agent --p2 random --n 1   # smoke test
```

Then train it on the **Train** page, benchmark it on **Simulate**, and play it on **Play**.

## Engines

Two interchangeable rules engines ship side by side: the default pure-Python one ([engine/PythonEngine/PaiShoGame.py](engine/PythonEngine/PaiShoGame.py)) and a faster Rust port under [engine/RustEngine/](engine/RustEngine/). Every page, the simulator (`--engine`), and training scripts can use either. The Rust engine is optional and needs a [Rust toolchain](https://rustup.rs/):

```bash
cd engine/RustEngine/crates/pybind
maturin build --release
pip install ../../target/wheels/*.whl
```

If you pick Rust without building it, you get an error that tells you how.

## Running with Docker

```bash
cp .env.example .env   # set TS_AUTHKEY (a Tailscale auth key: https://login.tailscale.com/admin/settings/keys)
docker compose up                                                   # GPU (NVIDIA Container Toolkit required)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up   # CPU only
```

This runs the app plus a Tailscale sidecar. The app is reachable at `http://localhost:5000` and at `http://mushibot:5000` from your tailnet; nothing is exposed publicly. `./data` is mounted, so weights and results persist. The image builds the Rust engine for you. To stop anyone else on your tailnet from starting training or simulation runs, set `MUSHIBOT_API_TOKEN` in `.env`, then visit `/train?api_token=<value>` once per browser.

## Testing

```bash
python tests/basic_tests.py   # test suite (what CI runs)
python -m Agents.registry     # registry validation
cd engine/RustEngine && cargo test --workspace   # Rust engine, if you have a toolchain
```

## Project layout

```
engine/PythonEngine/     Rules engine (PaiShoGame.py) + PSN notation (notation.py)
engine/RustEngine/       Optional Rust port, exposed to Python via PyO3/maturin
engine/engine_select.py  "python" | "rust" → the matching engine class
Agents/registry.py       Every agent's UI/training/CLI metadata; the single source of truth
Agents/classical/        basic_minimax (weightless template)
Agents/rl/               cnn_basic (trainable template)
Agents/training/         Training scripts + shared opponent loader
backend/ui/server.py     Flask app: Play, Simulate, Train, Leaderboard, Rules, Guide pages
backend/simulator.py     Headless game runner
scripts/new_agent.py     Agent scaffolder
data/params/             Saved weights
```

For the design behind the registry and the agent families, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Where this repo comes from

Pai Sho Lab is an **auto-published subset** of a private research codebase (MushiBot) with a much larger roster: full alpha-beta search, MCTS with RAVE, an NNUE-style quantised net, and NEAT. On every push there, CI runs a distillation script that:
- keeps only `random`, `basic_minimax`, and `cnn_basic`;
- strips private code, weights, and the deploy machinery;
- swaps in this README and a test-only CI;
- commits the result here with a `Synced from Amitaisela/MushiBot@<sha>` trailer.

What you're looking at is a small, working subset that's easy to read end to end and extend.

## Author

[Amitaisela](https://github.com/Amitaisela)
