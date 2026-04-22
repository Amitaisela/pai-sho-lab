# Pai Sho Lab

[![Tests](https://github.com/Amitaisela/pai-sho-lab/actions/workflows/test.yml/badge.svg)](https://github.com/Amitaisela/pai-sho-lab/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A minimal AI framework for building and training agents that play **Skud Pai Sho** — a two-player strategy board game on a circular 19×19 grid. Write an agent, wire it into the registry, train it from the browser, then pit it against other agents in simulation.

> Based on [Skud Pai Sho](https://skudpaisho.com/).

You start with three agents:

- `random` — picks a random legal move (baseline).
- `basic_minimax` — depth-limited alpha-beta minimax. Serves as the reference template for adding untrainable agents.
- `cnn_basic` — a trainable CNN. Serves as the reference template for adding trainable agents.

---

## What is this?

Pai Sho Lab is a local workbench for developing **Skud Pai Sho** AI agents. You write two Python files — an agent class and a training script — then add one entry to the registry describing their hyperparameters. From there the web UI generates a training form from your registry entry, spawns your training script as a subprocess, streams its progress, and makes the trained agent immediately available for simulation and play — no UI code needs to change.

Skud Pai Sho is a two-player game (a bit like chess or Go) played on a round board. Players take turns placing and moving flowers, and the first to surround the center of the board with a closed ring of matching flowers wins.

When you start Pai Sho Lab and open it in your browser, you get:

- **Train** — the main workflow. Pick an agent, fill in hyperparameters, kick off training, and watch live progress and metrics stream in.
- **Simulate** — run a tournament between two agents to measure how a training run actually moved the needle.
- **Leaderboard** — Elo ratings accumulated from simulation results, so improvement is legible over time.
- **Play** — a clickable board for sanity-checking an agent against a human, or spectating agent-vs-agent games.

Out of the box you can train `cnn_basic` and tune `basic_minimax`'s evaluation weights. To build up a roster of smarter agents, see [Adding a new model you can train](#adding-a-new-model-you-can-train) below.

---

## Setup

**Requirements:** Python 3.11+

```bash
git clone <url>
cd pai-sho-lab
python -m venv venv
.\venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

## Running

```bash
python ui/server.py
```

Open http://localhost:5000. Everything — training, simulation, leaderboard, play — is driven from the web UI.

First-time use: open the **Train** page, pick an agent, set hyperparameters, and start a run. When it finishes, head to **Simulate** to benchmark the trained agent against a baseline and watch the Elo update on the **Leaderboard**.

## Agents that ship with this template

| Agent | Description |
|-------|-------------|
| `random` | Picks a random legal move |
| `human` | You play via the browser |
| `basic_minimax` | Alpha-beta minimax with a tunable evaluation vector (reference template) |
| `cnn_basic` | Pretrained CNN value network with one-ply greedy selection |

### About `cnn_basic`

`cnn_basic` is a two-layer convolutional value network over an 8-channel 19×19 board encoding (piece planes, player-to-move, gate/zone masks). At play time the agent enumerates every legal action, applies it to a cloned game, scores the resulting state with the net, and picks the highest-scoring move — a one-ply greedy search, no lookahead.

The network ships with trained weights at [ai/params/CNNBasic/cnn_basic.pt](ai/params/CNNBasic/cnn_basic.pt), so it plays competently out of the box. If you want to improve on it, re-train via the Train page or directly:

```bash
python ai/training/cnn_basic_training.py --n 200 --lr 1e-3 --eps 0.5 --opponent self
```

It's a good template for anyone wiring up a small neural-net agent: the training loop, checkpoint format, and registry entry are all minimal and can be copied as-is.

---

## Developers

### Project main files

```
game/PaiShoGame.py       — Core rules engine (state, legal moves, harmony/clash/ring detection, clone())

ai/registry.py           — Single source of truth for every agent (UI config, training config, CLI mapping)
ai/training/             — One training script per trainable agent
ai/params/               — Saved weights / checkpoints (.pkl, .pt)
ai/utils.py              — Shared helpers
ai/elo.py                — Elo bookkeeping
ai/logging_utils.py      — logging

ui/server.py             — Flask app + REST endpoints
ui/simulate_manager.py   — Spawns simulator.py subprocesses for the Simulate page
ui/training_manager.py   — Spawns training scripts, parses their stdout for the Train page
ui/templates/            — index, simulate, train, leaderboard, guide, rules

simulator.py             — Headless game runner (subprocess target)
tests/                   — test.py (engine unit tests) + test_integration.py (end-to-end)
```

### Core conventions

- **Agent interface:** every agent implements `choose_action(game, verbose=False)` → action tuple.
- **Actions:** `('plant', flower, r, c)` or `('arrange', from_r, from_c, to_r, to_c)`.
- **Board state:** `dict[(row, col)] -> {'flower': name, 'player': 1|2, 'growing': bool}`.
- **Always clone before simulating:** `game.clone()` — `PaiShoGame` is mutable.

### The registry

[ai/registry.py](ai/registry.py) is the **single source of truth** for every agent. The game UI, Simulate page, Train page, and simulator all read from it. Adding a new agent = adding one entry to `AGENTS`. No UI code needs to change — forms, dropdowns, and CLI wiring are generated from the entry.

---

### Adding a new model you can train

Every piece below wires together through [ai/registry.py](ai/registry.py). The cleanest path is to **copy `basic_minimax` and rename** — it intentionally exercises every registry feature ([ai/classical/basic_minimax.py](ai/classical/basic_minimax.py)). There is also a walkthrough on the **Guide** page in the UI ([ui/templates/guide.html](ui/templates/guide.html)).

#### 1. Write the agent class

Create `ai/rl/my_agent.py`. It must:

- Define a class with a constructor accepting at minimum `player`, `load=True`, and any play-time knobs you expose.
- Implement `choose_action(self, game, verbose=False)` returning a valid action tuple.
- Provide `save(path)` and, if `load=True`, restore weights from `model_path` — silently fall back to random init when the file is absent (first-time training).

```python
class MyAgent:
    def __init__(self, player=1, load=True, temperature=0.0):
        self.player = player
        self.temperature = temperature
        self.model = build_model()
        if load and os.path.exists(MODEL_PATH):
            self.load(MODEL_PATH)

    def choose_action(self, game, verbose=False):
        legal = game.get_legal_actions()
        # score each resulting state with self.model, pick best
        ...

    def save(self, path): ...
    def load(self, path): ...
```

#### 2. Write the training script

Create `ai/training/my_agent_training.py`, if it's needed. It must:

- Accept its hyperparameters as CLI flags using `argparse` — the names must match the right-hand side of `training_cli_map` in your registry entry.
- Run self-play (or play against a fixed opponent), update the model, and **save checkpoints to `model_path`** periodically and at the end.
- Log progress with `ai/logging_utils.get_logger(name)` and `log_event(logger, "episode", episode=i, total=N, ...)` — the Train page reads `EVENT:{...}` lines on stdout to drive its progress bar.
- Support `--resume` to load the existing checkpoint and continue.

Minimal skeleton:

```python
import argparse
from ai.logging_utils import get_logger, log_event

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int,   default=5000)   # episodes
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--eps",     type=float, default=0.3)
    ap.add_argument("--resume",  action="store_true")
    args = ap.parse_args()

    log = get_logger("my_agent")
    agent = MyAgent(load=args.resume)

    for episode in range(1, args.n + 1):
        outcome, steps = self_play_one_episode(agent, args)
        log_event(log, "episode", episode=episode, total=args.n, outcome=outcome, steps=steps)
        if episode % 500 == 0:
            agent.save(MODEL_PATH)
    agent.save(MODEL_PATH)

if __name__ == "__main__":
    main()
```

#### 3. Register it

Append one entry to the `AGENTS` list in [ai/registry.py](ai/registry.py):

```python
{
    "key": "my_agent",
    "display_name": "My Agent",
    "description": "One-line summary shown in the UI",
    "kind": "class",
    "module": "ai.rl.my_agent",
    "class_name": "MyAgent",
    "play_kwargs": {"player": 1, "load": True},
    "needs_player": True,

    # Form fields in the game UI. Types: number | checkbox | text.
    "play_params": [
        {"key": "temperature", "label": "Temperature", "type": "number",
         "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
         "tooltip": "0 = greedy; >0 = softmax sampling"},
    ],

    # Where checkpoints live. The UI shows "Model trained ✓" if this exists.
    "model_path": "ai/params/MyAgent/model.pt",

    # Training script + the knobs that drive it.
    "training_script": "ai/training/my_agent_training.py",
    "training_params": [
        {"key": "episodes", "label": "Episodes",      "type": "number",   "default": 5000, "step": 500, "min": 1},
        {"key": "lr",       "label": "Learning Rate", "type": "number",   "default": 1e-3, "step": 1e-4, "min": 1e-5, "max": 0.1},
        {"key": "epsilon",  "label": "Epsilon",       "type": "number",   "default": 0.3,  "step": 0.01, "min": 0, "max": 1},
        {"key": "resume",   "label": "Resume",        "type": "checkbox", "default": False},
    ],

    # Form field key  →  CLI flag consumed by the training script.
    "training_cli_map": {
        "episodes": "--n", "lr": "--lr", "epsilon": "--eps", "resume": "--resume",
    },

    # Which form field is "total episodes" (drives the progress bar).
    "total_episodes_key": "episodes",

    # Name of the stdout line parser in ui/training_manager.py.
    # Use "basic_minimax" if your log line matches that format, or add a new parser.
    "log_parser": "basic_minimax",

    # Optional: path to a free-form config file editable from the Train page.
    "config_file": None,
},
```

#### Try it

The full lifecycle for adding an agent:

**write agent → write trainer → add registry entry → train agent → play agent against other agents / humans**

---

## Testing

```bash
python tests/test_integration.py  # end-to-end coverage
```

## Author

[Amitaisela](https://github.com/Amitaisela)
