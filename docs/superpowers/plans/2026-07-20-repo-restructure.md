# Repo Restructure: engine / backend / frontend + Docker Compose + Tailscale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize MushiBot into `engine/` (game + AI, dependency-free), `backend/` (Flask app, subprocess orchestration, simulator), and `frontend/` (templates + static assets) folders; consolidate runtime data into `data/`; add a `docker compose up` workflow with GPU-by-default/CPU-override and a Tailscale sidecar.

**Architecture:** Pure physical file reorganization — Python import names (`game`, `ai`, `ui`) do not change, only their on-disk location (via `pyproject.toml`'s `packages.find` `where`). Every literal filesystem-path string (model weights, training-script subprocess targets, config files, saved games, results CSVs, logs, Elo files) must be updated by hand since those are not import statements. A single `DATA_DIR` constant in `ai/utils.py` centralizes the repo-root-relative computation that seven agent modules currently duplicate via fragile `__file__`-relative dot-counting.

**Tech Stack:** Python 3.11+, Flask, setuptools (packages.find), Docker Compose v2, Tailscale sidecar container.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-20-repo-restructure-design.md` — read it before starting; this plan implements it exactly.
- Python import paths (`game`, `ai`, `ui`) must NOT change anywhere in the codebase — only `pyproject.toml`'s `where` changes.
- Every literal filesystem path (not an import) that currently assumes `ai/params`, `ai/checkpoints`, `ai/training/*.py` as a subprocess target, `SavedGames/`, or `results/` must be updated to the new `data/`-relative or `engine/`-relative location.
- No new abstractions beyond the one justified `DATA_DIR` constant (removes duplicated fragile path math already present in 7+ files — this is a strict simplification, not new complexity).
- All moves preserve git history — use plain `mv` (this repo's Bash tool runs git-bash) followed by `git add -A`, which lets git detect renames by content similarity across mixed tracked/gitignored trees (some directories like `ai/params` mix tracked config files with gitignored weights, so per-file `git mv` is impractical).

---

### Task 1: Move all files into the new layout

**Files:**
- Move: `game/` → `engine/game/`
- Move: `ai/` (minus `params/`, `checkpoints/`) → `engine/ai/`
- Move: `ai/params/` → `data/params/`
- Move: `ai/checkpoints/` → `data/checkpoints/`
- Move: `ui/templates/` → `frontend/templates/`
- Move: `ui/static/` → `frontend/static/`
- Move: `ui/` (remainder: `server.py`, `simulate_manager.py`, `training_manager.py`) → `backend/ui/`
- Move: `simulator.py` → `backend/simulator.py`
- Move: `SavedGames/` → `data/saved_games/`
- Move: `results/` → `data/results/`
- Create: `backend/evaluation/` (currently empty except gitignored `__pycache__`; recreated as an empty placeholder per the design)
- Modify: `.gitignore:3` (`ai/checkpoints` → `data/checkpoints`)

**Interfaces:**
- Produces: the physical directory tree every later task edits into. No code changes in this task — pure `mv`.

- [ ] **Step 1: Run the moves**

```bash
cd "c:/Users/amita/Documents/GitHub/MushiBot"
mkdir -p engine backend frontend data
mv game engine/game
mv ai/params data/params
mkdir -p data/checkpoints
mv ai/checkpoints/neat data/checkpoints/neat
rmdir ai/checkpoints
mv ai engine/ai
mv ui/templates frontend/templates
mv ui/static frontend/static
mv ui backend/ui
mv simulator.py backend/simulator.py
mv results data/results
mkdir -p data/saved_games backend/evaluation
rmdir SavedGames 2>/dev/null || true
```

- [ ] **Step 2: Verify the tree**

Run: `find engine backend frontend data -maxdepth 3 -not -path '*__pycache__*' | sort`
Expected: `engine/game/PaiShoGame.py`, `engine/game/notation.py`, `engine/ai/registry.py`, `engine/ai/elo.py`, `engine/ai/utils.py`, `engine/ai/logging_utils.py`, `engine/ai/classical/`, `engine/ai/rl/`, `engine/ai/training/`, `backend/ui/server.py`, `backend/ui/simulate_manager.py`, `backend/ui/training_manager.py`, `backend/simulator.py`, `backend/evaluation/`, `frontend/templates/*.html`, `frontend/static/tiles/*.png`, `data/params/...`, `data/checkpoints/neat/neat-checkpoint-5`, `data/results/*.csv`, `data/saved_games/` all present. No `ai/`, `ui/`, `game/`, `SavedGames/`, `results/` remaining at repo root.

- [ ] **Step 3: Update .gitignore**

Old (`.gitignore` line 3):
```
ai/checkpoints
```
New:
```
data/checkpoints
```

- [ ] **Step 4: Stage and commit**

```bash
git add -A
git status
git commit -m "Move game/ai/ui into engine/backend/frontend layout, consolidate data/"
```

Expected: `git status` before commit shows renames (`renamed:`) for the majority of tracked files (registry.py, elo.py, agent modules, templates, params configs, etc.), confirming history is preserved.

---

### Task 2: Update pyproject.toml package discovery + add DATA_DIR constant

**Files:**
- Modify: `pyproject.toml:39-41`
- Modify: `engine/ai/utils.py` (add `import os` and `DATA_DIR` before the existing functions)

**Interfaces:**
- Produces: `DATA_DIR` — absolute path constant, importable as `from ai.utils import DATA_DIR`, resolving to `<repo_root>/data`. Every later task that touches a weights/config path imports this instead of hand-computing `__file__`-relative dots.

- [ ] **Step 1: Update pyproject.toml**

Old:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["ai*", "game*", "ui*"]
```
New:
```toml
[tool.setuptools.packages.find]
where = ["engine", "backend"]
include = ["ai*", "game*", "ui*"]
```

- [ ] **Step 2: Add DATA_DIR to engine/ai/utils.py**

Old (top of file):
```python
def _ring_threat_level(harmonies):
```
New:
```python
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


def _ring_threat_level(harmonies):
```

- [ ] **Step 3: Reinstall editable and verify imports resolve from the new location**

```bash
pip install -e .
python -c "import game, ai, ui; from ai.utils import DATA_DIR; print(DATA_DIR)"
```
Expected: prints an absolute path ending in `...\MushiBot\data` (or `/data` on POSIX), no `ModuleNotFoundError`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml engine/ai/utils.py
git commit -m "Point package discovery at engine/backend; add shared DATA_DIR constant"
```

---

### Task 3: Fix engine/ai/elo.py and engine/ai/logging_utils.py root-relative paths

Both files compute a repo-root path via `os.path.dirname(__file__)` + a fixed number of `'..'` hops. Since `ai/elo.py` and `ai/logging_utils.py` moved one directory level deeper (from `ai/` to `engine/ai/`), each needs one more `'..'` hop. `elo_ratings.json`, `elo_history.json`, and `logs/` stay at the repo root — only the hop count changes, not the target filenames.

**Files:**
- Modify: `engine/ai/elo.py:8`
- Modify: `engine/ai/logging_utils.py:7`

**Interfaces:**
- Consumes: none new.
- Produces: `PROJECT_ROOT` (elo.py) and `_LOG_DIR` (logging_utils.py) both correctly resolve to the repo root / `<repo_root>/logs`.

- [ ] **Step 1: Fix elo.py**

Old:
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
```
New:
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
```

- [ ] **Step 2: Fix logging_utils.py**

Old:
```python
_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
```
New:
```python
_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
```

- [ ] **Step 3: Verify**

```bash
python -c "from ai.elo import PROJECT_ROOT; from ai.logging_utils import _LOG_DIR; print(PROJECT_ROOT); print(_LOG_DIR)"
```
Expected: `PROJECT_ROOT` prints the repo root (contains `pyproject.toml`); `_LOG_DIR` prints `<repo_root>/logs`.

- [ ] **Step 4: Commit**

```bash
git add engine/ai/elo.py engine/ai/logging_utils.py
git commit -m "Fix elo.py and logging_utils.py root-relative paths for the deeper engine/ai/ location"
```

---

### Task 4: Fix engine/ai/registry.py literal paths

`model_path`, `config_file`, and `training_script` are literal filesystem-path strings (not imports) consumed elsewhere as `os.path.join(PROJECT_ROOT, entry["model_path"])` and as a direct `subprocess.Popen` argument with `cwd=PROJECT_ROOT`. All three families of literal now need updated prefixes: `ai/params/...` → `data/params/...`, `ai/training/....py` → `engine/ai/training/....py`.

**Files:**
- Modify: `engine/ai/registry.py` (16 literal-string edits across the `AGENTS` list)

**Interfaces:**
- Consumes: none.
- Produces: `AGENTS` entries whose `model_path`/`config_file`/`training_script` values are valid relative paths from the repo root once `data/` and `engine/` exist (satisfied by Task 1).

- [ ] **Step 1: Apply all 16 literal-string edits**

| Line (pre-edit) | Old | New |
|---|---|---|
| 72 | `"model_path": "ai/params/MonteCarlo/q_table.pkl",` | `"model_path": "data/params/MonteCarlo/q_table.pkl",` |
| 73 | `"training_script": "ai/training/monte_carlo_training.py",` | `"training_script": "engine/ai/training/monte_carlo_training.py",` |
| 144 | `"model_path": "ai/params/TD/weights.pkl",` | `"model_path": "data/params/TD/weights.pkl",` |
| 145 | `"training_script": "ai/training/td_learning_training.py",` | `"training_script": "engine/ai/training/td_learning_training.py",` |
| 187 | `"model_path": "ai/params/neat/neat_winner.pkl",` | `"model_path": "data/params/neat/neat_winner.pkl",` |
| 188 | `"training_script": "ai/training/neat_train.py",` | `"training_script": "engine/ai/training/neat_train.py",` |
| 203 | `"config_file": "ai/params/neat/neat-config.txt",` | `"config_file": "data/params/neat/neat-config.txt",` |
| 226 | `"model_path": "ai/params/PPO/ppo_model.pt",` | `"model_path": "data/params/PPO/ppo_model.pt",` |
| 227 | `"training_script": "ai/training/ppo_training.py",` | `"training_script": "engine/ai/training/ppo_training.py",` |
| 308 | `"model_path": "ai/params/CNNBasic/cnn_basic.pt",` | `"model_path": "data/params/CNNBasic/cnn_basic.pt",` |
| 309 | `"training_script": "ai/training/cnn_basic_training.py",` | `"training_script": "engine/ai/training/cnn_basic_training.py",` |
| 331 | `"config_file": "ai/params/CNNBasic/example_config.txt",` | `"config_file": "data/params/CNNBasic/example_config.txt",` |
| 361 | `"model_path": "ai/params/NNEUBasic/nneu_basic.pt",` | `"model_path": "data/params/NNEUBasic/nneu_basic.pt",` |
| 362 | `"training_script": "ai/training/nneu_pipeline.py",` | `"training_script": "engine/ai/training/nneu_pipeline.py",` |
| 422 | `"model_path": "ai/params/SetTransformer/set_transformer_model.pt",` | `"model_path": "data/params/SetTransformer/set_transformer_model.pt",` |
| 423 | `"training_script": "ai/training/set_transformer_training.py",` | `"training_script": "engine/ai/training/set_transformer_training.py",` |

Use `replace_all: false` per-line edits (each `old_string` is unique in the file since it includes the specific agent's filename).

- [ ] **Step 2: Verify no stray `ai/params` or `ai/training` literals remain**

Run: `grep -n '"ai/' engine/ai/registry.py`
Expected: no output (all literals now start with `data/` or `engine/ai/`).

- [ ] **Step 3: Commit**

```bash
git add engine/ai/registry.py
git commit -m "Repoint registry.py model/config/training-script paths at data/ and engine/ai/"
```

---

### Task 5: Fix engine/ai/rl/*.py agent modules to use DATA_DIR

Seven agent modules each independently compute their weights path via `os.path.dirname(__file__)` + `'..'` + `'params/<Dir>'`, which worked when `params/` lived directly under `ai/`. Now that `params/` is `data/params/` (a sibling of `engine/`, not nested under `engine/ai/`), each of these needs to import the shared `DATA_DIR` constant from Task 2 instead of re-deriving the offset.

**Files:**
- Modify: `engine/ai/rl/monte_carlo.py:1-11,128-147`
- Modify: `engine/ai/rl/cnn_basic.py:15-30`
- Modify: `engine/ai/rl/nneu_basic.py:35-50`
- Modify: `engine/ai/rl/td_learning.py:1-11`
- Modify: `engine/ai/rl/neat_agent.py:1-20`
- Modify: `engine/ai/rl/ppo_agent.py:1-16`
- Modify: `engine/ai/rl/set_transformer_agent.py:1-19`

**Interfaces:**
- Consumes: `DATA_DIR` from `ai.utils` (Task 2).
- Produces: each module's `MODEL_PATH` / `WEIGHTS_PATH` / `GENOME_PATH` / `CONFIG_PATH` constant, unchanged in name and consuming code, now correctly resolving to `<repo_root>/data/params/<Dir>/...`. `engine/ai/training/neat_train.py` (Task 6) imports `GENOME_PATH` from here.

- [ ] **Step 1: engine/ai/rl/monte_carlo.py**

Old (imports, lines 1-6):
```python
import os
import pickle
from collections import defaultdict
import random

from game.PaiShoGame import VALID_SPACES, CIRCLE, ACCENT_TILES, SPECIAL_TILES
```
New:
```python
import os
import pickle
from collections import defaultdict
import random

from game.PaiShoGame import VALID_SPACES, CIRCLE, ACCENT_TILES, SPECIAL_TILES
from ai.utils import DATA_DIR
```

Old (`save_model`, lines 132-134):
```python
        current_dir = os.path.dirname(__file__)
        params_dir = os.path.abspath(os.path.join(current_dir, '..', 'params/MonteCarlo'))
        os.makedirs(params_dir, exist_ok=True)
```
New:
```python
        params_dir = os.path.join(DATA_DIR, 'params', 'MonteCarlo')
        os.makedirs(params_dir, exist_ok=True)
```

Old (`load_model`, lines 146-147):
```python
        current_dir = os.path.dirname(__file__)
        filepath = os.path.abspath(os.path.join(current_dir, '..', 'params/MonteCarlo', filename))
```
New:
```python
        filepath = os.path.join(DATA_DIR, 'params', 'MonteCarlo', filename)
```

- [ ] **Step 2: engine/ai/rl/cnn_basic.py**

Old (lines 23-30):
```python
from game.PaiShoGame import (
    BOARD_SIZE, VALID_SPACES, ACCENT_TILES, CIRCLE, SPECIAL_TILES,
)


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'params', 'CNNBasic', 'cnn_basic.pt'
)
```
New:
```python
from game.PaiShoGame import (
    BOARD_SIZE, VALID_SPACES, ACCENT_TILES, CIRCLE, SPECIAL_TILES,
)
from ai.utils import DATA_DIR


MODEL_PATH = os.path.join(DATA_DIR, 'params', 'CNNBasic', 'cnn_basic.pt')
```

- [ ] **Step 3: engine/ai/rl/nneu_basic.py**

Old (lines 43-50):
```python
from game.PaiShoGame import (
    BOARD_SIZE, CIRCLE, ACCENT_TILES, SPECIAL_TILES,
)


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'params', 'NNEUBasic', 'nneu_basic.pt'
)
```
New:
```python
from game.PaiShoGame import (
    BOARD_SIZE, CIRCLE, ACCENT_TILES, SPECIAL_TILES,
)
from ai.utils import DATA_DIR


MODEL_PATH = os.path.join(DATA_DIR, 'params', 'NNEUBasic', 'nneu_basic.pt')
```

- [ ] **Step 4: engine/ai/rl/td_learning.py**

Old (lines 7-11):
```python
from game.PaiShoGame import ACCENT_TILES, CIRCLE
from ai.utils import _ring_threat_level, _ring_completion_distance

N_FEATURES = 26
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'params', 'TD', 'weights.pkl')
```
New:
```python
from game.PaiShoGame import ACCENT_TILES, CIRCLE
from ai.utils import _ring_threat_level, _ring_completion_distance, DATA_DIR

N_FEATURES = 26
WEIGHTS_PATH = os.path.join(DATA_DIR, 'params', 'TD', 'weights.pkl')
```

- [ ] **Step 5: engine/ai/rl/neat_agent.py**

Old (lines 8-20):
```python
from ai.utils import _ring_threat_level, _ring_completion_distance
from game.PaiShoGame import (
    ACCENT_TILES, CIRCLE, GATES, SPECIAL_TILES,
    RADIUS, _CIRCLE_IDX, _GATES_SET,
)

N_FEATURES = 40
_CENTER = (RADIUS, RADIUS)

GENOME_PATH = os.path.join(os.path.dirname(__file__), '..', 'params', 'neat', 'neat_winner.pkl')
CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'params', 'neat', 'neat-config.txt')
)
```
New:
```python
from ai.utils import _ring_threat_level, _ring_completion_distance, DATA_DIR
from game.PaiShoGame import (
    ACCENT_TILES, CIRCLE, GATES, SPECIAL_TILES,
    RADIUS, _CIRCLE_IDX, _GATES_SET,
)

N_FEATURES = 40
_CENTER = (RADIUS, RADIUS)

GENOME_PATH = os.path.join(DATA_DIR, 'params', 'neat', 'neat_winner.pkl')
CONFIG_PATH = os.path.join(DATA_DIR, 'params', 'neat', 'neat-config.txt')
```

- [ ] **Step 6: engine/ai/rl/ppo_agent.py**

Old (lines 10-16):
```python
from ai.rl.neat_agent import extract_features, N_FEATURES
from game.PaiShoGame import (
    CIRCLE, FLOWER, ACCENT_TILES, SPECIAL_TILES, GATES, VALID_SPACES,
    _GATES_SET, _VALID_SPACES_SET,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'params', 'PPO', 'ppo_model.pt')
```
New:
```python
from ai.rl.neat_agent import extract_features, N_FEATURES
from ai.utils import DATA_DIR
from game.PaiShoGame import (
    CIRCLE, FLOWER, ACCENT_TILES, SPECIAL_TILES, GATES, VALID_SPACES,
    _GATES_SET, _VALID_SPACES_SET,
)

MODEL_PATH = os.path.join(DATA_DIR, 'params', 'PPO', 'ppo_model.pt')
```

- [ ] **Step 7: engine/ai/rl/set_transformer_agent.py**

Old (lines 10-19):
```python
from ai.rl.ppo_agent import _action_features, ACTION_FEATURE_DIM
from game.PaiShoGame import (
    CIRCLE, FLOWER, ACCENT_TILES, SPECIAL_TILES, GATES,
    _GATES_SET, _VALID_SPACES_SET, _GARDEN_OF,
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'params', 'SetTransformer',
    'set_transformer_model.pt',
)
```
New:
```python
from ai.rl.ppo_agent import _action_features, ACTION_FEATURE_DIM
from ai.utils import DATA_DIR
from game.PaiShoGame import (
    CIRCLE, FLOWER, ACCENT_TILES, SPECIAL_TILES, GATES,
    _GATES_SET, _VALID_SPACES_SET, _GARDEN_OF,
)

MODEL_PATH = os.path.join(DATA_DIR, 'params', 'SetTransformer', 'set_transformer_model.pt')
```

- [ ] **Step 8: Verify every agent module imports and resolves its path under data/params**

```bash
python -c "
from ai.rl.monte_carlo import MonteCarloAgent
from ai.rl.cnn_basic import MODEL_PATH as CNN_MP
from ai.rl.nneu_basic import MODEL_PATH as NNEU_MP
from ai.rl.td_learning import WEIGHTS_PATH
from ai.rl.neat_agent import GENOME_PATH, CONFIG_PATH
from ai.rl.ppo_agent import MODEL_PATH as PPO_MP
from ai.rl.set_transformer_agent import MODEL_PATH as ST_MP
for p in (CNN_MP, NNEU_MP, WEIGHTS_PATH, GENOME_PATH, CONFIG_PATH, PPO_MP, ST_MP):
    assert 'data' + __import__('os').sep + 'params' in p, p
    print(p)
"
```
Expected: seven paths printed, each containing `data/params` (or `data\params` on Windows), no exceptions.

- [ ] **Step 9: Commit**

```bash
git add engine/ai/rl
git commit -m "Point ai/rl agent modules at shared DATA_DIR instead of file-relative dot-counting"
```

---

### Task 6: Fix engine/ai/training/neat_train.py and nneu_pipeline.py

`neat_train.py` moved one level deeper (`ai/training/` → `engine/ai/training/`), so both its `sys.path.insert` bootstrap and its checkpoint-prefix root computation need one more `'..'` hop, and its `--config`/checkpoint-prefix defaults move from `ai/params`/`ai/checkpoints` to `data/params`/`data/checkpoints`. `nneu_pipeline.py` spawns two sibling training scripts via bare relative-path subprocess calls that must now include the `engine/` prefix.

**Files:**
- Modify: `engine/ai/training/neat_train.py:10,13,177-180,206-207,209,219-221`
- Modify: `engine/ai/training/nneu_pipeline.py:1-11,44,65`
- Modify: `engine/ai/training/cnn_basic_training.py:10` (docstring only)

**Interfaces:**
- Consumes: `GENOME_PATH` from `ai.rl.neat_agent` (Task 5).

- [ ] **Step 1: neat_train.py — fix sys.path bootstrap**

Old (line 10):
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
```
New:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
```

- [ ] **Step 2: neat_train.py — import GENOME_PATH and reuse it**

Old (line 13):
```python
from ai.rl.neat_agent import extract_features, N_FEATURES
```
New:
```python
from ai.rl.neat_agent import extract_features, N_FEATURES, GENOME_PATH
```

Old (lines 177-180):
```python
    winner_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'params', 'neat', 'neat_winner.pkl')
    )
    os.makedirs(os.path.dirname(winner_path), exist_ok=True)
```
New:
```python
    winner_path = GENOME_PATH
    os.makedirs(os.path.dirname(winner_path), exist_ok=True)
```

- [ ] **Step 3: neat_train.py — fix --config default and help text**

Old (lines 206-207):
```python
    parser.add_argument('--config', type=str, default=os.path.join('ai', 'params', 'neat', 'neat-config.txt'),
                        help='Path to NEAT config file (default: ai/params/neat/neat-config.txt)')
```
New:
```python
    parser.add_argument('--config', type=str, default=os.path.join('data', 'params', 'neat', 'neat-config.txt'),
                        help='Path to NEAT config file (default: data/params/neat/neat-config.txt)')
```

- [ ] **Step 4: neat_train.py — fix checkpoint-prefix help text and default computation**

Old (line 209):
```python
    parser.add_argument('--checkpoint-prefix', type=str, default=None,
                        help='Prefix for checkpoint files (default: ai/checkpoints/neat/neat-checkpoint-)')
```
New:
```python
    parser.add_argument('--checkpoint-prefix', type=str, default=None,
                        help='Prefix for checkpoint files (default: data/checkpoints/neat/neat-checkpoint-)')
```

Old (lines 219-221):
```python
    if args.checkpoint_prefix is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        args.checkpoint_prefix = os.path.join(project_root, 'ai', 'checkpoints', 'neat', 'neat-checkpoint-')
```
New:
```python
    if args.checkpoint_prefix is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        args.checkpoint_prefix = os.path.join(project_root, 'data', 'checkpoints', 'neat', 'neat-checkpoint-')
```

- [ ] **Step 5: nneu_pipeline.py — fix docstring and subprocess targets**

Old (lines 1-11):
```python
"""Iterative generate-then-train pipeline for NNEU Basic.

Each generation:
  1. Run `nneu_generate.py` with the latest checkpoint as the actor.
  2. Run `nneu_basic_training.py` over all accumulated shards.

Shards are written to `ai/params/NNEUBasic/data/gen{gg}.npz` and the
trained weights land at `ai/params/NNEUBasic/nneu_basic.pt` (the
standard `MODEL_PATH`) after each generation, so evaluation and the UI
pick them up automatically.
"""
```
New:
```python
"""Iterative generate-then-train pipeline for NNEU Basic.

Each generation:
  1. Run `nneu_generate.py` with the latest checkpoint as the actor.
  2. Run `nneu_basic_training.py` over all accumulated shards.

Shards are written to `data/params/NNEUBasic/data/gen{gg}.npz` and the
trained weights land at `data/params/NNEUBasic/nneu_basic.pt` (the
standard `MODEL_PATH`) after each generation, so evaluation and the UI
pick them up automatically.
"""
```

Old (line 44):
```python
            sys.executable, "-u", "ai/training/nneu_generate.py",
```
New:
```python
            sys.executable, "-u", "engine/ai/training/nneu_generate.py",
```

Old (line 65):
```python
            sys.executable, "-u", "ai/training/nneu_basic_training.py",
```
New:
```python
            sys.executable, "-u", "engine/ai/training/nneu_basic_training.py",
```

- [ ] **Step 6: cnn_basic_training.py — cosmetic docstring fix**

Old (line 10):
```python
Registry exposes ai/params/CNNBasic/example_config.txt as a
```
New:
```python
Registry exposes data/params/CNNBasic/example_config.txt as a
```

- [ ] **Step 7: Verify neat_train.py imports and argparse defaults resolve correctly**

```bash
python -c "
from ai.training.neat_train import _parse_args
import sys
sys.argv = ['neat_train.py']
args = _parse_args()
print(args.config)
"
```
Expected: prints `data/params/neat/neat-config.txt` (or the Windows-separator equivalent), no exceptions.

- [ ] **Step 8: Commit**

```bash
git add engine/ai/training/neat_train.py engine/ai/training/nneu_pipeline.py engine/ai/training/cnn_basic_training.py
git commit -m "Fix neat_train.py and nneu_pipeline.py paths for the engine/ai/training/ location"
```

---

### Task 7: Fix backend/ui/server.py

`server.py` moved from `ui/` to `backend/ui/` — one level deeper. Its `sys.path` bootstrap needs an extra `'..'` hop, its Flask app needs explicit `template_folder`/`static_folder` pointed at `frontend/` (since Flask's defaults look for `templates/`/`static/` next to the app file, which no longer exist there), every `send_from_directory('.', 'templates/X.html')` call needs to point at the new `frontend/templates` location, and the hardcoded `ai/...` source paths in the `/api/example/<filename>` download map need an `engine/` prefix.

**Files:**
- Modify: `backend/ui/server.py:10,30,127,326,576,581,587-590,719`

**Interfaces:**
- Consumes: none new (Flask, existing imports).
- Produces: `_FRONTEND_DIR` module constant used by every template-serving route in this file.

- [ ] **Step 1: Fix sys.path bootstrap**

Old (line 10):
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
```
New:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
```

- [ ] **Step 2: Point Flask at frontend/templates and frontend/static**

Old (line 30):
```python
app = Flask(__name__)
```
New:
```python
_FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))

app = Flask(
    __name__,
    template_folder=os.path.join(_FRONTEND_DIR, 'templates'),
    static_folder=os.path.join(_FRONTEND_DIR, 'static'),
)
```

- [ ] **Step 3: Fix the five send_from_directory calls**

Old (line 127):
```python
    return send_from_directory('.', 'templates/index.html')
```
New:
```python
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'index.html')
```

Old (line 326):
```python
    return send_from_directory('.', 'templates/leaderboard.html')
```
New:
```python
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'leaderboard.html')
```

Old (line 576):
```python
    return send_from_directory('.', 'templates/guide.html')
```
New:
```python
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'guide.html')
```

Old (line 581):
```python
    return send_from_directory('.', 'templates/rules.html')
```
New:
```python
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'rules.html')
```

Old (line 719):
```python
    return send_from_directory('.', 'templates/simulate.html')
```
New:
```python
    return send_from_directory(os.path.join(_FRONTEND_DIR, 'templates'), 'simulate.html')
```

- [ ] **Step 4: Fix the ALLOWED source-download map**

Old (lines 587-590):
```python
    ALLOWED = {
        'basic_minimax.py': os.path.join(PROJECT_ROOT, 'ai', 'classical', 'basic_minimax.py'),
        'cnn_basic.py': os.path.join(PROJECT_ROOT, 'ai', 'rl', 'cnn_basic.py'),
        'cnn_basic_training.py': os.path.join(PROJECT_ROOT, 'ai', 'training', 'cnn_basic_training.py'),
        'registry.py': os.path.join(PROJECT_ROOT, 'ai', 'registry.py'),
    }
```
New:
```python
    ALLOWED = {
        'basic_minimax.py': os.path.join(PROJECT_ROOT, 'engine', 'ai', 'classical', 'basic_minimax.py'),
        'cnn_basic.py': os.path.join(PROJECT_ROOT, 'engine', 'ai', 'rl', 'cnn_basic.py'),
        'cnn_basic_training.py': os.path.join(PROJECT_ROOT, 'engine', 'ai', 'training', 'cnn_basic_training.py'),
        'registry.py': os.path.join(PROJECT_ROOT, 'engine', 'ai', 'registry.py'),
    }
```

- [ ] **Step 5: Commit**

(Bundled with Task 8's commit since both files are verified together — see Task 8 Step 3.)

---

### Task 8: Fix backend/ui/training_manager.py and backend/ui/simulate_manager.py

Both modules moved one level deeper and independently compute their own `PROJECT_ROOT`; both need the same extra `'..'` hop. `simulate_manager.py` additionally hardcodes `simulator.py`'s path for its subprocess call, which must gain the `backend/` prefix since `simulator.py` moved there in Task 1.

**Files:**
- Modify: `backend/ui/training_manager.py:11`
- Modify: `backend/ui/simulate_manager.py:8,140`

**Interfaces:**
- Consumes: `entry["model_path"]` / `entry["training_script"]` from `ai.registry` (Task 4, already `data/`/`engine/`-prefixed).
- Produces: `PROJECT_ROOT` in both modules correctly resolving to the repo root.

- [ ] **Step 1: Fix training_manager.py PROJECT_ROOT**

Old (line 11):
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
```
New:
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
```

- [ ] **Step 2: Fix simulate_manager.py PROJECT_ROOT and simulator.py subprocess path**

Old (line 8):
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
```
New:
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
```

Old (line 140):
```python
        sys.executable, "-u", os.path.join(PROJECT_ROOT, "simulator.py"),
```
New:
```python
        sys.executable, "-u", os.path.join(PROJECT_ROOT, "backend", "simulator.py"),
```

- [ ] **Step 3: Verify PROJECT_ROOT resolves correctly in both managers**

Relies on the editable install from Task 2 (`pip install -e .`), which makes `ui` importable from any working directory — no manual `sys.path`/`chdir` needed.

```bash
python -c "
import os
from ui.training_manager import PROJECT_ROOT as TM_ROOT, MODEL_PATHS
from ui.simulate_manager import PROJECT_ROOT as SM_ROOT
assert os.path.exists(os.path.join(TM_ROOT, 'pyproject.toml')), TM_ROOT
assert os.path.exists(os.path.join(SM_ROOT, 'pyproject.toml')), SM_ROOT
print('TM_ROOT:', TM_ROOT)
print('SM_ROOT:', SM_ROOT)
print('sample MODEL_PATHS entry:', next(iter(MODEL_PATHS.items())))
"
```
Expected: both roots print the repo root (containing `pyproject.toml`), and the sample `MODEL_PATHS` entry's value contains `data/params` (or `data\params`).

- [ ] **Step 4: Commit (bundles Task 7 + Task 8)**

```bash
git add backend/ui/server.py backend/ui/training_manager.py backend/ui/simulate_manager.py
git commit -m "Fix backend/ui path constants and Flask template/static folders for the new layout"
```

---

### Task 9: Fix backend/simulator.py and tests/test_integration.py

`simulator.py`'s `results/` and `SavedGames/` literals are relative to the process's current working directory, which is always the repo root (both direct invocation from repo root per CLAUDE.md, and subprocess invocation via `cwd=PROJECT_ROOT` from `simulate_manager.py`/`training_manager.py`). They must be updated to the new `data/results/` and `data/saved_games/` locations. `test_integration.py`'s end-to-end test spawns `simulator.py` directly and must reference its new `backend/` location.

**Files:**
- Modify: `backend/simulator.py:108,124,136,340`
- Modify: `tests/test_integration.py:197`

**Interfaces:**
- Consumes: none new.
- Produces: games/results now land under `data/results/` and `data/saved_games/`.

- [ ] **Step 1: Fix save_result_to_csv**

Old (line 108):
```python
    file_path = f"results/{p1_name}_vs_{p2_name}_results.csv"
```
New:
```python
    file_path = f"data/results/{p1_name}_vs_{p2_name}_results.csv"
```

- [ ] **Step 2: Fix save_game_to_file and save_game_to_psn**

Old (line 124, in `save_game_to_file`):
```python
    dir_path = f"SavedGames/{p1_name}_vs_{p2_name}"
```
New:
```python
    dir_path = f"data/saved_games/{p1_name}_vs_{p2_name}"
```

Old (line 136, in `save_game_to_psn`):
```python
    dir_path = f"SavedGames/{p1_name}_vs_{p2_name}"
```
New:
```python
    dir_path = f"data/saved_games/{p1_name}_vs_{p2_name}"
```

- [ ] **Step 3: Fix the inline save path inside run_flask**

Old (line 340):
```python
                dir_path = f"SavedGames/{p1n}_vs_{p2n}"
```
New:
```python
                dir_path = f"data/saved_games/{p1n}_vs_{p2n}"
```

- [ ] **Step 4: Fix test_integration.py's simulator.py subprocess target**

Old (line 197):
```python
        [sys.executable, '-u', 'simulator.py', '--mode', 'local',
```
New:
```python
        [sys.executable, '-u', 'backend/simulator.py', '--mode', 'local',
```

- [ ] **Step 4b: Fix test_integration.py's direct `from simulator import ...` calls (discovered during execution)**

`simulator.py` was never part of the `ai*`/`game*`/`ui*` packages picked up by `packages.find` — it only worked as a bare `import simulator` because it sat at the repo root, which `test_integration.py` already adds to `sys.path`. Now that it lives at `backend/simulator.py`, `backend/` must be added to `sys.path` too so `from simulator import parse_model_spec` / `load_model` (used directly by several unit tests, not just the subprocess test) keep resolving.

Old (line 8):
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game.PaiShoGame import PaiShoGame, CIRCLE
```
New:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from game.PaiShoGame import PaiShoGame, CIRCLE
```

- [ ] **Step 5: Run the integration test to verify end-to-end**

Run: `python tests/test_integration.py`
Expected: all tests pass, including `test_simulator_local_mode_emits_game_end_event` (confirms `backend/simulator.py` runs correctly with `cwd=<repo root>` and emits `EVENT:` lines).

- [ ] **Step 6: Commit**

```bash
git add backend/simulator.py tests/test_integration.py
git commit -m "Point simulator.py's results/saved-games at data/, fix its new subprocess path in tests"
```

---

### Task 10: Update scripts/distill.py for the new layout

`distill.py` prunes a checkout of this repo down to the 3-agent public template before syncing to `pai-sho-lab`. Its `KEEP_FILES` allowlist and `cleanup_fs()` walk currently assume `ai/` sits at the repo root and that `ai/params/CNNBasic/*` survives pruning via the same walk. Both assumptions break now: the code tree is `engine/ai/`, and `params/` is a sibling top-level `data/params/` no longer touched by an `os.walk("ai", ...)` at all. This task splits the pruning into two walks — one over `engine/ai` (code) and one over `data/params` (the two files that must survive for the CNN template).

**Files:**
- Modify: `scripts/distill.py` (KEEP_FILES, REMOVE_DIRS, prune_registry, cleanup_fs)

**Interfaces:**
- Consumes: the new `engine/ai/`, `data/params/`, `data/checkpoints/` locations from Task 1.
- Produces: a distilled tree with the same *content* the old distill.py produced, just at the new paths — `.github/workflows/deploy.yml`'s post-distill smoke test (`from ai.registry import AGENTS`) needs no change since Python imports are unaffected.

- [ ] **Step 1: Update KEEP_FILES and add KEEP_DATA_FILES**

Old (lines 6-25):
```python
KEEP_AGENTS = {"random", "basic_minimax", "cnn_basic"}

KEEP_FILES = {
    "ai/__init__.py",
    "ai/registry.py",
    "ai/logging_utils.py",
    "ai/elo.py",
    "ai/utils.py",
    "ai/classical/__init__.py",
    "ai/classical/basic_minimax.py",
    "ai/rl/__init__.py",
    "ai/rl/cnn_basic.py",
    "ai/training/__init__.py",
    "ai/training/cnn_basic_training.py",
    "ai/training/opponent_utils.py",
    "ai/params/CNNBasic/example_config.txt",
    "ai/params/CNNBasic/cnn_basic.pt",
}

REMOVE_DIRS = ("ai/checkpoints", "scripts")
```
New:
```python
KEEP_AGENTS = {"random", "basic_minimax", "cnn_basic"}

KEEP_FILES = {
    "engine/ai/__init__.py",
    "engine/ai/registry.py",
    "engine/ai/logging_utils.py",
    "engine/ai/elo.py",
    "engine/ai/utils.py",
    "engine/ai/classical/__init__.py",
    "engine/ai/classical/basic_minimax.py",
    "engine/ai/rl/__init__.py",
    "engine/ai/rl/cnn_basic.py",
    "engine/ai/training/__init__.py",
    "engine/ai/training/cnn_basic_training.py",
    "engine/ai/training/opponent_utils.py",
}

KEEP_DATA_FILES = {
    "data/params/CNNBasic/example_config.txt",
    "data/params/CNNBasic/cnn_basic.pt",
}

REMOVE_DIRS = ("data/checkpoints", "scripts")
```

- [ ] **Step 2: Fix prune_registry's path**

Old (line 37):
```python
    path = "ai/registry.py"
```
New:
```python
    path = "engine/ai/registry.py"
```

- [ ] **Step 3: Fix cleanup_fs to walk engine/ai for code and data/params separately for weights**

Old (lines 79-92):
```python
def cleanup_fs():
    for d in REMOVE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)

    for root, _, files in os.walk("ai", topdown=False):
        for name in files:
            p = _norm(os.path.join(root, name))
            if p in KEEP_FILES or name == "__init__.py":
                continue
            os.remove(p)
        rel = _norm(os.path.relpath(root)).rstrip("/")
        if rel != "ai" and os.path.isdir(root) and not os.listdir(root):
            os.rmdir(root)
```
New:
```python
def cleanup_fs():
    for d in REMOVE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)

    for root, _, files in os.walk("engine/ai", topdown=False):
        for name in files:
            p = _norm(os.path.join(root, name))
            if p in KEEP_FILES or name == "__init__.py":
                continue
            os.remove(p)
        rel = _norm(os.path.relpath(root)).rstrip("/")
        if rel != "engine/ai" and os.path.isdir(root) and not os.listdir(root):
            os.rmdir(root)

    if os.path.isdir("data/params"):
        for root, _, files in os.walk("data/params", topdown=False):
            for name in files:
                p = _norm(os.path.join(root, name))
                if p in KEEP_DATA_FILES:
                    continue
                os.remove(p)
            rel = _norm(os.path.relpath(root)).rstrip("/")
            if rel != "data/params" and os.path.isdir(root) and not os.listdir(root):
                os.rmdir(root)
```

- [ ] **Step 4: Dry-run distill.py against a scratch copy to verify it produces the expected tree**

```bash
cd "c:/Users/amita/Documents/GitHub/MushiBot"
rm -rf /tmp/distill_check
git worktree add /tmp/distill_check HEAD
cd /tmp/distill_check
python scripts/distill.py
find engine/ai data/params -type f | sort
cd "c:/Users/amita/Documents/GitHub/MushiBot"
git worktree remove /tmp/distill_check --force
```
Expected `find` output: exactly `engine/ai/__init__.py`, `engine/ai/registry.py`, `engine/ai/logging_utils.py`, `engine/ai/elo.py`, `engine/ai/utils.py`, `engine/ai/classical/__init__.py`, `engine/ai/classical/basic_minimax.py`, `engine/ai/rl/__init__.py`, `engine/ai/rl/cnn_basic.py`, `engine/ai/training/__init__.py`, `engine/ai/training/cnn_basic_training.py`, `engine/ai/training/opponent_utils.py`, `data/params/CNNBasic/example_config.txt`, `data/params/CNNBasic/cnn_basic.pt` — nothing else. Also confirm `engine/ai/registry.py`'s `AGENTS` list contains only `random`, `basic_minimax`, `cnn_basic` keys (`python -c "from ai.registry import AGENTS; print(sorted(a['key'] for a in AGENTS))"` run from inside the worktree before removing it, if the assertion needs double-checking).

- [ ] **Step 5: Commit**

```bash
git add scripts/distill.py
git commit -m "Adapt distill.py's pruning to engine/ai/ code and data/params/ weights"
```

---

### Task 11: Write Dockerfile, docker-compose.yml, docker-compose.cpu.yml, .env.example

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `docker-compose.cpu.yml`
- Create: `.env.example`
- Modify: `.gitignore` (ensure `.env` is ignored — it already is, via the existing `# Environments` section's `.env` entry; verify only, no edit expected)

**Interfaces:**
- Produces: a working `docker compose up` (GPU default) and `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up` (CPU override), both serving the Flask app on `localhost:5000` and the Tailscale tailnet hostname `mushibot`.

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY engine/ engine/
COPY backend/ backend/
COPY frontend/ frontend/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .
ENV HOST=0.0.0.0
EXPOSE 5000
CMD ["python", "backend/ui/server.py"]
```

**Discovered during execution:** `backend/ui/server.py`'s `app.run()` reads `HOST` from the environment, defaulting to `127.0.0.1` — the correct default for local `python backend/ui/server.py` use, but a listener bound to loopback-only is unreachable via Docker's port publishing (which connects via the container's external interface, not loopback). `ENV HOST=0.0.0.0` in the Dockerfile fixes this without changing the app's local-dev default. Verified: without it, `curl`/`urlopen` against the published port got `RemoteDisconnected` with nothing reaching Flask; with it, both `/` and `/static/tiles/*` served correctly.

- [ ] **Step 2: Create docker-compose.yml**

```yaml
services:
  tailscale:
    image: tailscale/tailscale:latest
    hostname: mushibot
    environment:
      - TS_AUTHKEY=${TS_AUTHKEY}
      - TS_STATE_DIR=/var/lib/tailscale
    volumes:
      - tailscale-state:/var/lib/tailscale
    cap_add: [NET_ADMIN, NET_RAW]
    ports:
      - "5000:5000"
    restart: unless-stopped

  backend:
    build: .
    network_mode: service:tailscale
    depends_on: [tailscale]
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  tailscale-state:
```

- [ ] **Step 3: Create docker-compose.cpu.yml**

```yaml
services:
  backend:
    deploy:
      resources:
        reservations:
          devices: []
```

- [ ] **Step 4: Create .env.example**

```
TS_AUTHKEY=tskey-auth-xxxxx
```

- [ ] **Step 5: Verify .env is gitignored**

Run: `grep -n '^\.env$' .gitignore`
Expected: one match (already present in the "Environments" section; no edit needed).

- [ ] **Step 6: Build and smoke-test the image (CPU override, since CI/dev machines may lack a GPU)**

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
sleep 5
curl -sf http://localhost:5000/ -o /dev/null && echo "OK: server responded"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml down
```
Expected: build succeeds, `OK: server responded` prints (or, if `TS_AUTHKEY` is unset/invalid, the `tailscale` container may fail to authenticate but `backend` should still serve on `localhost:5000` via the published port — if the request fails entirely, check `docker compose logs` for the actual failure before treating this as passing).

- [ ] **Step 7: Commit**

```bash
git add Dockerfile docker-compose.yml docker-compose.cpu.yml .env.example
git commit -m "Add Dockerfile, docker-compose (GPU default + CPU override), and Tailscale sidecar"
```

---

### Task 12: Update documentation (CLAUDE.md, README.md, guide.html)

**Files:**
- Modify: `CLAUDE.md` (project structure tree, run commands)
- Modify: `README.md` (project structure section, run commands, model_path references, guide walkthrough)
- Modify: `frontend/templates/guide.html` (in-UI walkthrough path references)

**Interfaces:**
- Consumes: none (documentation only).
- Produces: docs that match the tree produced by Tasks 1-11, so a new contributor's first `pip install -e .` / `python backend/ui/server.py` actually works.

- [ ] **Step 1: Update CLAUDE.md's "Running" section**

Old:
```bash
# Web UI (open http://localhost:5000)
python ui/server.py
```
New:
```bash
# Web UI (open http://localhost:5000)
python backend/ui/server.py
```

Old:
```bash
# Headless batch runs (no UI)
python simulator.py --mode local --p1 minimax --p2 monte_carlo --n 10
```
New:
```bash
# Headless batch runs (no UI)
python backend/simulator.py --mode local --p1 minimax --p2 monte_carlo --n 10
```

- [ ] **Step 2: Update CLAUDE.md's Training section commands**

Old:
```bash
python ai/training/monte_carlo_training.py     --n 5000 --eps 0.9 --lr 0.1 --decay 0.995
python ai/training/td_learning_training.py     --n 5000 --lr 0.01 --lam 0.7 --eps 0.3
python ai/training/cnn_basic_training.py       --n 200  --lr 1e-3 --eps 0.5 --opponent self
python ai/training/ppo_training.py             --n 5000 --lr 3e-4 --clip 0.2 --epochs 4
python ai/training/set_transformer_training.py --n 10000 --lr 3e-4 --opponent self
python ai/training/neat_train.py               --generations 100 --games 10 --opponents random
python ai/training/nneu_pipeline.py            --generations 3 --games_per_gen 500 --epochs_per_gen 10
```
New:
```bash
python engine/ai/training/monte_carlo_training.py     --n 5000 --eps 0.9 --lr 0.1 --decay 0.995
python engine/ai/training/td_learning_training.py     --n 5000 --lr 0.01 --lam 0.7 --eps 0.3
python engine/ai/training/cnn_basic_training.py       --n 200  --lr 1e-3 --eps 0.5 --opponent self
python engine/ai/training/ppo_training.py             --n 5000 --lr 3e-4 --clip 0.2 --epochs 4
python engine/ai/training/set_transformer_training.py --n 10000 --lr 3e-4 --opponent self
python engine/ai/training/neat_train.py               --generations 100 --games 10 --opponents random
python engine/ai/training/nneu_pipeline.py            --generations 3 --games_per_gen 500 --epochs_per_gen 10
```

- [ ] **Step 3: Update CLAUDE.md's Testing section**

Old:
```bash
python tests/test.py              # engine unit tests, writes tests/test_report.txt
python tests/test_integration.py  # end-to-end
```
New (unchanged — `tests/` stays at the repo root; verify only, no edit needed).

- [ ] **Step 4: Replace CLAUDE.md's "Project structure" tree**

Old:
```
game/PaiShoGame.py                     — Game engine (rules, state, validation, harmony/clash/ring detection)
game/notation.py                       — PSN (Pai Sho Notation) serialization

ai/registry.py                         — Single source of truth: every agent's UI/training/CLI metadata
ai/utils.py                            — Shared eval helpers (ring threat, harmony distance)
ai/elo.py                              — Elo bookkeeping, leaderboard, history (resets ratings on weight changes)
ai/logging_utils.py                    — Structured JSON event logger used by training scripts

ai/classical/minimax.py                — Alpha-beta with iterative deepening, transposition table, ordered moves
ai/classical/basic_minimax.py          — Stripped-down alpha-beta; reference template for a weightless agent

ai/rl/mcts.py                          — MCTS with UCT, RAVE, progressive widening, heuristic leaf eval
ai/rl/monte_carlo.py                   — Tabular Q-learning with Zobrist hashing
ai/rl/td_learning.py                   — TD(λ) over a 26-dim hand-crafted feature vector
ai/rl/cnn_basic.py                     — Two-layer CNN value net on an 8-channel 19×19 board (one-ply greedy)
ai/rl/nneu_basic.py                    — NNUE-style dual-accumulator value net (CReLU + scalar head, int16 inference)
ai/rl/neat_agent.py                    — NEAT-evolved feed-forward network
ai/rl/ppo_agent.py                     — Actor-critic with PPO clipped objective and GAE-λ
ai/rl/set_transformer_agent.py         — Set Transformer encoder over piece set; PPO-trained policy + value heads

ai/training/monte_carlo_training.py    — Self-play Q-learning loop
ai/training/td_learning_training.py    — TD(λ) self-play with eligibility traces
ai/training/cnn_basic_training.py      — Self-play value regression for the CNN
ai/training/nneu_generate.py           — Self-play data generator → CSR-packed .npz shards
ai/training/nneu_basic_training.py     — Supervised regression on shards
ai/training/nneu_dataset.py            — DataLoader for sparse CSR shards
ai/training/nneu_pipeline.py           — Orchestrator: loops generate → train across generations
ai/training/neat_train.py              — NEAT evolution loop with tournament fitness
ai/training/ppo_training.py            — PPO + GAE training loop
ai/training/set_transformer_training.py — PPO loop wired to the Set Transformer encoder
ai/training/opponent_utils.py          — load_opponent(name) → callable; used by training scripts

ai/params/                             — Saved weights/checkpoints per agent (.pkl Q-tables, .pt neural)
                                         Subdirs: MonteCarlo, TD, CNNBasic, NNEUBasic, PPO, SetTransformer, neat

ui/server.py                           — Flask app + REST endpoints; serves all pages
ui/training_manager.py                 — Spawns training scripts, parses stdout, exposes status to the Train page
ui/simulate_manager.py                 — Spawns simulator.py subprocesses for the Simulate page; tracks Elo
ui/templates/index.html                — Play page (board UI, vs human / AI / AI vs AI)
ui/templates/simulate.html             — Simulate page (tournament runner, live stats)
ui/templates/train.html                — Train page (per-agent hyperparameter forms, progress, log tail)
ui/templates/leaderboard.html          — Elo leaderboard + history view
ui/templates/guide.html                — In-UI walkthrough for adding a new agent
ui/templates/rules.html                — Game rules reference

simulator.py                           — Main game runner (modes: flask, local). Parses 'agent:k=v,...' specs.
tests/test.py                          — Engine unit tests (custom runner)
tests/test_integration.py              — End-to-end tests
```
New:
```
engine/game/PaiShoGame.py              — Game engine (rules, state, validation, harmony/clash/ring detection)
engine/game/notation.py                — PSN (Pai Sho Notation) serialization

engine/ai/registry.py                  — Single source of truth: every agent's UI/training/CLI metadata
engine/ai/utils.py                     — Shared eval helpers (ring threat, harmony distance) + DATA_DIR constant
engine/ai/elo.py                       — Elo bookkeeping, leaderboard, history (resets ratings on weight changes)
engine/ai/logging_utils.py             — Structured JSON event logger used by training scripts

engine/ai/classical/minimax.py         — Alpha-beta with iterative deepening, transposition table, ordered moves
engine/ai/classical/basic_minimax.py   — Stripped-down alpha-beta; reference template for a weightless agent

engine/ai/rl/mcts.py                   — MCTS with UCT, RAVE, progressive widening, heuristic leaf eval
engine/ai/rl/monte_carlo.py            — Tabular Q-learning with Zobrist hashing
engine/ai/rl/td_learning.py            — TD(λ) over a 26-dim hand-crafted feature vector
engine/ai/rl/cnn_basic.py              — Two-layer CNN value net on an 8-channel 19×19 board (one-ply greedy)
engine/ai/rl/nneu_basic.py             — NNUE-style dual-accumulator value net (CReLU + scalar head, int16 inference)
engine/ai/rl/neat_agent.py             — NEAT-evolved feed-forward network
engine/ai/rl/ppo_agent.py              — Actor-critic with PPO clipped objective and GAE-λ
engine/ai/rl/set_transformer_agent.py  — Set Transformer encoder over piece set; PPO-trained policy + value heads

engine/ai/training/monte_carlo_training.py    — Self-play Q-learning loop
engine/ai/training/td_learning_training.py    — TD(λ) self-play with eligibility traces
engine/ai/training/cnn_basic_training.py      — Self-play value regression for the CNN
engine/ai/training/nneu_generate.py           — Self-play data generator → CSR-packed .npz shards
engine/ai/training/nneu_basic_training.py     — Supervised regression on shards
engine/ai/training/nneu_dataset.py            — DataLoader for sparse CSR shards
engine/ai/training/nneu_pipeline.py           — Orchestrator: loops generate → train across generations
engine/ai/training/neat_train.py              — NEAT evolution loop with tournament fitness
engine/ai/training/ppo_training.py            — PPO + GAE training loop
engine/ai/training/set_transformer_training.py — PPO loop wired to the Set Transformer encoder
engine/ai/training/opponent_utils.py          — load_opponent(name) → callable; used by training scripts

data/params/                           — Saved weights/checkpoints per agent (.pkl Q-tables, .pt neural)
                                         Subdirs: MonteCarlo, TD, CNNBasic, NNEUBasic, PPO, SetTransformer, neat
data/checkpoints/                      — Mid-training checkpoints (e.g. NEAT population checkpoints)
data/saved_games/                      — Saved game files/PSN from simulator.py
data/results/                          — Simulation result CSVs

backend/ui/server.py                   — Flask app + REST endpoints; serves all pages
backend/ui/training_manager.py         — Spawns training scripts, parses stdout, exposes status to the Train page
backend/ui/simulate_manager.py         — Spawns simulator.py subprocesses for the Simulate page; tracks Elo
backend/simulator.py                   — Main game runner (modes: flask, local). Parses 'agent:k=v,...' specs.

frontend/templates/index.html          — Play page (board UI, vs human / AI / AI vs AI)
frontend/templates/simulate.html       — Simulate page (tournament runner, live stats)
frontend/templates/train.html          — Train page (per-agent hyperparameter forms, progress, log tail)
frontend/templates/leaderboard.html    — Elo leaderboard + history view
frontend/templates/guide.html          — In-UI walkthrough for adding a new agent
frontend/templates/rules.html          — Game rules reference
frontend/static/                       — Board tile images

tests/test.py                          — Engine unit tests (custom runner)
tests/test_integration.py              — End-to-end tests

Dockerfile, docker-compose.yml, docker-compose.cpu.yml, .env.example — containerized run + Tailscale sidecar
```

- [ ] **Step 5: Update README.md's Running section**

Old:
```bash
python ui/server.py
```
New:
```bash
python backend/ui/server.py
```

- [ ] **Step 6: Update README.md's cnn_basic section**

Old:
```
The network ships with trained weights at [ai/params/CNNBasic/cnn_basic.pt](ai/params/CNNBasic/cnn_basic.pt), so it plays competently out of the box. If you want to improve on it, re-train via the Train page or directly:

```bash
python ai/training/cnn_basic_training.py --n 200 --lr 1e-3 --eps 0.5 --opponent self
```
```
New:
```
The network ships with trained weights at [data/params/CNNBasic/cnn_basic.pt](data/params/CNNBasic/cnn_basic.pt), so it plays competently out of the box. If you want to improve on it, re-train via the Train page or directly:

```bash
python engine/ai/training/cnn_basic_training.py --n 200 --lr 1e-3 --eps 0.5 --opponent self
```
```

- [ ] **Step 7: Replace README.md's "Project main files" tree**

Old:
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
New:
```
engine/game/PaiShoGame.py — Core rules engine (state, legal moves, harmony/clash/ring detection, clone())

engine/ai/registry.py    — Single source of truth for every agent (UI config, training config, CLI mapping)
engine/ai/training/      — One training script per trainable agent
engine/ai/utils.py       — Shared helpers + DATA_DIR constant
engine/ai/elo.py         — Elo bookkeeping
engine/ai/logging_utils.py — logging

data/params/             — Saved weights / checkpoints (.pkl, .pt)

backend/ui/server.py     — Flask app + REST endpoints
backend/ui/simulate_manager.py — Spawns simulator.py subprocesses for the Simulate page
backend/ui/training_manager.py — Spawns training scripts, parses their stdout for the Train page
backend/simulator.py     — Headless game runner (subprocess target)

frontend/templates/      — index, simulate, train, leaderboard, guide, rules

tests/                   — test.py (engine unit tests) + test_integration.py (end-to-end)
```

- [ ] **Step 8: Update README.md's registry/agent-authoring links and code samples**

Old:
```
[ai/registry.py](ai/registry.py) is the **single source of truth** for every agent.
```
New:
```
[engine/ai/registry.py](engine/ai/registry.py) is the **single source of truth** for every agent.
```

Old:
```
Every piece below wires together through [ai/registry.py](ai/registry.py). The cleanest path is to **copy `basic_minimax` and rename** — it intentionally exercises every registry feature ([ai/classical/basic_minimax.py](ai/classical/basic_minimax.py)). There is also a walkthrough on the **Guide** page in the UI ([ui/templates/guide.html](ui/templates/guide.html)).
```
New:
```
Every piece below wires together through [engine/ai/registry.py](engine/ai/registry.py). The cleanest path is to **copy `basic_minimax` and rename** — it intentionally exercises every registry feature ([engine/ai/classical/basic_minimax.py](engine/ai/classical/basic_minimax.py)). There is also a walkthrough on the **Guide** page in the UI ([frontend/templates/guide.html](frontend/templates/guide.html)).
```

Old:
```
Create `ai/rl/my_agent.py`. It must:
```
New:
```
Create `engine/ai/rl/my_agent.py`. It must:
```

Old:
```
Create `ai/training/my_agent_training.py`, if it's needed. It must:
```
New:
```
Create `engine/ai/training/my_agent_training.py`, if it's needed. It must:
```

Old:
```
- Log progress with `ai/logging_utils.get_logger(name)` and `log_event(logger, "episode", episode=i, total=N, ...)` — the Train page reads `EVENT:{...}` lines on stdout to drive its progress bar.
```
New:
```
- Log progress with `ai.logging_utils.get_logger(name)` and `log_event(logger, "episode", episode=i, total=N, ...)` — the Train page reads `EVENT:{...}` lines on stdout to drive its progress bar.
```
(Note: this corrects `ai/logging_utils` to the `ai.logging_utils` import-module form — it was always an import reference, not a file path, and the slash form was a pre-existing documentation inconsistency being cleaned up incidentally.)

Old:
```
Append one entry to the `AGENTS` list in [ai/registry.py](ai/registry.py):
```
New:
```
Append one entry to the `AGENTS` list in [engine/ai/registry.py](engine/ai/registry.py):
```

Old (inside the registry code sample):
```python
    # Where checkpoints live. The UI shows "Model trained ✓" if this exists.
    "model_path": "ai/params/MyAgent/model.pt",

    # Training script + the knobs that drive it.
    "training_script": "ai/training/my_agent_training.py",
```
New:
```python
    # Where checkpoints live. The UI shows "Model trained ✓" if this exists.
    "model_path": "data/params/MyAgent/model.pt",

    # Training script + the knobs that drive it.
    "training_script": "engine/ai/training/my_agent_training.py",
```

Old:
```
    # Name of the stdout line parser in ui/training_manager.py.
```
New:
```
    # Name of the stdout line parser in backend/ui/training_manager.py.
```

- [ ] **Step 9: Update README.md's Testing section**

Old:
```bash
python tests/test_integration.py  # end-to-end coverage
```
New (unchanged — `tests/` stays at the repo root; verify only, no edit needed).

- [ ] **Step 10: Update frontend/templates/guide.html's path references**

Old (line 342):
```html
    <pre><code><span class="dim"># ai/rl/my_agent.py</span>
```
New:
```html
    <pre><code><span class="dim"># engine/ai/rl/my_agent.py</span>
```

Old (line 375):
```html
    <p>If your model is trainable, add a script at <code>ai/training/my_agent_training.py</code>. It should:</p>
```
New:
```html
    <p>If your model is trainable, add a script at <code>engine/ai/training/my_agent_training.py</code>. It should:</p>
```

Old (line 379):
```html
        <li>Save the trained model to <code>ai/params/MyAgent/</code>.</li>
```
New:
```html
        <li>Save the trained model to <code>data/params/MyAgent/</code>.</li>
```

Old (line 402):
```html
    <h2><span class="step-num">3</span> Register in <code>ai/registry.py</code></h2>
```
New:
```html
    <h2><span class="step-num">3</span> Register in <code>engine/ai/registry.py</code></h2>
```

Old (lines 416-417):
```html
    "model_path": "ai/params/MyAgent/model.pt",   <span class="dim"># where weights live</span>
    "training_script": "ai/training/my_agent_training.py",
```
New:
```html
    "model_path": "data/params/MyAgent/model.pt",   <span class="dim"># where weights live</span>
    "training_script": "engine/ai/training/my_agent_training.py",
```

Old (line 472):
```html
python ui/server.py
```
New:
```html
python backend/ui/server.py
```

Old (lines 481-483):
```html
        <tr><td><code>ai/rl/my_agent.py</code></td><td>New &mdash; agent class with <code>choose_action()</code></td></tr>
        <tr><td><code>ai/training/my_agent_training.py</code></td><td>New &mdash; training script with CLI args (optional)</td></tr>
        <tr><td><code>ai/registry.py</code></td><td>Edit &mdash; add one dict entry to <code>AGENTS</code></td></tr>
```
New:
```html
        <tr><td><code>engine/ai/rl/my_agent.py</code></td><td>New &mdash; agent class with <code>choose_action()</code></td></tr>
        <tr><td><code>engine/ai/training/my_agent_training.py</code></td><td>New &mdash; training script with CLI args (optional)</td></tr>
        <tr><td><code>engine/ai/registry.py</code></td><td>Edit &mdash; add one dict entry to <code>AGENTS</code></td></tr>
```

- [ ] **Step 11: Commit**

```bash
git add CLAUDE.md README.md frontend/templates/guide.html
git commit -m "Update CLAUDE.md, README.md, and guide.html for the new engine/backend/frontend/data paths"
```

---

### Task 13: End-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Fresh editable install**

```bash
pip install -e .
pip install -r requirements.txt
```
Expected: no errors.

- [ ] **Step 2: Run the engine unit tests**

Run: `python tests/test.py`
Expected: all tests pass (writes `tests/test_report.txt` as before — unaffected by this restructure since `tests/` didn't move).

- [ ] **Step 3: Run the integration tests**

Run: `python tests/test_integration.py`
Expected: all tests pass, including the `backend/simulator.py` subprocess test from Task 9.

- [ ] **Step 4: Run the Flask app locally and confirm the Play page loads**

```bash
python backend/ui/server.py > /tmp/server.log 2>&1 &
SERVER_PID=$!
sleep 2
curl -sf http://localhost:5000/ -o /dev/null && echo "OK: index served"
curl -sf http://localhost:5000/static/tiles/GB.png -o /dev/null && echo "OK: static tile served"
kill $SERVER_PID
```
Expected: both `OK:` lines print — confirms `frontend/templates` and `frontend/static` are correctly wired into Flask.

- [ ] **Step 5: Run a tiny local simulation end-to-end**

Run: `python backend/simulator.py --mode local --p1 random --p2 random --n 2 --save_results 1`
Expected: completes without error; `data/results/random_vs_random_results.csv` gets a new row (or is created).

- [ ] **Step 6: Confirm a training script resolves its checkpoint under data/**

Run: `python -c "from ai.rl.cnn_basic import MODEL_PATH; import os; print(MODEL_PATH); print(os.path.exists(MODEL_PATH))"`
Expected: path contains `data/params/CNNBasic/cnn_basic.pt` (or Windows-separator equivalent) and `True` (the weights file moved there in Task 1 and still exists).

- [ ] **Step 7: docker compose build (CPU override) as the final smoke test**

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
```
Expected: image builds successfully with no missing-file errors (confirms the Dockerfile's `COPY engine/ backend/ frontend/` matches the actual tree).

- [ ] **Step 8: Report status**

If every step above passed, the restructure is complete. If any step failed, stop and fix before considering this plan done — do not proceed to a "cleanup" commit over a broken state.

---

## Post-plan note

This plan does not touch `evaluation/`'s contents (it was empty before and after — only `backend/evaluation/` exists as an empty placeholder directory per the design). If evaluation tooling is added later, it belongs there.
