# Repo Restructure: engine / backend / frontend + Docker Compose + Tailscale

Date: 2026-07-20
Status: Approved, ready for implementation planning

## Goal

Reorganize MushiBot's file layout into clearly separated `engine/`, `backend/`,
and `frontend/` folders, add a `docker compose up` workflow that builds and
runs the app, and expose it on the user's Tailscale tailnet via a sidecar
container — all with minimal risk to existing behavior.

## Non-goals

- Not decoupling into a separate SPA + JSON API. The frontend remains
  Flask-rendered Jinja templates (`ui/templates/*.html` with inline JS/CSS);
  this is a physical file reorganization, not an architecture rewrite.
- Not renaming Python import paths (`game`, `ai`, `ui` stay as-is). Only their
  on-disk locations change.
- Not building a JS toolchain, CDN, or build step for the frontend.

## New top-level layout

```
MushiBot/
├── engine/                  # pure game logic + AI agents, zero web/subprocess deps
│   ├── game/                #   was game/        (PaiShoGame.py, notation.py)
│   └── ai/                  #   was ai/          (registry.py, elo.py, logging_utils.py,
│                             #                     utils.py, classical/, rl/, training/)
├── backend/                 # Flask app + process orchestration
│   ├── ui/                  #   was ui/*.py      (server.py, simulate_manager.py,
│                             #                     training_manager.py)
│   ├── evaluation/          #   was evaluation/   (currently empty; moves as-is)
│   └── simulator.py         #   was simulator.py
├── frontend/                # templates + static assets, served by backend/ui/server.py
│   ├── templates/           #   was ui/templates/
│   └── static/              #   was ui/static/ (tiles/*.png)
├── data/                    # runtime artifacts; Docker volume target
│   ├── params/              #   was ai/params/     (tracked config files + gitignored
│                             #                       weights, same split as today)
│   ├── checkpoints/         #   was ai/checkpoints/ (gitignored)
│   ├── saved_games/         #   was SavedGames/
│   └── results/             #   was results/
├── tests/                   # unchanged, top-level (test.py, test_integration.py, basic_tests.py)
├── scripts/                 # unchanged, top-level (distill.py)
├── Dockerfile
├── docker-compose.yml
├── docker-compose.cpu.yml   # override to disable GPU reservation
├── .env.example             # TS_AUTHKEY placeholder (committed)
├── .env                     # actual secrets (gitignored, not committed)
├── pyproject.toml
├── requirements.txt
├── README.md
├── CLAUDE.md
└── LICENSE
```

## Decisions and rationale

These were resolved with the user before writing this spec:

1. **Reorg only, not a real frontend/backend decouple.** The existing
   templates have inline JS/CSS and no build step; splitting into a true
   SPA + API would require rewriting the frontend, which is out of scope.
2. **`engine/` = `game/` + `ai/` only.** `tests/` stays its own top-level
   folder rather than nesting under `engine/`.
3. **`simulator.py` moves to `backend/`.** It has a `--mode flask` path that
   talks to the web backend, so it belongs alongside backend tooling rather
   than inside the dependency-free engine package.
4. **Runtime data consolidates into a top-level `data/` folder** (`data/params`,
   `data/checkpoints`, `data/saved_games`, `data/results`), separating code
   from generated/tracked-config artifacts and giving Docker a single volume
   mount point.
5. **`scripts/` stays top-level; `evaluation/` moves under `backend/`.**
   `distill.py` prunes the whole repo for the public mirror and isn't
   engine/backend/frontend-specific.
6. **Python import paths do not change.** `game`, `ai`, `ui` remain the
   importable package names; only `pyproject.toml`'s package-discovery
   `where` paths change. This avoids touching every `import` statement
   across the game engine, AI agents, training scripts, UI, tests, and
   simulator — the single biggest risk-reduction decision in this spec.
7. **GPU access defaults on, with a CPU-only override file.** Training uses
   PyTorch (`cnn_basic`, `nneu_basic`, `ppo`, `set_transformer`); GPU should
   be available by default but easy to disable via
   `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up`.
8. **Tailscale via sidecar container**, not host-level Tailscale. The
   `backend` service joins the `tailscale` service's network namespace
   (`network_mode: service:tailscale`), so `docker compose up` brings up
   both together and the app is reachable at a stable MagicDNS tailnet
   hostname with no public port exposure.
9. **Also reachable at `localhost:5000` on the host**, not tailnet-only, by
   publishing port 5000 on the `tailscale` service (which owns the shared
   network namespace).

## Import mechanics

`pyproject.toml`'s `[tool.setuptools.packages.find]` changes from:

```toml
where = ["."]
include = ["ai*", "game*", "ui*"]
```

to:

```toml
where = ["engine", "backend"]
include = ["ai*", "game*", "ui*"]
```

Setuptools' `find` supports multiple `where` roots and matches `include`
patterns against whichever root they're found under. This means `game` and
`ai` are discovered under `engine/`, `ui` is discovered under `backend/`, and
every existing `from game.PaiShoGame import ...` / `from ai.registry import
...` / `import ui.server` statement keeps working with **zero import-statement
edits**. `pip install -e .` makes the packages importable regardless of
current working directory, exactly as it does today.

## Path constants that must change

These are string literals, not import statements, so they're unaffected by
the import-path decision above and must be updated by hand:

- `ai/registry.py`, `ai/elo.py`, and training scripts under `ai/training/` —
  any literal `ai/params/...` or `ai/checkpoints/...` path → `data/params/...`,
  `data/checkpoints/...`.
- `ui/server.py` — `SavedGames/` and `results/` references →
  `data/saved_games/`, `data/results/`.
- `.gitignore` — `ai/checkpoints` entry → `data/checkpoints`. Extension-based
  patterns (`*.pkl`, `*.pt`, `*.npz`, `*_results.csv`) are location-agnostic
  and need no change.
- `scripts/distill.py` — `KEEP_FILES` and `REMOVE_DIRS` currently hardcode
  paths like `ai/params/CNNBasic/cnn_basic.pt` and
  `REMOVE_DIRS = ("ai/checkpoints", "scripts")`. These become
  `engine/ai/params/CNNBasic/cnn_basic.pt` and `REMOVE_DIRS = ("data/checkpoints",
  "scripts")`. Without this update, the public-mirror CI job
  ([.github/workflows/deploy.yml](../../../.github/workflows/deploy.yml))
  either breaks or silently ships the wrong file set to
  `github.com/Amitaisela/pai-sho-lab`.
- `CLAUDE.md` — the project-structure tree and run commands (e.g.
  `python ui/server.py` → `python backend/ui/server.py`,
  `python simulator.py ...` → `python backend/simulator.py ...`) must be
  updated to match the new physical layout, since it's checked-in
  documentation that currently states the old paths as fact.
- `ui/server.py` Flask app construction — needs `template_folder=` and
  `static_folder=` pointed at `../frontend/templates` and `../frontend/static`
  (Flask defaults to `templates/`/`static/` next to the app file, which no
  longer holds since templates move out to `frontend/`).

## Dockerfile

Single image serves both CPU and GPU cases — PyPI's `torch` wheel bundles its
own CUDA runtime, so the only difference between CPU and GPU operation is
which device Docker grants the container, not the image itself.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY engine/ engine/
COPY backend/ backend/
COPY frontend/ frontend/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .
EXPOSE 5000
CMD ["python", "backend/ui/server.py"]
```

## docker-compose.yml

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
      - "5000:5000"        # host localhost access, in addition to tailnet
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

## docker-compose.cpu.yml (override)

```yaml
services:
  backend:
    deploy:
      resources:
        reservations:
          devices: []
```

Usage:
- GPU (default): `docker compose up`
- CPU-only: `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up`

## .env.example (committed)

```
TS_AUTHKEY=tskey-auth-xxxxx
```

`.env` itself (with the real key) is gitignored, not committed.

## Error handling / edge cases

- If `TS_AUTHKEY` is missing or invalid, the `tailscale` container will fail
  to authenticate; `backend` (sharing its network namespace) still starts but
  is only reachable via the `ports: ["5000:5000"]` mapping on localhost, not
  via the tailnet, until a valid key is supplied and the container restarted.
- GPU reservation on a host without `nvidia-container-toolkit` installed will
  fail `docker compose up` outright — this is why the CPU override file
  exists as an explicit, documented fallback rather than a silent capability
  check.
- Moving `ai/params` to `data/params` preserves the existing split between
  git-tracked config/example files (e.g. `example_config.txt`,
  `neat-config.txt`) and gitignored generated weights (`*.pt`, `*.pkl`) —
  no change to what's tracked, only where it lives.

## Testing / verification plan

1. After moving files and updating `pyproject.toml`, run `pip install -e .`
   from repo root.
2. Run `python tests/test.py` and `python tests/test_integration.py` —
   both must pass with no import errors, confirming the `where`-path change
   correctly resolves `game`/`ai`/`ui` from their new locations.
3. Run `python backend/ui/server.py` locally (outside Docker) and confirm
   the Play page loads, templates render, and static tile images load from
   `frontend/static/`.
4. Run `python backend/simulator.py --mode local --p1 random --p2 random --n 2`
   to confirm the engine and simulator still run end-to-end from the new
   location.
5. `docker compose build && docker compose up` — confirm the container
   builds, starts, and serves on `localhost:5000`.
6. Confirm a training script's `--resume` path still finds its checkpoint
   under `data/checkpoints/...` after the move.
7. Manually inspect `scripts/distill.py`'s updated paths against the actual
   new `engine/ai/` tree to confirm the public-mirror allowlist still
   resolves correctly (a dry-run/diff, not a full CI trigger).
