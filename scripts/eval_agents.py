"""Evaluate agents against an opponent with alternating seats.

score = wins + 0.5 * (ties + timeouts). Refuses to score an agent whose
weights didn't load: either the weights file is missing outright (checked
directly against the agent module's own path constant, since some agents
fall back to a fresh/default state with no printed warning at all), or the
file exists but didn't fit the current network shape (some agents print a
"don't fit" warning and keep randomly-initialised weights instead of
raising).

Usage: python scripts/eval_agents.py nneu_basic mcts --vs random --games 100 --engine rust --json out.json
"""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulator import get_action, load_model, parse_model_spec  # noqa: E402
from Agents.registry import get_agent  # noqa: E402
from engine.engine_select import game_class  # noqa: E402

# Messages each trained agent's load path prints when it could not use its
# saved weights and fell back to a default/random state. Checked case-
# insensitively against anything the load captured on stdout or logging.
# This alone is not sufficient (see _expected_weights_path below for the
# silent-fallback case), but it is what catches a shape mismatch on an
# existing file trained before the official-rules switch.
_FALLBACK_MARKERS = (
    "don't fit",
    "not found",
    "using random init",
    "random initialization",
    "no q-table found",
    "falling back to random",
)


def _agent_module(entry):
    return importlib.import_module(entry["module"])


def _expected_weights_path(name, entry):
    """Best-effort path to the weights/genome file this agent loads from.

    Agents expose a module-level MODEL_PATH/WEIGHTS_PATH/GENOME_PATH
    constant. Returns None when the agent has no persisted weights (e.g.
    random, basic_minimax), in which case no fallback check applies.
    """
    if entry["kind"] != "class":
        return None
    mod = _agent_module(entry)
    for attr in ("MODEL_PATH", "WEIGHTS_PATH", "GENOME_PATH"):
        path = getattr(mod, attr, None)
        if isinstance(path, str):
            return path
    return None


def _weights_info(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    return {"path": path, "mtime": os.path.getmtime(path), "sha256": hashlib.sha256(data).hexdigest()[:16]}


def _load(spec, allow_untrained):
    """Instantiate `spec`'s agent, refusing a silent or logged fallback.

    Two independent checks, since neither alone covers every agent:
      1. If the module tells us where its weights should be and that path
         doesn't exist, refuse before even trying to load (catches a
         *silent* fallback that prints nothing).
      2. Capture stdout + logging during the load and scan for a fallback
         message (catches a shape-mismatch fallback where the file exists
         but doesn't fit).
    """
    name, params = parse_model_spec(spec)
    entry = get_agent(name)
    if not entry:
        raise ValueError(f"unknown agent: {name!r}")

    expected_path = _expected_weights_path(name, entry)
    if expected_path and not os.path.exists(expected_path) and not allow_untrained:
        raise RuntimeError(
            f"{spec}: weights not found at {expected_path!r}; train it first or pass --allow-untrained"
        )

    stdout_buf = io.StringIO()
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        with contextlib.redirect_stdout(stdout_buf):
            agent = load_model(name, params, verbose=False)
    finally:
        root_logger.removeHandler(handler)

    out = (stdout_buf.getvalue() + log_buf.getvalue()).lower()
    if not allow_untrained and any(marker in out for marker in _FALLBACK_MARKERS):
        raise RuntimeError(f"{spec}: weights did not load ({out.strip()!r}); train it first or pass --allow-untrained")

    return name, params, agent, expected_path


def _play(G, seat_names, seat_params, seat_agents, max_steps):
    g = G()
    turn = 0
    while g.winner is None:
        turn += 1
        if turn > max_steps:
            break
        legal = g.get_legal_actions()
        if not legal:
            break
        p = g.current_player
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                action = get_action(g, seat_names[p], legal, seat_agents[p], seat_params[p], False)
        except Exception as e:  # noqa: BLE001 - a crashing agent forfeits, like an illegal move
            action, why = None, f"raised {e!r}"
        else:
            why = f"returned {action!r}"
        if action is None or tuple(action) not in {tuple(a) for a in legal}:
            # An agent that can't produce a legal move forfeits the game (end_reason
            # 'resign': no agent resigns on its own) instead of crashing the evaluation.
            print(f"FORFEIT: {seat_names[p]} (player {p}) {why}", file=sys.stderr, flush=True)
            g.resign(p)
            break
        g.step(action)
    return g


def evaluate(agent_spec, opponent_spec, games, engine="rust", max_steps=1000, seed=0, allow_untrained=False):
    random.seed(seed)
    G = game_class(engine)
    a_name, a_params, a_agent, a_path = _load(agent_spec, allow_untrained)
    o_name, o_params, o_agent, _ = _load(opponent_spec, True)

    r = {
        "agent": agent_spec, "opponent": opponent_spec, "games": games,
        "wins": 0, "losses": 0, "ties": 0, "timeouts": 0,
        "as_p1": {"wins": 0, "losses": 0, "ties": 0, "timeouts": 0},
        "as_p2": {"wins": 0, "losses": 0, "ties": 0, "timeouts": 0},
        "end_reasons": {}, "weights": _weights_info(a_path),
    }

    t0 = time.time()
    for i in range(games):
        seat = 1 if i % 2 == 0 else 2
        names = {seat: a_name, 3 - seat: o_name}
        params = {seat: a_params, 3 - seat: o_params}
        agents = {seat: a_agent, 3 - seat: o_agent}
        g = _play(G, names, params, agents, max_steps)

        if g.winner is None:
            key = "timeouts"
        elif g.winner == 0:
            key = "ties"
        elif g.winner == seat:
            key = "wins"
        else:
            key = "losses"
        r[key] += 1
        r["as_p1" if seat == 1 else "as_p2"][key] += 1

        reason = getattr(g, "end_reason", None) or "timeout"
        r["end_reasons"][reason] = r["end_reasons"].get(reason, 0) + 1

    r["seconds"] = round(time.time() - t0, 1)
    r["score"] = r["wins"] + 0.5 * (r["ties"] + r["timeouts"])
    r["score_pct"] = round(100.0 * r["score"] / games, 1) if games else 0.0
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="+")
    ap.add_argument("--vs", default="random")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--engine", choices=["python", "rust"], default="rust")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-untrained", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    results = []
    for spec in args.agents:
        r = evaluate(spec, args.vs, args.games, args.engine, args.max_steps, args.seed, args.allow_untrained)
        results.append(r)
        print(f"{spec:>16} vs {args.vs:<14} {r['score_pct']:5.1f}%  W{r['wins']} L{r['losses']} "
              f"T{r['ties']} TO{r['timeouts']}  ({r['seconds']}s)  {r['end_reasons']}", flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
