import os
import subprocess
import sys
import threading
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from ai import elo  # noqa: E402
from ai.logging_utils import parse_event_line  # noqa: E402

_lock = threading.Lock()
_state = {
    "process": None,
    "status": "idle",
    "total_games": 0,
    "current_game": 0,
    "p1_wins": 0,
    "p2_wins": 0,
    "draws": 0,
    "log_lines": [],
    "log_seq": 0,
    "start_time": None,
    "error": None,
    "p1_spec": None,
    "p2_spec": None,
    "rated": True,
    "p1_key": None,
    "p2_key": None,
    "p1_elo": None,
    "p2_elo": None,
    "elo_events": [],
}

_MAX_LOG_LINES = 1000


def _build_spec(model, params):
    if not params:
        return model
    parts = []
    for k, v in params.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}={v}")
    return f"{model}:{','.join(parts)}" if parts else model


def _reader_thread(process):
    try:
        for raw in iter(process.stdout.readline, ''):
            line = raw.rstrip('\n\r')
            if not line:
                continue
            ev = parse_event_line(line)
            with _lock:
                if ev is None:
                    if len(_state["log_lines"]) >= _MAX_LOG_LINES:
                        _state["log_lines"].pop(0)
                    _state["log_lines"].append(line)
                    _state["log_seq"] += 1
                    continue

                if ev.get("event") != "game_end":
                    continue

                winner = ev.get("winner")
                _state["current_game"] = int(ev.get("game_id", _state["current_game"] + 1))
                if winner == 1:
                    _state["p1_wins"] += 1
                elif winner == 2:
                    _state["p2_wins"] += 1
                else:
                    _state["draws"] += 1
                    winner = None
                if _state.get("rated") and _state.get("p1_key") and _state.get("p2_key"):
                    try:
                        res = elo.record_game(_state["p1_key"], _state["p2_key"], winner)
                        if res:
                            res["game"] = _state["current_game"]
                            _state["p1_elo"] = res["p1_after"]
                            _state["p2_elo"] = res["p2_after"]
                            _state["elo_events"].append(res)
                            if len(_state["elo_events"]) > 500:
                                _state["elo_events"].pop(0)
                    except Exception:
                        pass
    except Exception:
        pass

    retcode = process.wait()
    with _lock:
        if _state["status"] == "stopping":
            _state["status"] = "idle"
        elif retcode == 0:
            _state["status"] = "completed"
        else:
            _state["status"] = "error"
            _state["error"] = f"Process exited with code {retcode}"


def start_simulation(p1_model, p1_params, p2_model, p2_params,
                     n_games, save_results=False, save_games=False,
                     save_period=1, verbose=True, rated=True, max_steps=1000):
    with _lock:
        if _state["status"] == "running":
            raise ValueError("Simulation is already running")

    p1_spec = _build_spec(p1_model, p1_params)
    p2_spec = _build_spec(p2_model, p2_params)

    cmd = [
        sys.executable, "-u", os.path.join(PROJECT_ROOT, "simulator.py"),
        "--mode", "local",
        "--p1", p1_spec,
        "--p2", p2_spec,
        "--n", str(int(n_games)),
        "--save_results", "1" if save_results else "0",
        "--save_games", "1" if save_games else "0",
        "--save_period", str(int(save_period)),
        "--v", "1" if verbose else "0",
        "--max_steps", str(int(max_steps)),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_ROOT,
        bufsize=1,
        env=env,
    )

    display_cmd = ["python" if i == 0 else
                   (os.path.relpath(a, PROJECT_ROOT) if os.path.isabs(a) and a.startswith(PROJECT_ROOT) else a)
                   for i, a in enumerate(cmd)]
    cmd_line = "$ " + " ".join(display_cmd)

    p1_key = p1_model
    p2_key = p2_model
    p1_elo = elo.get_rating(p1_key) if p1_key else None
    p2_elo = elo.get_rating(p2_key) if p2_key else None

    with _lock:
        _state.update({
            "process": proc,
            "status": "running",
            "total_games": int(n_games),
            "current_game": 0,
            "p1_wins": 0,
            "p2_wins": 0,
            "draws": 0,
            "log_lines": [cmd_line],
            "log_seq": 1,
            "start_time": time.time(),
            "error": None,
            "p1_spec": p1_spec,
            "p2_spec": p2_spec,
            "rated": bool(rated),
            "p1_key": p1_key,
            "p2_key": p2_key,
            "p1_elo": p1_elo,
            "p2_elo": p2_elo,
            "elo_events": [],
        })

    t = threading.Thread(target=_reader_thread, args=(proc,), daemon=True)
    t.start()

    return get_status()


def stop_simulation():
    with _lock:
        proc = _state["process"]
        if proc is None or _state["status"] != "running":
            return get_status()
        _state["status"] = "stopping"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    with _lock:
        _state["status"] = "idle"
        _state["process"] = None

    return get_status()


def get_log_tail(since_seq):
    with _lock:
        seq = _state["log_seq"]
        lines = _state["log_lines"]
        if since_seq >= seq:
            return [], seq
        new_count = min(seq - since_seq, len(lines))
        return list(lines[-new_count:]), seq


def get_status():
    with _lock:
        elapsed = None
        if _state["start_time"] and _state["status"] == "running":
            elapsed = round(time.time() - _state["start_time"], 1)
        elif _state["start_time"]:
            elapsed = round(time.time() - _state["start_time"], 1)
        return {
            "status": _state["status"],
            "total_games": _state["total_games"],
            "current_game": _state["current_game"],
            "p1_wins": _state["p1_wins"],
            "p2_wins": _state["p2_wins"],
            "draws": _state["draws"],
            "log_lines": list(_state["log_lines"]),
            "log_seq": _state["log_seq"],
            "elapsed": elapsed,
            "error": _state["error"],
            "p1_spec": _state["p1_spec"],
            "p2_spec": _state["p2_spec"],
            "rated": _state.get("rated", True),
            "p1_key": _state.get("p1_key"),
            "p2_key": _state.get("p2_key"),
            "p1_elo": _state.get("p1_elo"),
            "p2_elo": _state.get("p2_elo"),
            "elo_events": list(_state.get("elo_events", [])),
        }
