import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from ai.registry import get_agent, trainable_agents
from ai.logging_utils import parse_event_line

_lock = threading.Lock()
_training_state = {
    "process": None,
    "model": None,
    "status": "idle",
    "total_episodes": 0,
    "current_episode": 0,
    "log_lines": [],
    "log_seq": 0,
    "parsed_stats": {},
    "start_time": None,
    "error": None,
}

MODEL_PATHS = {}
for _a in trainable_agents():
    if _a.get("model_path"):
        MODEL_PATHS[_a["key"]] = os.path.join(PROJECT_ROOT, _a["model_path"])

_MAX_LOG_LINES = 500


_RE_MC = re.compile(
    r"Episode ([\d,]+)/([\d,]+) \| Eps: ([\d.]+) \| (.+?) in ([\d,]+) steps \| "
    r"Q-Table: ([\d,]+) \(([^)]+)\) \| Took ([\d.]+)s"
)
_RE_TD = re.compile(
    r"Episode ([\d,]+)/([\d,]+) \| "
    r"eps=([\d.]+) \| (.+?) in (\d+) steps \| "
    r"W1=(\d+) W2=(\d+) D=(\d+) \| "
    r"([\d.]+)s"
)
_RE_NEAT_GEN = re.compile(r"Generation\s+(\d+)", re.IGNORECASE)
_RE_PPO = re.compile(
    r"Episode ([\d,]+)/([\d,]+) \| "
    r"eps=([\d.]+) \| (.+?) in (\d+) steps \| "
    r"W1=(\d+) W2=(\d+) D=(\d+) \| "
    r"PL=([\d.]+) VL=([\d.]+) \| "
    r"params=([\d,]+) \| "
    r"([\d.]+)s"
)
_RE_BASIC_MINIMAX = re.compile(
    r"Episode ([\d,]+)/([\d,]+) \| "
    r"eps=([\d.]+) \| (.+?) in (\d+) steps \| "
    r"W1=(\d+) W2=(\d+) D=(\d+) \| "
    r"wr=([\d.]+) \| weights=([\d,]+) \| "
    r"([\d.]+)s"
)


def _extract_mc(m):
    return {
        "episode": int(m.group(1).replace(",", "")),
        "total": int(m.group(2).replace(",", "")),
        "epsilon": float(m.group(3)),
        "outcome": m.group(4),
        "steps": int(m.group(5).replace(",", "")),
        "q_table_size": int(m.group(6).replace(",", "")),
        "q_table_delta": m.group(7).strip(),
        "episode_time": float(m.group(8)),
    }

def _extract_td(m):
    return {
        "episode": int(m.group(1).replace(",", "")),
        "total": int(m.group(2).replace(",", "")),
        "epsilon": float(m.group(3)),
        "outcome": m.group(4),
        "steps": int(m.group(5)),
        "p1_wins": int(m.group(6)),
        "p2_wins": int(m.group(7)),
        "draws": int(m.group(8)),
        "episode_time": float(m.group(9)),
    }

def _extract_neat(m):
    return {"episode": int(m.group(1))}

def _extract_ppo(m):
    return {
        "episode": int(m.group(1).replace(",", "")),
        "total": int(m.group(2).replace(",", "")),
        "epsilon": float(m.group(3)),
        "outcome": m.group(4),
        "steps": int(m.group(5)),
        "p1_wins": int(m.group(6)),
        "p2_wins": int(m.group(7)),
        "draws": int(m.group(8)),
        "policy_loss": float(m.group(9)),
        "value_loss": float(m.group(10)),
        "params": int(m.group(11).replace(",", "")),
        "episode_time": float(m.group(12)),
    }

def _extract_basic_minimax(m):
    return {
        "episode": int(m.group(1).replace(",", "")),
        "total": int(m.group(2).replace(",", "")),
        "epsilon": float(m.group(3)),
        "outcome": m.group(4),
        "steps": int(m.group(5)),
        "p1_wins": int(m.group(6)),
        "p2_wins": int(m.group(7)),
        "draws": int(m.group(8)),
        "win_rate": float(m.group(9)),
        "params": int(m.group(10).replace(",", "")),
        "episode_time": float(m.group(11)),
    }

_LOG_PARSERS = {
    "monte_carlo":   (_RE_MC, _extract_mc),
    "td_learning":   (_RE_TD, _extract_td),
    "neat":          (_RE_NEAT_GEN, _extract_neat),
    "ppo":           (_RE_PPO, _extract_ppo),
    "basic_minimax": (_RE_BASIC_MINIMAX, _extract_basic_minimax),
}


def _parse_line(model, line):
    """Parse one stdout line into stats. Structured EVENT lines win; regex is a fallback."""
    ev = parse_event_line(line)
    if ev is not None:
        evt = ev.get("event")
        payload = {k: v for k, v in ev.items() if k not in ("ts", "event")}
        if evt == "episode":
            return payload
        if evt == "epoch":
            # Pipeline training phase: drive progress bar from epoch number.
            payload["phase"] = "train"
            payload["episode"] = ev.get("epoch")
            return payload
        if evt == "gen_game":
            # Pipeline generation phase: drive progress bar from game number.
            payload["phase"] = "generate"
            payload["episode"] = ev.get("game")
            return payload
        if evt == "eval":
            # Strength eval result; merged into stats so the UI can show winrate.
            return payload
    entry = get_agent(model)
    parser_key = entry["log_parser"] if entry else model
    parser = _LOG_PARSERS.get(parser_key)
    if not parser:
        return None
    regex, extractor = parser
    m = regex.search(line)
    if m:
        return extractor(m)
    return None


def _reader_thread(process, model):
    try:
        for raw in iter(process.stdout.readline, ''):
            line = raw.rstrip('\n\r')
            if not line:
                continue
            parsed = _parse_line(model, line)
            is_event = line.startswith("EVENT:")
            with _lock:
                # Raw EVENT:{...} lines are machine-only; users see the human-readable line above.
                if not is_event:
                    if len(_training_state["log_lines"]) >= _MAX_LOG_LINES:
                        _training_state["log_lines"].pop(0)
                    _training_state["log_lines"].append(line)
                    _training_state["log_seq"] += 1
                if parsed:
                    _training_state["parsed_stats"].update(parsed)
                    if "episode" in parsed:
                        _training_state["current_episode"] = parsed["episode"]
                    if "total" in parsed:
                        _training_state["total_episodes"] = parsed["total"]
    except Exception:
        pass

    retcode = process.wait()
    with _lock:
        if _training_state["status"] == "stopping":
            _training_state["status"] = "idle"
        elif retcode == 0:
            _training_state["status"] = "completed"
        else:
            _training_state["status"] = "error"
            _training_state["error"] = f"Process exited with code {retcode}"


def start_training(model, params):
    with _lock:
        if _training_state["status"] == "running":
            raise ValueError("Training is already running")

    entry = get_agent(model)
    if not entry or not entry.get("training_script"):
        raise ValueError(f"Unknown or non-trainable model: {model}")

    python = sys.executable
    cmd = [python, "-u", entry["training_script"]]

    cli_map = entry.get("training_cli_map", {})
    training_params = entry.get("training_params", [])

    defaults = {tp["key"]: tp["default"] for tp in training_params}

    for param_key, cli_flag in cli_map.items():
        value = params.get(param_key, defaults.get(param_key))
        if value is None:
            continue
        # resume accepts either a bool (on/off) or a path to a checkpoint.
        if param_key == "resume":
            if isinstance(value, bool) or (isinstance(value, (int, float)) and value in (0, 1)):
                cmd += [cli_flag, "1" if value else "0"]
            elif value:
                cmd += [cli_flag, str(value)]
        else:
            cmd += [cli_flag, str(value)]

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

    ep_key = entry.get("total_episodes_key")
    total = params.get(ep_key, defaults.get(ep_key, 0)) if ep_key else 0

    with _lock:
        _training_state.update({
            "process": proc,
            "model": model,
            "status": "running",
            "total_episodes": int(total),
            "current_episode": 0,
            "log_lines": [cmd_line],
            "log_seq": 1,
            "parsed_stats": {},
            "start_time": time.time(),
            "error": None,
        })

    t = threading.Thread(target=_reader_thread, args=(proc, model), daemon=True)
    t.start()

    return get_status()


def stop_training():
    with _lock:
        proc = _training_state["process"]
        if proc is None or _training_state["status"] != "running":
            return get_status()
        _training_state["status"] = "stopping"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    with _lock:
        _training_state["status"] = "idle"
        _training_state["process"] = None

    return get_status()


def get_log_tail(since_seq):
    with _lock:
        seq = _training_state["log_seq"]
        lines = _training_state["log_lines"]
        if since_seq >= seq:
            return [], seq
        new_count = min(seq - since_seq, len(lines))
        return list(lines[-new_count:]), seq


def get_status():
    with _lock:
        elapsed = None
        if _training_state["start_time"] and _training_state["status"] == "running":
            elapsed = round(time.time() - _training_state["start_time"], 1)
        return {
            "model": _training_state["model"],
            "status": _training_state["status"],
            "total_episodes": _training_state["total_episodes"],
            "current_episode": _training_state["current_episode"],
            "log_lines": list(_training_state["log_lines"]),
            "log_seq": _training_state["log_seq"],
            "parsed_stats": dict(_training_state["parsed_stats"]),
            "elapsed": elapsed,
            "error": _training_state["error"],
        }


def get_model_info(model):
    path = MODEL_PATHS.get(model)
    if not path:
        return {"exists": False, "path": None}
    exists = os.path.exists(path)
    info = {
        "exists": exists,
        "path": os.path.relpath(path, PROJECT_ROOT),
    }
    if exists:
        stat = os.stat(path)
        info["size_bytes"] = stat.st_size
        info["size_human"] = _human_size(stat.st_size)
        info["modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return info


def _human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"
