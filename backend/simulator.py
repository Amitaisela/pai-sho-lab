import requests
import sys
import time
import os
import json
import argparse
import types
from tqdm import tqdm

# engine_select.py is a loose module directly under engine/, not a package matched by
# pyproject.toml's [tool.setuptools.packages.find] include patterns - see server.py's
# identical comment on this.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'engine'))

from PythonEngine.notation import game_to_psn
from Agents.actions import action_kind, arrange_parts, plant_parts
from Agents.registry import get_agent
from Agents.agent_loader import instantiate, act
from Agents.logging_utils import get_logger, log_event
from engine_select import DEFAULT_ENGINE, game_class

# --mode flask drives a running Lab: LAB_URL, else http://127.0.0.1:<LAB_PORT or 5001>.
SERVER_URL = os.environ.get("LAB_URL") or f"http://127.0.0.1:{os.environ.get('LAB_PORT', '5001')}"


def _auth_headers():
    """X-API-Token for the Lab's mutating endpoints, when MUSHIBOT_API_TOKEN is set."""
    token = os.environ.get("MUSHIBOT_API_TOKEN", "").strip()
    return {"X-API-Token": token} if token else {}


GAME_ID = "default"

log = get_logger("simulator")


def _coerce_scalar(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_model_spec(spec):
    """Parse "modelname" or "modelname:key=val,..." into (name, params)."""
    if ':' not in spec:
        name = spec.strip()
        if not name:
            raise ValueError(f"empty agent spec: {spec!r}")
        return name, {}
    name, params_str = spec.split(':', 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty agent name in spec: {spec!r}")
    params = {}
    for kv in params_str.split(','):
        kv = kv.strip()
        if not kv:
            continue
        if '=' not in kv:
            raise ValueError(f"bad param {kv!r} in agent spec {spec!r} (expected key=value)")
        k, v = kv.split('=', 1)
        k = k.strip()
        if not k:
            raise ValueError(f"empty key in agent spec {spec!r}")
        params[k] = _coerce_scalar(v.strip())
    return name, params


def load_model(model, params=None, verbose=True):
    params = params or {}
    entry = get_agent(model)
    if not entry:
        raise ValueError(f"unknown agent: {model!r}")
    if entry["kind"] != "class":
        return None

    if verbose:
        log.info(f"Loading {entry['display_name']} agent...")

    return instantiate(entry, params=params)


def get_action(game, model, legal_actions, agent=None, params=None, verbose=True):
    entry = get_agent(model)
    if not entry:
        raise ValueError(f"Model '{model}' is not registered in Agents/registry.py.")

    if entry["kind"] == "class" and agent is None:
        raise ValueError(f"{entry['display_name']} agent instance not provided to get_action.")

    return act(entry, game, agent=agent, params=params, verbose=verbose, legal_actions=legal_actions)


def save_result_to_csv(p1_model, p2_model, winner, turn_count, duration):
    p1_name = p1_model.split(':')[0]
    p2_name = p2_model.split(':')[0]
    file_path = f"data/results/{p1_name}_vs_{p2_name}_results.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0

    with open(file_path, "a") as f:
        if write_header:
            f.write("timestamp,winner,turns,duration_seconds,p1_model,p2_model\n")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # 0 = tie, empty = timeout (no winner) -- keep these distinguishable in the CSV.
        winner_status = '' if winner is None else winner
        f.write(f"{timestamp},{winner_status},{turn_count},{duration:.2f},{p1_model},{p2_model}\n")


def save_game_to_file(game, game_id, p1_spec, p2_spec):
    p1_name = p1_spec.split(':')[0]
    p2_name = p2_spec.split(':')[0]
    dir_path = f"data/saved_games/{p1_name}_vs_{p2_name}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"game_{game_id}.json")
    save_data = game.to_save_dict(p1_name=p1_spec, p2_name=p2_spec)
    with open(file_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    log.info(f"Game saved to {file_path}")


def save_game_to_psn(game, game_id, p1_spec, p2_spec):
    p1_name = p1_spec.split(':')[0]
    p2_name = p2_spec.split(':')[0]
    dir_path = f"data/saved_games/{p1_name}_vs_{p2_name}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"game_{game_id}.psn")
    with open(file_path, 'w') as f:
        f.write(game_to_psn(game, p1_name=p1_spec, p2_name=p2_spec))
    log.info(f"Game saved to {file_path}")


def play_move(action):
    kind = action_kind(action)
    if kind == 'plant':
        flower, r, c, displace = plant_parts(action)
        payload = {"flower": flower, "row": r, "col": c}
        if displace is not None:
            payload['displace_row'], payload['displace_col'] = displace
        res = requests.post(f"{SERVER_URL}/api/plant/{GAME_ID}", json=payload, headers=_auth_headers())
    elif kind == 'arrange':
        fr, fc, tr, tc = arrange_parts(action)
        payload = {"from_row": fr, "from_col": fc, "to_row": tr, "to_col": tc}
        res = requests.post(f"{SERVER_URL}/api/arrange/{GAME_ID}", json=payload, headers=_auth_headers())
    else:
        res = requests.post(f"{SERVER_URL}/api/skip_bonus/{GAME_ID}", headers=_auth_headers())
    return res.json()


_REASON_LABELS = {'ring': 'harmony_ring', 'last_basic_flower': 'last_basic_flower',
                  'no_moves': 'no_moves', 'resign': 'resign'}


def _win_reason(game):
    """Stable label for why a finished game ended, from the engine's end_reason."""
    return _REASON_LABELS.get(getattr(game, 'end_reason', None))


def print_report(results, p1_spec, p2_spec, total_time):
    p1_name = p1_spec.split(':')[0]
    p2_name = p2_spec.split(':')[0]
    n = len(results)

    p1_wins = sum(1 for r in results if r['winner'] == 1)
    p2_wins = sum(1 for r in results if r['winner'] == 2)
    # 'draws' keeps its original (pre-fix) meaning for compatibility: any game
    # that didn't end with a 1/2 winner, whether a genuine tie (winner == 0)
    # or a simulator-side timeout (winner is None, max_steps hit before the
    # engine reached an end state). 'ties' and 'timeouts' below split those
    # two apart for callers that care about the distinction.
    draws = sum(1 for r in results if r['winner'] is None or r['winner'] == 0)
    ties = sum(1 for r in results if r['winner'] == 0)
    timeouts = sum(1 for r in results if r['winner'] is None)

    # Tally every end_reason the engine can report (not just ring/last_basic_flower),
    # so resigns and no-legal-moves endings show up in the summary too.
    win_reason_counts = {1: {label: 0 for label in _REASON_LABELS.values()},
                         2: {label: 0 for label in _REASON_LABELS.values()}}
    for r in results:
        w = r.get('winner')
        if w not in (1, 2):
            continue
        kind = _win_reason(types.SimpleNamespace(end_reason=r.get('end_reason')))
        if kind in win_reason_counts[w]:
            win_reason_counts[w][kind] += 1

    turns = [r['turns'] for r in results]
    durations = [r['duration'] for r in results]

    avg_turns = sum(turns) / n
    min_turns = min(turns)
    max_turns = max(turns)
    avg_duration = sum(durations) / n

    log.info(f"\n{'=' * 50}")
    log.info("  SIMULATION REPORT")
    log.info(f"{'=' * 50}")
    log.info(f"  Matchup:  P1 ({p1_name}) vs P2 ({p2_name})")
    log.info(f"  Games played: {n}")
    log.info(f"{'─' * 50}")
    log.info(f"  P1 ({p1_name}) wins:  {p1_wins:>4}  ({p1_wins / n * 100:.1f}%)")
    log.info(f"    by Harmony Ring:         {win_reason_counts[1]['harmony_ring']:>4}")
    log.info(f"    by Last Basic Flower:    {win_reason_counts[1]['last_basic_flower']:>4}")
    log.info(f"    by No Legal Moves:       {win_reason_counts[1]['no_moves']:>4}")
    log.info(f"    by Resignation:          {win_reason_counts[1]['resign']:>4}")
    log.info(f"  P2 ({p2_name}) wins:  {p2_wins:>4}  ({p2_wins / n * 100:.1f}%)")
    log.info(f"    by Harmony Ring:         {win_reason_counts[2]['harmony_ring']:>4}")
    log.info(f"    by Last Basic Flower:    {win_reason_counts[2]['last_basic_flower']:>4}")
    log.info(f"    by No Legal Moves:       {win_reason_counts[2]['no_moves']:>4}")
    log.info(f"    by Resignation:          {win_reason_counts[2]['resign']:>4}")
    log.info(f"  Draws/Stalemates:     {draws:>4}  ({draws / n * 100:.1f}%)")
    log.info(f"    Ties:                    {ties:>4}")
    log.info(f"    Timeouts:                {timeouts:>4}")
    log.info(f"{'─' * 50}")
    log.info(f"  Turns   - avg: {avg_turns:.1f}  min: {min_turns}  max: {max_turns}")
    log.info(f"  Game duration - avg: {avg_duration:.2f}s  total: {total_time:.2f}s")
    if n > 1:
        most_turns = max(results, key=lambda r: r['turns'])
        fewest_turns = min(results, key=lambda r: r['turns'])
        longest_time = max(results, key=lambda r: r['duration'])
        shortest_time = min(results, key=lambda r: r['duration'])
        log.info(f"  Most turns:    #{most_turns['id']} ({most_turns['turns']} turns, {most_turns['duration']:.2f}s)")
        log.info(
            f"  Fewest turns:  #{fewest_turns['id']} ({fewest_turns['turns']} turns, {fewest_turns['duration']:.2f}s)")
        log.info(
            f"  Longest game:  #{longest_time['id']} ({longest_time['duration']:.2f}s, {longest_time['turns']} turns)")
        log.info(
            f"  Shortest game: #{shortest_time['id']} ({shortest_time['duration']:.2f}s, {shortest_time['turns']} turns)")
    log.info(f"{'=' * 50}\n")
    log_event(log, "match_summary",
              p1=p1_spec, p2=p2_spec, games=n,
              p1_wins=p1_wins, p2_wins=p2_wins, draws=draws,
              p1_harmony_ring_wins=win_reason_counts[1]['harmony_ring'],
              p1_last_basic_flower_wins=win_reason_counts[1]['last_basic_flower'],
              p2_harmony_ring_wins=win_reason_counts[2]['harmony_ring'],
              p2_last_basic_flower_wins=win_reason_counts[2]['last_basic_flower'],
              # New: every end_reason, plus ties/timeouts split out of 'draws'.
              p1_no_moves_wins=win_reason_counts[1]['no_moves'],
              p1_resign_wins=win_reason_counts[1]['resign'],
              p2_no_moves_wins=win_reason_counts[2]['no_moves'],
              p2_resign_wins=win_reason_counts[2]['resign'],
              ties=ties, timeouts=timeouts,
              avg_turns=round(avg_turns, 2),
              min_turns=min_turns, max_turns=max_turns,
              avg_duration=round(avg_duration, 3),
              total_time=round(total_time, 3))


def run_flask(iterations, p1_spec, p2_spec, save, delay, verbose,
              save_games=False, save_period=1, max_steps=1000, engine=DEFAULT_ENGINE):
    p1_name, p1_params = parse_model_spec(p1_spec)
    p2_name, p2_params = parse_model_spec(p2_spec)

    log.info(f"Connecting to Skud Pai Sho server at {SERVER_URL}...")
    log.info(f"Matchup: Player 1 ({p1_spec}) vs Player 2 ({p2_spec})")

    requests.post(f"{SERVER_URL}/api/set_agents", json={"p1": p1_name, "p2": p2_name}, headers=_auth_headers())
    agent_p1 = load_model(p1_name, p1_params, verbose)
    agent_p2 = load_model(p2_name, p2_params, verbose)

    overall_time = time.time()
    results = []
    p1_wins = p2_wins = 0

    pbar = tqdm(range(iterations), desc="Games", disable=None, dynamic_ncols=True)
    for i in pbar:
        start_time = time.time()
        turn_count = 0
        last_player_turn = None

        requests.post(f"{SERVER_URL}/api/new_game", json={"engine": engine}, headers=_auth_headers())
        log.info(f"\n=== Starting Game {i + 1}/{iterations} ===")

        while True:
            try:
                response = requests.get(f"{SERVER_URL}/api/state/{GAME_ID}", headers=_auth_headers())
                state = response.json()['state']
            except Exception as e:
                log.info(f"Error connecting to server: {e}. Retrying in 2 seconds...")
                time.sleep(2)
                continue

            if state['winner'] is not None:
                reason = state.get('message', '') or ''
                log.info(f"\n{'=' * 30}\nGAME OVER! Winner: {state['winner']}\nReason: {reason}\n{'=' * 30}")
                break

            current_p = state['current_player']
            current_name = p1_name if current_p == 1 else p2_name
            current_params = p1_params if current_p == 1 else p2_params

            if current_name.lower() == "human":
                if last_player_turn != current_p:
                    log.info(f"Waiting for Player {current_p} (Human) to play via the UI...")
                    last_player_turn = current_p
                time.sleep(1)
                continue

            if last_player_turn != current_p:
                log.info(f"AI Player {current_p} ({current_name}) is thinking...")
                last_player_turn = current_p
            elif state.get('bonus_turn') and verbose:
                log.info(f"  Bonus turn for Player {current_p}!")

            # state['engine'] reflects whatever the server actually used for /api/new_game,
            # which is authoritative over the `engine` this call was started with.
            local_game = game_class(state.get('engine', engine)).from_dict(state)
            legal_actions = local_game.get_legal_actions()

            if not legal_actions:
                log.info("No legal moves available. Stalemate!")
                break
            if turn_count >= max_steps:
                log.info(f"Stalemate! max length reached ({max_steps} steps).")
                break

            agent = agent_p1 if current_p == 1 else agent_p2
            best_move = get_action(local_game, current_name, legal_actions, agent, current_params, verbose=verbose)

            if best_move:
                log.info(f"AI performs: {best_move}")
                play_move(best_move)
                turn_count += 1
            else:
                log.info("Error: AI returned no move.")
                break

            time.sleep(delay)

        duration = time.time() - start_time
        game_message = state.get('message', '') or ''
        end_reason = state.get('end_reason')
        results.append({'id': i + 1, 'winner': state.get('winner'), 'turns': turn_count,
                        'duration': duration, 'message': game_message, 'end_reason': end_reason})
        log_event(log, "game_end", mode="flask", game_id=i + 1,
                  p1=p1_spec, p2=p2_spec,
                  winner=state.get('winner'), turns=turn_count,
                  duration=round(duration, 3),
                  reason=game_message,
                  win_reason=_win_reason(types.SimpleNamespace(end_reason=end_reason))
                  if state.get('winner') in (1, 2) else None)
        if state.get('winner') == 1:
            p1_wins += 1
        elif state.get('winner') == 2:
            p2_wins += 1
        pbar.set_postfix(p1=p1_wins, p2=p2_wins)

        if save:
            save_result_to_csv(p1_spec, p2_spec, state['winner'], turn_count, duration)

        if save_games and (i + 1) % save_period == 0:
            try:
                resp = requests.get(f"{SERVER_URL}/api/save/{GAME_ID}", headers=_auth_headers())
                save_data = resp.json()
                p1n = p1_spec.split(':')[0]
                p2n = p2_spec.split(':')[0]
                dir_path = f"data/saved_games/{p1n}_vs_{p2n}"
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, f"game_{i + 1}.json")
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                log.info(f"Game saved to {file_path}")
            except Exception as e:
                log.info(f"Warning: Could not save game: {e}")

    print_report(results, p1_spec, p2_spec, time.time() - overall_time)


def play_single_local_game(game_id, p1_name, p2_name, p1_params, p2_params,
                           agent_p1, agent_p2, verbose=True, max_steps=1000, engine=DEFAULT_ENGINE):
    start_time = time.time()
    game = game_class(engine)()
    turn_count = 0
    if verbose:
        log.info(f"Game {game_id} started...")

    while game.winner is None:
        turn_count += 1
        p = game.current_player
        name = p1_name if p == 1 else p2_name
        params = p1_params if p == 1 else p2_params
        agent = agent_p1 if p == 1 else agent_p2

        if verbose:
            log.info(f"\n--- Game {game_id} | Turn {turn_count} | Player {p}'s ({name}) move ---")

        legal_actions = game.get_legal_actions()
        if not legal_actions or turn_count >= max_steps:
            end_message = "max length reached." if turn_count >= max_steps else "No legal moves available."
            if verbose: log.info(f"Stalemate! {end_message}")
            break

        action = get_action(game, name, legal_actions, agent, params, verbose)
        if verbose: log.info(f"Player {p} ({name}) plays: {action}")
        game.step(action)

    duration = time.time() - start_time
    return game_id, game.winner, getattr(game, 'message', ''), turn_count, duration, game


def run_local(iterations, p1_spec, p2_spec, save, verbose=True, save_games=False, save_period=1, save_psn=False,
              max_steps=1000, engine=DEFAULT_ENGINE):
    p1_name, p1_params = parse_model_spec(p1_spec)
    p2_name, p2_params = parse_model_spec(p2_spec)

    overall_time = time.time()
    log.info(f"Starting local games: Player 1 ({p1_spec}) vs Player 2 ({p2_spec})")

    agent_p1 = load_model(p1_name, p1_params, verbose=verbose)
    agent_p2 = load_model(p2_name, p2_params, verbose=verbose)

    results = []
    p1_wins = p2_wins = 0

    pbar = tqdm(range(iterations), desc="Games", disable=None, dynamic_ncols=True)
    for i in pbar:
        game_id, winner, message, turn_count, duration, game = play_single_local_game(
            game_id=i + 1,
            p1_name=p1_name, p2_name=p2_name,
            p1_params=p1_params, p2_params=p2_params,
            agent_p1=agent_p1, agent_p2=agent_p2,
            verbose=verbose, max_steps=max_steps, engine=engine,
        )
        log.info(f"\n{'=' * 30}\nGAME {game_id} OVER\n{'=' * 30}")
        if winner in (1, 2):
            log.info(f"WINNER: Player {winner}\nMessage: {message}")
        elif winner == 0:
            log.info("It's a tie!")
        else:
            log.info("Stalemate! (timeout, no winner)")

        end_reason = getattr(game, 'end_reason', None)
        results.append({'id': game_id, 'winner': winner, 'turns': turn_count,
                        'duration': duration, 'message': message or '', 'end_reason': end_reason})
        log_event(log, "game_end", mode="local", game_id=game_id,
                  p1=p1_spec, p2=p2_spec,
                  winner=winner, turns=turn_count,
                  duration=round(duration, 3),
                  reason=message or '',
                  win_reason=_win_reason(game) if winner in (1, 2) else None)
        if winner == 1:
            p1_wins += 1
        elif winner == 2:
            p2_wins += 1
        pbar.set_postfix(p1=p1_wins, p2=p2_wins)

        if save:
            save_result_to_csv(p1_spec, p2_spec, winner, turn_count, duration)

        if save_games and (i + 1) % save_period == 0:
            save_game_to_file(game, game_id, p1_spec, p2_spec)
        if save_psn and (i + 1) % save_period == 0:
            save_game_to_psn(game, game_id, p1_spec, p2_spec)

    print_report(results, p1_spec, p2_spec, time.time() - overall_time)


def parse_params():
    parser = argparse.ArgumentParser(
        description="Skud Pai Sho bot matchups.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--mode", type=str, choices=['flask', 'local'], default='flask',
                        help="Execution mode: flask, local")
    parser.add_argument("--p1", type=str, default="random",
                        help="Player 1 model spec (e.g. 'minimax:time_budget=5')")
    parser.add_argument("--p2", type=str, default="random",
                        help="Player 2 model spec (e.g. 'mcts:time_budget=3')")
    parser.add_argument("--n", type=int, default=1, help="Number of games to play")
    parser.add_argument("--save_results", type=int, default=1, help="Save results to CSV (1=yes, 0=no)")
    parser.add_argument("--save_games", type=int, default=0, help="Save full game files with history (1=yes, 0=no)")
    parser.add_argument("--save_psn", type=int, default=0, help="Save games as PSN text (1=yes, 0=no)")
    parser.add_argument("--save_period", type=int, default=1,
                        help="Save a game file every N games (requires --save_games 1)")
    parser.add_argument("--v", type=int, default=1, help="Verbose output (1=yes, 0=no)")
    parser.add_argument("--delay", type=float, default=0,
                        help="Delay in seconds between AI moves (Flask mode only)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of steps per game before declaring a stalemate")
    parser.add_argument("--engine", type=str, choices=['python', 'rust'], default=DEFAULT_ENGINE,
                        help="Rules engine to play against (local mode) or request from the server "
                             "(flask mode). 'rust' requires crates/pybind to be built - see "
                             "engine/engine_select.py.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_params()
    if args.mode == 'flask':
        run_flask(args.n, args.p1, args.p2, bool(args.save_results), args.delay,
                  verbose=bool(args.v), save_games=bool(args.save_games),
                  save_period=args.save_period, max_steps=args.max_steps, engine=args.engine)
    elif args.mode == 'local':
        run_local(args.n, args.p1, args.p2, bool(args.save_results), verbose=bool(args.v),
                  save_games=bool(args.save_games), save_period=args.save_period,
                  save_psn=bool(args.save_psn), max_steps=args.max_steps, engine=args.engine)
