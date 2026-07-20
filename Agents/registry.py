import importlib
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def num_param(key, label, default, **kw):
    """A numeric play/training param. Extra kwargs: min, max, step, tooltip, cli_flag."""
    d = {'key': key, 'label': label, 'type': 'number', 'default': default}
    d.update(kw)
    return d

def text_param(key, label, default, **kw):
    """A free-text play/training param. Extra kwargs: tooltip, cli_flag."""
    d = {'key': key, 'label': label, 'type': 'text', 'default': default}
    d.update(kw)
    return d

def checkbox_param(key, label, default=False, **kw):
    """A boolean play/training param. Extra kwargs: tooltip, cli_flag."""
    d = {'key': key, 'label': label, 'type': 'checkbox', 'default': default}
    d.update(kw)
    return d
AGENTS = [{'key': 'random', 'display_name': 'Random', 'description': 'Picks a random legal action', 'architecture': 'A stateless baseline agent: on each turn it enumerates every legal action from the current position and samples one uniformly at random. There is no model, no search, and no learning — it exists purely as a lower bound for measuring the strength of every other agent and as a sanity-check opponent during training.', 'kind': 'inline', 'play_params': [], 'model_path': None, 'training_script': None, 'training_params': [], 'total_episodes_key': None, 'log_parser': None, 'config_file': None}, {'key': 'basic_minimax', 'display_name': 'Basic Minimax', 'description': 'Alpha-beta minimax with a fixed hand-crafted evaluation (no training required — reference template for a weightless agent)', 'architecture': 'Stripped-down alpha-beta minimax intended as a readable reference for any agent that does not need training. Search runs to a small fixed maximum depth under a per-move time budget, with simple alpha-beta pruning and no transposition table. The leaf evaluation is a fixed hand-crafted scoring function (not learned), so play strength comes entirely from search depth and the quality of that single eval — perfect as a template when prototyping a new agent that should compete without any weights.', 'kind': 'class', 'module': 'Agents.classical.basic_minimax', 'class_name': 'BasicMinimaxAgent', 'play_kwargs': {'player': 1}, 'needs_player': True, 'play_params': [num_param('time_budget', 'Time budget (s)', 1.0, step=0.1, min=0.1, max=30, tooltip='Per-move search time budget.'), num_param('max_depth', 'Max depth', 2, step=1, min=1, max=6, tooltip='Maximum alpha-beta search depth.')], 'model_path': None, 'training_script': None, 'training_params': [], 'total_episodes_key': None, 'log_parser': None, 'config_file': None}, {'key': 'cnn_basic', 'display_name': 'Basic CNN', 'description': 'Tiny two-layer convolutional value network (reference template for a trainable agent)', 'architecture': 'A small convolutional value network operating directly on the 19×19 board. The position is encoded as an 8-channel tensor (own/opponent circle, accent, and special tiles, a growing-state plane, and a static valid-square mask). Two 3×3 convolutional layers with ReLU extract local spatial patterns, then a linear head with tanh outputs a scalar value in [-1, 1]. At play time the agent rolls every legal move forward one ply, runs the CNN on the resulting board, and picks the action whose successor scores highest.', 'kind': 'class', 'module': 'Agents.rl.cnn_basic', 'class_name': 'CNNBasicAgent', 'play_kwargs': {'player': 1, 'load': True}, 'needs_player': True, 'play_params': [num_param('epsilon', 'Exploration (epsilon)', 0.0, step=0.01, min=0.0, max=1.0, tooltip='Play-time exploration probability (0 = greedy).'), text_param('device', 'Device', 'cpu', tooltip='cpu or cuda')], 'model_path': 'data/params/CNNBasic/cnn_basic.pt', 'training_script': 'Agents/training/cnn_basic_training.py', 'training_params': [num_param('episodes', 'Episodes', 200, step=50, min=1, tooltip='Number of self-play training episodes', cli_flag='--n'), num_param('lr', 'Learning Rate', 0.001, step=0.0001, min=1e-05, max=1, tooltip='Adam optimizer learning rate', cli_flag='--lr'), num_param('epsilon', 'Start Epsilon', 0.5, step=0.01, min=0, max=1, tooltip='Initial exploration rate', cli_flag='--eps'), num_param('min_epsilon', 'Min Epsilon', 0.05, step=0.01, min=0, max=1, tooltip='Floor for epsilon decay', cli_flag='--min_eps'), num_param('decay', 'Epsilon Decay', 0.995, step=0.001, min=0, max=1, tooltip='Multiply epsilon by this each episode', cli_flag='--decay'), num_param('max_steps', 'Max Steps', 300, step=50, min=50, tooltip='Max turns per self-play game', cli_flag='--ms'), num_param('batch_size', 'Batch Size', 32, step=8, min=1, tooltip='Minibatch size for the value-regression step', cli_flag='--batch'), text_param('opponent', 'Opponent', 'self', tooltip='Training opponent (filled in after registry is built)', cli_flag='--opponent'), text_param('device', 'Device', 'cpu', tooltip='cpu or cuda', cli_flag='--device'), checkbox_param('resume', 'Resume from checkpoint', False, tooltip='Load existing weights and continue training', cli_flag='--resume')], 'total_episodes_key': 'episodes', 'log_parser': None, 'config_file': 'data/params/CNNBasic/example_config.txt'}]

def _fill_opponent_tooltip(agent_key):
    """For agents with an `opponent`/`p2` training param, set tooltip to `self` +
    every other registry key, so the UI hover-tip stays in sync with the
    registry."""
    entry = next((a for a in AGENTS if a['key'] == agent_key), None)
    if not entry:
        return
    others = [a['key'] for a in AGENTS if a['key'] != agent_key]
    options = ['self'] + others
    for tp in entry.get('training_params', []):
        if tp.get('key') in ('opponent', 'p2'):
            tp['tooltip'] = 'Training opponent: ' + ', '.join(options)
for _a in AGENTS:
    if any((tp.get('key') in ('opponent', 'p2') for tp in _a.get('training_params', []))):
        _fill_opponent_tooltip(_a['key'])
_BY_KEY = {a['key']: a for a in AGENTS}

def get_agent(key):
    return _BY_KEY.get(key.lower())

def all_agents():
    return AGENTS

def playable_agents():
    return AGENTS

def trainable_agents():
    return [a for a in AGENTS if a.get('training_script')]

def validate_registry():
    """Sanity-check every AGENTS entry's shape, returning a list of error
    strings (empty = valid). Checks: unique keys, a supported `kind`, that
    `module`/`class_name`/`function_name` actually import and resolve, that
    every trainable param carries a `cli_flag`, and that `training_script`
    points at a real file. Does not check `model_path` (untrained agents
    legitimately have no weights on disk yet).

    Run via `python -m Agents.registry` or from tests/test.py.
    """
    errors = []
    seen_keys = set()
    valid_kinds = {'inline', 'function', 'class'}
    for a in AGENTS:
        key = a.get('key')
        label = key or '<missing key>'
        if not key:
            errors.append("entry missing required 'key' field")
        elif key in seen_keys:
            errors.append(f'{label}: duplicate key')
        seen_keys.add(key)
        kind = a.get('kind')
        if kind not in valid_kinds:
            errors.append(f'{label}: invalid kind {kind!r} (must be one of {sorted(valid_kinds)})')
            continue
        if kind in ('function', 'class'):
            module_name = a.get('module')
            attr_name = a.get('function_name') if kind == 'function' else a.get('class_name')
            attr_field = 'function_name' if kind == 'function' else 'class_name'
            if not module_name:
                errors.append(f"{label}: kind={kind} requires 'module'")
            elif not attr_name:
                errors.append(f"{label}: kind={kind} requires '{attr_field}'")
            else:
                try:
                    mod = importlib.import_module(module_name)
                except Exception as e:
                    errors.append(f"{label}: module '{module_name}' failed to import: {e}")
                else:
                    if not hasattr(mod, attr_name):
                        errors.append(f"{label}: '{module_name}' has no attribute '{attr_name}'")
        training_params = a.get('training_params', [])
        if a.get('training_script'):
            script_path = os.path.join(_REPO_ROOT, a['training_script'])
            if not os.path.exists(script_path):
                errors.append(f"{label}: training_script not found at {a['training_script']}")
            for tp in training_params:
                if not tp.get('cli_flag'):
                    errors.append(f"{label}: training_param '{tp.get('key')}' missing 'cli_flag'")
        elif training_params:
            errors.append(f'{label}: has training_params but no training_script')
    return errors
if __name__ == '__main__':
    _errors = validate_registry()
    if _errors:
        print(f'registry.py: {len(_errors)} error(s):')
        for _e in _errors:
            print(f'  - {_e}')
        raise SystemExit(1)
    print(f'registry.py: {len(AGENTS)} agents, all valid.')
