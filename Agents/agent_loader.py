"""Single source of truth for dispatching a registry entry's `kind`.

Every caller that needs to build an agent from a registry.py entry and ask it
for a move (the Play page, the simulator, and training scripts' opponent
loader) used to reimplement this import → instantiate → dispatch logic
independently, and the copies had already drifted. This module is the one
place that logic lives now; callers wire it into their own caching/session
semantics instead of re-deriving it.
"""

import importlib
import random as _random


def resolve(entry):
    """Import entry['module'] and return (module, class_or_function).

    Only valid for kind 'class' (looks up 'class_name') or 'function'
    (looks up 'function_name').
    """
    mod = importlib.import_module(entry["module"])
    if entry["kind"] == "class":
        return mod, getattr(mod, entry["class_name"])
    if entry["kind"] == "function":
        return mod, getattr(mod, entry["function_name"])
    raise ValueError(f"resolve() does not apply to kind '{entry['kind']}'")


def build_kwargs(entry, base_key, params=None):
    """Merge entry[base_key] (e.g. 'play_kwargs') with any `play_params`
    overrides present in `params` (a dict of key -> value from the caller)."""
    kwargs = dict(entry.get(base_key, {}))
    params = params or {}
    for pd in entry.get("play_params", []):
        if pd["key"] in params:
            kwargs[pd["key"]] = params[pd["key"]]
    return kwargs


def instantiate(entry, params=None, player=None):
    """Build a fresh agent instance for a kind='class' entry.

    `params` supplies play_params overrides on top of `play_kwargs`.
    `player` overrides the constructor's `player=` kwarg when the entry
    declares `needs_player`; pass None to leave play_kwargs' own value.
    """
    _, cls = resolve(entry)
    kwargs = build_kwargs(entry, "play_kwargs", params)
    if entry.get("needs_player") and player is not None:
        kwargs["player"] = player
    return cls(**kwargs)


def sync_agent(entry, agent, game, params=None):
    """Apply per-turn tunables to an existing agent instance in place,
    without rebuilding it: play_params overrides + needs_player's
    current-player sync."""
    params = params or {}
    for pd in entry.get("play_params", []):
        if pd["key"] in params:
            setattr(agent, pd["key"], params[pd["key"]])
    if entry.get("needs_player"):
        agent.player = game.current_player


def act(entry, game, agent=None, params=None, verbose=False, legal_actions=None):
    """Return one move, dispatching on entry['kind'].

    - 'inline': uniform-random legal action.
    - 'function': calls the registered function with function_kwargs + any
      play_params overrides found in `params`.
    - 'class': requires a pre-built `agent` (see `instantiate`). Syncs
      play_params/needs_player onto it, then calls `choose_action`.

    `legal_actions`, if the caller already computed it, avoids a redundant
    `game.get_legal_actions()` call for the 'inline' branch.
    """
    kind = entry["kind"]

    if kind == "inline":
        legal = legal_actions if legal_actions is not None else game.get_legal_actions()
        return _random.choice(legal) if legal else None

    if kind == "function":
        _, func = resolve(entry)
        kwargs = build_kwargs(entry, "function_kwargs", params)
        return func(game, **kwargs)

    if kind == "class":
        if agent is None:
            raise ValueError(
                f"'{entry.get('key')}' is kind=class but no agent instance was provided to act()."
            )
        sync_agent(entry, agent, game, params)
        return agent.choose_action(game, verbose)

    raise ValueError(f"Unsupported agent kind '{kind}' for '{entry.get('key')}'.")
