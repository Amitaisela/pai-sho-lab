"""Generic opponent loader used by training scripts.

Looks up any registry entry by key and returns a callable
`choose_action(game)` that plays a move as player 2. Special-cases
`'self'` (caller handles it — self-play) and `'random'` (uniform pick).
"""

import importlib
import random as _rnd

from ai.registry import get_agent


def load_opponent(name):
    """Return a callable `choose_action(game)` for the named opponent.

    `'self'` returns None — the caller should keep using the training agent
    for both sides.
    """
    if name is None or name == 'self':
        return None

    if name == 'random':
        def _random(game):
            legal = game.get_legal_actions()
            return _rnd.choice(legal) if legal else None
        return _random

    entry = get_agent(name)
    if not entry:
        raise ValueError(
            f"Unknown opponent '{name}'. Must be 'self', 'random', or a "
            f"registry key."
        )

    kind = entry.get("kind")

    if kind == "inline":
        def _inline(game):
            legal = game.get_legal_actions()
            return _rnd.choice(legal) if legal else None
        return _inline

    if kind == "function":
        mod = importlib.import_module(entry["module"])
        func = getattr(mod, entry["function_name"])
        kwargs = dict(entry.get("function_kwargs", {}))

        def _func(game):
            return func(game, **kwargs)
        return _func

    if kind == "class":
        mod = importlib.import_module(entry["module"])
        cls = getattr(mod, entry["class_name"])
        kwargs = dict(entry.get("play_kwargs", {}))
        if entry.get("needs_player"):
            kwargs["player"] = 2
        agent = cls(**kwargs)

        def _class(game):
            if entry.get("needs_player"):
                agent.player = game.current_player
            return agent.choose_action(game, verbose=False)
        return _class

    raise ValueError(f"Unsupported agent kind '{kind}' for opponent '{name}'.")
