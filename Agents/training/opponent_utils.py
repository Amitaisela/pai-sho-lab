"""Generic opponent loader used by training scripts.

Looks up any registry entry by key and returns a callable
`choose_action(game)` that plays a move as player 2. Special-cases
`'self'` (caller handles it — self-play) and `'random'` (uniform pick).
"""

import random as _rnd

from Agents.registry import get_agent
from Agents.agent_loader import instantiate, act


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

    if entry.get("kind") == "class":
        agent = instantiate(entry, player=2)

        def _class(game):
            return act(entry, game, agent=agent)
        return _class

    def _other(game):
        return act(entry, game)
    return _other
