"""Basic Minimax agent — weightless starter template.

Depth-limited alpha-beta search with a small, hand-crafted evaluation
function. There is nothing to train: the evaluation coefficients are
hardcoded. Use this as the reference for wiring up a non-trainable
agent in the registry.

For a trainable reference template, see ai/rl/cnn_basic.py.
"""

import random
import time

from game.PaiShoGame import ACCENT_TILES
from ai.utils import _ring_threat_level, _ring_completion_distance


TILE_VALUE = {
    'Orchid': 25, 'WhiteLotus': 20,
    'Rose': 10, 'Chrysanthemum': 10, 'Rhododendron': 10,
    'Jasmine': 10, 'Lily': 10, 'Jade': 10,
}

W_HARMONY = 40.0
W_MATERIAL = 10.0
W_RING_DIST = 80.0
W_OPP_THREAT = 200.0


class BasicMinimaxAgent:
    """Depth-limited alpha-beta minimax with a fixed linear evaluator."""

    def __init__(self, player=1, time_budget=1.0, max_depth=2):
        self.player = player
        self.time_budget = float(time_budget)
        self.max_depth = max(1, int(max_depth))

    def evaluate(self, game, maximizing_player):
        if game.winner == maximizing_player:
            return float('inf')
        if game.winner is not None:
            return float('-inf')

        score = 0.0
        opponent = 3 - maximizing_player

        for player, sign in ((maximizing_player, 1), (opponent, -1)):
            harmonies = game.find_harmonies(player)
            score += sign * len(harmonies) * W_HARMONY
            dist = _ring_completion_distance(harmonies)
            if dist <= 1:
                score += sign * W_RING_DIST
            for t in game.board.values():
                if t['player'] == player and t['flower'] not in ACCENT_TILES:
                    score += sign * TILE_VALUE.get(t['flower'], 10) * W_MATERIAL / 10.0

        opp_threat = _ring_threat_level(game.find_harmonies(opponent))
        score -= opp_threat * W_OPP_THREAT
        return score

    def _alphabeta(self, game, depth, alpha, beta, maximizing, maximizing_player, deadline):
        if time.time() >= deadline:
            return None
        if depth == 0 or game.winner is not None:
            return self.evaluate(game, maximizing_player)
        legal = game.get_legal_actions()
        if not legal:
            return self.evaluate(game, maximizing_player)
        random.shuffle(legal)

        if maximizing:
            best = float('-inf')
            for a in legal:
                child = game.clone()
                child.step(a)
                next_max = (child.current_player == maximizing_player)
                v = self._alphabeta(child, depth - 1, alpha, beta, next_max,
                                    maximizing_player, deadline)
                if v is None:
                    return None
                if v > best:
                    best = v
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return best

        best = float('inf')
        for a in legal:
            child = game.clone()
            child.step(a)
            next_max = (child.current_player == maximizing_player)
            v = self._alphabeta(child, depth - 1, alpha, beta, next_max,
                                maximizing_player, deadline)
            if v is None:
                return None
            if v < best:
                best = v
            beta = min(beta, v)
            if beta <= alpha:
                break
        return best

    def _score_root(self, game, depth, deadline):
        maximizing_player = game.current_player
        legal = game.get_legal_actions()
        if not legal:
            return []
        random.shuffle(legal)
        scored = []
        for a in legal:
            if time.time() >= deadline:
                break
            child = game.clone()
            child.step(a)
            next_max = (child.current_player == maximizing_player)
            v = self._alphabeta(child, depth - 1, float('-inf'), float('inf'),
                                next_max, maximizing_player, deadline)
            if v is None:
                break
            scored.append((a, v))
        return scored

    def choose_action(self, game, verbose=False):
        legal = game.get_legal_actions()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        deadline = time.time() + self.time_budget
        scored = []
        for d in range(1, self.max_depth + 1):
            if time.time() >= deadline:
                break
            depth_scored = self._score_root(game, d, deadline)
            if depth_scored:
                scored = depth_scored

        if not scored:
            return random.choice(legal)

        best_action, best_value = max(scored, key=lambda av: av[1])
        if verbose:
            print(f"BasicMinimax chose {best_action} (value={best_value:.4f})")
        return best_action
