"""Basic CNN agent — trainable starter template.

A minimal convolutional value network over the 19x19 Pai Sho board.
At play time the agent evaluates every legal action by rolling the move
forward, encoding the resulting board as a small tensor, and picking the
move with the highest CNN value from the current player's perspective.

The net is intentionally tiny (two conv layers + a linear head) so the
model trains in minutes and stays readable as a reference. It has NO
dependency on TD learning or any other agent's feature extractor.

Companion training script: ai/training/cnn_basic_training.py.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game.PaiShoGame import (
    BOARD_SIZE, VALID_SPACES, ACCENT_TILES, CIRCLE, SPECIAL_TILES,
)


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'params', 'CNNBasic', 'cnn_basic.pt'
)

IN_CHANNELS = 8
CONV_CHANNELS = 16

_VALID_MASK = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
for _r, _c in VALID_SPACES:
    _VALID_MASK[_r, _c] = 1.0


def encode_board(game, player):
    """Return an (IN_CHANNELS, 19, 19) float32 tensor from `player`'s view."""
    opponent = 3 - player
    x = np.zeros((IN_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Ch 7: static valid-square mask.
    x[7] = _VALID_MASK

    for (r, c), tile in game.board.items():
        flower = tile['flower']
        p = tile['player']
        mine = (p == player)
        growing = bool(tile.get('growing', False))

        if flower in CIRCLE:
            x[0 if mine else 1, r, c] = 1.0
        elif flower in ACCENT_TILES:
            x[2 if mine else 3, r, c] = 1.0
        elif flower in SPECIAL_TILES:
            x[4 if mine else 5, r, c] = 1.0

        if growing:
            x[6, r, c] = 1.0
    return x


class BasicCNN(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, hidden=CONV_CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden * BOARD_SIZE * BOARD_SIZE, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return torch.tanh(self.head(x)).squeeze(-1)


class CNNBasicAgent:
    """Greedy value-CNN player: pick the action leading to the highest-valued state."""

    def __init__(self, player=1, load=False, epsilon=0.0, device=None):
        self.player = player
        self.epsilon = float(epsilon)
        self.device = torch.device(device) if device else torch.device('cpu')
        self.net = BasicCNN().to(self.device)
        self.net.eval()
        if load:
            self.load_model()

    def _value(self, game, player):
        enc = encode_board(game, player)
        t = torch.from_numpy(enc).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.net(t).item()
        return v

    def choose_action(self, game, verbose=False):
        legal = game.get_legal_actions()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(legal)

        me = game.current_player
        self.player = me
        best_action = None
        best_value = float('-inf')
        for a in legal:
            child = game.clone()
            try:
                child.step(a)
            except Exception:
                continue
            if child.winner == me:
                v = float('inf')
            elif child.winner is not None:
                v = float('-inf')
            else:
                v = self._value(child, me)
            if v > best_value:
                best_value = v
                best_action = a
        if best_action is None:
            best_action = random.choice(legal)
        if verbose:
            print(f"CNNBasic chose {best_action} (value={best_value:.4f})")
        return best_action

    def save_model(self, path=None):
        p = path or MODEL_PATH
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save(self.net.state_dict(), p)

    def load_model(self, path=None):
        p = path or MODEL_PATH
        if not os.path.exists(p):
            print(f"CNNBasic weights not found at {p}, using randomly initialised net.")
            return
        try:
            state = torch.load(p, map_location=self.device)
            self.net.load_state_dict(state)
            self.net.eval()
        except Exception as e:
            print(f"CNNBasic failed to load weights from {p}: {e}. Using random init.")
