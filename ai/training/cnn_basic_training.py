"""Training script for the Basic CNN agent.

Self-play value-regression: each episode plays a game where both sides
use the current CNN (with epsilon-greedy exploration), logs every
position reached, then trains the net so each state's predicted value
moves toward the game's final outcome (+1 for the eventual winner,
-1 for the loser, 0 for a draw / step-limit timeout) from that
player's perspective. Simple, no TD, no bootstrapping.

Registry exposes ai/params/CNNBasic/example_config.txt as a
`config_file` for demonstration. This script intentionally does NOT
read it — the file exists only to show the registry feature.
"""

import argparse
import os
import random as _rnd
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from game.PaiShoGame import PaiShoGame
from ai.rl.cnn_basic import CNNBasicAgent, encode_board, MODEL_PATH
from ai.logging_utils import get_logger, log_event
from ai.training.opponent_utils import load_opponent


def _reward_from_player_view(winner, player):
    if winner == 0 or winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def _play_episode(agent, max_steps, epsilon, opp_fn=None):
    """Play one game, returning (transitions, winner, steps, message).

    When `opp_fn` is None the training agent plays both sides (self-play) and
    transitions are logged for every move. Otherwise the training agent plays
    as player 1, `opp_fn(game)` plays as player 2, and only P1 transitions
    are logged so the value regression targets stay on-policy.
    """
    game = PaiShoGame()
    agent.epsilon = epsilon
    transitions = []
    steps = 0
    while game.winner is None and steps < max_steps:
        p = game.current_player
        legal = game.get_legal_actions()
        if not legal:
            break
        if opp_fn is not None and p == 2:
            action = opp_fn(game)
            if action is None:
                break
            game.step(action)
            steps += 1
            continue
        transitions.append((encode_board(game, p), p))
        agent.player = p
        action = agent.choose_action(game)
        if action is None:
            break
        game.step(action)
        steps += 1
    message = getattr(game, 'message', '') or '' if game.winner else ''
    return transitions, game.winner, steps, message


def train_cnn(
    episodes,
    lr,
    epsilon,
    min_eps,
    decay,
    max_steps,
    batch_size,
    resume,
    opponent="self",
    device="cpu",
):
    log = get_logger("cnn_basic_training")
    log.info(f"Starting Basic CNN training for {episodes:,} episodes.")
    log.info(f"  lr={lr}  eps={epsilon}->{min_eps}  max_steps={max_steps}  "
             f"batch={batch_size}  opponent={opponent}  device={device}")
    log_event(log, "run_start", model="cnn_basic", episodes=episodes, lr=lr,
              epsilon=epsilon, min_epsilon=min_eps, decay_rate=decay,
              max_steps=max_steps, batch_size=batch_size, resume=bool(resume),
              opponent=opponent, device=device)

    opp_fn = load_opponent(opponent)

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    agent = CNNBasicAgent(player=1, load=bool(resume), device=dev)
    agent.net.train()
    optimizer = optim.Adam(agent.net.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    n_params = sum(p.numel() for p in agent.net.parameters())

    wins = {1: 0, 2: 0, 0: 0}

    for episode in tqdm(range(episodes), desc="Episodes", disable=None, dynamic_ncols=True):
        t0 = time.time()
        transitions, winner, steps, reason = _play_episode(agent, max_steps, epsilon, opp_fn)
        tallies = {1: 0, 2: 0, 0: 0}
        tallies[winner if winner else 0] += 1
        wins[winner if winner else 0] += 1

        loss_val = 0.0
        if transitions:
            xs = np.stack([t[0] for t in transitions], axis=0)
            ys = np.array(
                [_reward_from_player_view(winner, p) for _, p in transitions],
                dtype=np.float32,
            )
            x_tensor = torch.from_numpy(xs).to(dev)
            y_tensor = torch.from_numpy(ys).to(dev)

            agent.net.train()
            idxs = np.arange(len(transitions))
            np.random.shuffle(idxs)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(idxs), batch_size):
                batch = idxs[start:start + batch_size]
                pred = agent.net(x_tensor[batch])
                loss = loss_fn(pred, y_tensor[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            loss_val = total_loss / max(n_batches, 1)
            agent.net.eval()

        if winner == 1:
            outcome = "P1 wins"
        elif winner == 2:
            outcome = "P2 wins"
        else:
            outcome = "Draw"

        epsilon = max(min_eps, epsilon * decay)
        episode_time = time.time() - t0

        reason_suffix = f" | Reason: {reason}" if reason else ""
        log.info(
            f"Episode {episode + 1:,}/{episodes:,} | "
            f"eps={epsilon:.4f} | {outcome} in {steps} steps | "
            f"W1={wins[1]} W2={wins[2]} D={wins[0]} | "
            f"loss={loss_val:.4f} | params={n_params:,} | "
            f"{episode_time:.2f}s{reason_suffix}"
        )
        log_event(log, "episode",
                  episode=episode + 1, total=episodes,
                  epsilon=round(epsilon, 4),
                  outcome=outcome, steps=steps,
                  p1_wins=wins[1], p2_wins=wins[2], draws=wins[0],
                  loss=round(loss_val, 6),
                  params=n_params,
                  episode_time=round(episode_time, 4),
                  reason=reason)

        if (episode + 1) % 10 == 0:
            agent.save_model()
            log.info(f"  [Autosave] model saved at episode {episode + 1}")

    agent.save_model()
    log.info("Training finished!")
    log.info(f"Saved to {MODEL_PATH}")
    log_event(log, "run_end", status="completed",
              p1_wins=wins[1], p2_wins=wins[2], draws=wins[0])
    return agent


def parse_params():
    parser = argparse.ArgumentParser(description="Basic CNN training for Skud Pai Sho")
    parser.add_argument("--n", type=int, default=200, help="Number of self-play episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--eps", type=float, default=0.5, help="Start epsilon for exploration")
    parser.add_argument("--min_eps", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--decay", type=float, default=0.995, help="Epsilon decay per episode")
    parser.add_argument("--ms", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--batch", type=int, default=32, help="Minibatch size")
    parser.add_argument("--resume", type=int, default=0, help="Resume from saved weights (1=yes)")
    parser.add_argument("--opponent", type=str, default="self",
                        help="Training opponent: 'self', 'random', or any registry key")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_params()
    train_cnn(
        episodes=args.n,
        lr=args.lr,
        epsilon=args.eps,
        min_eps=args.min_eps,
        decay=args.decay,
        max_steps=args.ms,
        batch_size=args.batch,
        resume=args.resume,
        opponent=args.opponent,
        device=args.device,
    )
