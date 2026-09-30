"""Time get_legal_actions() on mid-game Rust positions.

Each timed call runs on a fresh clone whose legal-actions cache was cleared by
re-assigning current_player (a mutation), so nothing is served from the cache.
Bonus-turn positions (the only ones that generate accent placements) are timed
separately, and their share of total time in random play is reported.

Usage: python scripts/bench_legal_actions.py [--positions 200] [--seed 7] [--engine rust]
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine_select import game_class  # noqa: E402


def midgame_positions(G, n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        g = G()
        for _ in range(rng.randint(20, 80)):
            acts = g.get_legal_actions()
            if not acts or g.winner is not None:
                break
            g.step(rng.choice(acts))
        if g.winner is None:
            out.append(g)
    return out


def _time(games, reps):
    t0 = time.perf_counter()
    total = 0
    for _ in range(reps):
        for g in games:
            c = g.clone()
            c.current_player = c.current_player  # clears the cached legal list
            total += len(c.get_legal_actions())
    return (time.perf_counter() - t0) * 1000 / max(1, len(games) * reps), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--positions', type=int, default=400)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--engine', choices=['python', 'rust'], default='rust')
    args = ap.parse_args()
    G = game_class(args.engine)
    games = midgame_positions(G, args.positions, args.seed)
    bonus = [g for g in games if g.bonus_turn]
    normal = [g for g in games if not g.bonus_turn]
    mb, _ = _time(bonus, args.reps)
    mn, _ = _time(normal, args.reps)
    share = mb * len(bonus) / (mb * len(bonus) + mn * len(normal)) if games else 0
    print(f"{args.engine}: {len(games)} positions ({len(bonus)} bonus-turn); "
          f"bonus {mb:.3f} ms/pos, normal {mn:.3f} ms/pos; bonus-turn share of time {share:.1%}")


if __name__ == '__main__':
    main()
