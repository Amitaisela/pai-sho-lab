"""All-pairs round robin with a maximum-likelihood Elo fit and bootstrap CIs.

Plays every unordered pair among the given agent specs `games_per_pair` times
(alternating seats), fits Bradley-Terry ratings to the results, and reports
each rating with a 95% bootstrap confidence interval.

Usage: python scripts/round_robin.py SPEC [SPEC ...] --games 10 --workers 6 \
           --anchor random --json out.json --md out.md
"""
import argparse
import json
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine_select import game_class  # noqa: E402
from scripts.eval_agents import _load, _play  # noqa: E402

_K = math.log(10.0) / 400.0


def fit_elo(results, anchor=None, anchor_rating=0.0, iters=500, lr=0.5, l2=1e-6):
    """Maximum-likelihood Bradley-Terry fit.

    `results` is a list of (a, b, score_a) with score_a in {0, 0.5, 1}.
    Maximises sum(s*log(p) + (1-s)*log(1-p)) - l2*sum(r**2), where
    p = 1 / (1 + 10**((r_b - r_a) / 400)), by gradient ascent on NumPy
    arrays (a diagonal Newton preconditioner keeps convergence fast without
    changing the fixed point the plain gradient would reach). `l2` is kept
    small -- it exists only to keep an undefeated player's rating finite,
    not to meaningfully shrink ordinary estimates. Shifts the result so
    `anchor` lands at `anchor_rating`, or the mean is 0 with no anchor.
    """
    players = sorted({p for a, b, _s in results for p in (a, b)})
    if not players:
        return {}
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)

    ia = np.array([idx[a] for a, _b, _s in results], dtype=np.int64)
    ib = np.array([idx[b] for _a, b, _s in results], dtype=np.int64)
    s = np.array([sc for _a, _b, sc in results], dtype=np.float64)

    r = np.zeros(n, dtype=np.float64)
    for _ in range(iters):
        diff = r[ia] - r[ib]
        p = 1.0 / (1.0 + np.power(10.0, -diff / 400.0))
        per_game = _K * (s - p)
        grad = np.zeros(n, dtype=np.float64)
        np.add.at(grad, ia, per_game)
        np.add.at(grad, ib, -per_game)
        grad -= 2.0 * l2 * r

        # Diagonal Newton preconditioner: curvature of the per-player
        # log-likelihood is ~K^2 * p * (1-p) per game the player is in.
        # This is a step-size choice only -- it does not move the fixed
        # point, just reaches it in far fewer iterations than plain
        # gradient ascent would need.
        curve = np.zeros(n, dtype=np.float64)
        per_curve = _K * _K * p * (1.0 - p)
        np.add.at(curve, ia, per_curve)
        np.add.at(curve, ib, per_curve)
        curve += 2.0 * l2

        step = lr * grad / np.maximum(curve, 1e-9)
        r += np.clip(step, -50.0, 50.0)

    ratings = dict(zip(players, r.tolist()))
    if anchor is not None and anchor in ratings:
        shift = anchor_rating - ratings[anchor]
    else:
        shift = -float(np.mean(r))
    return {p: v + shift for p, v in ratings.items()}


def bootstrap_intervals(results, n=200, seed=0, **fit_kwargs):
    """Percentile bootstrap 95% CI for each player's fitted rating."""
    rng = random.Random(seed)
    m = len(results)
    players = sorted({p for a, b, _s in results for p in (a, b)})
    samples = {p: [] for p in players}

    for _ in range(n):
        resample = [results[rng.randrange(m)] for _ in range(m)] if m else []
        fit = fit_elo(resample, **fit_kwargs)
        for p in players:
            if p in fit:
                samples[p].append(fit[p])

    out = {}
    for p in players:
        vals = samples[p]
        if not vals:
            out[p] = (float('nan'), float('nan'))
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[p] = (float(lo), float(hi))
    return out


def _play_one(args):
    """Top-level, picklable worker: one game of spec a vs spec b, or None if it crashed.

    A crash drops that one game (reported on stderr) instead of the whole tournament."""
    try:
        return _play_one_inner(args)
    except Exception as e:  # noqa: BLE001 - any agent failure must not kill a multi-hour run
        print(f"GAME ERROR: {args[0]} vs {args[1]} (seed {args[4]}): {e!r}", file=sys.stderr, flush=True)
        return None


def _play_one_inner(args):
    """Play one game of spec a vs spec b and return (a, b, score_a)."""
    a_spec, b_spec, engine, max_steps, seed, allow_untrained, a_seat = args
    random.seed(seed)
    G = game_class(engine)
    a_name, a_params, a_agent, _ = _load(a_spec, allow_untrained)
    b_name, b_params, b_agent, _ = _load(b_spec, allow_untrained)

    names = {a_seat: a_name, 3 - a_seat: b_name}
    params = {a_seat: a_params, 3 - a_seat: b_params}
    agents = {a_seat: a_agent, 3 - a_seat: b_agent}
    g = _play(G, names, params, agents, max_steps)

    if g.winner is None:
        score_a = 0.5
    elif g.winner == 0:
        score_a = 0.5
    elif g.winner == a_seat:
        score_a = 1.0
    else:
        score_a = 0.0
    # end_reason 'resign' only happens when _play forfeits an agent for a bad move.
    forfeit = None
    if getattr(g, 'end_reason', None) == 'resign':
        forfeit = b_spec if g.winner == a_seat else a_spec
    return (a_spec, b_spec, score_a, forfeit)


def run_round_robin(specs, games_per_pair, engine='rust', max_steps=1000, workers=1, seed=0,
                     allow_untrained=False, progress=False, partial_path=None, stats=None):
    """Play every unordered pair `games_per_pair` times, alternating seats.

    Returns a list of (a, b, score_a) tuples, ordered by pair then game index,
    with `score_a` from spec a's point of view (1/0.5/0, timeouts as 0.5).
    A move that is None/illegal/raises forfeits that game (a loss for that agent).
    A game that crashes outside an agent's move is dropped. If `stats` is a dict it
    receives {'dropped': int, 'forfeits': {spec: int}}.

    Every spec and the engine are loaded once up front, so a typo or a missing Rust
    build fails immediately instead of producing an empty table.
    """
    game_class(engine)
    for spec in specs:
        _load(spec, allow_untrained)

    jobs = []
    job_idx = 0
    for i, a in enumerate(specs):
        for b in specs[i + 1:]:
            for g in range(games_per_pair):
                a_seat = 1 if g % 2 == 0 else 2
                job_seed = seed + job_idx
                jobs.append((a, b, engine, max_steps, job_seed, allow_untrained, a_seat))
                job_idx += 1

    def _progress(done, res):
        if progress:
            status = 'error' if res is None else f"{res[0]} vs {res[1]}: {res[2]}" + (
                f" (forfeit: {res[3]})" if res[3] else '')
            print(f"[{done}/{len(jobs)}] {status}", flush=True)
        if partial_path and res is not None:
            with open(partial_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(list(res)) + '\n')

    results = []
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            for i, res in enumerate(pool.imap(_play_one, jobs), start=1):
                _progress(i, res)
                results.append(res)
    else:
        for i, job in enumerate(jobs, start=1):
            res = _play_one(job)
            _progress(i, res)
            results.append(res)

    kept = [r for r in results if r is not None]
    if stats is not None:
        stats['dropped'] = len(results) - len(kept)
        forfeits = {}
        for r in kept:
            if r[3]:
                forfeits[r[3]] = forfeits.get(r[3], 0) + 1
        stats['forfeits'] = forfeits
    return [r[:3] for r in kept]


def _score_pct_table(results, ratings, ci):
    per_player = {p: {'wins': 0.0, 'games': 0} for p in ratings}
    for a, b, score_a in results:
        per_player[a]['wins'] += score_a
        per_player[a]['games'] += 1
        per_player[b]['wins'] += (1.0 - score_a)
        per_player[b]['games'] += 1

    rows = []
    for p in sorted(ratings, key=lambda k: -ratings[k]):
        g = per_player[p]['games']
        pct = 100.0 * per_player[p]['wins'] / g if g else 0.0
        lo, hi = ci.get(p, (float('nan'), float('nan')))
        rows.append({
            'spec': p, 'rating': ratings[p], 'ci_lo': lo, 'ci_hi': hi,
            'games': g, 'score_pct': pct,
        })
    return rows


def _pair_matrix(specs, results):
    tally = {(a, b): [0.0, 0] for a in specs for b in specs if a != b}
    for a, b, score_a in results:
        tally[(a, b)][0] += score_a
        tally[(a, b)][1] += 1
        tally[(b, a)][0] += (1.0 - score_a)
        tally[(b, a)][1] += 1
    matrix = {}
    for a in specs:
        matrix[a] = {}
        for b in specs:
            if a == b:
                matrix[a][b] = None
                continue
            wins, games = tally[(a, b)]
            matrix[a][b] = round(100.0 * wins / games, 1) if games else None
    return matrix


def render_markdown(specs, rows, matrix):
    lines = ['| Rank | Spec | Rating | 95% CI | Games | Score % |',
             '|---|---|---|---|---|---|']
    for i, row in enumerate(rows, 1):
        ci = f"[{row['ci_lo']:.0f}, {row['ci_hi']:.0f}]"
        lines.append(f"| {i} | {row['spec']} | {row['rating']:.1f} | {ci} | "
                      f"{row['games']} | {row['score_pct']:.1f}% |")

    lines.append('')
    lines.append('Pair matrix (row vs column, score %):')
    lines.append('')
    header = '| |' + '|'.join(specs) + '|'
    sep = '|---|' + '|'.join('---' for _ in specs) + '|'
    lines.append(header)
    lines.append(sep)
    for a in specs:
        cells = []
        for b in specs:
            v = matrix[a][b]
            cells.append('-' if v is None else f'{v:.1f}%')
        lines.append(f"| {a} |" + '|'.join(cells) + '|')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('specs', nargs='+')
    ap.add_argument('--games', type=int, default=10, help='games per pair')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--engine', choices=['python', 'rust'], default='rust')
    ap.add_argument('--max_steps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--anchor', default=None)
    ap.add_argument('--anchor_rating', type=float, default=0.0)
    ap.add_argument('--bootstrap', type=int, default=200)
    ap.add_argument('--allow-untrained', action='store_true')
    ap.add_argument('--json', default=None)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    anchor = args.anchor if args.anchor in args.specs or args.anchor is None else None

    partial = (args.json + '.partial.jsonl') if args.json else None
    stats = {}
    results = run_round_robin(args.specs, args.games, engine=args.engine, max_steps=args.max_steps,
                               workers=args.workers, seed=args.seed, allow_untrained=args.allow_untrained,
                               progress=True, partial_path=partial, stats=stats)
    if not results:
        sys.exit('round_robin: no games completed (see GAME ERROR lines above)')

    fit_kwargs = {'anchor': anchor, 'anchor_rating': args.anchor_rating}
    ratings = fit_elo(results, **fit_kwargs)
    ci = bootstrap_intervals(results, n=args.bootstrap, seed=args.seed, **fit_kwargs)
    rows = _score_pct_table(results, ratings, ci)
    matrix = _pair_matrix(args.specs, results)
    md = render_markdown(args.specs, rows, matrix)
    md += (f"\nDropped games (crashed outside an agent's move): {stats.get('dropped', 0)}. "
           f"Forfeits (None/illegal/raising move): {stats.get('forfeits') or 'none'}.\n")

    print(md)

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({
                'specs': args.specs, 'games_per_pair': args.games, 'engine': args.engine,
                'anchor': anchor, 'anchor_rating': args.anchor_rating,
                'results': results, 'table': rows, 'matrix': matrix,
                'dropped': stats.get('dropped', 0), 'forfeits': stats.get('forfeits', {}),
            }, f, indent=2)

    if args.md:
        with open(args.md, 'w', encoding='utf-8') as f:
            f.write(md + '\n')


if __name__ == '__main__':
    main()
