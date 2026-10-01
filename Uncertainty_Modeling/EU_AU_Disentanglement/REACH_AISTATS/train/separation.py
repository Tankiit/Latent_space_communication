"""Experiment 3b: weak embedding separation.

Sweeps the known-source tangent-space gap against the unlabelled log size. The claim under
test is NOT that widths stay small; it is that near-singular embeddings predict slow
resolution while validity is preserved. A representation that hides a small gap by
reporting narrow intervals would be a failure, not a success.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from data.worlds import make_separation_world, separation_gap
from data.acquisitions import draw_log, draw_labelled_set
from models.ball import MMDBall, weight_interval
from models.widths import label_intervals, credal_widths
import test.inference as inf

REGIONS = ('on_support', 'boundary')


def separation_rows(sep_grid, n_grid, seeds, alpha=.05, delta=.1, m=128):
    rows = []
    for sep in sep_grid:
        world = make_separation_world(sep, delta)
        gap = separation_gap(world)
        for seed in seeds:
            log = draw_log(world, max(n_grid), seed)
            # Labels are held fixed and noiseless: this sweep isolates the embedding geometry.
            lo, hi = label_intervals(draw_labelled_set(world, m, 0., seed), 0., alpha/2)
            for n in n_grid:
                ball = MMDBall.from_log(world.conditionals, log[:n], alpha/2)
                w0 = weight_interval(ball, 0)
                for cell, region in enumerate(world.regions):
                    row = dict(world_id=world.name, sep=float(sep), gap=gap, n=int(n), m=int(m),
                               eta=0., seed=int(seed), probe_id=cell, region=region,
                               eps_n=ball.eps_n, radius_over_gap=ball.eps_n/gap,
                               feasible=ball.feasible(),
                               ball_covers_truth=ball.covers_truth(world.marginal),
                               outer_covers_truth=ball.outer_covers(world.weights),
                               weight0_width=w0['width'], weight0_status=w0['status'],
                               resolution=ball.resolution([.5, .5, 0], [.9, .1, 0]),
                               true_target=world.target(cell))
                    row.update(credal_widths(ball, cell, lo, hi))
                    rows.append(row)
    return pd.DataFrame(rows)


def summarise(df, target_width=.2):
    """Per-separation resolution profile plus the smallest swept n reaching target_width."""
    out = []
    probes = df[df.region.isin(REGIONS)]
    for (sep, gap), block in probes.groupby(['sep', 'gap']):
        per_n = block.groupby('n').agg(W_cls=('W_cls', 'mean'), weight0_width=('weight0_width', 'mean'),
                                       radius_over_gap=('radius_over_gap', 'mean')).reset_index()
        reached = per_n[per_n.W_cls <= target_width].n
        unique = block.drop_duplicates(['n', 'seed'])
        coverage = inf.coverage_at_least_nominal(unique.ball_covers_truth, .975)
        out.append(dict(sep=float(sep), gap=float(gap), target_width=target_width,
                        smallest_n_reaching_target=int(reached.min()) if len(reached) else None,
                        profile=[dict(n=int(r.n), W_cls=float(r.W_cls),
                                      weight0_width=float(r.weight0_width),
                                      radius_over_gap=float(r.radius_over_gap))
                                 for r in per_n.itertuples()],
                        outer_coverage_rate=float(unique.outer_covers_truth.mean()),
                        ball_coverage=coverage))
    out.sort(key=lambda r: r['sep'])
    widest = max(out, key=lambda r: r['profile'][-1]['W_cls'])
    narrowest = min(out, key=lambda r: r['profile'][-1]['W_cls'])
    monotone = all(a['profile'][-1]['W_cls'] >= b['profile'][-1]['W_cls']
                   for a, b in zip(out, out[1:]))
    return dict(per_separation=out,
                width_decreases_with_separation_at_max_n=bool(monotone),
                worst_case=dict(sep=widest['sep'], W_cls_at_max_n=widest['profile'][-1]['W_cls']),
                best_case=dict(sep=narrowest['sep'], W_cls_at_max_n=narrowest['profile'][-1]['W_cls']),
                validity_preserved_at_all_separations=bool(
                    all(r['ball_coverage']['rate'] >= .975 for r in out)),
                interpretation='near-singular embeddings widen intervals; coverage must not degrade')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/separation')
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--seed-start', type=int, default=0,
                        help='first seed; give shards disjoint ranges so replications stay independent')
    parser.add_argument('--sep', type=float, nargs='+', default=[1., .5, .25, .1, .05])
    parser.add_argument('--n', type=int, nargs='+', default=[256, 1024, 4096, 16384])
    parser.add_argument('--target-width', type=float, default=.2)
    args = parser.parse_args(argv)
    if args.seeds < 2 or min(args.n) < 1 or args.seed_start < 0:
        parser.error('At least two seeds, a nonnegative seed start, and positive budgets required')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = separation_rows(args.sep, args.n, range(args.seed_start, args.seed_start+args.seeds))
    df.to_csv(out/'rows.csv', index=False)
    summary = summarise(df, args.target_width)
    summary['config'] = vars(args)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Separation sweep: {len(df)} rows over {len(args.sep)} gaps to {out}')
    print('Wide intervals at small gaps are the expected result, not a defect.')


if __name__ == '__main__':
    main()
