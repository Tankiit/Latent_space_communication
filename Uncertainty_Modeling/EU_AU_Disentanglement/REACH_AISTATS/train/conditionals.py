"""Experiment 6: estimated reference conditionals, reference error, and shift stress.

Three conditions share one engine:

positive_control   deployment is in the declared family and the reference radius is carried.
                   The true weight vector must stay in the outer set at about the nominal rate.
ignored_reference  identical, except the reference radius is dropped. Coverage is expected to
                   fail at small reference budgets. This is why the radius is not optional.
conditional_shift  deployment is generated from tilted known conditionals, so no weight vector
                   reproduces it. The mixture residual should exceed the declared radius.
                   A flagged incompatibility says the declared family is wrong; it does NOT
                   identify the tilt, or any other particular cause.

Alpha is split three ways as in the cached-feature path: log, references, labels.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from data.worlds import make_world, tilted_conditionals
from data.acquisitions import draw_reference, draw_log_from, draw_labelled_set
from models.ball import MMDBall, embedding_radius, weight_interval
from models.widths import label_intervals, credal_widths
from models.baselines import feature_match
import test.inference as inf

REGIONS = ('on_support', 'boundary')
CONDITIONS = ('positive_control', 'ignored_reference', 'conditional_shift')


def condition_rows(world, condition, r_grid, n_grid, seeds, shift=.25, alpha=.05, m=256, eta=0.):
    """One row per (reference budget, log size, seed): coverage, width, and compatibility."""
    carry_reference = condition != 'ignored_reference'
    applied_shift = shift if condition == 'conditional_shift' else 0.
    deployment = world.weights @ tilted_conditionals(world, applied_shift)
    rows = []
    for seed in seeds:
        log = draw_log_from(deployment, max(n_grid), seed)
        lo, hi = label_intervals(draw_labelled_set(world, m, eta, seed), eta, alpha/3)
        for r in r_grid:
            # Known rows are estimated from finite references; the novel row stays a declared candidate.
            estimated = np.vstack([draw_reference(world, r, seed), world.conditionals[-1]])
            radius = embedding_radius(r, alpha/6) if carry_reference else 0.
            for n in n_grid:
                ball = MMDBall.from_log(estimated, log[:n], alpha/3, radius)
                empirical = np.bincount(log[:n], minlength=estimated.shape[1])/n
                # Full declared family, including the candidate novel row: this is the
                # family whose adequacy is in question. The known-only residual mirrors
                # the cached-feature diagnostic and is also raised by ordinary novel mass.
                full = feature_match(estimated, empirical)
                known_only = feature_match(estimated[:2], empirical)
                novel = weight_interval(ball, 2)
                widths = [credal_widths(ball, cell, lo, hi) for cell, region
                          in enumerate(world.regions) if region in REGIONS]
                rows.append(dict(world_id=world.name, condition=condition, shift=applied_shift,
                                 carry_reference=carry_reference, r=int(r), n=int(n), m=int(m),
                                 eta=eta, seed=int(seed), eps_n=ball.eps_n,
                                 reference_radius=radius, radius=ball.radius,
                                 feasible=ball.feasible(),
                                 weights_in_outer_set=ball.outer_covers(world.weights),
                                 novel_lower=novel['lower'], novel_upper=novel['upper'],
                                 novel_width=novel['width'],
                                 novel_interval_covers=bool(novel['lower']-1e-9 <= world.weights[2]
                                                            <= novel['upper']+1e-9),
                                 declared_family_residual=full['residual'],
                                 known_mixture_residual=known_only['residual'],
                                 incompatible=bool(full['residual'] > ball.radius),
                                 known_only_incompatible=bool(known_only['residual'] > ball.radius),
                                 W_cls=float(np.mean([w['W_cls'] for w in widths])),
                                 W_total=float(np.mean([w['W_total'] for w in widths]))))
    return pd.DataFrame(rows)


def summarise(df):
    table = []
    for (condition, r, n), block in df.groupby(['condition', 'r', 'n']):
        coverage = inf.coverage_at_least_nominal(block.weights_in_outer_set, .95)
        table.append(dict(condition=condition, r=int(r), n=int(n),
                          weight_coverage_rate=coverage['rate'],
                          weight_coverage_lower=coverage['lower'],
                          independent_replicates=coverage['independent_replicates'],
                          novel_interval_coverage=float(block.novel_interval_covers.mean()),
                          mean_novel_width=float(block.novel_width.mean()),
                          mean_reference_radius=float(block.reference_radius.mean()),
                          mean_radius=float(block.radius.mean()),
                          mean_residual=float(block.declared_family_residual.mean()),
                          mean_known_only_residual=float(block.known_mixture_residual.mean()),
                          incompatible_rate=float(block.incompatible.mean()),
                          known_only_incompatible_rate=float(block.known_only_incompatible.mean()),
                          mean_W_cls=float(block.W_cls.mean()),
                          mean_W_total=float(block.W_total.mean())))
    table.sort(key=lambda row: (row['condition'], row['r'], row['n']))
    checks = {}
    for condition in df.condition.unique():
        rows = [row for row in table if row['condition'] == condition]
        checks[condition] = dict(min_weight_coverage=min(r['weight_coverage_rate'] for r in rows),
                                 max_incompatible_rate=max(r['incompatible_rate'] for r in rows))
    verdict = dict(
        positive_control_retains_coverage=bool(
            checks.get('positive_control', {}).get('min_weight_coverage', 0) >= .95),
        positive_control_rarely_flags_incompatible=bool(
            checks.get('positive_control', {}).get('max_incompatible_rate', 1) <= .05),
        dropping_reference_radius_loses_coverage=bool(
            checks.get('ignored_reference', {}).get('min_weight_coverage', 1) < .95),
        shift_is_detected=bool(
            checks.get('conditional_shift', {}).get('max_incompatible_rate', 0) >= .5))
    return dict(table=table, per_condition=checks, verdict=verdict,
                interpretation='incompatibility indicates the declared family is wrong, not its cause')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/conditionals')
    parser.add_argument('--world', default='separated')
    parser.add_argument('--seeds', type=int, default=64)
    parser.add_argument('--seed-start', type=int, default=0,
                        help='first seed; give shards disjoint ranges so replications stay independent')
    parser.add_argument('--r', type=int, nargs='+', default=[32, 128, 512, 2048, 8192])
    parser.add_argument('--n', type=int, nargs='+', default=[1024, 8192, 65536])
    parser.add_argument('--shift', type=float, default=.5)
    parser.add_argument('--eta', type=float, default=0.)
    parser.add_argument('--m', type=int, default=256)
    args = parser.parse_args(argv)
    if args.seeds < 2 or min(args.r+args.n) < 1 or not 0 < args.shift <= 1 or args.seed_start < 0:
        parser.error('At least two seeds, a nonnegative seed start, positive budgets, '
                     'and a shift in (0,1] required')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    world = make_world(args.world)
    seeds = range(args.seed_start, args.seed_start+args.seeds)
    frames = [condition_rows(world, condition, args.r, args.n, seeds,
                             args.shift, m=args.m, eta=args.eta) for condition in CONDITIONS]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out/'rows.csv', index=False)
    summary = summarise(df)
    summary['config'] = vars(args)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Estimated-conditionals table: {len(df)} rows to {out}')
    for key, value in summary['verdict'].items():
        print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
