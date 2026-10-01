"""Experiment 5: matched total width, different acquisition benefit.

Two arms are chosen to have comparable W_total at baseline while responding differently to
one extra unit of unlabelled log versus one extra unit of labels. The arm pair is SELECTED
using data from selection seeds and then re-estimated on disjoint evaluation seeds, so the
reported contrast is not the same quantity that drove the selection.

This demonstrates that a single total width does not determine which acquisition helps.
It is not a budget-allocation study and implies no optimal allocation.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from data.worlds import make_world
from data.acquisitions import draw_log, draw_labelled_set
from models.ball import MMDBall
from models.widths import label_intervals, credal_widths
import test.inference as inf
import train.predictions as P

REGIONS = ('on_support', 'boundary')
# Budget grid wide enough to contain both prevalence-limited and label-limited regimes.
CANDIDATES = ([dict(n=n, m=m, eta=.0) for n in (256, 1024, 4096, 16384, 65536)
               for m in (16, 64, 256, 1024)]
              + [dict(n=n, m=m, eta=.2) for n, m in ((1024, 256), (4096, 256),
                                                     (16384, 64), (65536, 64))])
# Arms whose baseline width is at the vacuous ceiling are excluded: matching two saturated
# arms is uninformative because no acquisition can move either one.
MAX_BASELINE = .92


def _mean_total(world, ball, lo, hi):
    widths = [credal_widths(ball, cell, lo, hi)['W_total']
              for cell, region in enumerate(world.regions) if region in REGIONS]
    return float(np.mean(widths))


def arm_widths(world, arm, factor, seed, alpha=.05):
    """Baseline width plus the two single-intervention upgrades, on one shared replicate."""
    n, m, eta = int(arm['n']), int(arm['m']), float(arm['eta'])
    log = draw_log(world, n*factor, seed)
    labels = draw_labelled_set(world, m*factor, eta, seed)
    ball = MMDBall.from_log(world.conditionals, log[:n], alpha/2)
    big_log = MMDBall.from_log(world.conditionals, log, alpha/2)
    lo, hi = label_intervals(labels[:, :m], eta, alpha/2)
    big_lo, big_hi = label_intervals(labels, eta, alpha/2)
    baseline = _mean_total(world, ball, lo, hi)
    plus_log = _mean_total(world, big_log, lo, hi)
    plus_label = _mean_total(world, ball, big_lo, big_hi)
    return dict(seed=int(seed), n=n, m=m, eta=eta, factor=factor, baseline=baseline,
                plus_log=plus_log, plus_label=plus_label,
                gain_log=baseline-plus_log, gain_label=baseline-plus_label)


def arm_frame(world, arms, factor, seeds, alpha=.05):
    rows = []
    for index, arm in enumerate(arms):
        for seed in seeds:
            rows.append(dict(arm_id=index, arm=f"n{arm['n']}_m{arm['m']}_eta{arm['eta']:g}",
                             **arm_widths(world, arm, factor, seed, alpha)))
    return pd.DataFrame(rows)


def select_pair(frame, band=P.EQUIV_BAND, max_baseline=MAX_BASELINE):
    """Most asymmetric matched pair; selection is exploratory and uses selection seeds only."""
    stats = frame.groupby(['arm_id', 'arm']).agg(baseline=('baseline', 'mean'),
                                                 gain_log=('gain_log', 'mean'),
                                                 gain_label=('gain_label', 'mean')).reset_index()
    stats = stats[stats.baseline <= max_baseline]
    best = None
    for a in stats.itertuples():
        for b in stats.itertuples():
            if a.arm_id >= b.arm_id or abs(a.baseline-b.baseline) > band:
                continue
            score = (a.gain_log-a.gain_label)-(b.gain_log-b.gain_label)
            if best is None or abs(score) > abs(best['score']):
                log_arm, label_arm = (a, b) if score > 0 else (b, a)
                best = dict(score=float(score), log_favoured_arm=int(log_arm.arm_id),
                            label_favoured_arm=int(label_arm.arm_id),
                            baseline_gap=float(abs(a.baseline-b.baseline)),
                            selection_kind='exploratory_selection_on_selection_seeds')
    return best


def evaluate_pair(frame, pair, n_boot=2000):
    """Paired per-seed contrasts on the evaluation seeds for the selected arms."""
    out = {}
    for role, arm_id in (('log_favoured', pair['log_favoured_arm']),
                         ('label_favoured', pair['label_favoured_arm'])):
        block = frame[frame.arm_id == arm_id]
        out[role] = dict(arm=block.arm.iloc[0], baseline=float(block.baseline.mean()),
                         gain_log=inf.seed_bootstrap_ci(block.gain_log, block.seed, n_boot=n_boot),
                         gain_label=inf.seed_bootstrap_ci(block.gain_label, block.seed, n_boot=n_boot))
    left = frame[frame.arm_id == pair['log_favoured_arm']].set_index('seed')
    right = frame[frame.arm_id == pair['label_favoured_arm']].set_index('seed')
    shared = left.index.intersection(right.index)
    baseline_gap = (left.loc[shared].baseline-right.loc[shared].baseline)
    asymmetry = ((left.loc[shared].gain_log-left.loc[shared].gain_label)
                 - (right.loc[shared].gain_log-right.loc[shared].gain_label))
    out['baseline_difference'] = inf.seed_bootstrap_ci(baseline_gap, shared, n_boot=n_boot)
    out['asymmetry'] = inf.seed_bootstrap_ci(asymmetry, shared, n_boot=n_boot)
    out['baseline_matched'] = inf.equivalence_flat(out['baseline_difference'], P.EQUIV_BAND)
    out['asymmetry_exceeds_margin'] = bool(np.isfinite(out['asymmetry']['lower'])
                                           and out['asymmetry']['lower'] > P.MIN_EFFECT)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/matched')
    parser.add_argument('--world', default='separated')
    parser.add_argument('--selection-seeds', type=int, default=12)
    parser.add_argument('--eval-seeds', type=int, default=32)
    parser.add_argument('--seed-start', type=int, default=0,
                        help='first selection seed; shards must not overlap, and the selection and '
                             'evaluation ranges stay disjoint within a shard')
    parser.add_argument('--max-baseline', type=float, default=MAX_BASELINE)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.eval_seeds < 2 or args.selection_seeds < 2 or args.factor < 2 or args.seed_start < 0:
        parser.error('At least two seeds per stage, a nonnegative seed start, '
                     'and an upgrade factor of two required')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    world = make_world(args.world)
    first_eval = args.seed_start+args.selection_seeds
    select_range = range(args.seed_start, first_eval)
    eval_range = range(first_eval, first_eval+args.eval_seeds)
    selection = arm_frame(world, CANDIDATES, args.factor, select_range)
    evaluation = arm_frame(world, CANDIDATES, args.factor, eval_range)
    selection.to_csv(out/'selection_rows.csv', index=False)
    evaluation.to_csv(out/'rows.csv', index=False)
    pair = select_pair(selection, max_baseline=args.max_baseline)
    summary = dict(config=vars(args), candidates=CANDIDATES, equivalence_band=P.EQUIV_BAND,
                   min_effect=P.MIN_EFFECT, selection=pair)
    if pair is None:
        summary['status'] = 'no_matched_pair_within_band'
    else:
        summary['status'] = 'ok'
        summary['evaluation'] = evaluate_pair(evaluation, pair, args.bootstrap)
        summary['interpretation'] = ('equal total width, unequal acquisition benefit; '
                                     'no claim about optimal budget allocation')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Matched-arm study written to {out} (status={summary["status"]})')


if __name__ == '__main__':
    main()
