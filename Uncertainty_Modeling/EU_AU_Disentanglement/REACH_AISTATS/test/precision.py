"""Replication planning for the paired acquisition contrasts.

The pilot's eight seeds were never a power justification. This module reports, per contrast,
the independent-seed standard deviation, the achieved bootstrap half-width, and the seed count
implied by a target half-width. A normal-approximation seed count is a planning figure, not a
guarantee about the bootstrap interval actually used for inference.
"""
import numpy as np
from test.inference import paired_contrast, seed_bootstrap_ci

REGIONS = ('on_support', 'boundary')


def contrast_precision(df, predicted, target_half_width=.01, n_boot=2000):
    rows = []
    for (width, knob, world_id), sign in predicted.items():
        sub = df[(df.world_id == world_id) & df.region.isin(REGIONS)]
        levels = sorted(sub[knob].unique())
        if len(levels) < 2:
            continue
        pairs = paired_contrast(sub, width, knob, levels[0], levels[-1])
        means = pairs.groupby('seed').difference.mean()
        sd = float(means.std(ddof=1)) if len(means) > 1 else np.nan
        ci = seed_bootstrap_ci(pairs.difference, pairs.seed, n_boot=n_boot)
        half = (ci['upper']-ci['lower'])/2 if np.isfinite(ci['lower']) else np.nan
        needed = int(np.ceil((1.96*sd/target_half_width)**2)) if np.isfinite(sd) and sd > 0 else 1
        rows.append(dict(width=width, knob=knob, world_id=world_id, predicted_sign=sign,
                         estimate=ci['estimate'], seed_sd=sd, independent_seeds=int(len(means)),
                         achieved_half_width=half, target_half_width=target_half_width,
                         seeds_for_target=needed,
                         target_met=bool(np.isfinite(half) and half <= target_half_width)))
    return rows


def summarise(rows, target_half_width=.01):
    usable = [r for r in rows if np.isfinite(r['achieved_half_width'])]
    return dict(per_contrast=rows, target_half_width=target_half_width,
                contrasts=len(rows),
                contrasts_meeting_target=sum(r['target_met'] for r in rows),
                worst_achieved_half_width=max((r['achieved_half_width'] for r in usable),
                                              default=None),
                max_seeds_for_target=max((r['seeds_for_target'] for r in rows), default=None),
                note='normal-approximation planning figure; the reported CIs are seed bootstrap')
