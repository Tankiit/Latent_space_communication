# Collaborator plan: self-contained synthetic experiments

This is the work that can start **today**, needs **no data, no encoder, and no GPU**, and is
embarrassingly parallel over seeds. Read `README.md` first for what the worlds and widths mean.

Everything below validates *machinery*: that the intervals cover truth, that they contract when
they should, and that the designed failures still fail. None of it settles the manuscript's
definitions — that is owner work, listed at the bottom.

## Compute is not the constraint

Measured on one laptop core, single process, no parallelism:

| Run | Wall clock |
| --- | --- |
| `train --seeds 8` (the old pilot) | ~6 s |
| `train --seeds 128` | ~80 s |
| `train --seeds 256` | ~2.7 min |
| `separation` at defaults (32 seeds, `n` to 16384) | ~6 s |
| `conditionals` at defaults (64 seeds) | ~19 s |
| `matched` at defaults (12 selection + 32 evaluation seeds) | ~16 s |
| `main.py test` (19 tests) | ~1 s |

So there is no reason for eight-seed evidence. At 256 seeds all ten paired contrasts already meet a
±0.01 bootstrap half-width (worst achieved: 0.0063). The binding constraint on this project is
deciding what the paper claims, not CPU time.

## How to shard

`train`, `separation`, `conditionals`, and `matched` all take `--seed-start`. **Shards must use
disjoint seed ranges** — without it every machine recomputes the same draws and "more seeds" buys
nothing:

```bash
python main.py train --seeds 128 --seed-start 0    --out results/A_shard0
python main.py train --seeds 128 --seed-start 1000 --out results/A_shard1   # different machine
```

Each shard is a **complete independent replication**, so the first-class check is whether two shards
agree on every decision label. Do not merge `summary.json` files by hand. If you want one pooled
analysis, concatenating `rows.csv` across disjoint-seed shards is valid — the seed is the bootstrap
unit — and you can re-analyse from Python:

```python
import pandas as pd, train.experiments as E, train.predictions as P
df = pd.concat([pd.read_csv(f'results/A_shard{i}/rows.csv') for i in (0, 1)], ignore_index=True)
contrasts, coverage = E.analyse(df)
```

For `matched`, `--seed-start` offsets the selection range and the evaluation range follows it, so the
two stay disjoint inside each shard. That split is the point of the experiment — never evaluate on
the seeds that selected the arm pair.

## A. Validity and acquisition contrasts at a defensible seed count

```bash
python main.py train --seeds 128 --seed-start 0    --out results/A_validity_s0
python main.py train --seeds 128 --seed-start 1000 --out results/A_validity_s1000
```

**Why 128:** the hardest contrast (`W_label` against `m`, seed sd ≈ 0.053) needs ~110 seeds for a
±0.01 half-width by normal approximation. `precision.json` recomputes this for your run — if it
reports a larger `max_seeds_for_target`, raise `--seeds` and say so.

**Check:** `coverage.json` — `rate` at or above nominal for every world and `n`, with
`demonstrates_at_least_nominal` true. `contrasts.json` — all ten decisions. `precision.json` —
`contrasts_meeting_target` equal to `contrasts`.

**Pass:** coverage holds at every budget, every contrast meets the precision target, and the two
shards agree on all ten decision labels.

**Report immediately, do not re-run with different margins:** any `opposite_effect`, any
`not_established` on a contrast predicted flat, or any shard disagreement.

## B. Separation sweep: does a near-singular embedding resolve slowly?

```bash
python main.py separation --seeds 128 --sep 1 .75 .5 .35 .25 .15 .1 .05 \
                          --n 1024 4096 16384 65536 --out results/B_separation
```

**Check:** `width_decreases_with_separation_at_max_n` and
`validity_preserved_at_all_separations` both true, plus `smallest_n_reaching_target` per gap — that
column *is* the resolution curve, and it is the deliverable.

**Pass:** widths widen monotonically as the gap shrinks **while coverage holds at every gap**. A
small gap reporting a narrow interval is a failure, not a success.

**Saturation trap:** below a budget threshold every width pins to the vacuous ceiling 1.0 and
becomes seed-invariant for structural reasons — at `sep=1`, `n=256` is already saturated. Equal
widths there mean nothing. Keep budgets past saturation; that is why the grid above starts at 1024.

## C. Reference error and the detection boundary

```bash
python main.py conditionals --seeds 256 --r 32 128 512 2048 8192 32768 \
                            --shift .5  --out results/C_shift050
python main.py conditionals --seeds 256 --r 32 128 512 2048 8192 32768 \
                            --shift .25 --out results/C_shift025
python main.py conditionals --seeds 256 --r 32 128 512 2048 8192 32768 \
                            --shift .1  --out results/C_shift010
```

**Check:** the four flags in `verdict`, and the per-`(r, n)` rows in `table`.

**Pass:** `positive_control_retains_coverage` and
`positive_control_rarely_flags_incompatible` true; `dropping_reference_radius_loses_coverage`
true (this is the negative control that proves the radius is not optional);
`shift_is_detected` true at adequate reference budgets.

**Known trap, and the actual deliverable:** at `r=32` the honest reference radius (≈0.72) exceeds
the tilt it is meant to expose, so `shift_is_detected` reads false. That is intended conservatism,
not a bug. The useful output is the **detection boundary**: the smallest `r` at which each shift
becomes detectable. Tabulate it as (shift × r → `incompatible_rate`).

## D. Matched arms across all three worlds

```bash
python main.py matched --world separated  --selection-seeds 24 --eval-seeds 128 --out results/D_separated
python main.py matched --world confounded --selection-seeds 24 --eval-seeds 128 --out results/D_confounded
python main.py matched --world absent     --selection-seeds 24 --eval-seeds 128 --out results/D_absent
```

**Check:** `status`, then `evaluation.baseline_matched` and
`evaluation.asymmetry_exceeds_margin`, with the `asymmetry` CI.

**Pass:** a matched pair exists whose asymmetry CI lower bound clears the margin **on evaluation
seeds**. That is the claim: equal total width does not tell you which acquisition helps.

`status: no_matched_pair_within_band` is a reportable result, not a failure — it means no two arms
in the candidate grid sit within the equivalence band. Report it rather than widening the grid.
`baseline_matched: false` with a large asymmetry means the arms were not really matched, so the
headline comparison does not hold; say so plainly.

## E. Negative controls: the designed failures must stay failures

Run this first, and again after any code change. If any of these starts "passing", it is a bug
report, not a result.

```bash
python main.py test                                   # 19 tests
python main.py train --seeds 32 --out results/E_controls
```

```python
import pandas as pd
df = pd.read_csv('results/E_controls/rows.csv')
# 1. Novelty must stay unidentified when q sits inside the known convex hull.
print(df[df.world_id == 'confounded'].groupby('n').W_unk.mean())      # must stay wide
# 2. Off-support probes must report the vacuous policy, not an estimate.
print(df[df.region == 'off_support'].W_total.unique())                # expect [1.]
# 3. No width may exceed 1 or go negative, in any world.
print(df[['W_cls', 'W_label', 'W_unk', 'W_total']].agg(['min', 'max']))
```

The third negative control for the reference radius lives in experiment C
(`dropping_reference_radius_loses_coverage` must be **true**).

## What to send back

Per run: the `summary.json` / `contrasts.json` / `coverage.json` / `precision.json` (all small), the
`config` block, and the `manifest.json` `predictions_sha256`. **Not** `rows.csv` — it is large and
regenerable from the config.

Call out explicitly: any status that is not `ok` or `optimal`, any `solver_residual` or
`solver_failure`, any shard disagreement, and any run where a verdict flipped relative to a smaller
grid.

## Rules that are not up for negotiation

- Never report `W_total ≈ W_cls + W_label + W_unk`. They are four distinct estimands.
- Do not change `EQUIV_BAND` or `MIN_EFFECT` in `train/predictions.py`, and do not widen a candidate
  grid, to make a decision come out `supported`. Those margins need owner sign-off.
- Do not quote a verdict from a smoke-sized grid, and always report the `config` next to it.
- Freeze `train/predictions.py` before confirmatory runs and report the hash the manifest records.
  If you inspected results before fixing the grid, say so — that makes the run exploratory.
- Report per-budget coverage as per-budget. It is not a simultaneous guarantee over the grid.

## Not collaborator work (owner-owned, blocking the paper)

These stay with the owner, and A–E are cheap enough to re-run once they land:

1. The manuscript's posterior map and the exact `W_label` / `W_unk` definitions. Today's widths are
   a declared anchor, so **every number from A–E is provisional on this**.
2. The paper's claim about how the widths relate — decomposition, qualified bound, nested
   allocation, or sensitivity. Decide before any run is called confirmatory.
3. The four open theory obligations in `README.md` (prevalence contraction, label-information limit,
   novelty identification, pointwise original-feature widths).
4. Real cached-feature data for `main.py evaluate`, with a frozen encoder and disjoint pools, plus
   the multiclass source-to-target map.
5. Baseline integrations matched on guarantee type (robust DFM, BBSE, PAC label sets).
