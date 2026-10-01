# REACH: a finite-cell prototype for acquisition-conditional epistemic uncertainty

Deployment data arrives as a mixture of known sources plus possibly one novel source. This code
asks, for a declared mixture family and a declared acquisition protocol, **which uncertainty about
target label probabilities contracts when you add unlabelled inputs, labels, or class discovery,
and which uncertainty stays unidentified.**

It is a deliberately narrow anchor: a fixed finite partition of input space, two known sources plus
one candidate novel source, and a binary target. Every quantity is an interval computed by linear
programming over an explicit uncertainty set, so a claim either survives the solver or it does not.

Scope is **epistemic uncertainty only**. Labels and label noise enter as information interventions;
there is no aleatoric-uncertainty estimator, and density-ratio estimation stays diagnostic. There is
no neural optimizer and no training loop — `train` means statistical fitting, not gradient descent.

## Get it running

```bash
python -m pip install -r requirements.txt
python main.py test          # 18 numerical and integration tests, ~1 s
```

Run every command from this directory. `--help` works on the top level and on each subcommand.

```bash
python main.py train        --out results/pilot        --seeds 8 --bootstrap 50
python main.py separation   --out results/separation
python main.py matched      --out results/matched
python main.py conditionals --out results/conditionals
python main.py evaluate     cache.npz --out results/deployment --cells 8 --eta 0
```

| Command | Question it asks | What a pass looks like |
| --- | --- | --- |
| `train` | Do the intervals cover truth, and do the widths respond to each acquisition? | Coverage at or above nominal per budget; signed contrasts in the predicted direction |
| `separation` | Does a near-singular source embedding resolve slowly? | `W_cls` widens as the tangent-space gap shrinks **while coverage holds at every gap** |
| `matched` | Does one total width tell you which acquisition helps? | Two arms matched on baseline `W_total` with an asymmetry in log-gain minus label-gain |
| `conditionals` | Is the reference radius optional, and is a wrong family detected? | Nominal weight coverage with the radius, lost coverage without it, residual above radius under tilt |
| `evaluate` | Does the pipeline run on real cached features? | Acceptance, accepted error, width, and a mixture-incompatibility flag |

Each writes `rows.csv` plus `summary.json`, or for `train` the pilot bundle
(`contrasts.json`, `coverage.json`, `precision.json`, `em_baselines.json`, `widths.png`) and a
manifest with a SHA-256 of every source file. A fast end-to-end check:

```bash
python main.py train --out results/smoke --seeds 2 --n 16 32 --m 8 --eta 0 \
                     --worlds separated --bootstrap 2
```

## Start here: the synthetic worlds

**The synthetic path needs nothing from you — no data, no encoder, no cache.** It is where the
ground truth is exact, so it is both the first thing to run and the only place a validity claim can
be checked directly. Real data (item 6 below) comes much later.

Every world has three sources — two known, one candidate novel — over four input cells. Known source
weights split 0.65 / 0.35 of the non-novel mass, the novel source carries `delta` (0.1 by default),
and the per-source clean target probabilities are `theta = (0.2, 0.8, 0.9)`.

| World | Novel source `q` | What it probes |
| --- | --- | --- |
| `separated` | Mass concentrated in its own cell, outside the known convex hull | The favourable case: novel mass is identifiable, so `W_unk` should contract as the log grows |
| `confounded` | The midpoint of the two known conditionals — *inside* the hull | A **designed failure**: logs cannot identify semantic novelty there, so `W_unk` must stay wide. Narrow intervals here would be a bug |
| `absent` | Same as `separated` but with `delta = 0` | No novel mass exists; checks the machinery does not manufacture it |
| `sep{gap}` family | Fixed, with symmetric known rows | Used by `main.py separation`; the tangent-space gap is exactly `0.4 * sep * sqrt(2)`, so separation is a dial rather than a side effect |

Each row of `rows.csv` is one probe cell, and its `region` tells you how to read it:

| Region | Meaning |
| --- | --- |
| `on_support` | Both known sources put mass here; the main estimation target |
| `boundary` | Known-source overlap, where novelty contraction is observable |
| `novel_region` | Only the novel source supports this cell — so in `confounded` and `absent`, where the novel source puts no mass here, it carries zero mass too |
| `off_support` | No declared source supports it — zero mass, so the conditional is undefined and the vacuous `[0, 1]` policy fires |

Contrast analysis averages over `on_support` and `boundary` probes within each independent seed.
Inspect the region-specific rows before trusting that average, and expect `off_support` to be flat.

The predictions those contrasts are scored against live in `train/predictions.py`: a signed
expectation per (width, knob, world), where `n` is the unlabelled log size, `m` the label count, and
`eta` the known symmetric flip rate. A `0` means invariance is predicted and is tested as
equivalence; `None` means exploratory, with no claim either way. Read that file before running
anything — it is the closest thing to a protocol this prototype has.

## What you need to do

**If you are the collaborator running experiments, go to [`COLLABORATOR_PLAN.md`](COLLABORATOR_PLAN.md)** —
it has the five self-contained synthetic studies with exact commands, seed counts, pass criteria, and
measured runtimes (none longer than three minutes). The list below is the full project view.

Item 0 is a sanity check you can finish in a minute. Items 1–3 block a submission. The rest is
execution.

0. **Reproduce the synthetic suite and confirm the designed failures still fail.** `main.py test`
   then `main.py train`; check that coverage holds per budget, that `W_unk` stays wide in
   `confounded`, and that `off_support` reports the vacuous interval. If any of those "pass" too
   cleanly, something is wrong with the code, not with the theory.

1. **Supply the manuscript's posterior map and the exact `W_label` / `W_unk` definitions.** What is
   here is a declared anchor, not the paper's widths — the original formulas were never supplied.
   Replacing the anchor is expected; preserve the four interfaces the code is built around: the
   target, the acquisition protocol, the LP solver contract, and the validity checks.
2. **Decide what the paper claims about how the widths relate** — an algebraic decomposition, an
   explicitly qualified upper bound, a nested allocation, or a sensitivity statement. Pick it
   *before* any confirmatory run, and only claim additivity with a theorem behind it.
3. **Close or drop the four open theory obligations.** Each needs a result, and each already has a
   test here that would expose its absence:
   - *Prevalence contraction* — bound the feasible-weight diameter from embedding separation on the
     simplex tangent space, and control denominators for conditional targets. Discriminating test:
     grow `n` while weakening separation (`main.py separation`). Near-singularity should predict slow
     resolution, not be hidden by representation tuning.
   - *Label-information limit* — define the acquisition, and if you claim a limit, exhibit two label
     laws with the same unlabelled observable law.
   - *Novelty identification* — state the admissible novel densities `q` and the conditions
     separating novel mass from changed known prevalences. One separated example proves nothing for
     all densities, and if `q` lies in the known convex hull the logs cannot in general identify its
     novelty. Novel input mass is not the same as discovering its label.
   - *Pointwise original-feature widths* — these need density regularity and a bump scale; MMD alone
     is insufficient. Drop the claim or add the sensitivity study.
4. **Ratify or replace the decision margins.** The equivalence band and minimum effect size in
   `train/predictions.py` are proposals, not science.
5. **Run confirmatory experiments at the planned seed counts.** `precision.json` reports the
   per-contrast seed standard deviation and the seed count implied by a target half-width. Freeze
   `train/predictions.py` and the config by hash first, and disclose that the design was inspected
   beforehand — the pilot is exploratory, and its eight seeds are not a power justification.
6. **Supply one real cached-feature dataset** per the contract below, with a frozen encoder and
   genuinely disjoint pools. For ordinary multiclass image posteriors, replace the toy binary
   source-to-target map with the manuscript's multiclass map; do not silently reinterpret source
   groups as noisy labels.
7. **Integrate any baseline you intend to claim a comparison against**, matched on guarantee type.
   Robust distribution feature matching, BBSE, and PAC label-set methods are *not* implemented. What
   exists is EM with a percentile bootstrap, an explicitly failing closed-set EM, an
   oracle-augmented EM (a favourable reference, not an open-world detector), and simplex
   least-squares feature matching.

## Before you report any number

- **The four widths are distinct estimands, not an additive decomposition.** `W_cls` is the width of
  `P(Z=0 \| cell, Z known)` over feasible mixture weights, `W_label` that of `P(Y=1 \| cell, Z known)`
  over label intervals at *fixed* reference weights, `W_unk` that of `P(Z=novel \| cell)`, and
  `W_total` that of `P(Y=1 \| cell)` over weights, label intervals, and a free novel label
  probability. Never report `W_total ≈ W_cls + W_label + W_unk`. `W_label`'s invariance to log size
  follows partly from its fixed-anchor construction and is not independent evidence of
  identification. Source-group prevalence equals class prevalence only if the manuscript's semantics
  justify that correspondence.
- **Confidence statements hold per budget, not simultaneously** across a grid. A uniform guarantee
  needs a separate allocation or confidence-sequence argument.
- **Verdict blocks in `summary.json` are config-dependent.** At small reference budgets the honest
  reference radius can exceed the tilt it is meant to expose, so `shift_is_detected` reads `false`.
  That is intended conservatism. Read every verdict next to its `config` block, and never quote one
  from a smoke-sized grid.
- **Incompatibility means the declared family is wrong, not why.** The residual diagnostic does not
  identify the tilt or any other particular cause.
- **Zero-mass cells return `[0, 1]`** under a stated vacuous policy; the conditional is undefined
  there. A flat interval is not evidence that an off-support conditional was estimated.
- **The feasible set is a coordinate outer approximation** of the MMD ball, so intervals are
  conservative and can stay wide. Wide intervals at a small embedding gap are the expected result,
  not a defect. Effect decisions use proposed margins, not a power calculation.

## Real-data cache contract (item 6, not needed for synthetic work)

Nothing above requires this. `evaluate` reads one NPZ with no pickled objects:

| Key | Shape / purpose |
| --- | --- |
| `X_partition` | `N_dev x d` independent features used **only** to fit cell centres |
| `X_reference` | `N_ref x d` source-stratified reference features |
| `source_reference` | `N_ref` source IDs 0 or 1, observed reference metadata |
| `target_reference` | `N_ref` noisy binary `Y` observations; the known flip rate `eta` is a CLI argument |
| `X_log` | `N_log x d` unlabelled deployment features |
| `X_eval` | `N_eval x d` independent evaluation features |
| `target_eval` | Optional `N_eval` clean binary labels, used only after fitting |
| `q_cells` | Optional length-`cells` candidate novel distribution, declared after the partition is fitted |

All pools must be disjoint and independent — the loader cannot verify this from arrays alone, and
the guarantee fails silently if they overlap. Freeze the encoder before producing them. The default
`q` is uniform over cells, which is an **assumption**. A supplied `q_cells` must refer to the
partition this run fitted, so export centres or reproduce the partition on development data first.
No evaluation label enters estimation, and the within-source target probability is assumed constant
across cells — violations invalidate the clean-target claim.

The acceptance rule picks a class only when its posterior interval lies entirely on one side of 0.5.
That is not a PAC selective-risk or conformal label-coverage guarantee. Real data cannot give the
true posterior at each image, so treat per-input calibration as a diagnostic and report
prevalence-interval coverage across repeated deployment samples instead. Changing the label count
also moves the estimated class-conditionals, so cross-effects are expected; report them separately.

## Layout

```text
main.py              Single command-line entry point for all five commands
models/              MMD feasible sets, LP posterior widths, baseline estimators
data/                Synthetic worlds, separation and tilt families, acquisition pools, cache loading
train/               Experiment entry points: experiments, separation, matched, conditionals, cache
test/                Held-out evaluation, paired contrasts, bootstrap, coverage, replication planning
tests/               Numerical and integration regression tests
original/            Original supplied skeletons, preserved verbatim and unmodified
requirements.txt     Runtime dependencies
```

`tests/` holds software tests and `test/` holds research evaluation code; the split is deliberate.
Evaluation labels are touched only in `test/evaluate.py`.

Results directories, cached feature bundles, and working documents are untracked by design — see
`.gitignore` one level up. Everything in `results/` is reproducible from the commands above, so
regenerate it rather than asking for a copy.
