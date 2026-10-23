"""
Seeded E2 hidden-present recovery run (registered negative control).

10 generator-resampled seeds, 6 arms, matched tuning per seed, decision rule and gain
attribution from e2_arms. Writes results/e2_effusion_seeds.json and prints the
comparison table. Effusion (idx 2), MCAR unknown rate 0.3, caches 8000/3000.
"""

import json
import os
from math import sqrt

import numpy as np

from experiments_e0_e2 import load_features, make_typed_annotation, ABSENT, PRESENT, UNKNOWN
from e2_arms import (ARMS, tune, arm_readout, hidden_present_recovery,
                     decision_rule, evaluate_decision, attribute_gain, N_SEEDS)

RATE = 0.3
FINDING = 2


def phantom_rates(readouts, true_state, eps=0.05):
    """Interface metrics on truly-ABSENT rows: fraction emitting a positive severity
    (> eps) vs fraction refusing (NaN). The typed/two_part/indicator contracts emit 0
    or NaN by design; pooled and regression_impute always emit a positive number."""
    absent = (true_state == ABSENT)
    out = {}
    for name, (m, _) in readouts.items():
        m = np.asarray(m, dtype=float)
        out[name] = {
            "emit_positive_rate": float((m[absent] > eps).mean()),
            "refusal_rate": float(np.isnan(m[absent]).mean()),
        }
    return out


def main():
    feats_tr, labels_tr = load_features("chestmnist", "train", max_samples=8000)
    feats_te, labels_te = load_features("chestmnist", "test", max_samples=3000)

    per_seed = []
    for seed in range(N_SEEDS):
        st_tr, mg_tr, ts_tr, _ = make_typed_annotation(
            labels_tr, finding_idx=FINDING, p_unknown=RATE, mechanism="mcar",
            feats=feats_tr, rng=np.random.default_rng(seed))
        st_te, mg_te, ts_te, latent_te = make_typed_annotation(
            labels_te, finding_idx=FINDING, p_unknown=RATE, mechanism="mcar",
            feats=feats_te, rng=np.random.default_rng(1000 + seed))

        readouts, rmses = {}, {}
        for name in ARMS:
            _, arm = tune(name, feats_tr, st_tr, mg_tr, seed=seed)
            readouts[name] = arm_readout(name, arm, feats_te)
        # true_mag MUST be the latent magnitudes: hidden rows are observed UNKNOWN, so
        # the observed magnitudes are NaN there and would poison every RMSE
        rec = hidden_present_recovery(readouts, ts_te, st_te, latent_te)
        ph = phantom_rates(readouts, ts_te)
        for name in ARMS:
            rmses[name] = rec[name]["rmse_hidden"]
        masked = ["pooled_masked", "regression_impute", "two_part"]
        strongest = min(masked, key=lambda a: rmses[a])
        per_seed.append({
            "rmses": rmses,
            "coverage": {a: rec[a]["coverage"] for a in ARMS},
            "baseline_rmse": rec["typed"]["baseline_rmse"],
            "gap_vs_strongest_masked": rmses[strongest] - rmses["typed"],
            "strongest_masked": strongest,
            "phantom": ph,
        })
        print(f"seed {seed}: " + " ".join(f"{a}={rmses[a]:.3f}" for a in ARMS)
              + f" | strongest_masked={strongest}", flush=True)

    def t95(xs):
        xs = np.asarray(xs, dtype=float)
        half = 2.262 * xs.std(ddof=1) / sqrt(len(xs))
        return [float(xs.mean() - half), float(xs.mean() + half)]

    summary = {
        "config": {"finding": "effusion", "finding_idx": FINDING, "rate": RATE,
                   "n_seeds": N_SEEDS, "caps": [8000, 3000], "mechanism": "mcar"},
        "mean_rmse": {a: float(np.mean([s["rmses"][a] for s in per_seed])) for a in ARMS},
        "rmse_ci95": {a: t95([s["rmses"][a] for s in per_seed]) for a in ARMS},
        "mean_coverage": {a: float(np.mean([s["coverage"][a] for s in per_seed])) for a in ARMS},
        "mean_baseline_rmse": float(np.mean([s["baseline_rmse"] for s in per_seed])),
        "decision": evaluate_decision(
            np.array([s["gap_vs_strongest_masked"] for s in per_seed]),
            per_seed[0]["strongest_masked"]),
        "attribution": attribute_gain(
            {a: float(np.mean([s["rmses"][a] for s in per_seed])) for a in ARMS}),
        "phantom_on_absent": {
            a: {
                "emit_positive_rate": float(np.mean([s["phantom"][a]["emit_positive_rate"] for s in per_seed])),
                "refusal_rate": float(np.mean([s["phantom"][a]["refusal_rate"] for s in per_seed])),
            } for a in ARMS},
        "rule": decision_rule()["rule"],
        "per_seed": per_seed,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/e2_effusion_seeds.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== E2 seeded summary (hidden-present RMSE, mean [95% CI] over 10 seeds) ===")
    for a in ARMS:
        lo, hi = summary["rmse_ci95"][a]
        print(f"  {a:<18} {summary['mean_rmse'][a]:.3f} [{lo:.3f}, {hi:.3f}]  cov={summary['mean_coverage'][a]:.2f}")
    print(f"  mean-magnitude baseline  {summary['mean_baseline_rmse']:.3f}")
    d = summary["decision"]
    print(f"\n  decision vs {per_seed[0]['strongest_masked']}: mean gap {d['mean_gap']:+.3f} "
          f"CI95 [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] -> {d['verdict']}")
    print("  attribution:", {k: round(v, 3) for k, v in summary["attribution"].items()})
    print("  phantom on ABSENT rows (emit-positive / refusal):")
    for a in ARMS:
        p = summary["phantom_on_absent"][a]
        print(f"    {a:<18} {p['emit_positive_rate']:.2f} / {p['refusal_rate']:.2f}")


if __name__ == "__main__":
    main()
