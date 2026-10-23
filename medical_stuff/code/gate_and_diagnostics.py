"""
Pre-E2 gates and generator diagnostics for the effusion switch (ChestMNIST idx 2).

Run BEFORE wiring the E2 baseline arms. Each check prints PASS/FAIL against a
pre-stated bar so the decision is mechanical:

  G0  state-signal gate: held-out AUROC (PRESENT vs ABSENT) + always-ABSENT baseline.
      Bar: AUROC >= 0.65 to proceed; AUROC <= 0.55 blocks all state-gated E2 claims.
  D1  MAR strength: corr(observed feats[:,0] projection, log-magnitude) on PRESENT rows.
      Bar: |r| >= 0.3, else re-keyed MAR again collapses to MCAR.
  D2  circularity check: held-out R^2 of log-magnitude on features (single vs all 512).
      Bar: R^2(all) <= 0.75 (noise share >= 25%), else hidden-present recovery cannot
      separate arms and the generator must change before running.
  D3  MNAR decomposition: does the MNAR mask correlate with the feature part, the noise
      part, or both? Feature-based arms can only correct the feature part.
"""

from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from experiments_e0_e2 import load_features, make_typed_annotation, PRESENT

FINDING_IDX = 2  # effusion


# ------------------------------------------------------------------ G0: state signal
def effusion_gate(
    ftr: np.ndarray, ltr: np.ndarray, fte: np.ndarray, lte: np.ndarray,
    idx: int = FINDING_IDX, n_boot: int = 1000, seed: int = 0,
) -> Tuple[float, Tuple[float, float], float, float]:
    """Held-out AUROC of PRESENT vs ABSENT (clean labels; UNKNOWN is a generator
    artifact injected later, so no rows are excluded here) + majority baseline."""
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(ftr, ltr[:, idx])
    scores = clf.predict_proba(fte)[:, 1]
    y = lte[:, idx]
    auroc = float(roc_auc_score(y, scores))
    bal = float(balanced_accuracy_score(y, scores >= 0.5))

    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], scores[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    majority = float(1.0 - y.mean())
    return auroc, (float(lo), float(hi)), majority, bal


# ------------------------------------------------------------------ D1: MAR strength
def mar_strength(feats: np.ndarray, labels: np.ndarray, idx: int = FINDING_IDX,
                 proj_idx: int = 0, seed: int = 42) -> float:
    """corr(feats[:, proj_idx], log-magnitude) on truly PRESENT rows, measured on the
    generator actually used by the experiments (p_unknown=0 -> no hiding)."""
    _, mag, true_state, _ = make_typed_annotation(
        labels, finding_idx=idx, p_unknown=0.0, mechanism="mcar", feats=feats,
        rng=np.random.default_rng(seed),
    )
    present = (true_state == PRESENT)
    r = float(np.corrcoef(feats[present, proj_idx], np.log(mag[present]))[0, 1])
    return r


# ------------------------------------------------------------------ D2: circularity
def feature_explained_share(
    ftr: np.ndarray, ltr: np.ndarray, fte: np.ndarray, lte: np.ndarray,
    idx: int = FINDING_IDX, seed_tr: int = 42, seed_te: int = 123,
) -> Tuple[float, float, float, float]:
    """Held-out R^2 of log-magnitude regressed on features (fit on train PRESENT rows,
    scored on test PRESENT rows): feats[:,0] alone, then all 512 via ridge (unregularized
    OLS at ~910x512 overfits to negative held-out R^2). Also returns the generator's
    theoretical ceiling: R^2 of any feature-based predictor is at most
    Var(0.5*tanh(proj)) / (Var(0.5*tanh(proj)) + 0.4^2); 1 - ceiling is the irreducible
    noise share no arm can recover."""
    _, mag_tr, ts_tr, _ = make_typed_annotation(
        ltr, finding_idx=idx, p_unknown=0.0, mechanism="mcar", feats=ftr,
        rng=np.random.default_rng(seed_tr))
    _, mag_te, ts_te, _ = make_typed_annotation(
        lte, finding_idx=idx, p_unknown=0.0, mechanism="mcar", feats=fte,
        rng=np.random.default_rng(seed_te))
    ptr, pte = (ts_tr == PRESENT), (ts_te == PRESENT)
    ytr, yte = np.log(mag_tr[ptr]), np.log(mag_te[pte])

    r2_single = Ridge(alpha=1.0).fit(ftr[ptr, :1], ytr).score(fte[pte, :1], yte)

    # drop near-dead ResNet dims (train-present std < 1e-3; they poison standardization),
    # then sweep shrinkage: with signal concentrated in ~1 direction, light ridge keeps
    # too much noise-direction weight and can score negative held-out R^2
    keep = ftr[ptr].std(0) > 1e-3
    mu, sd = ftr[ptr][:, keep].mean(0), ftr[ptr][:, keep].std(0)
    Xtr, Xte = (ftr[ptr][:, keep] - mu) / sd, (fte[pte][:, keep] - mu) / sd
    r2_all = max(
        Ridge(alpha=a).fit(Xtr, ytr).score(Xte, yte)
        for a in (1, 10, 100, 1000)
    )

    proj = ftr[:, 0] / (np.std(ftr[:, 0]) + 1e-6)
    v_feat = float(np.var(0.5 * np.tanh(proj)))
    ceiling = v_feat / (v_feat + 0.4 ** 2)
    return float(r2_single), float(r2_all), ceiling, 1.0 - ceiling


# ------------------------------------------------------------------ D3: MNAR decomposition
def mnar_decomposition(feats: np.ndarray, labels: np.ndarray, idx: int = FINDING_IDX,
                       rate: float = 0.3, n_seeds: int = 10) -> dict:
    """Split the MNAR mask's dependence into the feature part (proj) and the noise part
    (eps = log-mag minus its feature-determined mean). corr(mask, log-mag) shown too."""
    r_feat, r_eps, r_mag = [], [], []
    std = np.std(feats[:, 0]) + 1e-6
    proj = feats[:, 0] / std
    for seed in range(n_seeds):
        state, _, _, latent_mag = make_typed_annotation(
            labels, finding_idx=idx, p_unknown=rate, mechanism="mnar", feats=feats,
            rng=np.random.default_rng(seed))
        mask = (state == 0).astype(float)  # UNKNOWN id
        log_mag = np.log(latent_mag)
        eps = log_mag - (1.0 + 0.5 * np.tanh(proj))  # generator's exact noise term
        r_feat.append(np.corrcoef(mask, proj)[0, 1])
        r_eps.append(np.corrcoef(mask, eps)[0, 1])
        r_mag.append(np.corrcoef(mask, log_mag)[0, 1])
    return {
        "r_mask_proj": (float(np.mean(r_feat)), float(np.std(r_feat))),
        "r_mask_eps": (float(np.mean(r_eps)), float(np.std(r_eps))),
        "r_mask_logmag": (float(np.mean(r_mag)), float(np.std(r_mag))),
    }


# ------------------------------------------------------------------ driver
def main():
    ftr, ltr = load_features("chestmnist", "train", max_samples=8000)
    fte, lte = load_features("chestmnist", "test", max_samples=3000)
    n_ptr, n_pte = int(ltr[:, FINDING_IDX].sum()), int(lte[:, FINDING_IDX].sum())
    print(f"effusion (idx {FINDING_IDX}): train present {n_ptr}, test present {n_pte}")

    print("\n[G0] state-signal gate (present vs absent, held-out, clean labels)")
    auroc, (lo, hi), majority, bal = effusion_gate(ftr, ltr, fte, lte)
    print(f"  AUROC = {auroc:.3f}  (95% boot CI {lo:.3f}-{hi:.3f}) | bar: >=0.65 pass, <=0.55 block")
    print(f"  always-ABSENT accuracy = {majority:.3f} | balanced acc @0.5 = {bal:.3f}")
    g0 = "PASS" if auroc >= 0.65 else ("BLOCK" if auroc <= 0.55 else "MARGINAL")
    print(f"  >>> G0: {g0}")

    print("\n[D1] MAR strength: corr(feats[:,0], log-mag | present)")
    r1 = mar_strength(ftr, ltr)
    r1_te = mar_strength(fte, lte, seed=123)
    print(f"  train r = {r1:+.3f} | test r = {r1_te:+.3f} | bar: |r| >= 0.3")
    print(f"  >>> D1: {'PASS' if min(abs(r1), abs(r1_te)) >= 0.3 else 'FAIL'}")

    print("\n[D2] circularity: held-out R^2 of log-mag on features")
    r2_single, r2_all, ceiling, noise_share = feature_explained_share(ftr, ltr, fte, lte)
    print(f"  R^2(feats[:,0], ridge) = {r2_single:.3f} | R^2(all 512, ridge) = {r2_all:.3f}")
    print(f"  generator ceiling R^2 = {ceiling:.3f} (irreducible noise share = {noise_share:.3f})")
    print(f"  bar: ceiling <= 0.75")
    print(f"  >>> D2: {'PASS' if ceiling <= 0.75 else 'FAIL — change the generator before running recovery'}")

    print("\n[D3] MNAR decomposition (mask vs feature part / noise part, 10 seeds)")
    d3 = mnar_decomposition(ftr, ltr)
    for k, (m, s) in d3.items():
        print(f"  {k:<14} = {m:+.3f} +- {s:.3f}")


if __name__ == "__main__":
    main()
