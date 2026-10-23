"""
E2 arm registry, matched tuning, decision rule, and gain attribution.

Locked 2026-10-22 BEFORE the seeded effusion run, in response to review:
- `pooled_masked` and `regression_impute` join the registry so the typed arm must face
  the strongest cheap alternatives (masked loss / masked regression), not a zero-fill
  strawman. Pre-registered corollary: on magnitude estimation the typed arm is expected
  to TIE pooled_masked (its advantage there is the masked loss, which pooled_masked
  also has — typing adds no information).
- Every arm gets the same hyperparameter search space, the same trial budget, and the
  same validation split; selection criterion is identical (val masked RMSE on
  observed-PRESENT rows). Regularization dominated the diagnostics (R^2 -1.4 -> +0.14),
  so unmatched tuning would be the largest confound in the experiment.
- The decision rule (what counts as "typed wins") is fixed here before any seeded run.
- Gain attribution decomposes typed-vs-zero_fill into a masking term and a typing term;
  conditioning on features is held fixed by design (all arms see the same features).

Where typing IS expected to be categorical (registered here, tested elsewhere):
phantom-read rate on ABSENT rows, do(S=0) leakage, and estimand (b) — separability of
state and extent effects in the outcome model (make_outcome, still unused/undecided).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments_e0_e2 import PRESENT, TypedHead  # reuse the typed head as-is

ARMS: Dict[str, Dict[str, Any]] = {
    "zero_fill_pooled":  {"loss": "MSE on zero-imputed targets over non-absent rows (UNKNOWN->0)", "typed": False},
    "indicator":         {"loss": "zero-imputed magnitude MSE + BCE presence; readout p*m",        "typed": False},
    "pooled_masked":     {"loss": "MSE masked to observed-PRESENT rows only; no state head",       "typed": False},
    "regression_impute": {"loss": "masked ridge regression on features (estimator-level arm)",     "typed": False},
    "two_part":          {"loss": "state classifier + masked magnitude head (hurdle literature arm: ZiDAG-style); readout [S_hat, S_hat*M_hat] as plain scalars, no refusal", "typed": False},
    "typed":             {"loss": "3-state CE + masked magnitude MSE; readout gated by state",     "typed": True},
}

# Verified algebraically and numerically 2026-10-22 (n=400k, no hidden rows):
#   zero-filled M == S*M exactly, and [state, gated M] == [mask, zero-filled M] pointwise.
# Consequences registered BEFORE any run:
#   - the indicator arm recovers alpha and beta in Y ~ sigmoid(alpha*S + beta*S*M)
#     (alpha_hat 1.61 vs 1.5, beta_hat 0.76 vs 0.8 at n=400k; consistent), and its
#     do(S=absent) is well defined ([0,0] -> logit 0): estimand (b) as designed is a
#     predetermined tie and is NOT built.
#   - two_part and typed emit the same two scalars, so they tie wherever the query is
#     well typed. The one typed-exclusive behavior: ill-typed do(Extent=5, S=0) — the
#     indicator/two_part interface silently answers (logit beta*5, P=0.98 in the check);
#     typed syntax refuses. That is interface/type safety, not estimation.

# ---- matched tuning: identical space, budget, split, and criterion for every arm ----
VAL_FRACTION = 0.15
TUNE_GRID: Dict[str, List[float]] = {"lr": [1e-3, 3e-3], "weight_decay": [1e-4, 1e-3, 1e-2, 1e-1]}
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]   # same 8-trial budget as the neural grid
N_EPOCHS, BATCH = 40, 256

# ---- decision rule (pre-registered before the seeded run) ----
MIN_GAP = 0.10        # magnitude units; below this a gap is a tie
N_SEEDS = 10          # generator-resampled seeds (resample the generator, not the rows)
CI_LEVEL = 0.95


# ------------------------------------------------------------------ heads
class PooledHead(nn.Module):
    """Single magnitude head; used by both pooled arms (loss differs)."""
    def __init__(self, d_in: int):
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(feats)).squeeze(-1) + 1e-4


class IndicatorHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.mag = nn.Linear(d_in, 1)
        self.pres = nn.Linear(d_in, 1)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.mag(feats)).squeeze(-1) + 1e-4, self.pres(feats).squeeze(-1)


# ------------------------------------------------------------------ loss per arm
def arm_loss(name: str, module: nn.Module, feats: torch.Tensor,
             state: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    observed = (state == PRESENT) & ~torch.isnan(mag)
    if name == "zero_fill_pooled":
        non_absent = state != 1  # UNKNOWN(0) or PRESENT(2); ABSENT excluded
        z = torch.where(torch.isnan(mag), torch.zeros_like(mag), mag)
        return F.mse_loss(module(feats)[non_absent], z[non_absent])
    if name == "indicator":
        non_absent = state != 1
        z = torch.where(torch.isnan(mag), torch.zeros_like(mag), mag)
        m, logit = module(feats)
        return (F.mse_loss(m[non_absent], z[non_absent])
                + F.binary_cross_entropy_with_logits(logit[non_absent], (state[non_absent] == PRESENT).float()))
    if name == "pooled_masked":
        return F.mse_loss(module(feats)[observed], mag[observed])
    if name in ("typed", "two_part"):
        s_logits, m = module(embeddings=feats)
        m = m.squeeze(-1)                      # TypedHead emits (B, 1)
        ce = F.cross_entropy(s_logits, state)
        if observed.any():
            return ce + F.mse_loss(m[observed], mag[observed])
        return ce
    raise KeyError(name)


def train_arm(name: str, feats: np.ndarray, state: np.ndarray, mag: np.ndarray,
              hp: Dict[str, float], seed: int) -> Any:
    torch.manual_seed(seed)
    d = feats.shape[1]
    module: Any = (RidgeArm(alphas=RIDGE_ALPHAS) if name == "regression_impute"
                   else PooledHead(d) if name in ("zero_fill_pooled", "pooled_masked")
                   else IndicatorHead(d) if name == "indicator"
                   else TypedHead(d_in=d, n_states=3))   # typed and two_part share the head; readouts differ
    if name == "regression_impute":
        module.fit(feats, state, mag)
        return module

    tf = torch.from_numpy(feats).float()
    ts = torch.from_numpy(state).long()
    tm = torch.from_numpy(mag).float()          # NaN preserved; every loss masks it out
    opt = torch.optim.Adam(module.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    n = len(tf)
    for _ in range(N_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = arm_loss(name, module, tf[idx], ts[idx], tm[idx])
            loss.backward()
            opt.step()
    return module


class RidgeArm:
    """Masked ridge on standardized live features; alpha picked on the SAME val criterion."""
    def __init__(self, alphas: List[float]):
        from sklearn.linear_model import Ridge
        self.alphas, self.model, self.keep, self.mu, self.sd, self.alpha = alphas, None, None, None, None, None

    def fit(self, feats: np.ndarray, state: np.ndarray, mag: np.ndarray):
        from sklearn.linear_model import Ridge
        obs = (state == PRESENT) & ~np.isnan(mag)
        y = np.log(np.maximum(mag[obs], 1e-6))
        self.keep = feats.std(0) > 1e-3          # dead ResNet dims poison scaling (see D2)
        self.mu, self.sd = feats[:, self.keep].mean(0), feats[:, self.keep].std(0)
        X = (feats[:, self.keep] - self.mu) / self.sd
        self.model, self.alpha = Ridge(alpha=self.alphas[0]).fit(X[obs], y), self.alphas[0]
        return self

    def readout(self, feats: np.ndarray) -> np.ndarray:
        X = (feats[:, self.keep] - self.mu) / self.sd
        return np.exp(self.model.predict(X))


# ------------------------------------------------------------------ readouts per arm semantics
def arm_readout(name: str, module: Any, feats: np.ndarray,
                gated: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns (magnitude for every row, predicted-present mask or None).
    Typed semantics with gated=True: magnitude is EMITTED (non-NaN) only where predicted
    state == PRESENT. gated=False returns the ungated magnitude map — used as the tuning
    criterion, which must be identical across arms (magnitude quality on observed rows)
    and must not be NaN-ed by gate refusals. Indicator: emits p*m everywhere. Pooled: emits everywhere."""
    tf = torch.from_numpy(feats).float()
    with torch.no_grad():
        if name in ("typed", "two_part"):
            s_logits, m = module(embeddings=tf)
            m = m.squeeze(-1).numpy()
            if not gated:
                return m, None                  # tuning criterion: ungated magnitude map
            emitted = (s_logits.argmax(-1) == PRESENT).numpy()
            if name == "two_part":
                # hurdle-literature readout: plain scalars [S_hat, S_hat*M_hat], no refusal
                return emitted * m, emitted
            out = m.copy()
            out[~emitted] = np.nan              # typed: refusal instead of silent zero
            return out, emitted
        if name == "indicator":
            m, logit = module(tf)
            return (torch.sigmoid(logit) * m).numpy(), None
        if name in ("zero_fill_pooled", "pooled_masked"):
            return module(tf).numpy(), None
    if name == "regression_impute":
        return module.readout(feats), None
    raise KeyError(name)


# ------------------------------------------------------------------ tuning (matched)
def tune(name: str, feats: np.ndarray, state: np.ndarray, mag: np.ndarray,
         seed: int = 0) -> Tuple[Dict[str, float], Any]:
    """Same budget/criterion for every arm: grid over TUNE_GRID (or RIDGE_ALPHAS),
    selected by masked RMSE on observed-PRESENT rows of the held-out val split."""
    n = len(feats)
    rng = np.random.default_rng(seed)
    val_idx = rng.random(n) < VAL_FRACTION
    tr = ~val_idx
    obs_val = (state[val_idx] == PRESENT) & ~np.isnan(mag[val_idx])

    if name == "regression_impute":
        best, best_hp = None, {"alpha": None}
        best_val = np.inf
        for a in RIDGE_ALPHAS:
            arm = train_arm(name, feats[tr], state[tr], mag[tr], {"alpha": a}, seed)
            pred = arm.readout(feats[val_idx])
            v = float(np.sqrt(np.mean((pred[obs_val] - mag[val_idx][obs_val]) ** 2)))
            if v < best_val:
                best_val, best, best_hp = v, arm, {"alpha": a}
        return best_hp, best

    best, best_hp, best_val = None, None, np.inf
    for lr in TUNE_GRID["lr"]:
        for wd in TUNE_GRID["weight_decay"]:
            hp = {"lr": lr, "weight_decay": wd}
            arm = train_arm(name, feats[tr], state[tr], mag[tr], hp, seed)
            # ungated readout: the selection criterion is the magnitude map, identical
            # for every arm; gate refusals must not be able to NaN the criterion
            m_val, _ = arm_readout(name, arm, feats[val_idx], gated=False)
            v = float(np.sqrt(np.mean((m_val[obs_val] - mag[val_idx][obs_val]) ** 2)))
            if v < best_val:
                best_val, best, best_hp = v, arm, hp
    return best_hp, best


# ------------------------------------------------------------------ recovery + decisions
def hidden_present_recovery(readouts: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
                            true_state: np.ndarray, obs_state: np.ndarray,
                            true_mag: np.ndarray) -> Dict[str, Dict[str, float]]:
    """RMSE per arm on hidden rows (truly PRESENT, observed UNKNOWN).
    Refusal rule: rows where an arm emits NaN score the mean-magnitude baseline value
    (conservative imputation that penalizes wrong gating); coverage reports the refusal
    rate separately."""
    hidden = (true_state == PRESENT) & (obs_state == 0)          # UNKNOWN id
    hidden_idx = np.where(hidden)[0]
    baseline = float(np.nanmean(true_mag[true_state == PRESENT]))
    out: Dict[str, Dict[str, float]] = {}
    for name, (m, _) in readouts.items():
        m = np.asarray(m, dtype=float).copy()
        hid = m[hidden_idx]
        refused = np.isnan(hid)
        hid[refused] = baseline                  # refusal scores the mean baseline
        out[name] = {
            "rmse_hidden": float(np.sqrt(np.mean((hid - true_mag[hidden_idx]) ** 2))),
            "coverage": float(1.0 - refused.mean()),
            "baseline_rmse": float(np.sqrt(np.mean((baseline - true_mag[hidden_idx]) ** 2))),
        }
    return out


def decision_rule() -> Dict[str, Any]:
    return {"min_gap": MIN_GAP, "n_seeds": N_SEEDS, "ci": CI_LEVEL,
            "rule": ("'typed wins' estimation iff paired RMSE gap vs the best masked arm "
                     f"(pooled_masked / regression_impute / two_part) is >= {MIN_GAP} magnitude units "
                     f"with a {CI_LEVEL:.0%} t-interval across {N_SEEDS} generator-resampled "
                     "seeds excluding 0; anything smaller is a tie"),
            "tie_meaning": ("a tie vs pooled_masked/two_part is the pre-registered expectation: typing "
                            "adds no information for magnitude estimation; the contribution "
                            "then lives in the audit, phantom reads / do(S=0), and the "
                            "selection-bias result"),
            "persistent_loss_protocol": ("a typed loss is investigated, not dismissed: "
                                         "(1) tuning parity — confirm every arm ran the full grid and "
                                         "log selected hp per arm per seed; (2) leakage — val split "
                                         "disjoint from train, test untouched during tuning, generator "
                                         "seeds independent; (3) per-seed gap distribution — one-seed "
                                         "outlier vs consistent loss; (4) gating coverage on hidden "
                                         "rows — low coverage means refusal imputation, not bad "
                                         "magnitude, inflates typed RMSE. If the loss survives all "
                                         "four checks it is REPORTED AS A FINDING")}


def evaluate_decision(seed_gaps: np.ndarray, strongest_masked: str) -> Dict[str, Any]:
    """seed_gaps: per-seed paired RMSE (arm[strongest_masked] - arm['typed'])."""
    from math import sqrt
    gaps = np.asarray(seed_gaps, dtype=float)
    mean = float(gaps.mean())
    half = 2.262 * gaps.std(ddof=1) / sqrt(len(gaps))   # t_{0.975, 9}
    lo, hi = mean - half, mean + half
    if mean >= MIN_GAP and lo > 0:
        verdict = "TYPED WINS"
    elif abs(mean) < MIN_GAP:
        verdict = "TIE (expected vs masked baselines)"
    else:
        verdict = f"TYPED LOSES to {strongest_masked} — audit tuning/implementation before believing it"
    return {"mean_gap": mean, "ci95": (lo, hi), "verdict": verdict}


def attribute_gain(mean_rmse: Dict[str, float]) -> Dict[str, float]:
    """typed advantage over zero_fill_pooled = masking + conditioning + typing proper.
    Conditioning on features is held fixed by design (all arms see the same features),
    so that term is 0 and the decomposition is two-term."""
    return {
        "total": mean_rmse["zero_fill_pooled"] - mean_rmse["typed"],
        "masking": mean_rmse["zero_fill_pooled"] - mean_rmse["pooled_masked"],
        "conditioning": 0.0,   # by design: identical feature access in every arm
        "typing_proper": mean_rmse["pooled_masked"] - mean_rmse["typed"],
    }


# ------------------------------------------------------------------ identity on PREDICTED states
def identity_on_predicted_states(name: str, module: Any, feats: np.ndarray,
                                 obs_state: np.ndarray, obs_mag: np.ndarray,
                                 tau: float = 0.5) -> Dict[str, float]:
    """Does [state, gated M] == [mask, zero-filled M] survive when the state is
    PREDICTED rather than true? On hidden rows the model only has (pi_hat, m_hat);
    the two readout conventions are:
      hard gate (typed):    1{pi_hat > tau} * m_hat, refusal elsewhere
      soft product (ind.):  pi_hat * m_hat, no refusal
    The identity holds exactly only under matched HARD gating on both sides (then the
    vectors are identical). The soft product differs by m_hat * |pi_hat - 1{pi_hat>tau}|,
    largest near the decision boundary — a shrinkage/thresholding readout choice that
    BOTH arms could make, hence an estimator choice, not a typing property."""
    import torch
    tf = torch.from_numpy(feats).float()
    with torch.no_grad():
        s_logits, m = module(embeddings=tf)
        m = m.squeeze(-1).numpy()
        # PRESENT is class 2 in the 3-state head
        pi = torch.softmax(s_logits, -1).numpy()[:, 2]
    hard = (pi > tau).astype(float)
    hard_prod = hard * m
    soft_prod = pi * m
    diff = np.abs(hard_prod - soft_prod)
    return {
        "n": float(len(m)),
        "frac_hard_gate_off": float(1.0 - hard.mean()),
        "frac_diff_gt_0.01": float((diff > 0.01).mean()),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "max_abs_diff": float(diff.max()),
        "identity_under_matched_hard_gating": True,   # (hard, hard*m) == (mask, zero-filled m) by algebra
    }


# ------------------------------------------------------------------ legal-intervention statement
def legal_queries() -> Dict[str, Any]:
    """FORMAL STATEMENT (interface property, registered as a demonstration, not
    evidence — see E2_delta_leak_status). Schema: Omega = {ABSENT} U ({PRESENT} x E).
    Q_legal = {observe, do(S=0), do(S=1, E=e) for e in E} — extent is defined only
    under presence. Q_ill = {do(S=0) AND E=e} — ill-typed: assigns an extent to an
    absent finding. Per arm, which queries are expressible, and what happens to Q_ill:"""
    return {
        "schema": "Omega = {ABSENT} + ({PRESENT} x E), E subset (0, inf)",
        "Q_legal": ["observe", "do(S=absent)", "do(S=present, E=e)"],
        "Q_ill": ["do(S=absent) AND E=e>0 (extent assigned to absent finding)"],
        "arms": {
            "zero_fill_pooled": {
                "expressible": "do(score=c) for scalar c — NO state slot",
                "Q_legal_coverage": "cannot express do(S=absent) distinctly from present-at-zero-extent",
                "Q_ill": "not even well-defined enough to violate",
            },
            "indicator / two_part": {
                "expressible": "full Q_legal via [mask, filled] or [S_hat, S_hat*M_hat]",
                "Q_legal_coverage": "do(S=absent) = set both coords to 0 (verified: recovers alpha, beta)",
                "Q_ill": "ACCEPTED SILENTLY: [0, e] is a reachable input and yields a confident wrong answer "
                         "(P(Y=1)=0.98 at beta*e=4.0 in the verified check)",
            },
            "typed": {
                "expressible": "full Q_legal via the tagged union",
                "Q_legal_coverage": "do(S=absent) collapses extent to undefined by construction",
                "Q_ill": "UNREPRESENTABLE: the type system refuses to construct (ABSENT, e); the query "
                         "cannot be asked, not merely discouraged",
            },
        },
        "formal_claim": ("typed is the unique interface with Q_ill = empty AND Q_legal fully "
                         "expressible; every alternative either cannot express do(S=absent) "
                         "(pooled) or accepts ill-typed queries silently (indicator, two_part)"),
    }


# ------------------------------------------------------------------ smoke test (synthetic data only)
if __name__ == "__main__":
    print("SMOKE TEST on synthetic random data — mechanical validation only,")
    print("NOT the pre-registered effusion run.\n")
    rng = np.random.default_rng(0)
    n, d = 600, 32
    feats = rng.normal(size=(n, d)).astype(np.float32)
    true_state = rng.choice([0, 1, 2], size=n, p=[0.2, 0.65, 0.15])
    mag = np.full(n, np.nan, dtype=np.float32)
    present = true_state == PRESENT
    mag[present] = np.exp(rng.normal(1.2, 0.4, size=present.sum()))
    obs_state = true_state.copy()
    hide = rng.random(n) < 0.3
    obs_state[hide] = 0

    for name in ARMS:
        hp, arm = tune(name, feats, obs_state, mag, seed=0)
        m, emitted = arm_readout(name, arm, feats)
        assert m.shape == (n,), name
        print(f"  arm {name:<18} hp={hp} emitted={(~np.isnan(m)).mean():.2f}")

    readouts = {}
    for name in ARMS:
        _, arm = tune(name, feats, obs_state, mag, seed=0)
        readouts[name] = arm_readout(name, arm, feats)
    rec = hidden_present_recovery(readouts, true_state, obs_state, mag)
    print("\n  recovery:", {k: round(v["rmse_hidden"], 3) for k, v in rec.items()})
    print("  attribution:", {k: round(v, 3) for k, v in
          attribute_gain({k: v["rmse_hidden"] for k, v in rec.items()}).items()})
    print("  decision:", decision_rule()["rule"][:80], "...")
    print("\nSmoke test passed.")
