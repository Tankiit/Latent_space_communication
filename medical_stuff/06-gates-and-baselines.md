---
title: "06: Gates, Baselines, and Decision Checkpoints"
tags: [baselines, gates, decisions, experimental-plan, icml-2027]
status: living
prerequisites: ["[[01-claim-and-story]]", "[[04-hurdle-test-and-audit]]"]
related: ["[[05-datasets]]", "[[09-venue-icml]]"]
---

# 06: Gates, Baselines, and Decision Checkpoints

*(Carried over from the ICLR-era plan and re-evaluated for ICML 2027)*

---

## 1. Baseline Model Registry

To substantiate our claims at ICML 2027, Typed CBMs must be benchmarked against a rigorous hierarchy of prior representations:

| Baseline Model | Bottleneck Representation $\mathbf{z}_j$ | Concept Loss $\mathcal{L}_{\text{concept}}$ | Absence Semantics | Known Pathology |
| :--- | :--- | :--- | :--- | :--- |
| **Standard CBM** (Koh et al., 2020) | Scalar $\hat{c}_j \in [0, 1]$ or $\mathbb{R}$ | $\operatorname{BCE}$ or $\operatorname{MSE}$ | Extent collapsed to 0 | Conflates zero extent with absence; cannot express severity. |
| **Binary Hard CBM** | Discrete $\hat{s}_j \in \{0, 1\}$ | $\operatorname{BCE}$ | $\hat{s}_j = 0$ | Discards continuous severity; poor downstream task accuracy. |
| **Concept Embedding Model (CEM)** (Zarlenga et al., 2022) | Vector $\mathbf{e}_j \in \mathbb{R}^{d_c}$ via soft mixture | $\operatorname{BCE}(\hat{p}) + \mathcal{L}_{\text{task}}$ | $\mathbf{e}_j^- \in \mathbb{R}^{d_c}$ | **Severe Information Leakage:** Task head exploits latent dimensions `(ICLR-era)`. |
| **Uncoupled Two-Head CBM** | Pair $(\hat{s}_j, \hat{e}_j) \in \mathbb{R}^2$ | $\operatorname{BCE}(s) + \operatorname{MSE}(e)$ | Unconstrained | **Phantom States:** Predicts large extent even when $P(S=0) \approx 1$. |
| **Tobit Censored CBM** | Clamped scalar $\max(0, \mathbf{w}_j^\top x)$ | Censored Gaussian NLL | Continuous threshold | Assumes single latent index drives both onset and extent. |
| **Typed CBM** (Ours) | Tagged Union: $[s_j, s_j \cdot e_j]$ | Hurdle NLL (Two-part) | $E_j = \emptyset$ (Masked to 0) | **Zero Leakage:** Strictly respects tagged-union support. |

---

## 2. Reviewer-Driven Decision Gates

Carried over from the `(ICLR-era)` experiment sequence, we retain explicit scientific checkpoints to govern progression from synthetic prototypes to full-scale clinical experiments:

```text
Synthetic Implementation
          │
          ▼
   [ Gate 1: Synthetic Leakage Separation ]
   Does Typed CBM achieve Delta_leak = 0 while CEM and Two-Head fail?
          │
     Yes  ├───────────────────────────────┐ No: STOP and redesign bottleneck
          ▼                               │
   [ Gate 2: Hurdle Test Discrimination ]
   Does Vuong test accept Hurdle over Tobit on candidate datasets?
          │
     Yes  ├───────────────────────────────┐ No: Candidate variable is not typed
          ▼                               │
   [ Gate 3: Clinical Intervention Payoff ]
   Does do(S=0) and do(S=1, E=e) beat scalar CBM on downstream task?
          │
     Yes  ▼
   Final Paper Manuscript Assembly (ICML 2027)
```

### Gate 1: Synthetic Leakage Separation (Must pass before dataset scaling)
- **Scientific Question:** Does the multiplicative tagged-union bottleneck mathematically eliminate counterfactual leakage that plagues CEM and Uncoupled CBMs?
- **Pass Criterion:**
  - On the synthetic benchmark (`code/typed_cbm/synthetic.py`):
    - $\Delta_{\text{leak}}(\text{Typed CBM}) \equiv 0.000$ (machine precision).
    - $\Delta_{\text{leak}}(\text{Uncoupled CBM}) \ge 0.150$.
    - $\Delta_{\text{leak}}(\text{CEM}) \ge 0.200$.
- **Action if failed:** If Typed CBM exhibits non-zero leakage, verify that downstream head $h_\theta$ does not contain unmasked bias terms dependent on raw encoder activations.

### Gate 2: Hurdle Test Discrimination on Real Candidates
- **Pass Criterion:** Candidate concepts on MIMIC-CXR and MIMIC-IV must reject the single-index Tobit model in favor of the two-part Hurdle model with Vuong statistic $V > 1.96$ ($p < 0.05$).
- **Action if failed:** Discard candidate variables that fail the test (e.g., variables that are pure continuous indices) and retain only true tagged-union findings.

### Gate 3: Clinical Counterfactual Intervention Payoff
- **Pass Criterion:**
  - Under random concept interventions ($10\%$ to $50\%$ of concepts intervened), Typed CBM downstream task accuracy exceeds Standard CBM by $\ge 5\%$ macro F1.
  - Zero instances of phantom-state contradictions in clinical deployment audits.

---

## 3. Re-evaluating Dropped Constraints from ICLR and AISTATS

During the 2026 submission cycle, the research was divided across two separate venues:
1. **ICLR Track (Supervised_LVM):** Restricted exclusively to linear algebraic equivalence $R A = R$ and parameterization invariance; uncertainty quantification was forbidden by scope.
2. **AISTATS Track (EU_AU_Disentanglement / REACH):** Restricted exclusively to credal set bounds, finite-sample concentration, and diagnostic widths; concept bottleneck models were excluded.

### Unified Strategy for ICML 2027:
- **Both constraints are dropped.**
- The ICML 2027 paper integrates structural algebraic integrity (ensuring the tagged union is preserved under representation transformations) with principled uncertainty quantification (quantifying epistemic uncertainty over discrete states and aleatoric variance over extents).

---

---

## 4. Empirical Benchmark: MedMNIST (ChestMNIST) Experiments E0–E2

> [!WARNING]
> **2026-10-23 correction.** An earlier version of this section recorded single-seed cardiomegaly numbers ("drop-rows +0.55 to +0.59 ≈ unbiased under MCAR and MAR"; "slope bias −0.65 on β=0.80"; 4000/1500 caps). Those numbers carried two known defects and are RETIRED: (a) the drop-rows column was inflated by the wrong analytic truth constant `exp(1.08)=2.945` (the image-dependent generator's real target is ~3.71) — corrected MCAR bias is ≈ 0; (b) structural β=0.8 is not a valid OLS target for binary Y (sigmoid attenuation) — targets are per-seed population OLS slopes on truly-present rows. Current, corrected numbers below are 10-seed, effusion, per-seed empirical targets.

**E0 (hurdle property, effusion idx 2, 8000/3000 caps):** 0.0000 non-PRESENT rows with magnitude; exactly 1 state carries magnitude; all states ≥ 30 rows. PASSED.

**E1 (10 generator-resampled seeds; per-seed empirical targets):**
- Drop-rows mean bias: MCAR ≈ 0 (−0.02 to −0.00, CIs cover 0); **proj-keyed MAR +0.05 → +0.44**; **MNAR +0.10 → +0.92** (≈ 2× MAR at each rate). Zero-fill mean bias −1.9 → −3.4 (dominated by prevalence, shrinking as prevalence rises).
- Slope: drop-rows ≈ unbiased under MCAR, mild attenuation under MAR (range restriction), stronger under MNAR. Zero-fill slope bias is **positive** (+0.08 to +0.13, CIs exclude 0) — a registered prediction falsified in sign: imputation manufactures a zero-extent cluster that *steepens* the fitted dose-response.

**E2 (6-arm honest registry, 10 seeds, matched tuning; `results/e2_effusion_seeds.json`):** hidden-present RMSE — pooled_masked 1.722 [1.56, 1.88] · typed 1.810 [1.67, 1.95] · mean-baseline 1.807 · zero_fill_pooled 3.10 · indicator 3.77 · two_part 4.18 · regression_impute 858 (log-space extrapolation blow-up, disclosed). **Typed ties the masked arms as registered** (gap −0.088 < 0.10 band); attribution: masking +1.38 (headline), typing proper ≈ 0. Gate degeneracy: the argmax state head never predicts PRESENT at 13% prevalence → typed coverage 0, RMSE = refusal-imputed baseline. Phantom reads on absent rows: pooled/regression 100%, typed refuses 100%, indicator 63%. $\Delta_{\text{leak}} = 0$ for typed is a demonstration (unreachable input), not evidence.

---

## 5. Arm Registry and Pre-E2 Gates (2026-10-23)

| Arm | Loss | Readout | Registered expectation |
| :--- | :--- | :--- | :--- |
| `zero_fill_pooled` | zero-imputed MSE | scalar score | loses (imputation bias) |
| `indicator` | zero-imputed MSE + BCE presence | $p \cdot \hat{e}$ | loses |
| `pooled_masked` | MSE masked to observed-PRESENT | scalar score | **ties typed** (has the masked loss) |
| `regression_impute` | masked ridge on features | $\hat{e}(x)$ | ties in principle; extrapolation-prone in practice |
| `two_part` | state CE + masked MSE (hurdle-literature arm) | $[\hat{s}, \hat{s}\hat{e}]$ | **ties typed** (same head, no refusal) |
| `typed` | state CE + masked MSE | gated; refusal where not PRESENT | ties masked arms in estimation |

**Pre-E2 gates (all passed, `code/gate_and_diagnostics.py`, values in `GATES`):** G0 effusion state signal AUROC 0.750 [0.724, 0.774] (bar 0.65; cardiomegaly rejected — state acc sat at the 77% majority baseline). D1 proj↔log-extent corr 0.357/0.396. D2 generator ceiling R² = 0.20 (80% noise — the recovery comparison cannot be circular, and arm gaps are bounded by the 20% share). D3 MNAR mask: −0.141 feature part, −0.309 noise part.

**Verified algebra governing E2 (registered in `PREDICTIONS`):** zero-filled $E \equiv S\!\cdot\!E$ identically; $[\text{state}, \text{gated } E] \equiv [\text{mask}, \text{zero-filled } E]$ pointwise; the indicator arm recovers both $\alpha$ and $\beta$ (n=400k); estimand (b) as designed is a predetermined tie and is not built. Refusal rule, decision thresholds, and the persistent-loss investigation protocol are pre-registered in `e2_arms.decision_rule()`.

---

## Action Items & Status

- [x] Implement E0–E2 experimental suite in `code/experiments_e0_e2.py`.
- [x] Verify MedMNIST loader, label definitions, and resolution flags.
- [x] Pre-E2 gates G0–D3 passed; honest 6-arm registry with matched tuning; seeded E1/E2 complete (2026-10-23).
- `TODO:` Gate 1 synthetic battery vs CEM/Tobit (the table at the top of this note) remains unexecuted.
- `TODO:` Multi-finding bottleneck extension (5 major ChestMNIST findings).

