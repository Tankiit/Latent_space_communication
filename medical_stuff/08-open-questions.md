---
title: "08: Open Questions and Unresolved Decisions"
tags: [open-questions, decisions, backlog, owners-blank, icml-2027]
status: living
prerequisites: ["[[index]]"]
related: ["[[01-claim-and-story]]", "[[03-option-a-vs-b]]", "[[09-venue-icml]]"]
---

# 08: Open Questions and Unresolved Decisions

This note registers all active theoretical, algorithmic, and empirical uncertainties in the Typed Concepts project for ICML 2027. Per project conventions, **owners are left blank** for team assignment.

---

## 1. Theoretical Uncertainties

### Q1: Identifiability Conditions for Option B (Weakly-Supervised Extent)
- **Question:** If ground-truth extent is observed for only a small fraction $\rho \in (0, 1]$ of positive cases, what are the necessary and sufficient conditions on encoder representation rank and head convexity to guarantee identifiability of $E$?
- **Impact:** Determines whether Option B [[03-option-a-vs-b]] can be presented as a proved theorem or only an empirical heuristic.
- **Reference:** Connects to Kuroki & Pearl (2014) proxy models and `(ICLR-era)` nullity bounds.

### Q2: Measure-Theoretic Propagation of Credal / Epistemic Uncertainty
- **Question:** In the dropped AISTATS plan (REACH), uncertainty was bounded via empirical cell frequencies. In a tagged union $\Omega = \{0\} \amalg (\{1\} \times \mathbb{R}^+)$, how does a credal interval over the Bernoulli parameter $\pi \in [\underline{\pi}, \overline{\pi}]$ propagate through downstream predictor $h_\theta(s, s \cdot e)$?
- **Impact:** Unifies the REACH uncertainty track with the CBM structural track.

### Q3: Extension to Nested Multi-Tier Categorical Unions
- **Question:** In oncology (e.g., TNM staging), findings have multi-branch typed unions:
  $$\text{Tumor} \in \{\text{T0 (Absent)}\} \cup \{\text{T1-T4} \times \text{Size (mm)}\} \cup \{\text{Tis (In Situ)}\}$$
  Does our hurdle likelihood easily generalize to Dirichlet-Multinomial or Categorical-Hurdle trees without numerical instability?

---

## 2. Methodological & Algorithmic Uncertainties

### Q4: Optimization Through the Discrete Gate
- **Question:** During end-to-end training of encoder + bottleneck + task head, how should we backpropagate through the discrete state $S_j$?
  - *Option 1:* Straight-Through Gumbel-Softmax estimator.
  - *Option 2:* Continuous relaxation during training: $\mathbf{z}_j = [\hat{\pi}_j(x), \hat{\pi}_j(x) \cdot \hat{e}_j(x)]$, discretizing only at test time / intervention.
  - *Option 3:* Decoupled two-stage training (train encoder with hurdle NLL, freeze, then train $h_\theta$).
- **Status:** Option 2 implemented as default in `code/typed_cbm/layers.py`; empirical sensitivity needs benchmarking.

### Q5: Inter-Concept Covariance in Extent Spaces
- **Question:** If cardiomegaly and pleural effusion co-occur, are their extents conditionally independent given $x$, or must we parameterize a multivariate hurdle distribution (e.g., Gaussian Copula hurdle)?
- **Trade-off:** Parameter complexity vs. modeling fidelity.

---

## 3. Empirical & Benchmark Uncertainties

### Q6: Primary Lead Dataset: Imaging vs. Tabular
- **Question:** Should Figure 1 and the primary experimental narrative center on:
  - **Chest X-ray (MIMIC-CXR / CheXmask):** High visual impact, highly intuitive for clinical reviewers, but requires handling image segmentations.
  - **MIMIC-IV Tabular EHR:** Cleanest numerical extents (lab units, creatinine, troponin), zero computer-vision confounds, but less visually striking.
- **Impact:** Sets the tone for Section 5 of the manuscript.

### Q7: Automated Protocol for Counterfactual Leakage ($\Delta_{\text{leak}}$)
- **Question:** When auditing baseline models (CEM and Uncoupled CBMs), what is the standardized perturbation set for latent extent $e'$ to ensure fair and worst-case leakage bounds across diverse random seeds?

---

## 4. Master Decision & Task Register

*Updated 2026-10-23 against the CLeaR 2027 deadline (Nov 16, 2026; see [[09-venue-icml]]). Resolved items keep their ID with a dated status.*

| ID | Category | Question / Action Item | Priority | Target Date | Owner | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T-01** | Theory | Prove absence invariance of block-diagonal automorphism group | P2 | — | | `DEFERRED` (post-CLeaR; estimation ties make it secondary) |
| **T-02** | Theory | Draft formal identifiability proposition for Option B | P2 | — | | `DEFERRED` (Option B deferred, see [[03-option-a-vs-b]] resolution) |
| **M-01** | Method | Discrete-gate training (Gumbel vs soft relaxation) | P1 | 2026-11-08 | | `RESCOPED`: the seeded run found argmax gates never fire at 13% prevalence (coverage 0) — the live question is gate *thresholds/calibration* (e.g. prevalence-aware τ or cost-sensitive), not the gradient estimator |
| **M-02** | Method | Copula hurdle layer for correlated concepts | P2 | — | | `DEFERRED` |
| **D-01** | Data | CheXmask CTR/effusion extraction | P2 | — | | `DEFERRED` (superseded for CLeaR by PUMS + PAD-UFES-20 audits, D-03/D-04) |
| **D-02** | Data | MIMIC-IV cohort | P2 | — | | `DEFERRED` (PhysioNet access unverified) |
| **D-03** | Data | PAD-UFES-20 blank-vs-`UNK` encoding audit (needs raw Mendeley CSV, CC BY 4.0) | **P0** | 2026-11-01 | | `TODO` — the only missing real-data piece; ISIC partial audit done ([[04-hurdle-test-and-audit]]) |
| **D-04** | Data | PUMS WKHP audit | — | — | | `DONE 2026-10-23` (`results/pums_wkhp_audit.json`) |
| **E-01** | Expts | Gate 1 synthetic leakage audit vs CEM/Tobit, 10 seeds | P1 | 2026-11-10 | | `TODO` |
| **E-02** | Expts | Derm7pt comparison | P2 | — | | `DEFERRED` (PAD-UFES-20 took the dermatology slot) |
| **E-03** | Expts | Seeded E1 + honest-arms E2 on effusion | — | — | | `DONE 2026-10-23` (`results/e2_effusion_seeds.json`; E1 table in [[06-gates-and-baselines]]) |
| **E-04** | Expts | Formal Vuong non-nested LR test implementation (Test 1 full protocol) | P1 | 2026-11-08 | | `TODO` — current audits are structure checks only |
| **W-01** | Writing | Section 3 (schema & hurdle loss) | **P0** | 2026-11-08 | | `TODO` |
| **W-02** | Writing | CLeaR manuscript assembly (not-a-method-paper framing; abstract states it) | **P0** | 2026-11-13 | | `TODO` |
| **W-03** | Writing | Figure 1: patient-level refusal structure (PAD-UFES-20 joint missingness) + E1 MNAR drift | P1 | 2026-11-10 | | `TODO` |
| **X-01** | Admin | Verify Causal_CBM M4/M5 "completed" claims (dashboard marks future-dated milestones done) | P1 | 2026-10-28 | | `TODO` |
