---
title: "04: Hurdle Test and Audit Battery"
tags: [audit, hurdle-test, statistical-testing, verification, icml-2027]
status: living
prerequisites: ["[[02-typed-concept-schema]]", "[[03-option-a-vs-b]]"]
related: ["[[05-datasets]]", "[[06-gates-and-baselines]]"]
---

# 04: Hurdle Test and Audit Battery

## Purpose of the Hurdle Audit

Not every clinical or tabular variable is a typed concept. Applying a typed hurdle bottleneck to a standard continuous variable (e.g., patient age, systolic blood pressure) or a pure binary flag (e.g., sex, pacemaker presence) constitutes model misspecification.

The **Hurdle Audit** is an operational verification battery that every candidate concept variable $C_j$ and dataset must pass before inclusion in the ICML 2027 experimental matrix.

---

## The Five-Step Hurdle Audit Battery

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                       Hurdle Verification Battery                      │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
   ┌─────────────────────────────────┼─────────────────────────────────┐
   ▼                                 ▼                                 ▼
Test 1: Structural Zero          Test 2: Decoupled Extent       Test 3: Leakage Audit
(Vuong / Hurdle vs Tobit)       (Var(E|S=1) > eps)             (do(S=0) Invariance)
   │                                 │                                 │
   └─────────────────────────────────┼─────────────────────────────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
             Test 4: Equivalence                 Test 5: Conditional
             (Admissible Automorphism)           Calibration
```

---

### Test 1: Structural Zero Test (Hurdle vs. One-Part Tobit / Continuous)
- **Objective:** Verify that the point mass at zero reflects a genuine structural hurdle (absence) rather than low-value censoring of a single latent Gaussian process.
- **Protocol:**
  1. Fit a two-part Hurdle model (Bernoulli gate + LogNormal/Gamma extent on positive subset):
     $$\mathcal{M}_{\text{hurdle}}: \quad P(S=1) = \pi, \quad p(E \mid S=1) = f(e; \theta)$$
  2. Fit a standard single-index Tobit (left-censored at 0) or zero-truncated continuous model $\mathcal{M}_{\text{tobit}}$.
  3. Compute the **Vuong Likelihood Ratio Test** for non-nested models:
     $$V = \frac{\sqrt{N} \left( \frac{1}{N} \sum_{i=1}^N m_i \right)}{\sqrt{\frac{1}{N} \sum_{i=1}^N (m_i - \bar{m})^2}}, \quad m_i = \log \frac{p_{\text{hurdle}}(c_i)}{p_{\text{tobit}}(c_i)}$$
- **Admission Criterion:** Reject $H_0$ (equivalence) with $V > +1.96$ ($p < 0.05$), confirming that the two-part process significantly outperforms single-index continuous censoring.
- **Empirical Hurdle Condition:** Proportion of zeros must satisfy $0.10 \le P(C_j = 0) \le 0.90$.

---

### Test 2: State-Extent Decoupling Test
- **Objective:** Confirm that extent $E$ carries non-trivial continuous information conditional on presence ($S=1$).
- **Protocol:**
  1. Filter the dataset to the active subset $\mathcal{D}_+ = \{x : s_j = 1\}$.
  2. Compute the conditional coefficient of variation:
     $$\operatorname{CV}(E_j \mid S_j = 1) = \frac{\sigma(E_j \mid S_j = 1)}{\mu(E_j \mid S_j = 1)}$$
  3. Verify that $E_j$ is not a deterministic function of $S_j$.
- **Admission Criterion:** $\operatorname{CV}(E_j \mid S_j = 1) \ge 0.15$ and the unique value count $|\operatorname{unique}(E_j \mid S_j = 1)| \ge 10$. If a variable has only 1 or 2 positive levels, it is re-classified as a categorical/ordinal concept, not a continuous extent concept.

---

### Test 3: Absence Invariance Audit (Counterfactual Leakage)
- **Objective:** Ensure that intervening on absence $do(S_j = 0)$ strictly prohibits extent information from leaking into the task prediction $h_\theta$.
- **Protocol:**
  1. Sample an input $x$ where ground-truth $s_j = 1$ and extent is $e_j$.
  2. Perform intervention $do(S_j = 0)$.
  3. Sweep the latent extent encoder output over extreme perturbations: $e' \in \{\min(E), \operatorname{median}(E), \max(E), 10 \times \max(E)\}$.
  4. Measure the downstream prediction divergence:
     $$\Delta_{\text{leak}}(x) = \sup_{e'} \left\| h_\theta(\mathbf{z}(S_j = 0, E_j = e')) - h_\theta(\mathbf{z}(S_j = 0, E_j = \emptyset)) \right\|$$
- **Admission Criterion:** $\Delta_{\text{leak}} \equiv 0$ (exact machine zero, $< 10^{-7}$).
  - *Standard uncoupled two-head CBMs fail this test* ($\Delta_{\text{leak}} > 0.15$).
  - *CEMs fail this test* due to unconstrained linear mixing weights `(ICLR-era)`.

---

### Test 4: Admissible Transformation and Automorphism Audit `(ICLR-era)`
- **Background `(ICLR-era)`:** In linear readout CBMs, representations $c$ can be transformed by any $A \in \operatorname{GL}(d)$ preserving task logits $h(Ac) = h(c)$, creating severe parameter unidentifiability.
- **Protocol:**
  1. Characterize the symmetry group $\mathcal{G}_{\text{typed}}$ of linear transformations $A$ that preserve the tagged-union zero cone:
     $$\mathcal{Z}_0 = \{ \mathbf{z} \in \mathbb{R}^{2K} : z_{2j} = 0 \iff z_{2j-1} = 0 \}$$
  2. Verify that admissible $A$ must be block-diagonal with respect to the state-extent pairs:
     $$A = \operatorname{diag}(A_1, \dots, A_K), \quad A_j = \begin{bmatrix} a_{11}^{(j)} & 0 \\ a_{21}^{(j)} & a_{22}^{(j)} \end{bmatrix}$$
  3. Verify that zero-state preservation prevents cross-concept rotation leakage between unrelated clinical findings.
- **Admission Criterion:** No rotation $A$ mixing concept $j$ into concept $k$ can preserve zero-state invariance.

---

### Test 5: Conditional Calibration Audit
- **Objective:** Evaluate calibration separately on the discrete gate and the conditional continuous density.
- **Protocol:**
  1. **Gate Calibration:** Compute Expected Calibration Error (ECE) and Brier score on $P(S_j = 1 \mid x)$ using the entire validation split.
  2. **Extent Calibration:** Conditional on $S_j = 1$, evaluate the empirical coverage of $(1 - \alpha)$ prediction intervals:
     $$\operatorname{Coverage}_{1-\alpha} = \frac{1}{|\mathcal{D}_+|} \sum_{i \in \mathcal{D}_+} \mathbb{I}\left( e_i \in [\hat{q}_{\alpha/2}(x_i), \hat{q}_{1-\alpha/2}(x_i)] \right)$$
- **Admission Criterion:** $\operatorname{ECE}_{\text{gate}} \le 0.05$ and $|\operatorname{Coverage}_{0.90} - 0.90| \le 0.04$.

---

## Action Items & Status

- `TODO:` Implement automated audit script in `code/typed_cbm/hurdle_audit.py`.
- `TODO:` Run Vuong test across all candidate datasets listed in [[05-datasets]]. **Note 2026-10-23: the real-data audits below are hurdle-STRUCTURE checks (zero placement, decoupling, jointness); the formal Vuong non-nested LR test (Test 1's full protocol) is still unimplemented — do not conflate the two in the paper.**
- `VERIFY:` Confirm whether statsmodels / scipy has built-in support for Vuong non-nested hypothesis testing or write a standalone utility.

---

## Audit Results (2026-10-23)

All numbers reproducible; result files cited per the index convention.

### A. PUMS WKHP (ACS 2023 1-Year, WY + DC) — `results/pums_wkhp_audit.json`
- Source: `https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_p{ST}.zip` (postal codes; the 1-year PUMS is a ~1% sample — WY 6,024 / DC 6,735 person records). Unweighted.
- **Zeros never enter the active branch**: $P(\text{WKHP}=0 \mid \text{employed}) = 0/6{,}727$ across both states (binomial 95% upper bound ≈ 5.5×10⁻⁴). PASS.
- **"WKHP==0 equals non-workers" is FALSE as stated**: zeros are exclusively non-employed, but only 84.2% (WY) / 76.5% (DC) of non-employed are zero — WKHP is a past-12-months variable while ESR is current employment; ~16–24% of currently non-employed worked earlier in the year. The clean typed variable is state = *worked at all in the past year*, extent = WKHP | worked (hurdle then partly definitional; the non-trivial evidence is strict positivity + spread inside employment).
- Admission band: zero fractions 0.375 (WY) / 0.251 (DC) ∈ [0.10, 0.90]. Decoupling: CV 0.351 / 0.294, 65–72 unique values. Cross-state shift present (means 39.48 vs 40.45 hours).

### B. PAD-UFES-20 diameter + refusal structure — `results/pad_ufes_isic_partial_audit.json`
- Source: ISIC Archive API v2 collection 406 (`/api/v2/images/search/?collections=406`), 2,298 records, 1,373 patients (matches the data descriptor exactly). Raw CSV: Mendeley doi 10.17632/zr7vgbcyr2.1, **CC BY 4.0**; descriptor paper is *Data in Brief* doi 10.1016/j.dib.2020.106221.
- **Extent branch passes**: diameter present on 1,490 lesions, **0 zeros**, 1–100mm, mean 11.97, CV 0.72, 41 unique values.
- **Missingness is patient-level refusal**: sex / Fitzpatrick / diameter missing at 0.350 / 0.350 / 0.352; **joint fraction 0.995**; the sex- and Fitzpatrick-missing sets are identical; **all 579 affected patients are affected on every lesion (zero partial cases)**. A real-world UNKNOWN mechanism operating at the patient level — patient-clustered MNAR, connecting directly to the E1 selection-bias result.
- Caveat: ISIC harmonization drops the 22 boolean clinical fields and collapses raw `UNK` strings to nulls. `TODO:` blank-vs-`UNK` **encoding** audit needs the raw Mendeley CSV (browser download; then minutes of work).

### C. ChestMNIST generator (E0) — passes at 8000/3000 caps on effusion (idx 2): 0 non-PRESENT rows carry magnitude; exactly 1 state carries the magnitude slot; all three states ≥ 30 rows. See `code/experiments_e0_e2.py` driver.
