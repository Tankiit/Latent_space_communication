---
title: "09: Target Venue: ICML 2027 Strategy, Timeline, and Dropped Constraints"
tags: [venue, icml-2027, strategy, timeline, constraints]
status: living
prerequisites: ["[[index]]", "[[01-claim-and-story]]"]
related: ["[[03-option-a-vs-b]]", "[[06-gates-and-baselines]]", "[[08-open-questions]]"]
---

# 09: Target Venue: ICML 2027 Strategy, Timeline, and Dropped Constraints

## 1. Context: Dropping the ICLR and AISTATS Split

During the 2026 conference cycle, the research agenda was split across two distinct submissions with conflicting constraints:
1. **The ICLR Plan (`Supervised_LVM`):**
   - Focused strictly on linear algebraic equivalence $R A = R$, matrix rank/nullity, and coordinate-dependent diagnostic sensitivity.
   - *Artificial Constraint:* Strictly prohibited uncertainty quantification, credal sets, or non-linear downstream predictors to keep the matrix perturbation proofs simple.
2. **The AISTATS Plan (`REACH` / `EU_AU_Disentanglement`):**
   - Focused strictly on finite-sample concentration inequalities, empirical cell frequencies, and diagnostic widths for epistemic uncertainty.
   - *Artificial Constraint:* Excluded Concept Bottleneck Models entirely, treating representations only as abstract partitions.

### The ICML 2027 Synthesis
Both separate plans have been dropped. This allows us to unify the strongest elements of both tracks into a single, cohesive, high-impact submission for **ICML 2027**:
- The **algebraic structural integrity** from the ICLR track (ensuring concepts have well-defined geometric support that prohibits shortcut leakage).
- The **principled uncertainty quantification** from the AISTATS track (disentangling discrete classification uncertainty from continuous measurement uncertainty).

---

## 2. Re-Decision on Dropped Constraints

The table below catalogs constraints that existed solely to separate the two earlier papers, along with their resolution for ICML 2027:

| ICLR-era Constraint | Purpose in Old Plan | Status for ICML 2027 | Re-Decision Justification |
| :--- | :--- | :--- | :--- |
| **No Uncertainty Quantification** | Avoided overlap with AISTATS REACH paper. | **REMOVED** | ICML demands calibrated uncertainty; hurdle layers naturally provide separate state entropy and extent variance. |
| **Linear Task Heads Only ($h(c) = Wc$)** | Simplified exact matrix compensation proofs ($W A^{-1}$). | **REMOVED** | The tagged-union bottleneck enforces $\mathbf{z}(S=0, E) = \mathbf{0}$, which guarantees absence invariance for *arbitrary* non-linear predictors $h_\theta$. |
| **Synthetic-Only Proof Checkpoint** | Stopped after synthetic Step 4 if diagnostics didn't separate. | **MODIFIED** | Retained as Gate 1 (`[[06-gates-and-baselines]]`), but real medical data (MIMIC-CXR / CheXmask) is now an essential primary deliverable. |
| **No Real Medical Data** | Kept experiments restricted to CIFAR / CUB / Shapes3D. | **REMOVED** | Typed concepts are uniquely motivated by clinical pathology (state + extent); medical datasets provide the strongest scientific justification. |
| **Exclusion of Conformal Inference** | Reserved for a downstream calibrated uncertainty track. | **FLAGGED FOR DECISION** | `TODO:` Decide whether conformal prediction sets on tagged unions should be included in the main paper or left as future work. |

---

## 3. What ICML Reviewers Expect

To achieve a clear accept at ICML 2027, the manuscript must balance four core dimensions:

1. **Theoretical Foundations:**
   - Formal measure-theoretic definition of the tagged-union hurdle measure $\nu = \delta_0 + \lambda_+$.
   - Mathematical proof that absence invariance ($\Delta_{\text{leak}} = 0$) holds under the multiplicative bottleneck, whereas CEM and uncoupled CBMs have non-zero leakage bounds.
2. **Methodological Novelty:**
   - First paper to identify the "Categorical Fallacy" in CBMs and provide an algebraic sum-type solution.
   - Clean, modular PyTorch implementation that drops into existing CBM pipelines.
3. **Empirical Rigor & Benchmarking:**
   - Execution across $\ge 10$ random seeds with proper statistical error bars and multiple-testing corrections.
   - Strict adherence to the [[04-hurdle-test-and-audit]] before declaring variables as typed concepts.
   - Comparison against all established baselines: Standard CBM, Hard CBM, CEM, Uncoupled Two-Head, and Tobit.
4. **Clinical & Scientific Realism:**
   - Real-world validation on chest radiographs (pleural effusion / cardiomegaly) and ICU EHR data (acute kidney injury / troponin).

---

## 4. Near-Term Venue: CLeaR 2027 (verified 2026-10-23)

**The near-term submission is CLeaR 2027, deadline Nov 16, 2026 AoE** (official CFP, verified): reviews Jan 19, rebuttals Jan 26, decisions Feb 22, 2027, conference May 3–5 (Paris). Dual submissions to another venue's proceedings are banned during the whole review window — the *same* work cannot go to both CLeaR and ICML 2027 (ICML's projected ~Jan 22–28 deadline is aggregator-only, no official CFP). Accepted papers may opt out of proceedings for a 1-page extended abstract. The CFP states application-focused papers need not introduce new methodology — the typed-concepts paper is **not a method paper** (E1 selection bias with known truth + audit battery + schema + interface guarantees; estimation ties masked hurdle baselines), and should say so in the abstract.

**Remaining to Nov 16 (2026-10-23 status):** PAD-UFES-20 raw-CSV encoding audit (D-03) → Gate 1 synthetic battery (E-01) → Vuong implementation (E-04) → manuscript W-01/W-02/W-03, with the register in [[08-open-questions]] as the working checklist. Completed and citable: seeded E1 (incl. the slope falsification), seeded honest-arms E2, PUMS + PAD ISIC audits, gates G0–D3, verified algebra.

## 5. Master Timeline to ICML 2027

```text
2026                 2026                 2026                 2027
OCTOBER              NOVEMBER             DECEMBER             JANUARY
 ├── Vault Setup      ├── Baseline Coding  ├── Medical Runs     ├── Final Draft
 ├── Synthetic Code   ├── Gate 1 Passing   ├── Tabular Runs     ├── Visual Figures
 └── Datasets Audit   └── CheXmask Data    └── Ablation Sweeps  └── Submission
```

| Phase | Milestone Deliverables | Target Window | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1: Foundations** | Complete Vault, implement `code/typed_cbm/` package, synthetic generator. | Oct 20 - Oct 31, 2026 | 🔄 In Progress |
| **Phase 2: Gate 1 & Baselines** | Run Gate 1 leakage audit across all baselines; implement Hurdle Test battery. | Nov 01 - Nov 20, 2026 | ⏳ Upcoming |
| **Phase 3: Medical Extraction** | Extract CTR and effusion from CheXmask; build MIMIC-IV cohort. | Nov 15 - Dec 05, 2026 | ⏳ Upcoming |
| **Phase 4: Full Benchmark Matrix** | Execute multi-seed runs on synthetic, CheXmask, and MIMIC-IV. | Dec 05 - Dec 24, 2026 | ⏳ Upcoming |
| **Phase 5: Manuscript Assembly** | Draft full LaTeX manuscript; generate publication-grade figures. | Dec 25 - Jan 15, 2027 | ⏳ Upcoming |
| **Phase 6: Submission** | Final polish, proofreading, supplemental code bundle upload. | Late Jan 2027 | 🎯 Target |

---

## Action Items & Next Steps

- `TODO:` Establish LaTeX repository skeleton using standard ICML 2027 style files.
- `TODO:` Decide on conformal prediction inclusion with the research team.
- `VERIFY:` Confirm exact ICML 2027 submission deadline date once announced by IMLS.
