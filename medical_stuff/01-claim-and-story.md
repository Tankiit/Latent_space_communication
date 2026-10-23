---
title: "01: Candidate Claim and Five-Beat Story"
tags: [claim, narrative, icml-2027]
status: living
prerequisites: ["[[index]]"]
related: ["[[02-typed-concept-schema]]", "[[03-option-a-vs-b]]", "[[09-venue-icml]]"]
---

# 01: Candidate One-Sentence Claim and Five-Beat Story

## Candidate One-Sentence Claim

> **Standard Concept Bottleneck Models force heterogeneous clinical and tabular concepts into homogeneous scalar or unconstrained embedding spaces, conflating discrete presence/absence (state) with continuous magnitude/severity (extent); formalizing concepts as typed tagged-union random variables with hurdle likelihoods provides mathematically coherent support, eliminates phantom-state leakage under counterfactual interventions, and decouples discrete classification uncertainty from continuous measurement uncertainty.**

---

## Five-Beat Story

### Beat 1: The Categorical Fallacy in Concept Bottlenecks
Standard Concept Bottleneck Models (CBMs; Koh et al., 2020) and Concept Embedding Models (CEMs; Zarlenga et al., 2022) assume that all intermediate human-understandable concepts share identical semantic and mathematical types: either scalar probabilities $c_j \in [0, 1]$, unbounded real logits $\hat{c}_j \in \mathbb{R}$, or unconstrained vector embeddings $\mathbf{e}_j \in \mathbb{R}^d$. 

However, in real-world clinical pathology, radiology, and tabular risk modeling, concepts are inherently **heterogeneous and hierarchical**. A clinical biomarker or finding consists of a discrete qualitative condition (**state**, e.g., Absent vs. Present, or Benign vs. Malignant) and, conditionally upon being present, a quantitative metric (**extent**, e.g., nodule diameter in mm, pleural effusion volume, or percentage of arterial stenosis). 

### Beat 2: The Pathology of Existing Work (Leakage, Phantom States, and Bad Interventions)
Treating typed concepts under existing paradigms leads to three severe failure modes:
1. **The Censoring Trap (Scalar Collapse):** Encoding "absent" as extent $= 0$ assumes that the transition from absence to presence is governed by the same continuous linear index as the variation within presence (the classical Tobit restriction).
2. **The Phantom State Trap (Uncoupled Heads):** Using two independent prediction heads (one for binary presence, one for continuous extent) produces logically impossible predictions—such as an encoder predicting $P(\text{Pneumothorax} = \text{Absent}) = 0.99$ while simultaneously outputting an estimated lung collapse of $35\%$.
3. **The Counterfactual Leakage Trap:** In CEMs and continuous representations, performing an intervention $do(\text{State} = \text{Absent})$ leaves residual extent information in the latent subspace (or vice versa), allowing downstream task heads to exploit leaked shortcuts `(ICLR-era)`. Furthermore, intervening on extent when a disease is absent ($do(\text{Extent} = 15\text{mm})$ while $\text{State} = \text{Absent}$) is ill-typed and ill-posed.

### Beat 3: The Proposal — Typed Concepts as Tagged Unions with Hurdle Likelihoods
We formalize concepts as algebraic data types (sum types / tagged unions) residing on a disjoint union support:
$$C_j \in \{\text{Absent}\} \cup \Big(\{\text{Present}\} \times \mathcal{E}_j\Big)$$
We equip this support with a **hurdle distribution layer** that factors joint concept generation into:
1. A discrete gating probability $\pi_j(x) = P(S_j = 1 \mid x)$ over state.
2. A conditional continuous density $p(e_j \mid S_j = 1, x)$ over extent with support strictly on $\mathcal{E}_j \subseteq (0, \infty)$ or $(0, 1]$.

Downstream predictors receive a strictly typed representation where extent coordinates are masked or nullified when the state is absent ($s_j = 0$).

### Beat 4: Principled Uncertainty and Typed Interventions
Because the model respects the tagged-union geometry:
- **Disentangled Uncertainty:** We cleanly separate aleatoric/epistemic uncertainty over the discrete diagnostic gate (e.g., entropy or credal sets over presence) from heteroscedastic uncertainty over continuous severity (e.g., conditional variance or prediction intervals).
- **Semantically Valid Interventions:** Interventions operate at the type level:
  - $do(S_j = 0)$ deterministically collapses extent to the null state $\emptyset$, guaranteeing zero information leakage to downstream task heads.
  - $do(E_j = e)$ is valid if and only if $S_j = 1$ (or syntactically forces $S_j \leftarrow 1$).

### Beat 5: Empirical Payoff across Clinical Benchmarks and Audit Batteries
We evaluate Typed CBMs on controlled synthetic benchmarks with known structural generation, as well as candidate medical datasets (e.g., chest X-ray effusion/cardiomegaly, dermatology lesion margins, ICU acute kidney injury):
- **Audit Compliance:** Typed CBMs pass the **Hurdle Test** and the **Admissible Transformation Audit**, preventing the rotation and leakage vulnerabilities identified in `(ICLR-era)` analyses.
- **Intervention Accuracy:** Counterfactual task accuracy under concept interventions significantly outperforms scalar CBMs and CEM baselines without sacrificing observational accuracy on the primary task.

---

## Open Decisions & Notes

- `TODO:` Decide whether the narrative foregrounds **medical safety / counterfactual validity** (Option A) or **general structural representation learning / algebraic constraints** (Option B). See [[03-option-a-vs-b]].
- `VERIFY:` Confirm whether CheXpert or MIMIC-CXR radiologist severity annotations provide sufficient sample size for continuous extent validation. See [[05-datasets]].
- Constraints that previously existed only to split the paper between ICLR (algebraic equivalence) and AISTATS (uncertainty decomposition) are removed; both facets are synthesized here for ICML 2027.

---

## Revision 2026-10-23: the claim after the honest-arms experiments

The candidate claim above is **superseded in part**. Verified facts (all reproducible; see [[06-gates-and-baselines]] and `results/`):

1. **Estimation ties.** Zero-filled extent satisfies $S \cdot E$ identically, and the typed readout $[\hat{s}, \hat{s}\hat{e}]$ equals the indicator readout $[\text{mask}, \text{zero-filled } E]$ pointwise. The indicator arm recovers both $\alpha$ and $\beta$ in $Y \sim \sigma(\alpha S + \beta S E)$ (n=400k check), and its $do(S{=}\text{absent})$ is well defined. A masked two-part (hurdle) model matches the typed arm on every well-typed query — confirmed by the 10-seed run: typed vs pooled_masked hidden-present RMSE gap −0.09, inside the registered 0.10 tie band. **The paper is not a method paper; the abstract must say so** (CLeaR's CFP explicitly admits application papers without new methodology).
2. **Where the contribution actually sits:**
   - **Selection bias with known truth** (strongest result): E1, 10 generator-resampled seeds on ChestMNIST effusion — drop-rows bias ≈ 0 under MCAR, +0.05→+0.44 under feature-keyed MAR, +0.10→+0.92 under MNAR; and a *falsified* prediction worth reporting: zero-fill **inflates** the fitted dose-response slope (+0.08 to +0.13) instead of attenuating it, by manufacturing a zero-magnitude cluster (`results/`, E1 table).
   - **Audit battery, two real-data passes** ([[04-hurdle-test-and-audit]]): PUMS WKHP (zeros never inside employment: 0/6,727; ESR-gate temporal misalignment documented) and PAD-UFES-20 diameter (0 zeros in 1,490 present; patient-level refusal: 99.5% joint missingness, zero partial patients).
   - **Interface guarantees of the typed schema** — the only typed-exclusive behaviors, formalized in `legal_queries()` (`code/e2_arms.py`): ill-typed queries $do(S{=}0, E{=}e)$ are *unrepresentable* (indicator/two-part answer them silently, P=0.98 in the check; pooled cannot even express $do(S{=}0)$), and phantom severity reads on absent findings are eliminated (pooled arms emit positive severity on 100% of absent rows).
3. **Registered negative control.** The seeded E2 comparison stands as pre-registered evidence that typing adds no *estimation* information (masking is the headline: +1.38 RMSE improvement attribution), with two disclosed artifacts: gate degeneracy at 13% prevalence (the argmax state head never predicts PRESENT; typed abstains globally — a threshold/interface finding, not a loss property) and the log-space ridge arm's extrapolation blow-up.
4. **Not built, on purpose:** estimand (b) as originally designed ($Y \sim \sigma(\alpha S + \beta S E)$ recovery) is a predetermined tie per the algebra above. Anything that is a function of $(S, S\!\cdot\!E)$ — including $E^2$, since $(SE)^2 = SE^2$ — is equally available to both arms.
