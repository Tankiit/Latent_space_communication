---
title: "03: Option A vs Option B: Typing in the Data vs Typing in the Model"
tags: [framing, design-fork, identifiability, icml-2027]
status: living
prerequisites: ["[[01-claim-and-story]]", "[[02-typed-concept-schema]]"]
related: ["[[04-hurdle-test-and-audit]]", "[[05-datasets]]", "[[08-open-questions]]"]
---

# 03: Option A vs Option B: Typing in the Data vs Typing in the Model

## The Framing Fork

A central strategic question for the ICML 2027 paper is the locus of "typing":
- **Is typing an ontological property of the ground-truth data?** (Option A)
- **Is typing an inductive bias / representational property enforced by the model architecture?** (Option B)

```text
                  ┌────────────────────────────────────────┐
                  │      Where does Typing Originate?      │
                  └───────────────────┬────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│       Option A:       │                           │       Option B:       │
│  Typing in the Data   │                           │  Typing in the Model  │
│ (Supervised Hurdle)   │                           │  (Structural Latent)  │
└───────────────────────┘                           └───────────────────────┘
```

---

## Detailed Comparison

| Axis | Option A: Typing in the Data | Option B: Typing in the Model |
| :--- | :--- | :--- |
| **Premise** | Clinical reality is tagged-union structured $(S, E)$. The dataset explicitly provides paired $(s_j, e_j)$ with structural missingness ($e_j = \emptyset$ when $s_j = 0$). | Concepts in the wild are messy or only partially annotated (e.g., binary flags only). The model's bottleneck architecture forces a $(S, E)$ decomposition. |
| **Model Role** | Learn calibrated hurdle distributions $p(s, e \mid x)$ and maintain structural invariance downstream. | Discover or disentangle discrete states and continuous extents from unstructured or weakly-supervised data. |
| **Loss Formulation** | Explicit hurdle negative log-likelihood: $\operatorname{BCE}(s, \pi) + s \cdot \mathcal{L}_{\text{extent}}(e, \hat{e})$. | Joint variational or adversarial loss; hurdle gate regularized by sparsity, mutual information, or reconstruction. |
| **Identifiability** | **Trivially Identifiable:** Both state and extent have explicit empirical anchors in the training distribution. | **Severe Identifiability Hazards:** Without extent ground truth, $\pi(x) \cdot e(x)$ suffers from scaling indeterminacy $\pi \to 1, e \to \pi \cdot e$. |
| **Benchmark Feasibility** | Requires datasets with genuine severity/extent annotations (e.g., radiologic diameter, cardiothoracic ratio, ICU creatinine). | Can be applied immediately to legacy CBM benchmarks (CUB-200, CEBaB, AwA2). |
| **Reviewer Reception** | Clean, mathematically grounded statistical contribution. Highly convincing for clinical and safety reviewers. | Exciting representation-learning story for ML theorists, but risks being attacked on identifiability and degenerate modes. |

---

## Option A Deep Dive: The Supervised Hurdle Approach

### Mechanism
In Option A, we assert that the failure of standard CBMs is **measurement misspecification**: researchers treat variables that live in tagged unions as if they were real numbers.
- When an annotator marks "Cardiomegaly = Absent", assigning $e = 0$ is a distortion.
- Option A provides the correct statistical machinery: two-part hurdle likelihoods, masked gradients, and gated downstream representations.

### Pros
1. Strong, unassailable statistical foundation.
2. Directly solves real clinical needs where doctors already record (State + Extent) in clinical forms (TNM staging, BI-RADS, acute organ injury scores).
3. The Hurdle Test [[04-hurdle-test-and-audit]] acts as an empirical validation filter.

### Cons
- Restricts experiments to datasets that contain genuine extent measurements (see [[05-datasets]]). Legacy computer-vision CBM benchmarks (CUB attributes) cannot be used without synthetic augmentation.

---

## Option B Deep Dive: The Structural Latent Typing Approach

### Mechanism
In Option B, we propose a new family of bottleneck layers for arbitrary concept models. Even if the practitioner only provides binary concept labels $s_j \in \{0, 1\}$, the model instantiates an internal tagged union:
$$z_j = \hat{s}_j \cdot \mathbf{u}_j(x)$$
where $\hat{s}_j$ is supervised by the binary label, but $\mathbf{u}_j(x) \in \mathbb{R}^d$ or $\mathbb{R}^+$ is an unobserved extent / latent modifier discovered via self-supervision or downstream task reconstruction.

### Pros
1. Can be run on any existing CBM benchmark (e.g., CUB bird colors, where binary label is "wing color = red", and latent extent discovers color saturation or patch size).
2. Appeals to the representation-learning community at ICML.

### Cons
1. **Identifiability Trap:** Why should $\mathbf{u}_j(x)$ capture "extent" rather than task-leakage or shortcut features? `(ICLR-era)` showed that unconstrained latent blocks in CEMs act as backdoor communication channels.
2. Interventions on discovered extents lack human interpretability unless constrained by strict causal or structural assumptions.

---

## Synthesis: The Recommended Path for ICML 2027

To maximize both scientific rigor and empirical scope, we propose a **Two-Tier Hybrid Strategy**:

1. **Primary Anchor (Option A):** The paper's theoretical framework, primary claims, and core benchmark results are established under **Option A**, using medical and tabular datasets with verified (state + extent) annotations, supported by the Hurdle Test.
2. **Generalization Extension (Option B):** As an extension, we formulate **Weakly-Supervised Typed Bottlenecks** for settings where extent is observed on only a small subset of samples ($10\%$) or derived via bounding-box / segmentation proxies. We prove that minimal anchor supervision ($>0\%$) resolves the identifiability collapse.

---

## Open Decisions & Notes

- `TODO:` Formalize the identifiability condition for Option B: what minimum supervision or inductive bias prevents the gate $\pi$ from degenerating?
- `TODO:` Pick the definitive framing choice for the manuscript introduction by November 2026.
- `VERIFY:` Review AISTATS EU/AU feedback to see if reviewers doubted unsupervised extent discovery.

---

## Resolution 2026-10-23

**Resolved for this paper: Option A (supervised hurdle) is the anchor; Option B is deferred.** Two findings forced the decision:

1. The honest-arms experiments showed the typed model *ties* masked two-part/hurdle baselines in estimation (see [[06-gates-and-baselines]]). Option B's pitch — discovering extent latently and beating supervised representations — is not available even under full supervision, so it cannot carry the paper.
2. The paper's identity shifted to selection bias + audit + schema/interface (see [[01-claim-and-story]] revision), all of which are Option-A constructs.

The recommended two-tier hybrid above stands as written but is now future work: weakly-supervised extent (Option B) is where a *follow-up* method paper could live, contingent on the identifiability formalization still marked TODO.
