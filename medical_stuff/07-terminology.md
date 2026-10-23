---
title: "07: Terminology: Typed vs Mixed vs Mixture (CEM Vocabulary Trap)"
tags: [terminology, disambiguation, cem, hurdle-models, vocabulary]
status: living
prerequisites: ["[[01-claim-and-story]]", "[[02-typed-concept-schema]]"]
related: ["[[04-hurdle-test-and-audit]]", "[[06-gates-and-baselines]]"]
---

# 07: Terminology: Typed vs Mixed vs Mixture (CEM Vocabulary Trap)

Precise mathematical vocabulary is critical to prevent reviewers from conflating our typed tagged-union formulation with existing heuristic approaches. This note establishes the canonical terminology and explains the **CEM Vocabulary Trap**.

---

## 1. Disambiguation Table

| Term | Formal Definition | Mathematical Support | Key Distinction |
| :--- | :--- | :--- | :--- |
| **Typed Concept** *(Ours)* | Algebraic sum type (tagged union) coupling a discrete state with a conditional continuous extent. | $C_j \in \mathcal{S} \amalg (\mathcal{S}^+ \times \mathcal{E})$ with hurdle measure $\delta_0 + \lambda_+$. | Internal structural typing within a single semantic concept. |
| **Mixed Concepts** *(Heterogeneous Bottleneck)* | A bottleneck layer containing distinct concept nodes of different types side-by-side. | $\mathcal{C} = \{0, 1\}^{K_1} \times \mathbb{R}^{K_2} \times \{1,\dots,C\}^{K_3}$. | Cartesian product of standard scalars; each individual concept is still flat/scalar. |
| **Mixture Concepts** *(CEM Embedding Trap)* | Convex combination of two learned vector embeddings in an unconstrained latent space. | $\mathbf{c}_j = \hat{p}_j \mathbf{c}_j^+ + (1 - \hat{p}_j) \mathbf{c}_j^-$, where $\mathbf{c}_j^+, \mathbf{c}_j^- \in \mathbb{R}^d$. | Overparameterized continuous subspace; does NOT represent severity; violates absence invariance. |
| **Hurdle Model** *(Two-Part Process)* | Statistical distribution where zeros originate exclusively from a single discrete selection gate. | $p(c) = (1-\pi)^{\mathbb{I}(c=0)} \cdot [\pi f(c)]^{\mathbb{I}(c>0)}$, with $\operatorname{supp}(f) \subseteq (0, \infty)$. | Only one source of zeros; matches clinical reality (a present lesion cannot have 0 mm size). |
| **Zero-Inflated Model** *(Dual-Process Zero)* | Distribution where zeros arise from two distinct sources: a point mass AND chance zeros of a continuous density. | $p(c) = \pi \delta_0(c) + (1-\pi) g(c)$, where $g(0) > 0$. | Inappropriate for physical extents; creates identifiability ambiguity between the two zero sources. |

---

## 2. The CEM Vocabulary Trap

Concept Embedding Models (CEM; Zarlenga et al., 2022) popularized the phrase *"concept representations with mixed or mixture embeddings"*. Reviewers often ask: *"How does a typed concept differ from CEM's positive/negative embedding mixture?"*

We must explicitly expose the three fundamental flaws of the CEM mixture:

### 1. Extent Conflation
In CEM, the representation is:
$$\mathbf{c}_j(x) = \hat{p}_j(x) \mathbf{c}_j^+(x) + (1 - \hat{p}_j(x)) \mathbf{c}_j^-(x)$$
Here, $\hat{p}_j \in [0, 1]$ is supervised as concept presence. But $\mathbf{c}_j^+$ is **not** an extent; it is a $d$-dimensional embedding vector produced by an MLP. The length or orientation of $\mathbf{c}_j^+$ has no units, no physical scale, and no correspondence to clinical severity (e.g., tumor diameter or fluid volume).

### 2. The Negative Embedding Failure (Absence Leakage)
In CEM, when a concept is completely absent ($\hat{p}_j = 0$), the bottleneck does **not** collapse to zero. Instead, it emits $\mathbf{c}_j^-(x) \in \mathbb{R}^d$. Because $\mathbf{c}_j^-(x)$ is a function of the input image $x$, the encoder can encode arbitrary downstream task features into $\mathbf{c}_j^-$, completely bypassing the concept bottleneck `(ICLR-era)`.
> In Typed CBMs, when $S_j = 0$, the representation is algebraically collapsed: $\mathbf{z}_j = [0, 0]$, mathematically guaranteeing zero information leakage.

### 3. Ill-Posed Interventions
In CEM, intervening on a concept means setting $\hat{p}_j \leftarrow 1$ or $\hat{p}_j \leftarrow 0$. There is no mechanism to intervene on **severity** while holding presence fixed, or to query the counterfactual: *"What if the patient had the lesion, but with half the diameter?"*

---

## 3. Hurdle vs. Zero-Inflation: Why Hurdle is the Correct Choice

In the statistics literature, two-part hurdle models (Cragg, 1971; Mullahy, 1986) are frequently confused with zero-inflated models (Lambert, 1992):

```text
Hurdle Model (Two-Part):
  Selection: [ Absent (C = 0) ]  ────── crossed ──────>  [ Present: Continuous Extent (C > 0) ]
  (Zeros occur exclusively at the gate)

Zero-Inflated Model:
  Gate:      [ Structural Zero ]
             [ Non-Structural Process ]  ─────────────>  [ Density can STILL produce C = 0 ]
  (Zeros are ambiguous: is it absent, or present with zero magnitude?)
```

In medical ontology:
- A lung nodule is either absent ($S=0$), or present with diameter $>0$ mm ($S=1, E > 0$).
- There is no clinical entity of a "nodule that is present but measures exactly 0.000 mm".
- Therefore, the **Hurdle formulation** is strictly correct, and the **Zero-Inflated formulation** is statistically misspecified for physical extent concepts.

---

## Summary of Manuscript Conventions

- Use **"Typed Concept"** when referring to our tagged-union $(S, E)$ formulation.
- Use **"Hurdle distribution layer"** when describing the encoder's probabilistic output.
- Refer to CEM strictly as **"Continuous Embedding Mixture"** to highlight its lack of type safety.
- Explicitly contrast **"absence invariance"** with CEM's "negative embedding leakage".
