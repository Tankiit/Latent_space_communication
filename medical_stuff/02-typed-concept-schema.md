---
title: "02: Typed Concept Schema, Tagged-Union Support, and Distribution Layer"
tags: [schema, tagged-union, hurdle-model, math, distribution-layer]
status: living
prerequisites: ["[[01-claim-and-story]]"]
related: ["[[03-option-a-vs-b]]", "[[04-hurdle-test-and-audit]]", "[[07-terminology]]"]
---

# 02: Typed Concept Schema, Tagged-Union Support, and Distribution Layer

## 1. Mathematical Formalism: Tagged-Union Support

In standard CBMs, the bottleneck space is $\mathcal{C} = \mathbb{R}^K$ or $[0, 1]^K$. For typed concepts, each concept $C_j$ lives in an **algebraic sum type** (tagged union):

$$\mathcal{C}_j = \mathcal{S}_j \amalg \Big(\mathcal{S}_j^+ \times \mathcal{E}_j\Big)$$

For the canonical binary-state case (Presence/Absence):
- State space: $\mathcal{S}_j = \{0, 1\}$, where $0 \equiv \text{Absent}$ and $1 \equiv \text{Present}$.
- Active states: $\mathcal{S}_j^+ = \{1\}$.
- Extent space: $\mathcal{E}_j \subseteq \mathbb{R}^+$ (e.g., $(0, \infty)$ for unbounded metrics like volume/mass, or $(0, 1]$ for normalized ratios like cardiothoracic ratio or stenosis percentage).

When $S_j = 0$, the extent is syntactically **undefined** or structurally null: $E_j = \emptyset$. Thus, the support of $C_j = (S_j, E_j)$ is:
$$\operatorname{supp}(C_j) = \{ (0, \emptyset) \} \cup \{ (1, e) : e \in \mathcal{E}_j \}$$

### Measure-Theoretic Foundation
The sample space $\Omega_j = \{0\} \cup (\{1\} \times \mathcal{E}_j)$ is endowed with the base measure:
$$\nu_j = \delta_{(0, \emptyset)} + \Big(\delta_1 \otimes \lambda_{\mathcal{E}_j}\Big)$$
where $\delta$ is the Dirac measure and $\lambda_{\mathcal{E}_j}$ is the Lebesgue measure on $\mathcal{E}_j$. This formalizes the two-part hurdle measure: a point mass at absence and an absolutely continuous density on presence.

---

## 2. Hurdle Distribution Layer

The neural encoder $g_\phi: \mathcal{X} \to \mathbb{R}^{d_{\text{enc}}}$ outputs parameters for the hurdle distribution:
$$g_\phi(x) \mapsto \Big(\operatorname{logit}_\pi(x), \mu_e(x), \log \sigma_e(x)\Big)$$

1. **State Gating Probability:**
   $$\pi_j(x) = P(S_j = 1 \mid x) = \sigma(\operatorname{logit}_\pi(x))$$

2. **Conditional Extent Density:**
   Conditional on $S_j = 1$, extent follows a parameterized density $p(e_j \mid S_j = 1, x)$. Typical parametric families:
   - **Log-Normal:** $e_j \sim \operatorname{LogNormal}(\mu_e(x), \sigma_e^2(x))$ for positive physical measurements (mm, mL).
   - **Beta / Truncated Normal:** $e_j \sim \operatorname{Beta}(\alpha_j(x), \beta_j(x))$ for bounded ratios in $(0, 1)$.

### Likelihood and Loss Function
For ground-truth supervision $(s_j, e_j) \in \Omega_j$, the exact negative log-likelihood (NLL) decomposes additively into hurdle components:
$$\mathcal{L}_{\text{typed}}(x, s_j, e_j) = -\log p(s_j, e_j \mid x)$$

$$\mathcal{L}_{\text{typed}} = \begin{cases} 
-\log(1 - \pi_j(x)) & \text{if } s_j = 0 \\
-\log \pi_j(x) - \log p(e_j \mid S_j = 1, x) & \text{if } s_j = 1
\end{cases}$$

Rewriting in indicator form:
$$\mathcal{L}_{\text{typed}} = \operatorname{BCE}(\pi_j(x), s_j) + s_j \cdot \mathcal{L}_{\text{extent}}(e_j; \mu_e(x), \sigma_e(x))$$

> [!IMPORTANT]
> The extent loss is multiplied by $s_j$. When $s_j = 0$, the encoder receives zero gradient from the extent loss. This prevents the extent head from attempting to fit arbitrary or imputed values on absent samples.

---

## 3. Downstream Task Interface $h_\theta(C)$

A critical vulnerability of naive two-head models is that downstream predictor $h_\theta$ may learn to attend to $E_j$ even when $S_j = 0$, causing phantom-state leakage `(ICLR-era)`.

### Admissible Bottleneck Embeddings
To preserve the tagged-union semantics downstream, the bottleneck representation $\mathbf{z}_j$ presented to the task predictor must satisfy the **Absence Invariance Condition**:
$$\mathbf{z}_j(S_j = 0, E_j = e) = \mathbf{z}_j(S_j = 0, E_j = e') = \mathbf{z}_{\emptyset, j} \quad \forall e, e'$$

We realize this via two architectural designs:

#### Design A: Structural Multiplicative Gating
$$\mathbf{z}_j = \begin{bmatrix} s_j \\ s_j \cdot e_j \end{bmatrix} \quad \text{or for soft relaxation:} \quad \mathbf{z}_j = \begin{bmatrix} \hat{\pi}_j \\ \hat{\pi}_j \cdot \hat{e}_j \end{bmatrix}$$
When $s_j = 0$, the second coordinate is identically zero regardless of the value produced by the extent estimator.

#### Design B: Tagged Embedding Blocks (CEM-Alternative)
$$\mathbf{z}_j = s_j \cdot \mathbf{v}_{\text{present}}(e_j) + (1 - s_j) \cdot \mathbf{v}_{\text{absent}}$$
where $\mathbf{v}_{\text{absent}}$ is a fixed or learned vector shared across all absent instances, and $\mathbf{v}_{\text{present}}(e_j)$ is an embedding function of extent.

---

## 4. Worked Example: Pleural Effusion in Chest Radiography

Consider concept $C_{\text{effusion}}$ representing pleural effusion in chest X-ray reading:

1. **State:**
   - $S = 0$: No pleural effusion.
   - $S = 1$: Pleural effusion present.
2. **Extent:**
   - If $S = 0$: $E = \emptyset$.
   - If $S = 1$: $E \in (0, 1]$, measuring estimated fluid volume / blunting index (normalized hemithorax fraction).
3. **Encoder Forward Pass on Image $X$:**
   - Raw outputs: $\operatorname{logit}_\pi = 1.8$, $\mu_e = 0.45$, $\sigma_e = 0.12$.
   - Gate probability: $\pi(X) = \sigma(1.8) \approx 0.858$.
   - Extent distribution: $\mathcal{N}(0.45, 0.12^2)$ truncated to $(0, 1]$.
4. **Prediction:**
   - Expected state: $\hat{s} = 1$ (since $\pi > 0.5$).
   - Expected extent: $\hat{e} = 0.45$.
   - Bottleneck vector: $\mathbf{z} = [0.858, 0.858 \times 0.45] = [0.858, 0.386]$.
5. **Counterfactual Intervention:**
   - Clinician applies $do(S = 0)$ ("simulate patient without effusion").
   - Model clamps $s \leftarrow 0$.
   - Extent automatically collapses: $s \cdot e \leftarrow 0 \cdot 0.45 = 0$.
   - Clamped bottleneck vector: $\mathbf{z}_{do(S=0)} = [0, 0]$.
   - The task head evaluates $h_\theta([0, 0])$, completely removing effusion severity from the clinical risk calculation.

---

## 5. Notes & Verification Items

- `TODO:` Extend the schema from binary state $\{0, 1\}$ to categorical states (e.g., $\{ \text{Absent}, \text{Benign}, \text{Malignant} \}$).
- `VERIFY:` Check whether PyC (`torch_concepts`) supports multiplicative masked bottlenecks or if custom autograd functions are required.
- `(ICLR-era):` Contrast with naive linear readout $\mathbf{w}^\top \mathbf{c}$, which allows rank-deficient leakage unless exact projection kernels are maintained.

### Verified identities and interface facts (2026-10-23)

- **Readout identity:** zero-filled extent satisfies $S \cdot E$ identically, and the gated typed readout $[s, s \cdot e]$ equals the indicator readout $[\text{mask}, \text{zero-filled } E]$ **pointwise**. Consequence: any downstream function of $(S, S\!\cdot\!E)$ — including $E^2$, since $(SE)^2 = SE^2$ — is equally available to typed and untyped arms; the typed schema's advantages are *interface-level*, not estimation-level (see [[01-claim-and-story]] revision).
- **Ill-typed queries:** $do(S{=}0, E{=}e)$ is unrepresentable under the tagged union (the type system refuses to construct $(\text{Absent}, e)$), while $[\text{mask}, \text{filled}]$ interfaces accept it silently. Formalized in `legal_queries()` (`code/e2_arms.py`).
- **3-state schema in the wild:** PAD-UFES-20 realizes {UNK-refused, absent, present × diameter} with patient-level refusal (99.5% joint missingness; [[04-hurdle-test-and-audit]]) — the UNKNOWN state is a real clinical mechanism, not a modeling convenience.
