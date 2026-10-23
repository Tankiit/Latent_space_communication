"""
Synthetic Medical Data Generator with Known Ground-Truth Hurdle Structure.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch


def generate_synthetic_medical_data(
    n_samples: int = 2000,
    d_features: int = 16,
    num_concepts: int = 4,
    num_classes: int = 2,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generates a controlled synthetic medical dataset with explicit (State, Extent) hurdle properties.

    Ground-Truth Generative Process:
    1. Patient biomarkers X ~ N(0, I_d)
    2. For each concept j in 1..K:
       - State Logit: z_s,j = w_s,j^T X + b_s,j
       - State: S_j ~ Bernoulli(sigmoid(z_s,j))
       - Latent Extent Index: z_e,j = w_e,j^T X + b_e,j
       - Extent: E_j = exp(z_e,j + noise) if S_j == 1 else 0.0
    3. Disease Risk / Outcome Y:
       - Logit: sum_j [alpha_j * S_j + beta_j * (S_j * E_j)] + gamma^T X_nuisance
       - Y ~ Bernoulli(sigmoid(Logit))
    """
    rng = np.random.RandomState(seed)

    # 1. Input Features X (e.g., age, vitals, lab values, image embeddings)
    X = rng.randn(n_samples, d_features).astype(np.float32)

    # Concept weights
    W_s = rng.randn(d_features, num_concepts).astype(np.float32) * 0.7
    b_s = rng.randn(num_concepts).astype(np.float32) * 0.2

    W_e = rng.randn(d_features, num_concepts).astype(np.float32) * 0.5
    b_e = rng.uniform(-0.5, 0.5, size=num_concepts).astype(np.float32)

    # 2. Generate States
    logits_s = X @ W_s + b_s
    probs_s = 1.0 / (1.0 + np.exp(-logits_s))
    S = (rng.rand(n_samples, num_concepts) < probs_s).astype(np.float32)

    # 3. Generate Extents (Strict LogNormal Hurdle)
    noise_e = rng.randn(n_samples, num_concepts).astype(np.float32) * 0.3
    latent_e = np.exp(X @ W_e + b_e + noise_e)
    # Mask extents strictly where state == 0
    E = np.where(S > 0.5, latent_e, 0.0).astype(np.float32)

    # 4. Generate Task Outcome Y
    # Task relies on both presence (alpha) and severity/extent (beta)
    alpha = np.array([1.2, -0.8, 1.5, 0.5], dtype=np.float32)[:num_concepts]
    beta = np.array([0.9, 0.0, 1.1, -0.4], dtype=np.float32)[:num_concepts]

    # True interaction signal: S * E
    gated_extent = S * E
    task_latent = (S @ alpha) + (gated_extent @ beta) + (X[:, :2] @ np.array([0.3, -0.2], dtype=np.float32))
    task_probs = 1.0 / (1.0 + np.exp(-task_latent))
    Y = (rng.rand(n_samples) < task_probs).astype(np.int64)

    return {
        "x": torch.from_numpy(X),
        "states": torch.from_numpy(S),
        "extents": torch.from_numpy(E),
        "y": torch.from_numpy(Y),
        "concept_names": [f"Finding_{j+1}" for j in range(num_concepts)],
    }
