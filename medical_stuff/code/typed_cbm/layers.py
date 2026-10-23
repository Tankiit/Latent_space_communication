"""
Hurdle Distribution Layers and Typed Bottlenecks in PyTorch and PyC (torch_concepts).
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_concepts as pyc
from .schema import TaggedUnionBatch


class HurdleEmbeddingToConcept(pyc.nn.BaseConceptLayer):
    """
    Parametric Hurdle layer extending PyC (torch_concepts) BaseConceptLayer.
    Transforms raw embeddings into hurdle parameters using PyC linear encoders:
    - logit_pi: Logit for Bernoulli state gate P(S=1 | embeddings)
    - mu_extent: Location parameter for LogNormal extent
    - log_sigma_extent: Scale parameter for LogNormal extent
    """
    def __init__(
        self,
        in_embeddings: int,
        out_concepts: int,
        min_sigma: float = 1e-2,
    ):
        super().__init__(
            out_concepts=out_concepts,
            in_embeddings=in_embeddings,
        )
        self.min_sigma = min_sigma

        # Head 1: State gate encoder
        self.state_head = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        # Head 2: Extent location encoder
        self.extent_mu_head = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        # Head 3: Extent scale (log sigma) encoder
        self.extent_log_sigma_head = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: Input tensor of shape (..., in_embeddings)
        Returns:
            logit_pi: (..., out_concepts)
            mu_extent: (..., out_concepts)
            log_sigma_extent: (..., out_concepts)
        """
        logit_pi = self.state_head(embeddings=embeddings)
        mu_extent = self.extent_mu_head(embeddings=embeddings)
        log_sigma = self.extent_log_sigma_head(embeddings=embeddings)
        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, min=math.log(self.min_sigma), max=2.0)
        return logit_pi, mu_extent, log_sigma


# Alias for backwards compatibility
HurdleLayer = HurdleEmbeddingToConcept


class TypedBottleneckLayer(pyc.nn.BaseConceptLayer):
    """
    Bottleneck layer that packages hurdle distributions into structurally typed representations.
    Extends PyC (torch_concepts) BaseConceptLayer to tagged-union concepts.
    Ensures that when S=0, extent is strictly zeroed downstream (Absence Invariance).
    """
    def __init__(
        self,
        in_embeddings: int,
        num_concepts: int,
        use_soft_relaxation: bool = True,
        tau: float = 1.0,
    ):
        # In typed concepts, each concept is represented by 2 coordinates: (s, s * e)
        super().__init__(
            out_concepts=2 * num_concepts,
            in_embeddings=in_embeddings,
        )
        self.num_concepts = num_concepts
        self.use_soft_relaxation = use_soft_relaxation
        self.tau = tau
        self.hurdle = HurdleEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=num_concepts,
        )

    @property
    def bottleneck_dim(self) -> int:
        """Each typed concept occupies 2 dimensions: (state, state * extent)."""
        return 2 * self.num_concepts

    def forward(
        self,
        embeddings: torch.Tensor,
        hard: bool = False,
        state_intervention: Optional[Dict[int, float]] = None,
        extent_intervention: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, TaggedUnionBatch, Dict[str, torch.Tensor]]:
        """
        Args:
            embeddings: Encoder feature tensor of shape (B, in_embeddings)
            hard: Whether to threshold states to discrete {0, 1}
            state_intervention: Dict mapping concept_idx -> value (0.0 or 1.0)
            extent_intervention: Dict mapping concept_idx -> value (positive float)
        Returns:
            concepts: Bottleneck concept vector for task predictor of shape (B, 2*K)
            batch: TaggedUnionBatch object
            params: Dictionary of raw hurdle parameters
        """
        logit_pi, mu_extent, log_sigma = self.hurdle(embeddings=embeddings)
        pi = torch.sigmoid(logit_pi)
        sigma = torch.exp(log_sigma)

        # Expected extent under LogNormal: E[e | S=1] = exp(mu + sigma^2 / 2)
        expected_extent = torch.exp(mu_extent + 0.5 * sigma.pow(2))

        # Determine effective states
        if hard or not self.training:
            states = (pi >= 0.5).float()
        else:
            if self.use_soft_relaxation:
                # Soft relaxation for gradient flow
                states = pi
            else:
                # Gumbel-Softmax straight-through
                logits_2d = torch.stack([-logit_pi, logit_pi], dim=-1)
                gumbel_sample = F.gumbel_softmax(logits_2d, tau=self.tau, hard=True)
                states = gumbel_sample[..., 1]

        extents = expected_extent

        # Apply do() Interventions
        if state_intervention is not None:
            states = states.clone()
            for k, val in state_intervention.items():
                states[:, k] = float(val)

        if extent_intervention is not None:
            extents = extents.clone()
            for k, val in extent_intervention.items():
                extents[:, k] = float(val)

        # STRICT MULTIPLICATIVE ABSENCE INVARIANCE:
        # Extent coordinate is gated by state:
        # z_j = [s_j, s_j * e_j]
        # When s_j = 0, the second coordinate is identically 0.
        gated_extent = states * extents

        B = embeddings.size(0)
        concepts = torch.stack([states, gated_extent], dim=-1).view(B, 2 * self.num_concepts)

        batch = TaggedUnionBatch(
            states=states,
            extents=extents,
        )

        params = {
            "logit_pi": logit_pi,
            "pi": pi,
            "mu_extent": mu_extent,
            "log_sigma": log_sigma,
            "sigma": sigma,
        }

        return concepts, batch, params


class LogNormalHurdleLoss(nn.Module):
    """
    Exact negative log-likelihood for two-part LogNormal hurdle model:
    L = BCEWithLogits(logit_pi, S) + S * NLL_LogNormal(E; mu, sigma)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        params: Dict[str, torch.Tensor],
        target_states: torch.Tensor,
        target_extents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            params: Dictionary containing 'logit_pi', 'mu_extent', 'log_sigma'
            target_states: Binary ground truth (B, K) in {0, 1}
            target_extents: Continuous ground truth (B, K), positive when S=1
        Returns:
            total_loss: L_gate + L_extent
            gate_loss: BCE loss
            extent_loss: Masked LogNormal NLL
        """
        logit_pi = params["logit_pi"]
        mu = params["mu_extent"]
        log_sigma = params["log_sigma"]
        sigma = torch.exp(log_sigma)

        # 1. State Gate Loss (BCE across all samples)
        gate_loss = F.binary_cross_entropy_with_logits(logit_pi, target_states)

        # 2. Extent Loss (Masked strictly to active subset S = 1)
        active_mask = (target_states > 0.5)

        if active_mask.any():
            active_e = target_extents[active_mask]
            active_mu = mu[active_mask]
            active_log_sigma = log_sigma[active_mask]
            active_sigma = sigma[active_mask]

            # Avoid log(0)
            clamped_e = torch.clamp(active_e, min=self.eps)
            log_e = torch.log(clamped_e)

            # LogNormal negative log density:
            # -log p(e) = 0.5 * log(2*pi) + log(sigma) + log(e) + 0.5 * ((log(e) - mu) / sigma)^2
            quad = 0.5 * ((log_e - active_mu) / active_sigma).pow(2)
            nll_element = 0.5 * math.log(2 * math.pi) + active_log_sigma + log_e + quad
            extent_loss = nll_element.mean()
        else:
            extent_loss = torch.tensor(0.0, device=logit_pi.device)

        total_loss = gate_loss + extent_loss
        return total_loss, gate_loss, extent_loss
