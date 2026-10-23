"""
Concept Bottleneck Model Architectures built with PyC (torch_concepts):
- TypedCBM (Proposed Tagged-Union Hurdle CBM)
- StandardCBM (Standard PyC Linear CBM)
- UncoupledTwoHeadCBM (Two-Head Uncoupled Baseline)
- CEMBaseline (Concept Embedding Model Baseline)
"""

from typing import Optional, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_concepts as pyc

from .layers import TypedBottleneckLayer, LogNormalHurdleLoss
from .schema import TaggedUnionBatch


class TypedCBM(nn.Module):
    """
    Typed Concept Bottleneck Model using PyC (torch_concepts).

    Architecture:
        embeddings -> TypedBottleneckLayer -> concepts [s, s*e] -> LinearConceptToConcept -> tasks
    """
    def __init__(
        self,
        in_embeddings: int,
        num_concepts: int,
        num_classes: int = 2,
        use_soft_relaxation: bool = True,
    ):
        super().__init__()
        self.in_embeddings = in_embeddings
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        # Typed hurdle bottleneck: in_embeddings -> 2*K concepts (state, state * extent)
        self.bottleneck = TypedBottleneckLayer(
            in_embeddings=in_embeddings,
            num_concepts=num_concepts,
            use_soft_relaxation=use_soft_relaxation,
        )

        # PyC Linear Predictor: 2*K concepts -> num_classes tasks
        self.predictor = pyc.nn.LinearConceptToConcept(
            in_concepts=self.bottleneck.bottleneck_dim,
            out_concepts=num_classes,
        )
        self.hurdle_loss_fn = LogNormalHurdleLoss()

    def forward(
        self,
        embeddings: torch.Tensor,
        hard: bool = False,
        state_intervention: Optional[Dict[int, float]] = None,
        extent_intervention: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, TaggedUnionBatch, Dict[str, torch.Tensor]]:
        """
        Forward pass following PyC calling style:
            concepts = bottleneck(embeddings=embeddings)
            tasks = predictor(concepts=concepts)
        """
        concepts, batch, params = self.bottleneck(
            embeddings=embeddings,
            hard=hard,
            state_intervention=state_intervention,
            extent_intervention=extent_intervention,
        )
        tasks = self.predictor(concepts=concepts)
        return tasks, concepts, batch, params

    def compute_loss(
        self,
        tasks: torch.Tensor,
        hurdle_params: Dict[str, torch.Tensor],
        target_y: torch.Tensor,
        target_states: torch.Tensor,
        target_extents: torch.Tensor,
        concept_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes joint task + typed hurdle concept loss."""
        task_loss = F.cross_entropy(tasks, target_y)
        total_concept_loss, gate_loss, extent_loss = self.hurdle_loss_fn(
            hurdle_params, target_states, target_extents
        )
        total_loss = task_loss + concept_weight * total_concept_loss

        metrics = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "concept_loss": total_concept_loss.item(),
            "gate_loss": gate_loss.item(),
            "extent_loss": extent_loss.item(),
        }
        return total_loss, metrics


class StandardCBM(nn.Module):
    """
    Standard linear Concept Bottleneck Model using native PyC layers.
    Following the user specification:
        encoder = pyc.nn.LinearEmbeddingToConcept(in_embeddings=..., out_concepts=...)
        predictor = pyc.nn.LinearConceptToConcept(in_concepts=..., out_concepts=...)
        concepts = encoder(embeddings=embeddings)
        tasks = predictor(concepts=concepts)
    """
    def __init__(
        self,
        in_embeddings: int,
        num_concepts: int,
        num_classes: int = 2,
    ):
        super().__init__()
        self.encoder = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=num_concepts,
        )
        self.predictor = pyc.nn.LinearConceptToConcept(
            in_concepts=num_concepts,
            out_concepts=num_classes,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        concept_intervention: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        concepts = self.encoder(embeddings=embeddings)
        c_probs = torch.sigmoid(concepts)

        if concept_intervention is not None:
            c_probs = c_probs.clone()
            for k, val in concept_intervention.items():
                c_probs[:, k] = float(val)

        tasks = self.predictor(concepts=c_probs)
        return tasks, c_probs


class UncoupledTwoHeadCBM(nn.Module):
    """
    Uncoupled Two-Head CBM baseline using PyC layers.
    Independently encodes binary state and continuous extent:
        state_concepts = state_encoder(embeddings=embeddings)
        extent_concepts = extent_encoder(embeddings=embeddings)
        tasks = predictor(concepts=[state, extent])
    Vulnerable to phantom states (predicting extent when S=0).
    """
    def __init__(
        self,
        in_embeddings: int,
        num_concepts: int,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.state_encoder = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=num_concepts,
        )
        self.extent_encoder = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=num_concepts,
        )
        self.predictor = pyc.nn.LinearConceptToConcept(
            in_concepts=2 * num_concepts,
            out_concepts=num_classes,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        state_intervention: Optional[Dict[int, float]] = None,
        extent_intervention: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_concepts = torch.sigmoid(self.state_encoder(embeddings=embeddings))
        extent_concepts = F.softplus(self.extent_encoder(embeddings=embeddings))

        if state_intervention is not None:
            state_concepts = state_concepts.clone()
            for k, val in state_intervention.items():
                state_concepts[:, k] = float(val)

        if extent_intervention is not None:
            extent_concepts = extent_concepts.clone()
            for k, val in extent_intervention.items():
                extent_concepts[:, k] = float(val)

        # UNCOUPLED: Predictor receives extent even when state is 0!
        concepts = torch.cat([state_concepts, extent_concepts], dim=-1)
        tasks = self.predictor(concepts=concepts)
        return tasks, concepts, state_concepts, extent_concepts


class CEMBaseline(nn.Module):
    """
    Concept Embedding Model (CEM) style baseline implemented with PyC encoders.
    Bottleneck mixes positive and negative embedding vectors:
        c_j = p_j * c_j^+ + (1 - p_j) * c_j^-
    Vulnerable to absence leakage via c_j^-.
    """
    def __init__(
        self,
        in_embeddings: int,
        num_concepts: int,
        concept_dim: int = 8,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim

        self.prob_encoder = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=num_concepts,
        )
        self.pos_encoder = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=num_concepts * concept_dim,
        )
        self.neg_encoder = pyc.nn.LinearEmbeddingToConcept(
            in_embeddings=in_embeddings,
            out_concepts=num_concepts * concept_dim,
        )
        self.predictor = pyc.nn.LinearConceptToConcept(
            in_concepts=num_concepts * concept_dim,
            out_concepts=num_classes,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        state_intervention: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(self.prob_encoder(embeddings=embeddings))  # (B, K)

        if state_intervention is not None:
            probs = probs.clone()
            for k, val in state_intervention.items():
                probs[:, k] = float(val)

        B = embeddings.size(0)
        c_pos = self.pos_encoder(embeddings=embeddings).view(B, self.num_concepts, self.concept_dim)
        c_neg = self.neg_encoder(embeddings=embeddings).view(B, self.num_concepts, self.concept_dim)

        p_expanded = probs.unsqueeze(-1)  # (B, K, 1)
        c_mixed = p_expanded * c_pos + (1.0 - p_expanded) * c_neg  # (B, K, D)
        concepts = c_mixed.view(B, self.num_concepts * self.concept_dim)

        tasks = self.predictor(concepts=concepts)
        return tasks, concepts, probs
