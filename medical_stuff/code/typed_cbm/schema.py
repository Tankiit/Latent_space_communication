"""
Schema and Tagged-Union Data Structures for Typed Concepts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import torch
from torch_concepts.nn import MixConceptEmbeddingToConcept

class ConceptType(Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    TYPED_HURDLE = "typed_hurdle"


class ExtentDistribution(Enum):
    LOGNORMAL = "lognormal"
    BETA = "beta"
    TRUNCATED_NORMAL = "truncated_normal"


@dataclass
class TypedConceptSpec:
    name: str
    concept_type: ConceptType = ConceptType.TYPED_HURDLE
    extent_dist: ExtentDistribution = ExtentDistribution.LOGNORMAL
    extent_min: float = 0.0
    extent_max: Optional[float] = None
    unit: str = "arbitrary"
    description: str = ""

    def validate(self, state: torch.Tensor, extent: torch.Tensor) -> bool:
        """
        Validates tagged-union invariant:
        If state == 0, extent must be either NaN, 0, or masked.
        """
        if self.concept_type == ConceptType.TYPED_HURDLE:
            absent_mask = (state == 0)
            if absent_mask.any():
                absent_extents = extent[absent_mask]
                valid_zero = (absent_extents == 0.0) | torch.isnan(absent_extents)
                return bool(valid_zero.all().item())
        return True


@dataclass
class TaggedUnionBatch:
    """
    Batched representation of K concepts for B samples.
    """
    states: torch.Tensor  # Shape (B, K), values in {0, 1} or [0, 1]
    extents: torch.Tensor  # Shape (B, K), values in R^+ (or 0/NaN when state == 0)
    specs: Optional[List[TypedConceptSpec]] = None

    @property
    def active_mask(self) -> torch.Tensor:
        """Boolean mask where state == 1."""
        return self.states > 0.5

    def get_masked_extents(self, fill_value: float = 0.0) -> torch.Tensor:
        """
        Returns extent tensor with absent instances strictly replaced by fill_value.
        Guarantees that state == 0 samples contain no latent extent signals.
        """
        out = self.extents.clone()
        out[~self.active_mask] = fill_value
        return out

    def to_bottleneck_representation(self) -> torch.Tensor:
        """
        Constructs the canonical 2K-dimensional bottleneck vector:
        z = [s_1, s_1 * e_1, s_2, s_2 * e_2, ..., s_K, s_K * e_K]
        """
        masked_e = self.get_masked_extents(fill_value=0.0)
        # Interleave state and masked extent: (B, 2K)
        B, K = self.states.shape
        stacked = torch.stack([self.states, masked_e], dim=-1)  # (B, K, 2)
        return stacked.view(B, 2 * K)
