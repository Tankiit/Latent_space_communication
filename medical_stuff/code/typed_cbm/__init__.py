"""
Typed Concept Bottleneck Models (Typed CBMs)
ICML 2027 Reference Implementation
"""

from .schema import TypedConceptSpec, ConceptType, TaggedUnionBatch
from .layers import HurdleLayer, HurdleEmbeddingToConcept, TypedBottleneckLayer, LogNormalHurdleLoss
from .models import TypedCBM, StandardCBM, UncoupledTwoHeadCBM, CEMBaseline
from .interventions import apply_typed_intervention, evaluate_leakage
from .hurdle_audit import HurdleAuditor
from .synthetic import generate_synthetic_medical_data
from .medmnist_loader import get_medmnist_loaders, try_medmnist_datasets
from .concept_registry import (
    CONCEPTS,
    usable_as_typed,
    audit_summary,
    get_example_annotations,
    get_typed_medical_annotations,
    demo_mix_concept_predictor,
)

__all__ = [
    "TypedConceptSpec",
    "ConceptType",
    "TaggedUnionBatch",
    "HurdleLayer",
    "HurdleEmbeddingToConcept",
    "TypedBottleneckLayer",
    "LogNormalHurdleLoss",
    "TypedCBM",
    "StandardCBM",
    "UncoupledTwoHeadCBM",
    "CEMBaseline",
    "apply_typed_intervention",
    "evaluate_leakage",
    "HurdleAuditor",
    "generate_synthetic_medical_data",
    "get_medmnist_loaders",
    "try_medmnist_datasets",
    "CONCEPTS",
    "usable_as_typed",
    "audit_summary",
    "get_example_annotations",
    "get_typed_medical_annotations",
    "demo_mix_concept_predictor",
]
