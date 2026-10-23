"""
Concept Registry and Usability Auditor for Typed Concepts.

Audits candidate concepts across real and semi-synthetic medical datasets:
  - chestmnist:mass
  - retinamnist:grade
  - chestmnist:source_bbox

Provides type-aware pyc.Annotations integration with MixConceptEmbeddingToConcept.
"""

from typing import Dict, Any, Tuple, List, Optional
import torch
import torch_concepts as pyc
from torch_concepts.nn import MixConceptEmbeddingToConcept


# ------------------------------------------------------------------ Candidate Concepts
CONCEPTS: Dict[str, Dict[str, Any]] = {
    "chestmnist:mass": {
        "dataset": "chestmnist",
        "finding": "mass",
        "states_native": ["present", "absent"],
        "magnitude_native": None,
        "unknown": "inject",                      # Injected via MCAR / MAR / MNAR
        "magnitude": "inject: lesion diameter (mm) ~ LogNormal(mu(x), sigma^2) on PRESENT",
        "label_noise": "text-mined",              # ~10-15% NLP CheXpert extraction noise
        "n_total": 78468,                         # ChestMNIST total dataset size
        "n_min_state": 230,                       # In 4000 subset, >= 230 present cases
        "circular": False,                        # Extent is an independent continuous measurement
        "extent_carrying_states": ["present"],    # 1 state carries magnitude
    },
    "retinamnist:grade": {
        "dataset": "retinamnist",
        "finding": "diabetic_retinopathy",
        "states_native": ["G0", "G1", "G2", "G3", "G4"],
        "magnitude_native": "grade 1-4 (same label as state?)",
        "unknown": "none / marginal",
        "magnitude": "grade 1-4",
        "circular": True,                         # Circular: M == int(S), not an independent extent measurement
        "label_noise": "low (adjudicated by retina specialists)",
        "n": 1600,                                # Total across train/val/test
        "n_total": 1600,
        "n_min_state": 35,                        # G4 has only ~35 samples in train split
        "extent_carrying_states": ["G1", "G2", "G3", "G4"],  # 4 states, but collinear with state
    },
    "chestmnist:source_bbox": {
        "dataset": "chestmnist",
        "finding": "source_bbox_mass",
        "states_native": ["present", "absent"],
        "magnitude_native": "box area (source data, subset only)",
        "unknown": "none",
        "magnitude": "bbox area (w * h in pixels or cm^2)",
        "verify": "ID matching, subset size: NIH ChestX-ray14 BBox_List_2017.csv contains 984 images",
        "label_noise": "gold-standard radiologist bounding boxes",
        "n_total": 984,
        "n_min_state": 85,                        # ~85 mass bounding boxes in NIH 984 subset
        "circular": False,                        # Physical 2D area is independent of binary diagnosis
        "extent_carrying_states": ["present"],    # 1 state carries magnitude
    },
}


# ------------------------------------------------------------------ Usability Auditor
def usable_as_typed(c: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """
    Audits whether a candidate concept is mathematically and statistically usable as a typed concept:
      1. independent_magnitude: Is the magnitude an independent measurement, separate from the discrete state?
                                (Fails if magnitude is circular / deterministic relabeling of the state).
      2. two_extent_states:      Does the schema have >= 2 states with a magnitude?
                                (Required for inter-state kappa extent decoupling; hurdle models use >= 1).
      3. enough_rows:            Are there >= ~30 rows per state after missingness/injection?
                                (Guarantees statistical power for calibration and decodability tests).

    Returns:
      (independent_magnitude, two_extent_states, enough_rows)
    """
    # 1. Independent magnitude: not circular, and either native or injected continuous extent exists
    is_circular = bool(c.get("circular", False)) or (c.get("magnitude_native") == "grade 1-4 (same label as state?)")
    has_magnitude = (c.get("magnitude_native") is not None) or (c.get("magnitude") is not None and c.get("magnitude") != "none")
    independent_magnitude = has_magnitude and (not is_circular)

    # 2. Two extent states: kappa needs >= 2 states with a magnitude
    extent_states = c.get("extent_carrying_states", [])
    two_extent_states = len(extent_states) >= 2

    # 3. Enough rows: >= ~30 per state after any injection / subsetting
    min_count = c.get("n_min_state", 0)
    # If unknown injection reduces count by ~20%, effective count is ~0.8 * min_count
    if c.get("unknown") == "inject":
        effective_count = min_count * 0.8
    else:
        effective_count = min_count
    enough_rows = effective_count >= 30

    return independent_magnitude, two_extent_states, enough_rows


def audit_summary(c: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a full structured audit report for a candidate concept."""
    ind_mag, two_ext, enough = usable_as_typed(c)
    hurdle_usable = ind_mag and enough  # Single-hurdle requires independent magnitude and >= 30 rows
    multi_state_usable = ind_mag and two_ext and enough  # Multi-state kappa requires >= 2 extent states
    return {
        "independent_magnitude": ind_mag,
        "two_extent_states": two_ext,
        "enough_rows": enough,
        "hurdle_usable": hurdle_usable,
        "multi_state_kappa_usable": multi_state_usable,
    }


# ------------------------------------------------------------------ Type-Aware Annotations (PyC)
def get_example_annotations() -> pyc.Annotations:
    """
    User-specified standard type-aware annotations in PyC:
      - smoking: binary (cardinality 1)
      - genotype: categorical (cardinality 3)
      - tar: continuous (cardinality 1)
      - cancer: binary (cardinality 1)
    """
    return pyc.Annotations(
        labels=["smoking", "genotype", "tar", "cancer"],
        cardinalities=[1, 3, 1, 1],
        types=["binary", "categorical", "continuous", "binary"],
    )


def get_typed_medical_annotations() -> pyc.Annotations:
    """
    Type-aware concept annotations for typed medical findings:
      - mass_state: categorical in {UNKNOWN, ABSENT, PRESENT} (cardinality 3)
      - mass_magnitude: continuous lesion diameter (cardinality 1)
      - cardiomegaly_state: categorical in {UNKNOWN, ABSENT, PRESENT} (cardinality 3)
      - cardiomegaly_magnitude: continuous cardiothoracic ratio (cardinality 1)
    """
    return pyc.Annotations(
        labels=["mass_state", "mass_magnitude", "cardiomegaly_state", "cardiomegaly_magnitude"],
        cardinalities=[3, 1, 3, 1],
        types=["categorical", "continuous", "categorical", "continuous"],
    )


def demo_mix_concept_predictor(
    annotations: Optional[pyc.Annotations] = None,
    batch_size: int = 32,
    emb_size: int = 16,
    out_concepts: int = 1,
) -> torch.Tensor:
    """
    Demonstrates MixConceptEmbeddingToConcept with type-aware pyc.Annotations.
    """
    if annotations is None:
        annotations = get_example_annotations()

    predictor = MixConceptEmbeddingToConcept(
        in_concepts=annotations,   # type-aware: uses cardinalities and types
        in_embeddings=emb_size,
        out_concepts=out_concepts,
    )

    total_dim = sum(annotations.cardinalities)
    concepts = torch.randn(batch_size, total_dim)
    embeddings = torch.randn(batch_size, total_dim, emb_size)

    tasks = predictor(concepts=concepts, embeddings=embeddings)
    return tasks
