"""
Intervention Mechanics and Absence Invariance Leakage Evaluation with PyC style.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from .models import TypedCBM, UncoupledTwoHeadCBM


def apply_typed_intervention(
    model: nn.Module,
    embeddings: torch.Tensor,
    concept_idx: int,
    state_val: Optional[float] = None,
    extent_val: Optional[float] = None,
) -> torch.Tensor:
    """
    Executes a typed intervention on concept `concept_idx`.
    Tagged Union Rules:
    - If state_val == 0: extent_val is automatically forced to 0 (Absence).
    - If extent_val > 0 is specified without state: state_val is forced to 1.0.
    """
    state_dict = {}
    extent_dict = {}

    if state_val is not None:
        state_dict[concept_idx] = float(state_val)
        if state_val == 0.0:
            extent_dict[concept_idx] = 0.0  # Force absence collapse

    if extent_val is not None and (state_val is None or state_val > 0.0):
        extent_dict[concept_idx] = float(extent_val)
        state_dict[concept_idx] = 1.0

    if isinstance(model, TypedCBM):
        tasks, _, _, _ = model(
            embeddings=embeddings,
            hard=True,
            state_intervention=state_dict,
            extent_intervention=extent_dict,
        )
        return tasks
    elif isinstance(model, UncoupledTwoHeadCBM):
        tasks, _, _, _ = model(
            embeddings=embeddings,
            state_intervention=state_dict,
            extent_intervention=extent_dict,
        )
        return tasks
    else:
        raise ValueError(f"Unsupported model type for typed intervention: {type(model)}")


def evaluate_leakage(
    model: nn.Module,
    embeddings: torch.Tensor,
    concept_idx: int,
    perturbation_values: Optional[List[float]] = None,
) -> float:
    """
    Evaluates Absence Invariance Leakage:
        Delta_leak = max_{e'} || tasks(S=0, E=e') - tasks(S=0, E=0) ||
    Under a truly typed model, Delta_leak must be exactly 0.0.
    Under an uncoupled model, Delta_leak > 0 because the task predictor attends to extent
    even when state is 0.
    """
    model.eval()
    if perturbation_values is None:
        perturbation_values = [0.01, 0.5, 1.0, 5.0, 20.0, 100.0]

    with torch.no_grad():
        # Baseline: State = 0, Extent = 0
        if isinstance(model, TypedCBM):
            base_tasks, _, _, _ = model(
                embeddings=embeddings,
                hard=True,
                state_intervention={concept_idx: 0.0},
                extent_intervention={concept_idx: 0.0},
            )
        elif isinstance(model, UncoupledTwoHeadCBM):
            base_tasks, _, _, _ = model(
                embeddings=embeddings,
                state_intervention={concept_idx: 0.0},
                extent_intervention={concept_idx: 0.0},
            )
        else:
            return 0.0

        max_diff = 0.0
        for val in perturbation_values:
            if isinstance(model, TypedCBM):
                pert_tasks, _, _, _ = model(
                    embeddings=embeddings,
                    hard=True,
                    state_intervention={concept_idx: 0.0},
                    extent_intervention={concept_idx: val},  # Masked to 0 in typed model!
                )
            elif isinstance(model, UncoupledTwoHeadCBM):
                pert_tasks, _, _, _ = model(
                    embeddings=embeddings,
                    state_intervention={concept_idx: 0.0},
                    extent_intervention={concept_idx: val},  # Leaks to predictor in uncoupled model!
                )

            diff = torch.norm(pert_tasks - base_tasks, p=float("inf"), dim=-1).max().item()
            if diff > max_diff:
                max_diff = diff

    return max_diff
