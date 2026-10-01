"""inference.py — how results are judged. Validity, specificity, informativeness.

Design rules: fixed probe ids; nested sample prefixes within each seed (so n=1000 contains
the n=500 sample); resample WORLDS/seeds, never cells, preserving every paired condition.
"""
import numpy as np


def paired_contrast(df, width: str, knob: str, lo, hi, by=("world_id", "probe_id", "seed")):
    """Primary readout: width at knob=hi minus width at knob=lo, paired within `by`.
    Slopes are summaries, not the test. TODO: return per-pair differences."""
    ...


def seed_bootstrap_ci(diffs, group, n_boot: int = 2000, level: float = 0.95):
    """Bootstrap over independent worlds/seeds. TODO."""
    ...


def equivalence_flat(diffs_ci, band: float) -> bool:
    """'Flat' is an EQUIVALENCE claim: the whole CI must sit inside +/- band. A non-significant
    slope does not show invariance. Applies to W_label vs n (Prop B) and W_unk off-support."""
    ...


def coverage_at_least_nominal(covered, nominal: float, level: float = 0.95):
    """Is coverage >= nominal? One-sided interval on the coverage rate. Report beside mean width."""
    ...
