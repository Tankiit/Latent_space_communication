"""Read cached arrays without allowing pickled objects.

The caller must ensure independence of the development, reference, log,
and evaluation pools; array loading cannot verify this assumption.
"""
import numpy as np


def load_cache(path):
    """Return an NPZ context manager. Schema validation occurs during fitting."""
    return np.load(path, allow_pickle=False)
