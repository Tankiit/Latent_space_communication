"""Finite-cell, three-source worlds; source identity is not the binary target Y."""
from dataclasses import dataclass
import numpy as np

@dataclass
class World:
    name: str
    conditionals: np.ndarray  # source x input cell; last source is candidate novelty
    weights: np.ndarray
    theta: np.ndarray        # binary clean-target probability per source
    regions: tuple
    def __post_init__(self):
        self.conditionals = np.asarray(self.conditionals, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.theta = np.asarray(self.theta, dtype=float)
        k, d = self.conditionals.shape
        if k != 3 or len(self.weights) != k or len(self.theta) != k or len(self.regions) != d:
            raise ValueError('Prototype requires two known sources plus one candidate novel source')
        if not np.all(self.conditionals >= 0) or not np.allclose(self.conditionals.sum(1), 1):
            raise ValueError('Each source conditional must be a probability vector')
        if np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1):
            raise ValueError('Invalid source weights')
        if np.any((self.theta < 0) | (self.theta > 1)):
            raise ValueError('Invalid target probabilities')
    @property
    def marginal(self):
        return self.weights @ self.conditionals
    def target(self, cell):
        den = self.marginal[cell]
        return float((self.weights * self.conditionals[:,cell]) @ self.theta / den) if den > 0 else np.nan


def make_separation_world(sep, delta=0.1, novel=(.1, .1, .8, 0.)):
    """Known-source embeddings with controllable separation on the simplex tangent space.

    sep=1 gives two clearly distinct known conditionals; sep near 0 makes them nearly
    collinear, so the mixture weights approach non-identifiability. This is a different
    world family from make_world: the known rows are symmetric by construction so that
    the tangent-space gap is exactly ||a_0 - a_1||_2 = 0.4*sep*sqrt(2).
    """
    if not 0 < sep <= 1:
        raise ValueError('sep must lie in (0,1]')
    tilt = .2*sep
    known = np.array([[.5+tilt, .5-tilt, 0, 0], [.5-tilt, .5+tilt, 0, 0]])
    return World(f'sep{sep:g}', np.vstack([known, np.asarray(novel, dtype=float)]),
                 np.array([.65*(1-delta), .35*(1-delta), delta]), np.array([.2, .8, .9]),
                 ('on_support', 'boundary', 'novel_region', 'off_support'))


def separation_gap(world):
    """Euclidean distance between the two known source embeddings."""
    return float(np.linalg.norm(world.conditionals[0]-world.conditionals[1]))


def tilted_conditionals(world, shift, known=2):
    """Move known-source mass into the cell no declared source supports.

    Redistributing mass between cells 0 and 1 would stay inside the span of the two known
    rows and so remain exactly representable; such a tilt is undetectable by construction
    and is not a test of anything. Moving mass to the unsupported last cell genuinely
    leaves the declared family, so the mixture residual is bounded below by the moved mass.
    A detected incompatibility does not identify this tilt, or any other cause.
    """
    if not 0 <= shift <= 1:
        raise ValueError('shift must lie in [0,1]')
    a = world.conditionals.copy()
    if np.any(a[:, -1] > 0):
        raise ValueError('Last cell must be unsupported by every declared source')
    moved = shift*a[:known, 0]
    a[:known, 0] -= moved
    a[:known, -1] += moved
    return a


def make_world(name='separated', delta=0.1):
    if not 0 <= delta < 1:
        raise ValueError('delta must be in [0,1)')
    known = np.array([[.7,.3,0,0], [.2,.8,0,0]])
    # Overlap at known-region probes makes novelty contraction observable there;
    # positive mass at cell 2 keeps q outside the known convex hull.
    q = np.array([.1,.1,.8,0], dtype=float)
    if name == 'confounded':
        q = .5 * (known[0] + known[1])
    elif name == 'absent':
        delta = 0.
    elif name != 'separated':
        raise ValueError(name)
    return World(name, np.vstack([known,q]), np.array([.65*(1-delta),.35*(1-delta),delta]),
                 np.array([.2,.8,.9]), ('on_support','boundary','novel_region','off_support'))
