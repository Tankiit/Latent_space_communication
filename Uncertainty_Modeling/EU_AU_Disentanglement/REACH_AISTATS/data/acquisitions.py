"""Independent pools with prefixes shared across interventions within each replicate."""
import numpy as np

def draw_log(world, max_n, seed):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 101]))
    return rng.choice(len(world.marginal), max_n, p=world.marginal)

def draw_reference(world, r, seed, sources=2):
    """Finite source-stratified reference draws; r observations per known source.

    Returns estimated conditionals, so downstream sets must carry an embedding radius.
    """
    if r < 1:
        raise ValueError('Reference budget must be positive')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 303]))
    d = world.conditionals.shape[1]
    counts = np.vstack([np.bincount(rng.choice(d, int(r), p=world.conditionals[k]), minlength=d)
                        for k in range(sources)])
    return counts/float(r)


def draw_log_from(marginal, n, seed):
    """Unlabelled log from an explicit cell law, which need not be in the declared family."""
    marginal = np.asarray(marginal, dtype=float)
    if marginal.ndim != 1 or np.any(marginal < 0) or not np.isclose(marginal.sum(), 1):
        raise ValueError('marginal must be a probability vector')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 404]))
    return rng.choice(len(marginal), int(n), p=marginal)


def draw_labelled_set(world, max_m, eta, seed):
    if not 0 <= eta < .5:
        raise ValueError('Known symmetric binary flip rate must satisfy 0 <= eta < .5')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 202]))
    clean = rng.random((2,max_m)) < world.theta[:2,None]
    uniforms = rng.random((2,max_m))
    return np.logical_xor(clean, uniforms < eta).astype(float)
