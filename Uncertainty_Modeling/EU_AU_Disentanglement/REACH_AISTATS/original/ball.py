"""ball.py — the MMD ball of input densities the log cannot exclude.

P_n = { p : MMD(p, p_hat_n) <= eps_n }, admitting (a) mixtures of the frozen known-class
conditionals with free prevalences, and (b) those mixtures plus a delta-slice of an
unseen class. Everything downstream (widths.py) is a spread over this set.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class MMDBall:
    log_features: np.ndarray          # (n, d) the unlabelled log, in the ball's space
    class_samples: list               # per-class samples from the labelled set, same space
    kernel: str                       # TODO: bandwidth rule; median in 512-d saturates (see notes)
    space: str                        # "pca32" | "class_loglik" — high-d RBF cannot see prevalence
    eps_n: float
    eps_const: float                  # the constant in eps_n = eps_const * n**(-1/2)?  TODO: state on paper
    bump_scale: float                 # smoothness of the unseen-class slice; DEFINITIONAL, log it

    def resolution(self, w_from, w_to) -> float:
        """ball_resolution, defined relative to a STATED prevalence change.
        resolution = MMD(P_{w_to}, P_{w_from}) / eps_n, e.g. w_from=(.5,.5), w_to=(.9,.1).
        > 1 means the ball can tell that change apart from sampling noise at this n.
        A failure at the smallest n describes the method's operating range, not necessarily
        a broken study: report the n at which resolution crosses 1.
        TODO: fix the representation and kernel BEFORE evaluation; PCA / class scores change
        what the ball can distinguish."""
        ...

    def admissible_prevalences(self, n_grid: int = 200) -> np.ndarray:
        """Prevalence vectors w on the simplex with MMD(sum_c w_c P_c, p_hat_n) <= eps_n.
        TODO: grid / projected solver; K=2 can be a 1-D line, K>2 needs a sampler."""
        ...

    def admissible_novel_slices(self, delta_n: float) -> list:
        """Densities of the form (1-delta_n)*mixture + delta_n*q still inside the ball.
        TODO: q ranges over smooth bumps, supremum limited by bump_scale (honesty clause 2)."""
        ...

    def covers_truth(self, true_log_sampler) -> bool:
        """A12: is the *true* deployment density inside the ball? Run over many seeds; the
        claim is AT LEAST nominal coverage (a conservative ball may overcover), so report
        coverage together with width: validity from a saturated ball is vacuous.
        TODO: MMD(p*, p_hat_n) against eps_n. If p* is approximated by a large reference
        sample, account for that approximation error in the comparison."""
        ...
