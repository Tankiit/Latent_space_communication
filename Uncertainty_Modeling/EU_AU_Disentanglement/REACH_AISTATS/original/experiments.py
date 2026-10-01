"""experiments.py — the real experiments, one function per claim.

What each one is for (in the order they should run; a failed gate stops the ladder):

  1. gate_ball_resolution            Can the kernel see prevalence at the smallest n at all?
  2. ball_coverage_rate              Does the true density sit inside the ball at the nominal rate?  (A12)
  3. prevalence_response_to_log      Does W_cls shrink as the unlabelled log grows?
  4. label_width_ignores_log         Is W_label flat in n and shrinking in m, growing in eta?      (Prop B)
  5. coverage_on_vs_off_support      Does W_unk shrink with n on-support and stay flat off-support? (Prop C)
  6. em_bootstrap_baseline           Is W_cls more than bootstrap variance of an EM prevalence?      (A6)
  7. frozen_conditionals_diagnostic  How badly does the frozen-conditionals assumption fail?         (A3)
  8. decomposition_gap               How far are the parts from the joint width? (proof first)
     + absent_region_arm / present_unlabelled_arm / hull_confounded_arm / density_only_baseline
  9. response_matrix                 Assemble 3-8 into the signed-slope table vs predictions.py.
"""
import numpy as np
from worlds import World
from acquisitions import draw_log, draw_labelled_set
from ball import MMDBall
import inference as inf
from widths import credal_widths
import predictions as P


def gate_ball_resolution(world: World, n_min: int, seeds) -> float:
    """TODO: build the ball at n_min over seeds; return the minimum resolution.
    Kill rule: if <= P.MIN_RESOLUTION, fix the ball construction before anything else."""
    ...


def ball_coverage_rate(world: World, n_grid, seeds) -> dict:
    """TODO: for each n, fraction of seeds with ball.covers_truth(...). Report vs nominal."""
    ...


def prevalence_response_to_log(world: World, n_grid, m: int, seeds) -> "pd.DataFrame":
    """TODO: hold labels fixed, sweep n, record W_cls at boundary / class-centre probes."""
    ...


def label_width_ignores_log(world: World, n_grid, m_grid, eta_grid, seeds) -> "pd.DataFrame":
    """TODO: three sweeps, each moving one knob. W_label vs n is the Proposition B test:
    a slight shrink in n is either a real failure or the leak from non-frozen conditionals;
    the A3 diagnostic below is what tells you which."""
    ...


def coverage_on_vs_off_support(world: World, n_grid, seeds) -> "pd.DataFrame":
    """TODO: two logs, one containing the unseen-class slice and one not (include_novel flag),
    W_unk at on-support and off-support probes. Needs delta and separation s defined."""
    ...


def em_bootstrap_baseline(world: World, n_grid, seeds) -> "pd.DataFrame":
    """TODO: EMQ prevalence on the log, bootstrap its spread, push through the classifier,
    compare to W_cls at the same probes. Report ratio and rank correlation."""
    ...


def frozen_conditionals_diagnostic(world: World, n: int, seeds) -> "pd.DataFrame":
    """TODO: density-ratio estimate of labelled vs deployment inputs; `prevalence_explained`
    = fraction of the shift a pure prevalence change accounts for. 1.0 means frozen holds."""
    ...


def decomposition_gap(world: World, n: int, m: int, seeds) -> "pd.DataFrame":
    """NOT a pass/fail check. Widths of separately varied sources need not add to the joint
    width (f(a,b)=ab on [0,1]^2: joint 1, two marginals 1 each). First decide, on paper, which
    claim REACH makes: marginal sensitivities / a proved inequality / nested increments
    (order-dependent) / an exact identity. Only the last justifies the word 'decomposition'.
    TODO: report the gap W_total - sum(parts) per cell, so the wording can be fitted to it."""
    ...


def absent_region_arm(world: World, n_grid, seeds) -> "pd.DataFrame":
    """delta = 0; probes outside the known support (use COMPACT-support worlds for literal
    off-support; Gaussian tails only give low density). What stays unresolved with no data there?"""
    ...


def present_unlabelled_arm(world: World, n_grid, seeds) -> "pd.DataFrame":
    """delta > 0, the log samples the full mixture. Can the log separate unexplained input mass
    from known-class prevalence? Note: this reveals input mass, NOT the unseen class's label,
    so state which of the two W_unk measures."""
    ...


def hull_confounded_arm(world: World, n_grid, seeds) -> "pd.DataFrame":
    """q_s inside the convex hull of the known class-conditionals, so novel mass can be absorbed
    into changed prevalences. Observational equivalence: W_unk should NOT resolve. This is the
    stated identification limit, shown rather than asserted."""
    ...


def density_only_baseline(world: World, n_grid, seeds) -> "pd.DataFrame":
    """KDE / Mahalanobis score at the same probes. W_unk is algebraically monotone in the log's
    density at x (honesty clause 1), so the rank correlation is a sanity check, not a result.
    TODO: report what the calibration and the set-validity add beyond this ranking."""
    ...


def response_matrix(world: World, grid: dict, seeds) -> "pd.DataFrame":
    """TODO: one row per (n, m, eta, region, seed) matching SCHEMA in the plan doc; then
    signed slopes with bootstrap CIs, compared to P.PREDICTED within P.TOL."""
    ...


if __name__ == "__main__":
    # TODO: order = gate -> coverage -> responses -> baselines -> diagnostics -> matrix.
    # Stop at the first failed gate and print which rung failed.
    ...
