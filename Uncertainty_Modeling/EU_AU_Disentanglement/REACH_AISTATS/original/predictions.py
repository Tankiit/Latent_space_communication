"""predictions.py — pre-registration. The PI commits this file BEFORE the grid runs.

Sign in {-1, 0, +1} for 'width moves with knob', or None for 'exploratory, no claim'.
The analysis imports this and produces pass/fail; nobody edits it after the first full run.
"""
# PROTOCOL — these predictions are conditional on it, say so in the paper:
#  * labelled set drawn under a REFERENCE prevalence, so labels carry no information about
#    deployment prevalence (deployment-drawn labels would also shrink W_cls);
#  * exact anchor: oracle conditionals. Estimated-conditionals is a SEPARATE table.
#  * label noise: symmetric binary flip. The (1 - 2*eta) law is not claimed beyond that.
PROTOCOL = dict(labelled_prevalence="reference", conditionals="oracle", noise="binary_symmetric_flip")

PREDICTED = {
    # (width, knob, region): sign
    ("W_cls",   "log_n", "boundary"):    ...,   # TODO: -1 expected
    ("W_cls",   "m",     "boundary"):    ...,   # TODO: ?
    ("W_cls",   "eta",   "boundary"):    ...,   # TODO: ?
    ("W_label", "log_n", "boundary"):    ...,   # TODO: 0 expected (Proposition B)
    ("W_label", "m",     "boundary"):    ...,   # TODO: -1 expected
    ("W_label", "eta",   "boundary"):    ...,   # TODO: +1 expected
    ("W_unk",   "log_n", "on_support"):  ...,   # TODO: -1 expected
    ("W_unk",   "log_n", "off_support"): ...,   # TODO: 0 expected (Proposition C)
    ("W_unk",   "m",     "on_support"):  ...,   # TODO: ?
    ("W_unk",   "eta",   "on_support"):  ...,   # TODO: ?
}

EQUIV_BAND = ...   # TODO: practically negligible change, chosen from scientific meaning;
                   # the pilot null band informs PRECISION (seeds), not what counts as negligible
TOL = ...   # TODO: from the split-half null band at the smallest n, not by eye
MIN_RESOLUTION = 1.0   # ball_resolution gate
