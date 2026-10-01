"""Prospective prototype hypotheses; this file is not a historical preregistration.
Already inspected prior ideas/results must be disclosed. Freeze this file and config before
new confirmatory runs. No prediction here is an asserted theorem: see the open theory
obligations in README.md before treating any of them as established.
"""
PROTOCOL=dict(labels='reference_source_stratified',conditionals='oracle_anchor',
              noise='known_binary_symmetric_flip',widths='distinct_estimands_not_additive',
              set='coordinate_outer_approximation_to_categorical_MMD')
PREDICTED={
 ('W_cls','n','separated'):-1,
 ('W_label','n','separated'):0,
 ('W_unk','n','separated'):-1,
 ('W_cls','m','separated'):0,
 ('W_unk','m','separated'):0,
 ('W_label','m','separated'):-1,
 ('W_cls','eta','separated'):0,
 ('W_unk','eta','separated'):0,
 ('W_label','eta','separated'):1,
 ('W_unk','n','confounded'):None,
}
# Proposed margin on probability-width difference; scientific approval needed before confirmation.
EQUIV_BAND=.02
MIN_EFFECT=.02
