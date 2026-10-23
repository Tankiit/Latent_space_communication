"""
Unit tests for experiments_e0_e2.py functions and models.
"""

import unittest
import numpy as np
import torch
from experiments_e0_e2 import (
    UNKNOWN, ABSENT, PRESENT, HAS_MAG,
    make_typed_annotation, make_outcome,
    e0_hurdle_audit, e1_estimators,
    TypedHead, typed_loss,
    IndicatorBaselineHead, PooledBaselineHead,
    evaluate_absence_invariance,
)
import torch_concepts as pyc


class TestExperimentsE0E2(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        # Synthetic binary labels for testing
        self.labels = self.rng.integers(0, 2, size=(200, 5))
        self.feats = self.rng.standard_normal(size=(200, 32))

    def test_e0_hurdle_property(self):
        state, mag, *_ = make_typed_annotation(
            self.labels, finding_idx=0, p_unknown=0.2,
            image_dependent=False, rng=self.rng, mechanism="mcar"
        )
        absent_with_mag, states_with_mag, n_per_state = e0_hurdle_audit(state, mag)
        self.assertEqual(absent_with_mag, 0.0)
        self.assertEqual(states_with_mag, 1)
        self.assertGreater(n_per_state[UNKNOWN], 0)
        self.assertGreater(n_per_state[ABSENT], 0)
        self.assertGreater(n_per_state[PRESENT], 0)

    def test_e1_estimators_bias(self):
        state, mag, *_ = make_typed_annotation(
            self.labels, finding_idx=0, p_unknown=0.2,
            image_dependent=False, rng=self.rng, mechanism="mcar"
        )
        y = make_outcome(state, mag, rng=self.rng)
        mean_est, slope_est = e1_estimators(state, mag, y=y)

        # Zero-fill must be strictly smaller than drop-rows mean
        self.assertLess(mean_est["zero_fill"], mean_est["drop_rows"])
        self.assertIn("truth", mean_est)
        self.assertIn("zero_fill", slope_est)

    def test_typed_head_forward_and_loss(self):
        d_in = 32
        head = TypedHead(d_in=d_in, n_states=3)
        x = torch.randn(16, d_in)
        s_logits, m_pred = head(embeddings=x)
        self.assertEqual(s_logits.shape, (16, 3))
        self.assertEqual(m_pred.shape, (16, 1))

        # Test loss
        s_true = torch.randint(0, 3, (16,))
        m_true = torch.tensor([1.5 if s == PRESENT else float("nan") for s in s_true])
        loss, ce, mse = typed_loss(s_logits, m_pred, s_true, m_true)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_absence_invariance_comparison(self):
        d_in = 16
        # Typed head + downstream
        t_head = TypedHead(d_in=d_in, n_states=3)
        t_pred = pyc.nn.LinearConceptToConcept(in_concepts=4, out_concepts=1)
        x = torch.randn(10, d_in)

        typed_leak = evaluate_absence_invariance("typed", t_head, t_pred, x)
        self.assertEqual(typed_leak, 0.0)

        # Indicator head + downstream with active weight on magnitude
        ind_head = IndicatorBaselineHead(d_in=d_in)
        ind_pred = pyc.nn.LinearConceptToConcept(in_concepts=2, out_concepts=1)
        # Manually set downstream weight to guarantee non-zero magnitude contribution
        ind_pred.predictor.weight.data = torch.tensor([[1.5, 0.5]])

        ind_leak = evaluate_absence_invariance("indicator", ind_head, ind_pred, x)
        self.assertGreater(ind_leak, 0.0)


if __name__ == "__main__":
    unittest.main()
