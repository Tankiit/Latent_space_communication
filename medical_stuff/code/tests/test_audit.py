"""
Unit tests for HurdleAuditor.
"""

import unittest
import numpy as np
from typed_cbm.hurdle_audit import HurdleAuditor


class TestHurdleAuditor(unittest.TestCase):
    def test_structural_zero_vuong(self):
        np.random.seed(42)
        n = 1000
        # True hurdle data: 40% zeros, 60% LogNormal(1.0, 0.4)
        zeros = np.zeros(400)
        pos = np.exp(np.random.normal(1.0, 0.4, 600))
        data = np.concatenate([zeros, pos])

        res = HurdleAuditor.test_structural_zero(data)
        self.assertTrue(res["passed"])
        self.assertGreater(res["vuong_stat"], 1.96)
        self.assertAlmostEqual(res["prop_zeros"], 0.40, places=2)

    def test_extent_decoupling(self):
        states = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        extents = np.array([0, 0, 1.2, 3.4, 2.1, 5.5, 4.3, 2.9, 1.8, 4.0, 3.1, 2.2])

        res = HurdleAuditor.test_extent_decoupling(states, extents)
        self.assertTrue(res["passed"])
        self.assertGreater(res["cv"], 0.15)
        self.assertGreaterEqual(res["num_unique"], 10)

    def test_conditional_calibration(self):
        np.random.seed(42)
        n = 500
        true_s = (np.random.rand(n) > 0.5).astype(float)
        pred_pi = np.clip(true_s + np.random.normal(0, 0.1, n), 0.01, 0.99)

        true_e = np.exp(np.random.normal(1.0, 0.3, n))
        pred_mu = np.ones(n) * 1.0
        pred_sigma = np.ones(n) * 0.3

        res = HurdleAuditor.test_conditional_calibration(
            pred_pi, true_s, pred_mu, pred_sigma, true_e
        )
        self.assertTrue(res["passed"])
        self.assertLess(res["gate_ece"], 0.08)
        self.assertAlmostEqual(res["extent_coverage_90"], 0.90, delta=0.06)


if __name__ == "__main__":
    unittest.main()
