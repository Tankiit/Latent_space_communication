"""
Unit tests for HurdleEmbeddingToConcept, TypedBottleneckLayer, and LogNormalHurdleLoss.
"""

import unittest
import torch
import torch_concepts as pyc
from typed_cbm.layers import HurdleEmbeddingToConcept, TypedBottleneckLayer, LogNormalHurdleLoss


class TestLayers(unittest.TestCase):
    def test_hurdle_layer_forward(self):
        B, D, K = 8, 16, 3
        layer = HurdleEmbeddingToConcept(in_embeddings=D, out_concepts=K)
        x = torch.randn(B, D)
        logit_pi, mu, log_sigma = layer(embeddings=x)

        self.assertEqual(logit_pi.shape, (B, K))
        self.assertEqual(mu.shape, (B, K))
        self.assertEqual(log_sigma.shape, (B, K))

    def test_typed_bottleneck_absence_invariance(self):
        B, D, K = 5, 12, 2
        layer = TypedBottleneckLayer(in_embeddings=D, num_concepts=K)
        x = torch.randn(B, D)

        # Force concept 0 to be state = 0
        concepts, batch, params = layer(embeddings=x, hard=True, state_intervention={0: 0.0})

        # Bottleneck concept vector is (B, 4)
        # Coordinate 0: state 0 (must be 0.0)
        # Coordinate 1: state 0 * extent 0 (must be 0.0)
        self.assertTrue((concepts[:, 0] == 0.0).all())
        self.assertTrue((concepts[:, 1] == 0.0).all())

    def test_hurdle_loss_masked_extent(self):
        B, K = 6, 2
        loss_fn = LogNormalHurdleLoss()

        params = {
            "logit_pi": torch.zeros(B, K, requires_grad=True),
            "mu_extent": torch.zeros(B, K, requires_grad=True),
            "log_sigma": torch.zeros(B, K, requires_grad=True),
        }

        # Half absent, half present
        target_s = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        target_e = torch.tensor([
            [2.0, 0.0],
            [1.5, 0.0],
            [3.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])

        total_loss, gate_loss, extent_loss = loss_fn(params, target_s, target_e)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertGreater(gate_loss.item(), 0.0)
        self.assertGreater(extent_loss.item(), 0.0)

        # Backward check
        total_loss.backward()
        # Gradient for mu_extent on concept 1 (which was all absent) should be ZERO!
        self.assertEqual(params["mu_extent"].grad[:, 1].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
