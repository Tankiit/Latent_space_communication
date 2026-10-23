"""
Unit tests for models and intervention leakage using PyC style.
"""

import unittest
import torch
import torch_concepts as pyc
from typed_cbm.models import TypedCBM, StandardCBM, UncoupledTwoHeadCBM, CEMBaseline
from typed_cbm.interventions import evaluate_leakage, apply_typed_intervention


class TestModelsAndInterventions(unittest.TestCase):
    def setUp(self):
        self.B = 10
        self.D = 16
        self.K = 3
        self.num_classes = 2
        self.embeddings = torch.randn(self.B, self.D)

    def test_pyc_standard_cbm_style(self):
        """
        Tests the exact PyC style requested by the user:
            encoder = pyc.nn.LinearEmbeddingToConcept(in_embeddings=..., out_concepts=...)
            predictor = pyc.nn.LinearConceptToConcept(in_concepts=..., out_concepts=...)
            concepts = encoder(embeddings=x)
            tasks = predictor(concepts=concepts)
        """
        encoder = pyc.nn.LinearEmbeddingToConcept(in_embeddings=self.D, out_concepts=self.K)
        predictor = pyc.nn.LinearConceptToConcept(in_concepts=self.K, out_concepts=self.num_classes)

        concepts = encoder(embeddings=self.embeddings)
        tasks = predictor(concepts=concepts)

        self.assertEqual(concepts.shape, (self.B, self.K))
        self.assertEqual(tasks.shape, (self.B, self.num_classes))

    def test_typed_cbm_forward_and_loss(self):
        model = TypedCBM(in_embeddings=self.D, num_concepts=self.K, num_classes=self.num_classes)
        tasks, concepts, batch, params = model(embeddings=self.embeddings)

        self.assertEqual(tasks.shape, (self.B, self.num_classes))
        self.assertEqual(concepts.shape, (self.B, 2 * self.K))

        target_y = torch.randint(0, self.num_classes, (self.B,))
        target_s = torch.randint(0, 2, (self.B, self.K)).float()
        target_e = torch.rand(self.B, self.K) * target_s

        total_loss, metrics = model.compute_loss(tasks, params, target_y, target_s, target_e)
        self.assertGreater(total_loss.item(), 0.0)

    def test_absence_invariance_zero_leakage(self):
        """
        Critical Test:
        TypedCBM must have delta_leak == 0.0 under do(S=0) even when extent is perturbed.
        """
        typed_model = TypedCBM(in_embeddings=self.D, num_concepts=self.K, num_classes=self.num_classes)
        # Ensure predictor has random non-zero weights
        torch.nn.init.normal_(typed_model.predictor.predictor.weight, mean=1.0, std=0.5)

        leakage = evaluate_leakage(typed_model, self.embeddings, concept_idx=0)
        self.assertAlmostEqual(leakage, 0.0, places=6)

    def test_uncoupled_model_has_leakage(self):
        """
        UncoupledTwoHeadCBM will leak extent because predictor directly receives extent coordinate.
        """
        uncoupled = UncoupledTwoHeadCBM(in_embeddings=self.D, num_concepts=self.K, num_classes=self.num_classes)
        with torch.no_grad():
            uncoupled.predictor.predictor.weight.fill_(1.0)

        leakage = evaluate_leakage(uncoupled, self.embeddings, concept_idx=0)
        self.assertGreater(leakage, 0.01)

    def test_typed_intervention_helper(self):
        model = TypedCBM(in_embeddings=self.D, num_concepts=self.K, num_classes=self.num_classes)
        out = apply_typed_intervention(model, self.embeddings, concept_idx=1, state_val=0.0)
        self.assertEqual(out.shape, (self.B, self.num_classes))


if __name__ == "__main__":
    unittest.main()
