"""
Unit tests for Concept Registry, Usability Auditor, and PyC Type-Aware Annotations.
"""

import unittest
import torch
import torch_concepts as pyc
from torch_concepts.nn import MixConceptEmbeddingToConcept

from typed_cbm.concept_registry import (
    CONCEPTS,
    usable_as_typed,
    audit_summary,
    get_example_annotations,
    get_typed_medical_annotations,
    demo_mix_concept_predictor,
)


class TestConceptRegistry(unittest.TestCase):
    def test_concepts_keys_and_schema(self):
        self.assertIn("chestmnist:mass", CONCEPTS)
        self.assertIn("retinamnist:grade", CONCEPTS)
        self.assertIn("chestmnist:source_bbox", CONCEPTS)

        for key, c in CONCEPTS.items():
            self.assertIn("states_native", c)
            self.assertIn("magnitude_native", c)

    def test_usable_as_typed_evaluations(self):
        # 1. chestmnist:mass
        ind_mass, two_mass, enough_mass = usable_as_typed(CONCEPTS["chestmnist:mass"])
        self.assertTrue(ind_mass)
        self.assertFalse(two_mass)  # Only PRESENT has extent (single-hurdle)
        self.assertTrue(enough_mass)

        # 2. retinamnist:grade
        ind_ret, two_ret, enough_ret = usable_as_typed(CONCEPTS["retinamnist:grade"])
        self.assertFalse(ind_ret)   # Circular: magnitude == grade integer
        self.assertTrue(two_ret)    # G1-G4 are 4 states

        # 3. chestmnist:source_bbox
        ind_bbox, two_bbox, enough_bbox = usable_as_typed(CONCEPTS["chestmnist:source_bbox"])
        self.assertTrue(ind_bbox)   # BBox area is independent physical measurement
        self.assertFalse(two_bbox)  # Only PRESENT has bbox
        self.assertTrue(enough_bbox)

    def test_user_requested_annotations_and_mix_predictor(self):
        annotations = pyc.Annotations(
            labels=["smoking", "genotype", "tar", "cancer"],
            cardinalities=[1, 3, 1, 1],
            types=["binary", "categorical", "continuous", "binary"],
        )
        self.assertEqual(sum(annotations.cardinalities), 6)

        predictor = MixConceptEmbeddingToConcept(
            in_concepts=annotations,
            in_embeddings=16,
            out_concepts=1,
        )

        concepts = torch.randn(32, 6)
        embeddings = torch.randn(32, 6, 16)
        tasks = predictor(concepts=concepts, embeddings=embeddings)

        self.assertEqual(tasks.shape, (32, 1))

    def test_typed_medical_annotations(self):
        med_annotations = get_typed_medical_annotations()
        self.assertEqual(sum(med_annotations.cardinalities), 8)

        med_tasks = demo_mix_concept_predictor(
            annotations=med_annotations,
            batch_size=16,
            emb_size=16,
            out_concepts=1,
        )
        self.assertEqual(med_tasks.shape, (16, 1))


if __name__ == "__main__":
    unittest.main()
