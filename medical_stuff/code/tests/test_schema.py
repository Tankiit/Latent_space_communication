"""
Unit tests for schema and TaggedUnionBatch.
"""

import unittest
import torch
from typed_cbm.schema import TypedConceptSpec, ConceptType, TaggedUnionBatch


class TestSchema(unittest.TestCase):
    def test_spec_validation(self):
        spec = TypedConceptSpec(name="Effusion", concept_type=ConceptType.TYPED_HURDLE)
        # Valid: state 0 has 0 extent
        s_valid = torch.tensor([0.0, 1.0])
        e_valid = torch.tensor([0.0, 5.2])
        self.assertTrue(spec.validate(s_valid, e_valid))

        # Invalid: state 0 has non-zero extent
        s_invalid = torch.tensor([0.0, 1.0])
        e_invalid = torch.tensor([3.1, 5.2])
        self.assertFalse(spec.validate(s_invalid, e_invalid))

    def test_tagged_union_batch(self):
        B, K = 4, 3
        states = torch.tensor([
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        extents = torch.tensor([
            [2.5, 9.9, 1.2],
            [8.8, 7.7, 3.4],
            [1.1, 4.4, 6.6],
            [5.5, 2.2, 3.3],
        ])
        batch = TaggedUnionBatch(states=states, extents=extents)

        # get_masked_extents should zero out absent slots
        masked = batch.get_masked_extents(fill_value=0.0)
        self.assertEqual(masked[0, 1].item(), 0.0)
        self.assertEqual(masked[1, 0].item(), 0.0)
        self.assertEqual(masked[1, 1].item(), 0.0)
        self.assertEqual(masked[0, 0].item(), 2.5)

        # Bottleneck vector should have shape (B, 2*K) = (4, 6)
        z = batch.to_bottleneck_representation()
        self.assertEqual(z.shape, (B, 2 * K))
        # Check interleaving: z[0] = [s0, s0*e0, s1, s1*e1, s2, s2*e2]
        self.assertEqual(z[0, 0].item(), 1.0)
        self.assertEqual(z[0, 1].item(), 2.5)
        self.assertEqual(z[0, 2].item(), 0.0)
        self.assertEqual(z[0, 3].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
