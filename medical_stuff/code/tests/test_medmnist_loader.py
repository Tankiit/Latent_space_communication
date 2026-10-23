"""
Unit tests for MedMNIST loader.
"""

import unittest
import torch
from typed_cbm.medmnist_loader import get_medmnist_loaders


class TestMedMNISTLoader(unittest.TestCase):
    def test_retinamnist_loaders(self):
        bundle = get_medmnist_loaders(dataset_name="retinamnist", batch_size=16, download=False)
        self.assertIn("train_loader", bundle)
        self.assertIn("test_loader", bundle)
        self.assertIn("train_loader_at_eval", bundle)
        self.assertIn("pil_dataset", bundle)

        train_loader = bundle["train_loader"]
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape[0], 16)
        self.assertEqual(images.shape[1], 3)  # RGB
        self.assertEqual(images.shape[2], 28)
        self.assertEqual(images.shape[3], 28)

        eval_loader = bundle["train_loader_at_eval"]
        eval_images, _ = next(iter(eval_loader))
        self.assertEqual(eval_images.shape[0], 32)  # 2 * BATCH_SIZE

    def test_bloodmnist_loaders(self):
        bundle = get_medmnist_loaders(dataset_name="bloodmnist", batch_size=8, download=False)
        train_loader = bundle["train_loader"]
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (8, 3, 28, 28))
        self.assertEqual(bundle["n_classes"], 8)

    def test_pathmnist_loaders(self):
        bundle = get_medmnist_loaders(dataset_name="pathmnist", batch_size=8, download=False)
        train_loader = bundle["train_loader"]
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (8, 3, 28, 28))
        self.assertEqual(bundle["n_classes"], 9)


if __name__ == "__main__":
    unittest.main()
