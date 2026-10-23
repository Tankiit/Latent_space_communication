"""
Demonstration and Verification Script for MedMNIST DataLoaders:
- PathMNIST
- BloodMNIST
- RetinaMNIST
"""

import sys
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import medmnist
from medmnist import INFO

from typed_cbm.medmnist_loader import get_medmnist_loaders


def main():
    print("=" * 75)
    print("MEDMNIST DATALOADERS & PREPROCESSING VERIFICATION")
    print("=" * 75)

    datasets_to_try = ["pathmnist", "bloodmnist", "retinamnist"]
    BATCH_SIZE = 64
    download = True

    for name in datasets_to_try:
        print(f"\n--- Loading and Evaluating: {name.upper()} ---")

        # 1. Exact user requested pattern:
        info = INFO[name]
        DataClass = getattr(medmnist, info["python_class"])

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ])

        # Load datasets
        train_dataset = DataClass(split="train", transform=data_transform, download=download)
        test_dataset = DataClass(split="test", transform=data_transform, download=download)
        pil_dataset = DataClass(split="train", download=download)

        # Encapsulate data into dataloader form
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)

        # 2. Check shapes and metadata
        print(f"  Class: {info['python_class']} | Task: {info['task']}")
        print(f"  Channels: {info['n_channels']} | Num Classes: {len(info['label'])}")
        print(f"  Train samples: {len(train_dataset):,} | Test samples: {len(test_dataset):,}")
        print(f"  PIL dataset sample 0 type: {type(pil_dataset[0][0])}")

        # 3. Fetch sample batches
        images, labels = next(iter(train_loader))
        print(f"  Train Batch - Images: {images.shape}, Range: [{images.min():.2f}, {images.max():.2f}], Labels: {labels.shape}")

        eval_images, eval_labels = next(iter(train_loader_at_eval))
        print(f"  Eval Batch  - Images: {eval_images.shape}, Labels: {eval_labels.shape}")

        test_images, test_labels = next(iter(test_loader))
        print(f"  Test Batch  - Images: {test_images.shape}, Labels: {test_labels.shape}")

        # Verification check
        assert images.shape[0] == BATCH_SIZE
        assert eval_images.shape[0] == 2 * BATCH_SIZE or eval_images.shape[0] == len(train_dataset)
        assert test_images.shape[0] == 2 * BATCH_SIZE or test_images.shape[0] == len(test_dataset)
        print(f"  Status: [OK] Successfully verified {name}.")

    print("\n" + "=" * 75)
    print("All MedMNIST datasets (PathMNIST, BloodMNIST, RetinaMNIST) successfully loaded and verified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
