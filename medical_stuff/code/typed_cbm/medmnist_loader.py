"""
MedMNIST DataLoaders and Preprocessing for PathMNIST, BloodMNIST, and RetinaMNIST.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO


def get_medmnist_loaders(
    dataset_name: str = "pathmnist",
    batch_size: int = 128,
    download: bool = True,
    root: Optional[str] = None,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Constructs MedMNIST train, eval, and test DataLoaders with the standard preprocessing pipeline.

    Preprocessing:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    Datasets loaded:
        train_dataset = DataClass(split='train', transform=data_transform, download=download)
        test_dataset = DataClass(split='test', transform=data_transform, download=download)
        val_dataset = DataClass(split='val', transform=data_transform, download=download)
        pil_dataset = DataClass(split='train', download=download)

    DataLoaders encapsulated:
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    Args:
        dataset_name: One of 'pathmnist', 'bloodmnist', 'retinamnist', or other MedMNIST flag.
        batch_size: Batch size for training.
        download: Whether to download dataset if not cached.
        root: Storage path for datasets (defaults to MedMNIST standard ~/.medmnist).
        num_workers: Number of workers for DataLoader.

    Returns:
        Dictionary containing dataloaders, datasets, and metadata.
    """
    flag = dataset_name.lower().strip()
    if flag not in INFO:
        raise ValueError(
            f"Unknown dataset '{flag}'. Supported MedMNIST datasets include: "
            f"{list(INFO.keys())}"
        )

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])

    # Preprocessing transform as specified
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    kwargs: Dict[str, Any] = {"download": download}
    if root is not None:
        kwargs["root"] = root

    # Load datasets
    train_dataset = DataClass(split="train", transform=data_transform, **kwargs)
    test_dataset = DataClass(split="test", transform=data_transform, **kwargs)
    val_dataset = DataClass(split="val", transform=data_transform, **kwargs)

    # PIL dataset without tensor transforms (useful for visualization and inspection)
    pil_dataset = DataClass(split="train", **kwargs)

    # Encapsulate data into dataloader form
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    train_loader_at_eval = data.DataLoader(
        dataset=train_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "flag": flag,
        "info": info,
        "DataClass": DataClass,
        "data_transform": data_transform,
        # Datasets
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "val_dataset": val_dataset,
        "pil_dataset": pil_dataset,
        # Loaders
        "train_loader": train_loader,
        "train_loader_at_eval": train_loader_at_eval,
        "test_loader": test_loader,
        "val_loader": val_loader,
        # Dimensions
        "n_channels": info["n_channels"],
        "n_classes": len(info["label"]),
        "task": info["task"],
        "labels_dict": info["label"],
    }


def try_medmnist_datasets(
    datasets: Optional[List[str]] = None,
    batch_size: int = 64,
    download: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Utility to load and verify multiple MedMNIST datasets (defaults to pathmnist, bloodmnist, retinamnist).
    """
    if datasets is None:
        datasets = ["pathmnist", "bloodmnist", "retinamnist"]

    results = {}
    for name in datasets:
        bundle = get_medmnist_loaders(dataset_name=name, batch_size=batch_size, download=download)
        results[name] = bundle
    return results
