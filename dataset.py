"""
dataset.py

PyTorch Dataset for interior design style classification.
Expects data organized as:

    data/images/
    ├── mid_century_modern/
    │   ├── img_001.jpg
    │   └── ...
    ├── scandinavian/
    └── ...

Applies train/val augmentation strategies appropriate for style classification:
- Train: random crops, horizontal flip, color jitter, random rotation
- Val:   center crop, no augmentation
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# ImageNet normalization — required for pretrained EfficientNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """Transforms for single-image inference (same as val)."""
    return get_val_transforms()


class InteriorStyleDataset(Dataset):
    """
    Dataset for interior style classification.

    Args:
        root_dir: Path to the images directory (contains one subfolder per class)
        transform: torchvision transforms to apply
        class_names: Optional list of class names to use (filters and orders classes)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Discover classes from folder names
        discovered = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if class_names is not None:
            # Validate requested classes exist
            missing = set(class_names) - set(discovered)
            if missing:
                raise ValueError(f"Classes not found in {root_dir}: {missing}")
            self.class_names = class_names
        else:
            self.class_names = discovered

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        # Build flat list of (image_path, label) pairs
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                for img_path in sorted(cls_dir.glob(ext)):
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. "
                "Run: python data/scripts/download_dataset.py"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def class_counts(self) -> dict:
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {self.class_names[idx]: count for idx, count in sorted(counts.items())}


def build_dataloaders(
    data_dir: str = "data/images",
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train and validation dataloaders from a directory.

    Returns:
        train_loader, val_loader, class_names
    """
    # Full dataset (no transforms yet, we split first)
    full_dataset = InteriorStyleDataset(data_dir, transform=None)
    class_names = full_dataset.class_names

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Apply different transforms to each split
    train_dataset = _SubsetWithTransform(train_subset, get_train_transforms())
    val_dataset = _SubsetWithTransform(val_subset, get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset: {len(full_dataset)} images across {len(class_names)} classes")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Classes: {class_names}")

    return train_loader, val_loader, class_names


class _SubsetWithTransform(Dataset):
    """Wrapper that applies transforms to a Subset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # img is a PIL Image here (no transform was applied to the base dataset)
        if self.transform:
            img = self.transform(img)
        return img, label
