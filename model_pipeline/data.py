"""Shared CIFAR-100 transforms and deterministic train/validation split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def training_transform(image_size: int = 128):
    return transforms.Compose(
        [
            transforms.Resize(image_size + 16, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def evaluation_transform(image_size: int = 128):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def stratified_indices(targets, validation_size: int, seed: int):
    """Return stable train/validation indices with equal class representation."""
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.unique(targets)
    if validation_size <= 0 or validation_size >= len(targets):
        raise ValueError("validation_size must be between 1 and dataset_size - 1")
    if validation_size % len(classes):
        raise ValueError("validation_size must be divisible by the number of classes")

    rng = np.random.default_rng(seed)
    per_class = validation_size // len(classes)
    validation = []
    training = []
    for class_id in classes:
        class_indices = np.flatnonzero(targets == class_id)
        rng.shuffle(class_indices)
        validation.extend(class_indices[:per_class].tolist())
        training.extend(class_indices[per_class:].tolist())
    return sorted(training), sorted(validation)


def split_fingerprint(train_indices, validation_indices) -> str:
    payload = json.dumps(
        {"train": list(train_indices), "validation": list(validation_indices)},
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_split_manifest(path: Path, train_indices, validation_indices, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "CIFAR-100",
        "seed": seed,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "sha256": split_fingerprint(train_indices, validation_indices),
        "train_indices": list(train_indices),
        "validation_indices": list(validation_indices),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_datasets(root: Path, image_size: int, validation_size: int, seed: int, download: bool):
    untransformed = CIFAR100(root=root, train=True, download=download)
    train_indices, validation_indices = stratified_indices(
        untransformed.targets, validation_size, seed
    )
    train_full = CIFAR100(
        root=root, train=True, download=False, transform=training_transform(image_size)
    )
    validation_full = CIFAR100(
        root=root, train=True, download=False, transform=evaluation_transform(image_size)
    )
    test = CIFAR100(
        root=root, train=False, download=download, transform=evaluation_transform(image_size)
    )
    return (
        Subset(train_full, train_indices),
        Subset(validation_full, validation_indices),
        test,
        train_indices,
        validation_indices,
    )
