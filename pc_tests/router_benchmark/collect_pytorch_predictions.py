"""Collect deterministic per-image predictions from whole-model checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode
import timm  # noqa: F401 - registers classes used by whole-model checkpoints

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class IndexedCIFAR100(CIFAR100):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image, target, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True,
                        help="CSV with model_id,checkpoint columns")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Base directory for relative checkpoint paths in the model CSV",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", nargs="+", choices=["train", "test"],
                        default=["train", "test"])
    parser.add_argument("--quantized-backend", default="onednn")
    return parser.parse_args()


def load_models(manifest: pd.DataFrame, device: torch.device, checkpoint_root: Path | None):
    missing = {"model_id", "checkpoint"}.difference(manifest.columns)
    if missing:
        raise ValueError(f"Model manifest is missing columns: {sorted(missing)}")
    if manifest["model_id"].duplicated().any():
        raise ValueError("Each model_id must appear exactly once")
    models = []
    for row in manifest.itertuples(index=False):
        path = Path(row.checkpoint).expanduser()
        if not path.is_absolute():
            if checkpoint_root is None:
                raise ValueError(
                    "Relative checkpoint paths require --checkpoint-root"
                )
            path = checkpoint_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"Loading {row.model_id}: {path}")
        model = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(model, nn.Module):
            raise TypeError(f"Checkpoint is not a torch.nn.Module: {path}")
        models.append((str(row.model_id), model.float().eval().to(device)))
    return models


@torch.inference_mode()
def collect_split(split, dataset, models, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    rows = []
    for model_id, model in models:
        correct = seen = 0
        for images, targets, indices in loader:
            predicted = model(images.to(device, non_blocking=True)).argmax(1).cpu()
            correct += int((predicted == targets).sum())
            seen += len(targets)
            rows.extend({
                "image_id": f"{split}:{int(index):05d}",
                "model_id": model_id,
                "true_label": int(target),
                "predicted_label": int(prediction),
            } for index, target, prediction in zip(indices, targets, predicted, strict=True))
        print(f"{model_id} | {split} accuracy: {100.0 * correct / seen:.2f}%")
    return rows


def main() -> None:
    args = parse_args()
    if args.quantized_backend in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = args.quantized_backend
        print(f"Quantized backend: {torch.backends.quantized.engine}")
    device = torch.device(args.device)
    models = load_models(pd.read_csv(args.models), device, args.checkpoint_root)
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.image_size), transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    rows = []
    available_splits = {"train": True, "test": False}
    for split in args.splits:
        train = available_splits[split]
        dataset = IndexedCIFAR100(root=args.data_dir, train=train, download=False,
                                  transform=transform)
        rows.extend(collect_split(split, dataset, models, device,
                                  args.batch_size, args.num_workers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} prediction rows to {args.output}")


if __name__ == "__main__":
    main()
