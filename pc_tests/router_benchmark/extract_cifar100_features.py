"""Extract handcrafted features for the CIFAR-100 train and test sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torchvision.datasets import CIFAR100

from features import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features.csv"))
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for split, train in (("train", True), ("test", False)):
        dataset = CIFAR100(root=args.data_dir, train=train, download=args.download)
        for index, (image, target) in enumerate(dataset):
            rows.append({
                "image_id": f"{split}:{index:05d}",
                "split": split,
                "true_label": int(target),
                **extract_features(image),
            })
            if (index + 1) % 5000 == 0:
                print(f"{split}: extracted {index + 1}/{len(dataset)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
