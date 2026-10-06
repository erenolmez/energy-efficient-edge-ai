"""Train reproducible, unpruned CIFAR-100 baselines for the architecture sweep."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from model_pipeline.data import build_datasets, write_split_manifest  # noqa: E402
from model_pipeline.models import ARCHITECTURES, create_classifier, normalize_architecture  # noqa: E402
from model_pipeline.naming import model_name  # noqa: E402
from model_pipeline.reproducibility import seed_everything, seed_worker, torch_generator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-batches", type=int, default=0,
                        help="Limit each epoch/evaluation to N batches for pipeline checks.")
    return parser.parse_args()


def make_loader(dataset, batch_size, workers, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=torch_generator(seed),
    )


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch = labels.size(0)
            total_loss += float(loss.detach()) * batch
            correct += int((logits.argmax(1) == labels).sum())
            total += batch
    return {"loss": total_loss / total, "accuracy": 100.0 * correct / total, "images": total}


def save_deployment_model(model, path: Path, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    original_device = next(model.parameters()).device
    model.cpu().eval()
    torch.save(model, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model.to(original_device)


def train_architecture(architecture, args, loaders, split_hash):
    device = torch.device(args.device)
    model = create_classifier(architecture, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_accuracy = -1.0
    best_state = None
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, optimizer, args.smoke_batches
        )
        validation_metrics = run_epoch(
            model, loaders["validation"], criterion, device, max_batches=args.smoke_batches
        )
        row = {
            "architecture": architecture,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row))
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()

    model.load_state_dict(best_state)
    test_metrics = run_epoch(
        model, loaders["test"], criterion, device, max_batches=args.smoke_batches
    )
    stem = model_name(architecture, prune=0, precision="fp32")
    metadata = {
        "architecture": architecture,
        "dataset": "CIFAR-100",
        "num_classes": 100,
        "image_size": args.image_size,
        "prune": 0,
        "precision": "fp32",
        "seed": args.seed,
        "split_sha256": split_hash,
        "pretrained_initialization": args.pretrained,
        "best_validation_accuracy": best_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "training_seconds": time.perf_counter() - started,
        "smoke_test": bool(args.smoke_batches),
    }
    save_deployment_model(model, args.output_dir / f"{stem}.pth", metadata)
    with (args.output_dir / f"{stem}_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return metadata


def main():
    args = parse_args()
    architectures = [normalize_architecture(x) for x in args.architectures.split(",") if x.strip()]
    if len(set(architectures)) != len(architectures):
        raise ValueError("--architectures contains duplicates")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test, train_indices, validation_indices = build_datasets(
        args.data_dir, args.image_size, args.validation_size, args.seed, args.download
    )
    split_path = args.output_dir / f"cifar100_split_seed{args.seed}.json"
    write_split_manifest(split_path, train_indices, validation_indices, args.seed)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    summaries = []
    for architecture in architectures:
        seed_everything(args.seed)
        # Recreate the loaders so every architecture sees the same shuffle order.
        loaders = {
            "train": make_loader(train, args.batch_size, args.num_workers, True, args.seed),
            "validation": make_loader(
                validation, args.batch_size, args.num_workers, False, args.seed + 1
            ),
            "test": make_loader(test, args.batch_size, args.num_workers, False, args.seed + 2),
        }
        summaries.append(
            train_architecture(architecture, args, loaders, split_payload["sha256"])
        )
    (args.output_dir / "baseline_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
