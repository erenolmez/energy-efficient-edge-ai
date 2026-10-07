"""Structurally prune trained CIFAR-100 classifiers and fine-tune them."""
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
import torch_pruning as tp
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from model_pipeline.data import build_datasets, split_fingerprint  # noqa: E402
from model_pipeline.models import ARCHITECTURES, normalize_architecture  # noqa: E402
from model_pipeline.naming import model_name  # noqa: E402
from model_pipeline.reproducibility import seed_everything, seed_worker, torch_generator  # noqa: E402
from pc_tests.model_pipeline.train_baselines import run_epoch, save_deployment_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pruned"))
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--pruning-levels", default="10,20,30,40,50,60,70,80,90")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--max-pruning-ratio", type=float, default=0.95)
    parser.add_argument("--global-pruning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isomorphic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument(
        "--structure-only", action="store_true",
        help="Validate pruning and forward passes without fine-tuning or saving models.",
    )
    return parser.parse_args()


def parse_levels(value: str) -> list[int]:
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("--pruning-levels must contain unique integer percentages")
    if any(level <= 0 or level >= 100 for level in levels):
        raise ValueError("pruning levels must be between 1 and 99")
    return levels


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


def classifier_layers(model: nn.Module) -> list[nn.Module]:
    candidates = [
        module for module in model.modules()
        if isinstance(module, nn.Linear) and module.out_features == 100
    ]
    if not candidates:
        raise RuntimeError("Could not identify the 100-class output layer")
    return candidates


def model_complexity(model, image_size, device):
    example = torch.randn(1, 3, image_size, image_size, device=device)
    macs, parameters = tp.utils.count_ops_and_params(model, example)
    return float(macs), int(parameters)


def structurally_prune(model, ratio, args, device):
    example = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    baseline_macs, baseline_parameters = model_complexity(model, args.image_size, device)
    pruner = tp.pruner.BasePruner(
        model=model,
        example_inputs=example,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=classifier_layers(model),
        round_to=args.round_to or None,
        global_pruning=args.global_pruning,
        isomorphic=args.isomorphic,
        max_pruning_ratio=args.max_pruning_ratio,
    )
    pruner.step()
    with torch.inference_mode():
        output = model(example)
    if output.shape != (1, 100):
        raise RuntimeError(f"Pruned model returned unexpected shape {tuple(output.shape)}")
    pruned_macs, pruned_parameters = model_complexity(model, args.image_size, device)
    return {
        "baseline_macs": baseline_macs,
        "pruned_macs": pruned_macs,
        "mac_reduction_percent": 100.0 * (1.0 - pruned_macs / baseline_macs),
        "baseline_parameters": baseline_parameters,
        "pruned_parameters": pruned_parameters,
        "parameter_reduction_percent": 100.0 * (
            1.0 - pruned_parameters / baseline_parameters
        ),
    }


def load_baseline(architecture: str, baseline_dir: Path, device):
    path = baseline_dir / f"{model_name(architecture, prune=0, precision='fp32')}.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Missing baseline model: {path}")
    return torch.load(path, map_location=device, weights_only=False), path


def fine_tune(architecture, level, args, loaders, split_hash):
    device = torch.device(args.device)
    stem = model_name(architecture, prune=level, precision="fp32")
    final_path = args.output_dir / f"{stem}.pth"
    resume_path = args.output_dir / f"{stem}__training.pt"
    if final_path.is_file() and not args.force:
        metadata_path = final_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise RuntimeError(f"Final model exists without metadata: {final_path}")
        print(f"Skipping completed model: {stem}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    elapsed_before_resume = 0.0
    history = []
    best_accuracy = -1.0
    best_state = None
    start_epoch = 1
    if args.resume and resume_path.is_file() and not args.force:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model = checkpoint["model"].to(device)
        complexity = checkpoint["complexity"]
        history = checkpoint["history"]
        best_accuracy = checkpoint["best_accuracy"]
        best_state = checkpoint["best_state"]
        start_epoch = checkpoint["epoch"] + 1
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        print(f"Resuming {stem} at epoch {start_epoch}")
    else:
        model, _ = load_baseline(architecture, args.baseline_dir, device)
        complexity = structurally_prune(model, level / 100.0, args, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    if args.resume and resume_path.is_file() and not args.force:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, optimizer, args.smoke_batches,
            scaler=scaler, use_amp=args.amp,
        )
        validation_metrics = run_epoch(
            model, loaders["validation"], criterion, device,
            max_batches=args.smoke_batches, use_amp=args.amp,
        )
        row = {
            "architecture": architecture,
            "prune": level,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
        model_for_resume = copy.deepcopy(model).cpu()
        torch.save(
            {
                "architecture": architecture,
                "prune": level,
                "epoch": epoch,
                "model": model_for_resume,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_accuracy": best_accuracy,
                "best_state": best_state,
                "complexity": complexity,
                "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
            },
            resume_path,
        )
        del model_for_resume

    model.load_state_dict(best_state)
    test_metrics = run_epoch(
        model, loaders["test"], criterion, device,
        max_batches=args.smoke_batches, use_amp=args.amp,
    )
    metadata = {
        "architecture": architecture,
        "dataset": "CIFAR-100",
        "num_classes": 100,
        "image_size": args.image_size,
        "requested_prune_percent": level,
        "precision": "fp32",
        "seed": args.seed,
        "split_sha256": split_hash,
        "best_validation_accuracy": best_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "fine_tuning_epochs": args.epochs,
        "fine_tuning_seconds": elapsed_before_resume + time.perf_counter() - started,
        "smoke_test": bool(args.smoke_batches),
        **complexity,
    }
    save_deployment_model(model, final_path, metadata)
    with (args.output_dir / f"{stem}_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return metadata


def main():
    args = parse_args()
    architectures = [
        normalize_architecture(value)
        for value in args.architectures.split(",") if value.strip()
    ]
    levels = parse_levels(args.pruning_levels)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.structure_only:
        results = []
        device = torch.device(args.device)
        for architecture in architectures:
            for level in levels:
                model, _ = load_baseline(architecture, args.baseline_dir, device)
                complexity = structurally_prune(model, level / 100.0, args, device)
                result = {"architecture": architecture, "prune": level, **complexity}
                print(json.dumps(result), flush=True)
                results.append(result)
                del model
        (args.output_dir / "structure_check.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        return

    train, validation, test, train_indices, validation_indices = build_datasets(
        args.data_dir, args.image_size, args.validation_size, args.seed, args.download
    )
    split_hash = split_fingerprint(train_indices, validation_indices)
    summaries = []
    for architecture in architectures:
        for level in levels:
            seed_everything(args.seed)
            loaders = {
                "train": make_loader(train, args.batch_size, args.num_workers, True, args.seed),
                "validation": make_loader(
                    validation, args.batch_size, args.num_workers, False, args.seed + 1
                ),
                "test": make_loader(test, args.batch_size, args.num_workers, False, args.seed + 2),
            }
            summaries.append(fine_tune(architecture, level, args, loaders, split_hash))
            (args.output_dir / "pruning_summary.json").write_text(
                json.dumps(summaries, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
