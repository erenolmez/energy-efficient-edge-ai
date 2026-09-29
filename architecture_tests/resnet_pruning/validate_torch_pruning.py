"""Structural Torch-Pruning validation for CIFAR-100 ResNet architectures.

This is deliberately not an accuracy experiment: randomly initialized models
are enough to verify that dependency-aware channel pruning produces a valid
network and real MAC/parameter reductions. Accuracy requires trained baseline
checkpoints followed by fine-tuning.
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch_pruning as tp


def parse_csv(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", default="resnet34,resnet50")
    parser.add_argument("--ratios", default="0.3,0.7,0.9")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    architectures = parse_csv(args.architectures, str)
    ratios = parse_csv(args.ratios, float)
    rows = []

    print(f"Torch: {torch.__version__}; Torch-Pruning: {tp.__version__}")
    print(f"Device: {device}; input: 3x{args.image_size}x{args.image_size}")

    for architecture in architectures:
        baseline = timm.create_model(
            architecture, pretrained=False, num_classes=args.num_classes
        ).to(device).eval()
        example = torch.randn(1, 3, args.image_size, args.image_size, device=device)
        base_macs, base_params = tp.utils.count_ops_and_params(baseline, example)

        for ratio in ratios:
            model = copy.deepcopy(baseline)
            classifier = model.get_classifier()
            ignored = [classifier] if isinstance(classifier, nn.Module) else []
            pruner = tp.pruner.BasePruner(
                model=model,
                example_inputs=example,
                importance=tp.importance.GroupMagnitudeImportance(p=2),
                pruning_ratio=ratio,
                ignored_layers=ignored,
                round_to=args.round_to,
                global_pruning=False,
                isomorphic=False,
                max_pruning_ratio=0.95,
            )

            status, error = "pass", ""
            try:
                pruner.step()
                with torch.inference_mode():
                    output = model(example)
                if tuple(output.shape) != (1, args.num_classes):
                    raise RuntimeError(f"unexpected output shape {tuple(output.shape)}")
                pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example)
            except Exception as exc:  # preserve failure details in the CSV
                status, error = "fail", f"{type(exc).__name__}: {exc}"
                pruned_macs = pruned_params = float("nan")

            row = {
                "architecture": architecture,
                "requested_pruning_ratio": ratio,
                "status": status,
                "base_macs": base_macs,
                "pruned_macs": pruned_macs,
                "mac_reduction_percent": 100 * (1 - pruned_macs / base_macs),
                "base_params": base_params,
                "pruned_params": pruned_params,
                "parameter_reduction_percent": 100 * (1 - pruned_params / base_params),
                "output_classes": args.num_classes,
                "error": error,
            }
            rows.append(row)
            print(
                f"{architecture:8s} ratio={ratio:.1f} {status.upper():4s} | "
                f"MAC reduction={row['mac_reduction_percent']:.2f}% | "
                f"parameter reduction={row['parameter_reduction_percent']:.2f}%"
            )
            del model

        del baseline

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
