#!/usr/bin/env python3
"""Measure complete input-router + selected TensorRT model systems on Jetson."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.modules.setdefault("onnxruntime", None)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.datasets import CIFAR100
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_trt import CIFAR100_MEAN, CIFAR100_STD, PowerMonitor, TRTInfer
from router_features import extract_feature_vector


MODELS = [f"fp32_p{prune}" for prune in range(0, 100, 10)]


def save_reports(raw, objective, output_dir):
    """Write a baseline-relative table and comparison plots."""
    excluded = {"system", "repeat", "batch_size", "difficulty_percentile"}
    numeric = [column for column in raw.columns if column not in excluded]
    keys = ["difficulty_percentile", "batch_size", "system"]
    means = raw.groupby(keys, sort=False)[numeric].mean().reset_index()
    reports = []
    for (_, _), group in means.groupby(
        ["difficulty_percentile", "batch_size"], sort=False
    ):
        baseline = group.loc[group["system"] == "fp32_p0_baseline"].iloc[0]
        group = group.copy()
        group["accuracy_delta_vs_p0_pp"] = (
            group["accuracy_percent"] - baseline["accuracy_percent"]
        )
        group["latency_saving_vs_p0_percent"] = 100 * (
            baseline["latency_e2e_ms_per_image"] - group["latency_e2e_ms_per_image"]
        ) / baseline["latency_e2e_ms_per_image"]
        group["energy_saving_vs_p0_percent"] = 100 * (
            baseline["vdd_in_energy_mj_per_image"] - group["vdd_in_energy_mj_per_image"]
        ) / baseline["vdd_in_energy_mj_per_image"]
        reports.append(group)
    report = pd.concat(reports, ignore_index=True)
    report.to_csv(output_dir / f"{objective}_baseline_comparison.csv", index=False)

    for percentile, percentile_means in means.groupby(
        "difficulty_percentile", sort=False
    ):
        labels = list(dict.fromkeys(percentile_means["system"]))
        batch_sizes = list(dict.fromkeys(percentile_means["batch_size"]))
        x = np.arange(len(batch_sizes))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        panels = [
            ("accuracy_percent", "Accuracy (%)", "Accuracy"),
            ("latency_e2e_ms_per_image", "Latency (ms/image)", "End-to-end latency"),
            ("vdd_in_energy_mj_per_image", "Energy (mJ/image)", "VDD_IN energy"),
        ]
        for axis, (metric, ylabel, title) in zip(axes, panels):
            for label in labels:
                values = (
                    percentile_means.loc[percentile_means["system"] == label]
                    .set_index("batch_size").loc[batch_sizes]
                )
                axis.plot(x, values[metric], marker="o", label=label)
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.set_xticks(x, batch_sizes)
            axis.set_xlabel("Batch size")
            axis.grid(axis="y", alpha=0.25)
        axes[-1].legend(fontsize=7, ncol=2)
        fig.suptitle(
            f"P{percentile:g} batch routing ({objective}): same 2,000 CIFAR-100 images"
        )
        stem = f"{objective}_p{percentile:g}"
        fig.savefig(output_dir / f"{stem}_system_comparison.png", dpi=180)
        plt.close(fig)

        selected = [f"selected_{model}" for model in MODELS]
        routing = percentile_means.loc[
            percentile_means["system"] != "fp32_p0_baseline"
        ].copy()
        selection_labels = [
            f"B{int(row.batch_size)} {row.system}" for row in routing.itertuples()
        ]
        selection_x = np.arange(len(routing))
        fig, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
        bottom = np.zeros(len(routing))
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(MODELS)))
        for model, column, color in zip(MODELS, selected, colors):
            values = routing[column].to_numpy()
            axis.bar(selection_x, values, bottom=bottom, label=model, color=color)
            bottom += values
        axis.set_ylabel("Images selected (out of 2,000)")
        axis.set_title(f"P{percentile:g} batch selections: {objective} objective")
        axis.set_xticks(selection_x, selection_labels, rotation=55, ha="right")
        axis.legend(ncol=5, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        fig.savefig(output_dir / f"{stem}_model_selection_counts.png", dpi=180)
        plt.close(fig)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 24, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24),
            nn.Conv2d(24, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32),
            nn.Conv2d(32, 48, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 1),
        )

    def forward(self, value):
        return self.net(value).squeeze(1)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trt-engine-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results/dynamic_routing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-sizes", default="1",
                        help="Comma-separated TensorRT batch sizes.")
    parser.add_argument("--difficulty-percentiles", default="90",
                        help="Comma-separated batch difficulty percentiles.")
    parser.add_argument("--candidate-models", default=",".join(MODELS),
                        help="Comma-separated model IDs available to the policy.")
    return parser.parse_args()


def evaluation_positions(labels, seed):
    positions = np.arange(len(labels))
    _, rest = train_test_split(positions, train_size=0.6, random_state=seed,
                               stratify=labels)
    _, evaluation = train_test_split(rest, train_size=0.5, random_state=seed + 1,
                                     stratify=labels[rest])
    return evaluation


def model_engine_name(model_id):
    prune = int(model_id.removeprefix("fp32_p"))
    stem = "distilled_resnet18" if prune == 0 else f"structured_{prune}"
    return f"{stem}_fp32.engine"


def image_batch_tensor(image_batch, size):
    value = torch.from_numpy(np.asarray(image_batch).copy()).permute(0, 3, 1, 2).float() / 255
    if size != 32:
        value = F.interpolate(value, size=(size, size), mode="bilinear",
                              align_corners=False, antialias=True)
    mean = torch.tensor(CIFAR100_MEAN)[None, :, None, None]
    std = torch.tensor(CIFAR100_STD)[None, :, None, None]
    return (value - mean) / std


def load_router(entry, objective):
    artifact = Path(entry["artifact"])
    metadata = json.loads(Path(entry["metadata"]).read_text(encoding="utf-8"))
    policy = metadata["policies"][objective]
    if artifact.suffix == ".joblib":
        package = joblib.load(artifact)
        model = package["model"]
        feature_set = package["feature_set"]
        columns = package.get("selected_columns")

        def predict_batch(images):
            values = np.stack([
                extract_feature_vector(image, feature_set) for image in images
            ])
            if columns is not None:
                values = values[:, columns]
            return np.asarray(model.predict(values), dtype=float)
    elif artifact.suffix == ".pth":
        package = torch.load(artifact, map_location="cuda", weights_only=False)
        size = int(package["size"])
        model = TinyCNN().cuda().eval()
        model.load_state_dict(package["state_dict"])

        def predict_batch(images):
            with torch.inference_mode():
                values = image_batch_tensor(images, size).cuda(non_blocking=True)
                return model(values).detach().cpu().numpy().astype(float)
    else:
        raise ValueError(f"Unsupported router artifact: {artifact}")
    return predict_batch, policy


def choose_model(score, candidate_models, costs, policy):
    if policy["fallback"]:
        return "fp32_p0"
    order = np.argsort(costs, kind="stable")
    thresholds = (np.arange(len(costs) - 1) + 0.5) / (len(costs) - 1) - policy["bias"]
    return candidate_models[int(order[np.digitize([score], thresholds)[0]])]


def benchmark(name, predict_batch, policy, candidate_models, costs, images, labels, positions, runners,
              batch_size, difficulty_percentile, repeats):
    rows = []
    for repeat in range(repeats):
        selections = Counter()
        selected_batches = Counter()
        correct = 0
        router_seconds = 0.0
        torch.cuda.synchronize()
        monitor = PowerMonitor(interval_ms=50)
        monitor.start()
        start = time.time()
        for offset in range(0, len(positions), batch_size):
            batch_positions = positions[offset:offset + batch_size]
            image_batch = images[batch_positions]
            if name == "fp32_p0_baseline":
                model_id = "fp32_p0"
            else:
                router_start = time.perf_counter()
                scores = predict_batch(image_batch)
                difficulty = float(np.percentile(scores, difficulty_percentile))
                model_id = choose_model(difficulty, candidate_models, costs, policy)
                torch.cuda.synchronize()
                router_seconds += time.perf_counter() - router_start
            selections[model_id] += len(batch_positions)
            selected_batches[model_id] += 1
            tensor = image_batch_tensor(image_batch, 128)
            output = runners[model_id].infer(tensor)
            torch.cuda.synchronize()
            predicted = output.argmax(1).detach().cpu().numpy()
            correct += int((predicted == labels[batch_positions]).sum())
        torch.cuda.synchronize()
        end = time.time()
        power = monitor.stop_window(start, end)
        elapsed = end - start
        row = {
            "system": name, "repeat": repeat, "images": len(positions),
            "batch_size": batch_size,
            "difficulty_percentile": difficulty_percentile,
            "accuracy_percent": 100 * correct / len(positions),
            "latency_e2e_ms_per_image": 1000 * elapsed / len(positions),
            "throughput_e2e_images_s": len(positions) / elapsed,
            "router_ms_per_image": 1000 * router_seconds / len(positions),
            "router_ms_per_batch": 1000 * router_seconds / int(np.ceil(len(positions) / batch_size)),
            "vdd_in_power_w": power[0], "compute_rail_power_w": power[1],
            "vdd_in_energy_mj_per_image": 1000 * power[2] / len(positions),
            "compute_rail_energy_mj_per_image": 1000 * power[3] / len(positions),
            "gpu_temp_mean_c": power[4], "gpu_temp_max_c": power[5],
        }
        row.update({f"selected_{model}": selections[model] for model in MODELS})
        row.update({f"selected_batches_{model}": selected_batches[model] for model in MODELS})
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    return rows


def main():
    args = parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    difficulty_percentiles = [
        float(value) for value in args.difficulty_percentiles.split(",")
    ]
    candidate_models = [value.strip() for value in args.candidate_models.split(",")]
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if not difficulty_percentiles or any(
        not 0 <= value <= 100 for value in difficulty_percentiles
    ):
        raise ValueError("--difficulty-percentiles must be between 0 and 100")
    if (
        not candidate_models
        or len(candidate_models) != len(set(candidate_models))
        or any(value not in MODELS for value in candidate_models)
    ):
        raise ValueError(f"--candidate-models must be selected from {MODELS}")
    if "fp32_p0" not in candidate_models:
        raise ValueError("--candidate-models must include fp32_p0")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    objective = manifest["objective"]
    costs_frame = pd.read_csv(Path(manifest["costs_csv"]))
    costs = (
        costs_frame.set_index("model_id").loc[candidate_models, "cost"].to_numpy(float)
    )
    dataset = CIFAR100(root=args.data_dir, train=False, download=False)
    images = dataset.data
    labels = np.asarray(dataset.targets)
    positions = evaluation_positions(labels, args.split_seed)
    runners = {}
    for model_id in candidate_models:
        engine = args.trt_engine_dir / model_engine_name(model_id)
        if not engine.exists():
            raise FileNotFoundError(engine)
        runners[model_id] = TRTInfer(engine)
    all_rows = []
    baseline_predict = lambda image_batch: np.ones(len(image_batch), dtype=float)
    baseline_policy = {"fallback": True, "bias": 0.0}
    loaded_routers = [(entry["name"], *load_router(entry, objective))
                      for entry in manifest["routers"]]
    for batch_size in batch_sizes:
        # Warm every engine at the measured shape outside the power window.
        warm_images = images[positions[:batch_size]]
        warm = image_batch_tensor(warm_images, 128)
        for runner in runners.values():
            for _ in range(5):
                runner.infer(warm)
        torch.cuda.synchronize()
        # The static p0 route is independent of the difficulty percentile. Measure
        # it once per batch size, then reuse the same observations for every
        # percentile so comparisons share an identical, non-duplicated baseline.
        baseline_rows = benchmark(
            "fp32_p0_baseline", baseline_predict, baseline_policy,
            candidate_models, costs, images, labels, positions, runners,
            batch_size, difficulty_percentiles[0], args.repeats,
        )
        for percentile in difficulty_percentiles:
            all_rows.extend([
                {**row, "difficulty_percentile": percentile}
                for row in baseline_rows
            ])
            for name, predict_batch, policy in loaded_routers:
                all_rows.extend(benchmark(
                    name, predict_batch, policy, candidate_models, costs, images,
                    labels, positions, runners, batch_size, percentile, args.repeats,
                ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(args.output_dir / f"{objective}_runs.csv", index=False)
    metrics = [column for column in raw.columns
               if column not in {"system", "repeat", "batch_size", "difficulty_percentile"}]
    summary = raw.groupby(
        ["difficulty_percentile", "batch_size", "system"], sort=False
    )[metrics].agg(["mean", "std"])
    summary.to_csv(args.output_dir / f"{objective}_summary.csv")
    save_reports(raw, objective, args.output_dir)


if __name__ == "__main__":
    main()
