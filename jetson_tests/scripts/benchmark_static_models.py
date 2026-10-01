#!/usr/bin/env python3
"""Benchmark fixed TensorRT model choices before constructing a router policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.modules.setdefault("onnxruntime", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dynamic_routing import image_batch_tensor, model_engine_name
from evaluate_trt import PowerMonitor, TRTInfer


DEFAULT_MODELS = ["fp32_p0", "fp32_p30", "fp32_p50", "fp32_p70"]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trt-engine-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "results/static_models")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--batch-sizes", default="1,8,16,32,64,128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def evaluation_positions(labels, seed):
    positions = np.arange(len(labels))
    _, rest = train_test_split(
        positions, train_size=0.6, random_state=seed, stratify=labels
    )
    _, evaluation = train_test_split(
        rest, train_size=0.5, random_state=seed + 1, stratify=labels[rest]
    )
    return evaluation


def benchmark(model_id, runner, images, labels, positions, batch_size, repeats):
    warm = image_batch_tensor(images[positions[:batch_size]], 128)
    for _ in range(5):
        runner.infer(warm)
    torch.cuda.synchronize()

    rows = []
    for repeat in range(repeats):
        correct = 0
        torch.cuda.synchronize()
        monitor = PowerMonitor(interval_ms=50)
        monitor.start()
        start = time.time()
        for offset in range(0, len(positions), batch_size):
            batch_positions = positions[offset:offset + batch_size]
            tensor = image_batch_tensor(images[batch_positions], 128)
            output = runner.infer(tensor)
            torch.cuda.synchronize()
            predicted = output.argmax(1).detach().cpu().numpy()
            correct += int((predicted == labels[batch_positions]).sum())
        torch.cuda.synchronize()
        end = time.time()
        power = monitor.stop_window(start, end)
        elapsed = end - start
        row = {
            "model_id": model_id,
            "batch_size": batch_size,
            "repeat": repeat,
            "images": len(positions),
            "accuracy_percent": 100 * correct / len(positions),
            "latency_e2e_ms_per_image": 1000 * elapsed / len(positions),
            "throughput_e2e_images_s": len(positions) / elapsed,
            "vdd_in_power_w": power[0],
            "compute_rail_power_w": power[1],
            "vdd_in_energy_mj_per_image": 1000 * power[2] / len(positions),
            "compute_rail_energy_mj_per_image": 1000 * power[3] / len(positions),
            "gpu_temp_mean_c": power[4],
            "gpu_temp_max_c": power[5],
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    return rows


def mark_pareto(summary):
    summary = summary.copy()
    summary["pareto_efficient"] = True
    for batch_size, group in summary.groupby("batch_size", sort=False):
        for index, row in group.iterrows():
            dominates = (
                (group["accuracy_percent"] >= row["accuracy_percent"])
                & (group["latency_e2e_ms_per_image"] <= row["latency_e2e_ms_per_image"])
                & (group["vdd_in_energy_mj_per_image"] <= row["vdd_in_energy_mj_per_image"])
                & (
                    (group["accuracy_percent"] > row["accuracy_percent"])
                    | (group["latency_e2e_ms_per_image"] < row["latency_e2e_ms_per_image"])
                    | (group["vdd_in_energy_mj_per_image"] < row["vdd_in_energy_mj_per_image"])
                )
            )
            if dominates.any():
                summary.loc[index, "pareto_efficient"] = False
    return summary


def save_plots(summary, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = [
        ("accuracy_percent", "Accuracy (%)", "Accuracy"),
        ("latency_e2e_ms_per_image", "Latency (ms/image)", "End-to-end latency"),
        ("vdd_in_energy_mj_per_image", "Energy (mJ/image)", "VDD_IN energy"),
    ]
    for axis, (metric, ylabel, title) in zip(axes, panels):
        for model_id, group in summary.groupby("model_id", sort=False):
            axis.plot(group["batch_size"], group[metric], marker="o", label=model_id)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(summary["batch_size"].unique()))
        axis.get_xaxis().set_major_formatter(ScalarFormatter())
        axis.set_xlabel("Batch size")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].legend()
    fig.suptitle("Static FP32 model baselines: same 2,000 CIFAR-100 images")
    fig.savefig(output_dir / "static_model_comparison.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    models = [value.strip() for value in args.models.split(",")]
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not models or len(models) != len(set(models)) or "fp32_p0" not in models:
        raise ValueError("--models must contain fp32_p0")
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")

    dataset = CIFAR100(root=args.data_dir, train=False, download=False)
    images = dataset.data
    labels = np.asarray(dataset.targets)
    positions = evaluation_positions(labels, args.split_seed)

    runners = {}
    for model_id in models:
        engine = args.trt_engine_dir / model_engine_name(model_id)
        if not engine.exists():
            raise FileNotFoundError(engine)
        runners[model_id] = TRTInfer(engine)

    rows = []
    for batch_size in batch_sizes:
        for model_id in models:
            rows.extend(benchmark(
                model_id, runners[model_id], images, labels, positions,
                batch_size, args.repeats,
            ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "static_model_runs.csv", index=False)
    metrics = [
        column for column in raw.columns
        if column not in {"model_id", "batch_size", "repeat"}
    ]
    summary = (
        raw.groupby(["batch_size", "model_id"], sort=False)[metrics]
        .mean().reset_index()
    )
    summary = mark_pareto(summary)
    summary.to_csv(args.output_dir / "static_model_summary.csv", index=False)
    save_plots(summary, args.output_dir)


if __name__ == "__main__":
    main()
