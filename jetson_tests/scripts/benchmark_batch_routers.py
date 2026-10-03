#!/usr/bin/env python3
"""Measure one-decision-per-batch routers against the direct p0 baseline."""

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
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.datasets import CIFAR100

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_router_features import batch_feature_vector, thumbnail_grid
from benchmark_dynamic_routing import image_batch_tensor, model_engine_name
from evaluate_trt import PowerMonitor, TRTInfer


MODELS = ["fp32_p0", "fp32_p10", "fp32_p20", "fp32_p30"]


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="32,64,128")
    parser.add_argument("--budgets-pp", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore an existing runs CSV and start the sweep again.",
    )
    return parser.parse_args()


def evaluation_positions(labels, seed):
    positions = np.arange(len(labels))
    _, rest = train_test_split(positions, train_size=0.6, random_state=seed,
                               stratify=labels)
    _, evaluation = train_test_split(rest, train_size=0.5, random_state=seed + 1,
                                     stratify=labels[rest])
    return evaluation


def budget_key(metadata, budget):
    keys = metadata["accuracy_budgets_pp"]
    matched = min(keys, key=lambda value: abs(float(value) - budget))
    if abs(float(matched) - budget) > 1e-9:
        raise ValueError(f"Budget {budget} is not present in {metadata['name']}")
    return str(matched)


def load_router(entry):
    artifact = Path(entry["artifact"])
    metadata = json.loads(Path(entry["metadata"]).read_text())
    runtime = entry.get("runtime", "auto")
    if artifact.suffix == ".joblib":
        package = joblib.load(artifact)
        model = package["model"]
        feature_names = package["feature_names"]
        selected_columns = (
            [feature_names.index("gradient_mean")]
            if package["kind"] == "gradient_stump" else None
        )

        def predict(images):
            values = batch_feature_vector(images)[None]
            if selected_columns is not None:
                values = values[:, selected_columns]
            return float(model.predict(values)[0])
    elif artifact.suffix == ".pth":
        package = torch.load(artifact, map_location="cuda", weights_only=False)
        model = TinyCNN().cuda().eval()
        model.load_state_dict(package["state_dict"])

        def predict(images):
            with torch.inference_mode():
                value = model(thumbnail_grid(images).cuda(non_blocking=True))
                return float(value.detach().cpu().item())
    elif artifact.suffix == ".engine":
        runner = TRTInfer(artifact)

        def predict(images):
            value = runner.infer(thumbnail_grid(images))
            return float(value.detach().cpu().item())
    else:
        raise ValueError(f"Unsupported router artifact: {artifact}")
    return metadata, predict, runtime


def choose_model(score, costs, policy):
    if policy["fallback"]:
        return "fp32_p0"
    order = np.argsort(costs, kind="stable")
    thresholds = (np.arange(len(costs) - 1) + 0.5) / (len(costs) - 1) - policy["bias"]
    return MODELS[int(order[np.digitize([score], thresholds)[0]])]


def benchmark(system, predict, policy, costs, images, labels, positions, runners,
              batch_size, budget, repeat_indices):
    rows = []
    for repeat in repeat_indices:
        correct = 0
        selections, selected_batches = Counter(), Counter()
        router_seconds = 0.0
        torch.cuda.synchronize()
        monitor = PowerMonitor(interval_ms=50)
        monitor.start()
        started = time.time()
        for offset in range(0, len(positions), batch_size):
            batch_positions = positions[offset:offset + batch_size]
            image_batch = images[batch_positions]
            if system == "fp32_p0_baseline":
                model_id = "fp32_p0"
            else:
                torch.cuda.synchronize()
                router_started = time.perf_counter()
                score = predict(image_batch)
                model_id = choose_model(score, costs, policy)
                torch.cuda.synchronize()
                router_seconds += time.perf_counter() - router_started
            selections[model_id] += len(batch_positions)
            selected_batches[model_id] += 1
            output = runners[model_id].infer(image_batch_tensor(image_batch, 128))
            torch.cuda.synchronize()
            prediction = output.argmax(1).detach().cpu().numpy()
            correct += int((prediction == labels[batch_positions]).sum())
        torch.cuda.synchronize()
        ended = time.time()
        power = monitor.stop_window(started, ended)
        elapsed = ended - started
        batches = int(np.ceil(len(positions) / batch_size))
        row = {
            "system": system, "repeat": repeat, "batch_size": batch_size,
            "accuracy_budget_pp": budget, "images": len(positions),
            "accuracy_percent": 100 * correct / len(positions),
            "latency_e2e_ms_per_image": 1000 * elapsed / len(positions),
            "throughput_e2e_images_s": len(positions) / elapsed,
            "router_ms_per_image": 1000 * router_seconds / len(positions),
            "router_ms_per_batch": 1000 * router_seconds / batches,
            "vdd_in_power_w": power[0], "compute_rail_power_w": power[1],
            "vdd_in_energy_mj_per_image": 1000 * power[2] / len(positions),
            "compute_rail_energy_mj_per_image": 1000 * power[3] / len(positions),
            "gpu_temp_mean_c": power[4], "gpu_temp_max_c": power[5],
        }
        row.update({f"selected_{model}": selections[model] for model in MODELS})
        row.update({f"selected_batches_{model}": selected_batches[model]
                    for model in MODELS})
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    return rows


def save_reports(raw, objective, output_dir):
    excluded = {"system", "repeat", "batch_size", "accuracy_budget_pp"}
    numeric = [column for column in raw if column not in excluded]
    keys = ["accuracy_budget_pp", "batch_size", "system"]
    means = raw.groupby(keys, sort=False)[numeric].mean().reset_index()
    reports = []
    for _, group in means.groupby(["accuracy_budget_pp", "batch_size"], sort=False):
        baseline = group[group.system == "fp32_p0_baseline"].iloc[0]
        group = group.copy()
        group["accuracy_delta_vs_p0_pp"] = group.accuracy_percent - baseline.accuracy_percent
        group["latency_saving_vs_p0_percent"] = 100 * (
            baseline.latency_e2e_ms_per_image - group.latency_e2e_ms_per_image
        ) / baseline.latency_e2e_ms_per_image
        group["energy_saving_vs_p0_percent"] = 100 * (
            baseline.vdd_in_energy_mj_per_image - group.vdd_in_energy_mj_per_image
        ) / baseline.vdd_in_energy_mj_per_image
        reports.append(group)
    report = pd.concat(reports, ignore_index=True)
    report.to_csv(output_dir / f"{objective}_batch_router_comparison.csv", index=False)

    routed = report[report.system != "fp32_p0_baseline"].copy()
    metric = f"{objective}_saving_vs_p0_percent"
    valid = routed[routed.accuracy_delta_vs_p0_pp >= -routed.accuracy_budget_pp - 1e-12]
    best = (valid.sort_values(metric, ascending=False)
            .groupby(["accuracy_budget_pp", "batch_size"], as_index=False).first())
    best.to_csv(output_dir / f"{objective}_best_by_budget_batch.csv", index=False)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for budget, group in best.groupby("accuracy_budget_pp"):
        axis.plot(group.batch_size, group[metric], marker="o", label=f"{budget:g} pp")
    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(best.batch_size.unique()))
    axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.set_xlabel("Batch size")
    axis.set_ylabel(f"{objective.title()} saving vs p0 (%)")
    axis.set_title(f"Best measured {objective} route at each accuracy budget")
    axis.grid(alpha=0.25)
    axis.legend(title="Maximum accuracy drop")
    fig.savefig(output_dir / f"{objective}_accuracy_budget_tradeoff.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    objective = manifest["objective"]
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    budgets = [float(value) for value in args.budgets_pp.split(",")]
    costs = (pd.read_csv(manifest["costs_csv"]).set_index("model_id")
             .loc[MODELS, "cost"].to_numpy(float))
    dataset = CIFAR100(root=args.data_dir, train=False, download=False)
    images, labels = dataset.data, np.asarray(dataset.targets)
    positions = evaluation_positions(labels, args.split_seed)
    runners = {
        model: TRTInfer(args.trt_engine_dir / model_engine_name(model))
        for model in MODELS
    }
    routers = [(entry["name"], *load_router(entry)) for entry in manifest["routers"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / f"{objective}_batch_router_runs.csv"
    if runs_path.exists() and not args.no_resume:
        raw = pd.read_csv(runs_path)
        rows = raw.to_dict("records")
        print(f"Resuming from {len(rows)} completed rows in {runs_path}", flush=True)
    else:
        rows = []

    def row_key(row):
        return (
            str(row["system"]), int(row["repeat"]), int(row["batch_size"]),
            float(row["accuracy_budget_pp"]),
        )

    completed = {row_key(row) for row in rows}

    def save_rows(new_rows):
        if not new_rows:
            return
        rows.extend(new_rows)
        completed.update(row_key(row) for row in new_rows)
        pd.DataFrame(rows).to_csv(runs_path, index=False)

    for batch_size in batch_sizes:
        warm = image_batch_tensor(images[positions[:batch_size]], 128)
        for runner in runners.values():
            for _ in range(5):
                runner.infer(warm)
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            source = next((
                row for row in rows
                if str(row["system"]) == "fp32_p0_baseline"
                and int(row["repeat"]) == repeat
                and int(row["batch_size"]) == batch_size
            ), None)
            if source is None:
                source = benchmark(
                    "fp32_p0_baseline", None, {"fallback": True}, costs,
                    images, labels, positions, runners, batch_size, budgets[0], [repeat],
                )[0]
            save_rows([
                {**source, "accuracy_budget_pp": budget}
                for budget in budgets
                if ("fp32_p0_baseline", repeat, batch_size, budget) not in completed
            ])
        for budget in budgets:
            for name, metadata, predict, runtime in routers:
                pending = [
                    repeat for repeat in range(args.repeats)
                    if (name, repeat, batch_size, budget) not in completed
                ]
                if not pending:
                    continue
                key = budget_key(metadata, budget)
                policy = metadata["policies"][str(batch_size)][objective][key]
                save_rows(benchmark(
                    name, predict, policy, costs, images, labels, positions,
                    runners, batch_size, budget, pending,
                ))
    raw = pd.DataFrame(rows)
    raw.to_csv(runs_path, index=False)
    save_reports(raw, objective, args.output_dir)


if __name__ == "__main__":
    main()
