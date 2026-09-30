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
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.datasets import CIFAR100
from torchvision import transforms
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_trt import CIFAR100_MEAN, CIFAR100_STD, PowerMonitor, TRTInfer
from router_features import extract_feature_vector


MODELS = [f"fp32_p{prune}" for prune in range(0, 100, 10)]


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


def image_tensor(image, size):
    value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()[None] / 255
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

        def predict(image):
            values = extract_feature_vector(image, feature_set).reshape(1, -1)
            if columns is not None:
                values = values[:, columns]
            return float(model.predict(values)[0])
    elif artifact.suffix == ".pth":
        package = torch.load(artifact, map_location="cuda", weights_only=False)
        size = int(package["size"])
        model = TinyCNN().cuda().eval()
        model.load_state_dict(package["state_dict"])

        def predict(image):
            with torch.inference_mode():
                return float(model(image_tensor(image, size).cuda()).item())
    else:
        raise ValueError(f"Unsupported router artifact: {artifact}")
    return predict, policy


def choose_model(score, costs, policy):
    if policy["fallback"]:
        return "fp32_p0"
    order = np.argsort(costs, kind="stable")
    thresholds = (np.arange(len(costs) - 1) + 0.5) / (len(costs) - 1) - policy["bias"]
    return MODELS[int(order[np.digitize([score], thresholds)[0]])]


def benchmark(name, predict, policy, costs, images, labels, positions, runners,
              network_transform, repeats):
    rows = []
    for repeat in range(repeats):
        selections = Counter()
        correct = 0
        router_seconds = 0.0
        torch.cuda.synchronize()
        monitor = PowerMonitor(interval_ms=50)
        monitor.start()
        start = time.time()
        for position in positions:
            image = images[position]
            router_start = time.perf_counter()
            model_id = choose_model(predict(image), costs, policy)
            torch.cuda.synchronize()
            router_seconds += time.perf_counter() - router_start
            selections[model_id] += 1
            tensor = network_transform(Image.fromarray(image))[None]
            output = runners[model_id].infer(tensor)
            torch.cuda.synchronize()
            correct += int(output.argmax(1).item() == int(labels[position]))
        torch.cuda.synchronize()
        end = time.time()
        power = monitor.stop_window(start, end)
        elapsed = end - start
        row = {
            "system": name, "repeat": repeat, "images": len(positions),
            "accuracy_percent": 100 * correct / len(positions),
            "latency_e2e_ms_per_image": 1000 * elapsed / len(positions),
            "throughput_e2e_images_s": len(positions) / elapsed,
            "router_ms_per_image": 1000 * router_seconds / len(positions),
            "vdd_in_power_w": power[0], "compute_rail_power_w": power[1],
            "vdd_in_energy_mj_per_image": 1000 * power[2] / len(positions),
            "compute_rail_energy_mj_per_image": 1000 * power[3] / len(positions),
            "gpu_temp_mean_c": power[4], "gpu_temp_max_c": power[5],
        }
        row.update({f"selected_{model}": selections[model] for model in MODELS})
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    return rows


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    objective = manifest["objective"]
    costs_frame = pd.read_csv(Path(manifest["costs_csv"]))
    costs = costs_frame.set_index("model_id").loc[MODELS, "cost"].to_numpy(float)
    dataset = CIFAR100(root=args.data_dir, train=False, download=False)
    images = dataset.data
    labels = np.asarray(dataset.targets)
    positions = evaluation_positions(labels, args.split_seed)
    network_transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    runners = {}
    for model_id in MODELS:
        engine = args.trt_engine_dir / model_engine_name(model_id)
        if not engine.exists():
            raise FileNotFoundError(engine)
        runners[model_id] = TRTInfer(engine)
    # Warm every engine outside the measured window.
    warm = network_transform(Image.fromarray(images[positions[0]]))[None]
    for runner in runners.values():
        for _ in range(5):
            runner.infer(warm)
    torch.cuda.synchronize()

    all_rows = []
    baseline_predict = lambda image: 1.0
    baseline_policy = {"fallback": True, "bias": 0.0}
    all_rows.extend(benchmark("fp32_p0_baseline", baseline_predict, baseline_policy,
                              costs, images, labels, positions, runners,
                              network_transform, args.repeats))
    for entry in manifest["routers"]:
        predict, policy = load_router(entry, objective)
        all_rows.extend(benchmark(entry["name"], predict, policy, costs, images,
                                  labels, positions, runners, network_transform,
                                  args.repeats))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(args.output_dir / f"{objective}_runs.csv", index=False)
    metrics = [column for column in raw.columns if column not in {"system", "repeat"}]
    summary = raw.groupby("system", sort=False)[metrics].agg(["mean", "std"])
    summary.to_csv(args.output_dir / f"{objective}_summary.csv")


if __name__ == "__main__":
    main()
