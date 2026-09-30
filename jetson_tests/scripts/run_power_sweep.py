#!/usr/bin/env python3
"""Run the CIFAR-100 TensorRT evaluation under explicit Jetson power modes.

The configuration maps board-supported nvpmodel IDs to human-readable labels.
Jetson nvpmodel controls a module profile, not an exact GPU-only watt cap. The
actual VDD_IN power measured by tegrastats is therefore always recorded and is
the value used to calculate energy per image.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=root)
    parser.add_argument(
        "--profile-config",
        type=Path,
        default=root / "manifests" / "power_profiles.jetson.json",
    )
    parser.add_argument("--precisions", default="fp32,fp16,int8")
    parser.add_argument("--n-subsets", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--settle-seconds", type=float, default=15.0)
    parser.add_argument("--no-lock-clocks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(
    command: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
        input=input_text,
    )


def current_nvpmodel_id() -> int | None:
    result = run(["sudo", "nvpmodel", "-q"], check=False, capture=True)
    numbers = re.findall(r"(?m)^\s*(\d+)\s*$", result.stdout)
    return int(numbers[-1]) if numbers else None


def load_profiles(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Power profile configuration not found: {path}\n"
            "Copy power_profiles.example.json, then replace its mode IDs with "
            "IDs reported by `sudo nvpmodel -q --verbose` on this Jetson."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", [])
    if not profiles:
        raise ValueError(f"No profiles defined in {path}")
    required = {"nvpmodel_id", "label"}
    for profile in profiles:
        missing = required.difference(profile)
        if missing:
            raise ValueError(f"Profile is missing {sorted(missing)}: {profile}")
    return profiles


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def set_gpu_frequency(freq_hz: int) -> None:
    base = Path("/sys/devices/platform/17000000.gpu/devfreq_dev")
    available = [int(value) for value in (base / "available_frequencies").read_text().split()]
    if freq_hz not in available:
        raise ValueError(f"Unsupported GPU frequency {freq_hz}; available values: {available}")

    # Lower min first so reducing max cannot conflict with the current bound.
    run(["sudo", "tee", str(base / "min_freq")], input_text=f"{min(available)}\n")
    run(["sudo", "tee", str(base / "max_freq")], input_text=f"{freq_hz}\n")
    run(["sudo", "tee", str(base / "min_freq")], input_text=f"{freq_hz}\n")


def annotate_results(csv_path: Path, profile: dict, mode_query: str):
    import pandas as pd

    frame = pd.read_csv(csv_path)
    frame.insert(0, "power_profile", profile["label"])
    frame.insert(1, "nvpmodel_id", int(profile["nvpmodel_id"]))
    frame.insert(2, "requested_watts", profile.get("requested_watts"))
    frame.insert(3, "gpu_freq_hz", profile.get("gpu_freq_hz"))
    frame.insert(4, "nvpmodel_query", " ".join(mode_query.split()))
    frame["energy_e2e_mj_per_image"] = frame["power_tegra"] * frame["latency_e2e"]
    frame["energy_pure_mj_per_image"] = frame["power_tegra"] * frame["latency_trt"]
    if "power_cpu_gpu_cv" in frame:
        frame["energy_cpu_gpu_cv_mj_per_image"] = frame["power_cpu_gpu_cv"] * frame["latency_e2e"]
    frame.to_csv(csv_path, index=False)
    return frame


def save_combined_outputs(frame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "power_sweep_all_runs.csv", index=False)
    group_columns = ["power_profile", "nvpmodel_id", "requested_watts", "gpu_freq_hz", "precision", "prune", "model_name"]
    summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            accuracy_mean=("acc", "mean"),
            latency_e2e_ms_mean=("latency_e2e", "mean"),
            throughput_e2e_mean=("throughput_e2e", "mean"),
            power_w_mean=("power_tegra", "mean"),
            compute_rail_power_w_mean=("power_cpu_gpu_cv", "mean"),
            energy_e2e_mj_per_image_mean=("energy_e2e_mj_per_image", "mean"),
            compute_rail_energy_mj_per_image_mean=("energy_cpu_gpu_cv_mj_per_image", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "power_sweep_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    for metric, ylabel, filename in (
        ("accuracy_mean", "Top-1 accuracy (%)", "accuracy_by_power.png"),
        ("latency_e2e_ms_mean", "End-to-end latency (ms/image)", "latency_by_power.png"),
        ("throughput_e2e_mean", "End-to-end throughput (images/s)", "throughput_by_power.png"),
        ("power_w_mean", "Measured VDD_IN power (W)", "measured_power.png"),
        ("compute_rail_power_w_mean", "Measured CPU/GPU/CV rail power (W)", "compute_rail_power.png"),
        ("energy_e2e_mj_per_image_mean", "End-to-end energy (mJ/image)", "energy_by_power.png"),
        ("compute_rail_energy_mj_per_image_mean", "CPU/GPU/CV rail energy (mJ/image)", "compute_rail_energy.png"),
    ):
        plot_data = summary[summary["precision"] == "fp32"]
        plt.figure(figsize=(11, 6))
        sns.lineplot(
            data=plot_data,
            x="prune",
            y=metric,
            hue="power_profile",
            marker="o",
        )
        plt.xlabel("Structured pruning ratio (%)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    profiles = load_profiles(args.profile_config.resolve())
    evaluator = base_dir / "scripts" / "evaluate_trt.py"
    combined = []
    original_mode = current_nvpmodel_id() if not args.dry_run else None

    try:
        for profile in profiles:
            mode_id = int(profile["nvpmodel_id"])
            label = str(profile["label"])
            profile_dir = base_dir / "results" / "power_sweep" / f"mode_{mode_id}_{slug(label)}"

            if not args.dry_run:
                run(["sudo", "nvpmodel", "-m", str(mode_id)])
                if not args.no_lock_clocks:
                    run(["sudo", "jetson_clocks"])
                if profile.get("gpu_freq_hz") is not None:
                    set_gpu_frequency(int(profile["gpu_freq_hz"]))
                time.sleep(args.settle_seconds)
                query = run(["sudo", "nvpmodel", "-q"], capture=True).stdout
            else:
                query = f"DRY RUN mode {mode_id}"

            command = [
                sys.executable,
                str(evaluator),
                "--base-dir", str(base_dir),
                "--fp32-dir", "models/fp32",
                "--trt-dir", "outputs/tensorrt",
                "--results-dir", str(profile_dir),
                "--data-dir", "data",
                "--precisions", args.precisions,
                "--n-subsets", str(args.n_subsets),
                "--subset-size", str(args.subset_size),
                "--batch-size", str(args.batch_size),
                "--no-download",
            ]
            if args.dry_run:
                print("+", " ".join(command))
                continue

            run(command)
            raw_csv = profile_dir / "tensorrt_results_all_runs.csv"
            combined.append(annotate_results(raw_csv, profile, query))

        if combined:
            import pandas as pd

            save_combined_outputs(pd.concat(combined, ignore_index=True), base_dir / "results" / "power_sweep")
    finally:
        if original_mode is not None and not args.dry_run:
            print(f"Restoring original nvpmodel mode {original_mode}.")
            run(["sudo", "nvpmodel", "-m", str(original_mode)], check=False)


if __name__ == "__main__":
    main()
