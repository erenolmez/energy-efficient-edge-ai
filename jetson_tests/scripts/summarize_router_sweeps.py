#!/usr/bin/env python3
"""Select baseline-valid energy and latency routes from measured Jetson sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", type=Path, required=True,
                        help="Energy baseline-comparison CSV.")
    parser.add_argument("--latency", type=Path, required=True,
                        help="Latency baseline-comparison CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-accuracy-drop-pp", type=float, default=0.5)
    return parser.parse_args()


def select_results(path, objective, maximum_drop):
    frame = pd.read_csv(path)
    required = {
        "difficulty_percentile", "batch_size", "system",
        "accuracy_delta_vs_p0_pp", "latency_saving_vs_p0_percent",
        "energy_saving_vs_p0_percent",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    routed = frame.loc[frame["system"] != "fp32_p0_baseline"].copy()
    routed["objective"] = objective
    routed["meets_accuracy_budget"] = (
        routed["accuracy_delta_vs_p0_pp"] >= -maximum_drop - 1e-12
    )
    metric = f"{objective}_saving_vs_p0_percent"
    eligible = routed.loc[routed["meets_accuracy_budget"]].copy()
    if eligible.empty:
        return routed, eligible, eligible
    by_percentile = (
        eligible.sort_values(metric, ascending=False)
        .groupby(["batch_size", "difficulty_percentile"], as_index=False, sort=True)
        .first()
    )
    best = (
        eligible.sort_values(metric, ascending=False)
        .groupby("batch_size", as_index=False, sort=True)
        .first()
    )
    best["positive_measured_saving"] = best[metric] > 0
    return routed, by_percentile, best


def save_plot(energy_best, latency_best, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, frame, objective, color in (
        (axes[0], energy_best, "energy", "#2a9d8f"),
        (axes[1], latency_best, "latency", "#457b9d"),
    ):
        metric = f"{objective}_saving_vs_p0_percent"
        axis.axhline(0, color="black", linewidth=0.9)
        axis.bar(frame["batch_size"].astype(str), frame[metric], color=color)
        for index, row in enumerate(frame.itertuples()):
            value = getattr(row, metric)
            axis.text(index, value, f"P{row.difficulty_percentile:g}\n{value:.1f}%",
                      ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        axis.set_xlabel("Batch size")
        axis.set_ylabel(f"{objective.title()} saving vs p0 (%)")
        axis.set_title(f"Best valid {objective} route")
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.12)
    fig.suptitle("Measured router savings within 0.5 pp of the p0 accuracy baseline")
    fig.savefig(output_dir / "best_router_savings_vs_p0.png", dpi=180)
    plt.close(fig)


def save_selection_plot(best_frames, output_dir):
    combined = pd.concat(best_frames, ignore_index=True)
    model_columns = [
        column for column in combined.columns
        if column.startswith("selected_fp32_") and combined[column].sum() > 0
    ]
    labels = [
        f"{row.objective} B{int(row.batch_size)} P{row.difficulty_percentile:g}"
        for row in combined.itertuples()
    ]
    x = np.arange(len(combined))
    bottom = np.zeros(len(combined))
    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(model_columns)))
    for column, color in zip(model_columns, colors):
        values = combined[column].to_numpy(float)
        axis.bar(x, values, bottom=bottom, label=column.removeprefix("selected_"),
                 color=color)
        bottom += values
    axis.set_ylabel("Selected images (out of 2,000)")
    axis.set_title("Model counts for the best accuracy-valid route at each batch size")
    axis.set_xticks(x, labels, rotation=50, ha="right")
    axis.legend(ncol=4, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "best_router_model_counts.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for objective, path in (("energy", args.energy), ("latency", args.latency)):
        routed, by_percentile, best = select_results(
            path, objective, args.max_accuracy_drop_pp
        )
        routed.to_csv(args.output_dir / f"{objective}_all_routes.csv", index=False)
        by_percentile.to_csv(
            args.output_dir / f"{objective}_best_by_batch_percentile.csv", index=False
        )
        best.to_csv(args.output_dir / f"{objective}_best_by_batch.csv", index=False)
        outputs[objective] = best
    save_plot(outputs["energy"], outputs["latency"], args.output_dir)
    save_selection_plot([outputs["energy"], outputs["latency"]], args.output_dir)


if __name__ == "__main__":
    main()
