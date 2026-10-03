#!/usr/bin/env python3
"""Combine separate energy and latency batch-router measurements."""

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
    parser.add_argument("--energy", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--static-energy", type=Path, required=True)
    parser.add_argument("--static-latency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def best_routes(path, objective):
    frame = pd.read_csv(path)
    routed = frame[frame.system != "fp32_p0_baseline"].copy()
    routed["objective"] = objective
    routed["within_budget"] = (
        routed.accuracy_delta_vs_p0_pp >= -routed.accuracy_budget_pp - 1e-12
    )
    metric = f"{objective}_saving_vs_p0_percent"
    best = (routed[routed.within_budget].sort_values(metric, ascending=False)
            .groupby(["accuracy_budget_pp", "batch_size"], as_index=False).first())
    best["positive_saving"] = best[metric] > 0
    return routed, best


def best_static_routes(path, objective, budgets):
    frame = pd.read_csv(path)
    metric = (
        "vdd_in_energy_mj_per_image"
        if objective == "energy" else "latency_e2e_ms_per_image"
    )
    rows = []
    for batch_size, group in frame.groupby("batch_size", sort=True):
        baseline = group[group.model_id == "fp32_p0"].iloc[0]
        for budget in budgets:
            eligible = group[
                group.accuracy_percent >= baseline.accuracy_percent - budget - 1e-12
            ]
            selected = eligible.sort_values(metric).iloc[0].copy()
            selected["accuracy_budget_pp"] = budget
            selected["accuracy_delta_vs_p0_pp"] = (
                selected.accuracy_percent - baseline.accuracy_percent
            )
            selected[f"{objective}_saving_vs_p0_percent"] = 100 * (
                baseline[metric] - selected[metric]
            ) / baseline[metric]
            rows.append(selected)
    return pd.DataFrame(rows)


def save_best_plot(energy, latency, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, frame, objective in (
        (axes[0], energy, "energy"), (axes[1], latency, "latency")
    ):
        metric = f"{objective}_saving_vs_p0_percent"
        for budget, group in frame.groupby("accuracy_budget_pp"):
            axis.plot(group.batch_size, group[metric], marker="o", label=f"{budget:g} pp")
        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(frame.batch_size.unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("Batch size")
        axis.set_ylabel(f"{objective.title()} saving vs p0 (%)")
        axis.set_title(f"Best measured {objective} route")
        axis.grid(alpha=0.25)
        axis.legend(title="Maximum accuracy drop")
    fig.savefig(output_dir / "best_savings_by_accuracy_budget.png", dpi=180)
    plt.close(fig)


def save_overhead_plot(routed, output_dir):
    strict = routed[np.isclose(routed.accuracy_budget_pp, 0.5)]
    summary = (strict.groupby(["objective", "system"], as_index=False)
               .router_ms_per_batch.mean())
    systems = list(dict.fromkeys(summary.system))
    x = np.arange(len(systems))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for offset, (objective, group) in zip((-width / 2, width / 2), summary.groupby("objective")):
        values = group.set_index("system").reindex(systems).router_ms_per_batch
        axis.bar(x + offset, values, width, label=objective)
    axis.set_xticks(x, systems, rotation=30, ha="right")
    axis.set_ylabel("Router overhead (ms/batch)")
    axis.set_title("Measured router overhead at the 0.5 pp policy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "router_overhead_by_method.png", dpi=180)
    plt.close(fig)


def save_static_router_plot(router_energy, router_latency, static_energy,
                            static_latency, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, router, static, objective in (
        (axes[0], router_energy, static_energy, "energy"),
        (axes[1], router_latency, static_latency, "latency"),
    ):
        metric = f"{objective}_saving_vs_p0_percent"
        for batch_size in sorted(router.batch_size.unique()):
            routed = router[router.batch_size == batch_size].sort_values(
                "accuracy_budget_pp"
            )
            fixed = static[static.batch_size == batch_size].sort_values(
                "accuracy_budget_pp"
            )
            line = axis.plot(
                routed.accuracy_budget_pp, routed[metric], marker="o",
                label=f"router, batch {batch_size}",
            )[0]
            axis.plot(
                fixed.accuracy_budget_pp, fixed[metric], marker="s", linestyle="--",
                color=line.get_color(), label=f"static, batch {batch_size}",
            )
        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_xlabel("Maximum accuracy drop (percentage points)")
        axis.set_ylabel(f"{objective.title()} saving vs p0 (%)")
        axis.set_title(f"Best router vs best static model: {objective}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "best_router_vs_static.png", dpi=180)
    plt.close(fig)


def save_selection_plot(energy, latency, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    selection_columns = [
        "selected_fp32_p0", "selected_fp32_p10",
        "selected_fp32_p20", "selected_fp32_p30",
    ]
    labels = ["p0", "p10", "p20", "p30"]
    for axis, frame, objective in (
        (axes[0], energy, "energy"), (axes[1], latency, "latency")
    ):
        strict = frame[np.isclose(frame.accuracy_budget_pp, 0.5)].sort_values(
            "batch_size"
        )
        left = np.zeros(len(strict))
        for column, label in zip(selection_columns, labels):
            values = strict[column].to_numpy(float)
            axis.barh(strict.batch_size.astype(str), values, left=left, label=label)
            left += values
        route_names = ", ".join(strict.system.tolist())
        axis.set_xlabel("Images routed")
        axis.set_ylabel("Batch size")
        axis.set_title(f"Strict 0.5 pp {objective} routes\n{route_names}")
        axis.legend(title="Selected model", ncol=4, fontsize=8)
    fig.savefig(output_dir / "strict_policy_model_selections.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    energy_all, energy_best = best_routes(args.energy, "energy")
    latency_all, latency_best = best_routes(args.latency, "latency")
    budgets = sorted(energy_best.accuracy_budget_pp.unique())
    static_energy = best_static_routes(args.static_energy, "energy", budgets)
    static_latency = best_static_routes(args.static_latency, "latency", budgets)
    energy_all.to_csv(args.output_dir / "energy_all_routes.csv", index=False)
    latency_all.to_csv(args.output_dir / "latency_all_routes.csv", index=False)
    energy_best.to_csv(args.output_dir / "energy_best_routes.csv", index=False)
    latency_best.to_csv(args.output_dir / "latency_best_routes.csv", index=False)
    static_energy.to_csv(args.output_dir / "energy_best_static.csv", index=False)
    static_latency.to_csv(args.output_dir / "latency_best_static.csv", index=False)
    pd.read_csv(args.static_energy).to_csv(
        args.output_dir / "static_energy_summary.csv", index=False
    )
    pd.read_csv(args.static_latency).to_csv(
        args.output_dir / "static_latency_summary.csv", index=False
    )
    save_best_plot(energy_best, latency_best, args.output_dir)
    save_overhead_plot(pd.concat((energy_all, latency_all)), args.output_dir)
    save_static_router_plot(
        energy_best, latency_best, static_energy, static_latency, args.output_dir
    )
    save_selection_plot(energy_best, latency_best, args.output_dir)


if __name__ == "__main__":
    main()
