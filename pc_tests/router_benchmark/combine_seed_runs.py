"""Combine independent seed batches and publish one objective-specific report."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("router", as_index=False, sort=False).agg(
        runs=("seed", "count"),
        accuracy_mean=("accuracy_percent", "mean"),
        accuracy_std=("accuracy_percent", "std"),
        accuracy_delta_pp_mean=("accuracy_delta_pp", "mean"),
        candidate_saving_mean=("candidate_saving_percent", "mean"),
        candidate_saving_std=("candidate_saving_percent", "std"),
        router_latency_pc_mean_ms=("router_mean_ms_pc", "mean"),
        passing_seeds=("meets_eval_1pp_budget", "sum"),
    )


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Router | Accuracy | Difference vs p0 | Candidate saving | PC router overhead | Passes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.router} | {row.accuracy_mean:.2f}% ± {row.accuracy_std:.2f} | "
            f"{row.accuracy_delta_pp_mean:+.2f} pp | "
            f"{row.candidate_saving_mean:.2f}% ± {row.candidate_saving_std:.2f} | "
            f"{row.router_latency_pc_mean_ms:.2f} ms | "
            f"{int(row.passing_seeds)}/{int(row.runs)} |"
        )
    return "\n".join(lines)


def selection_table(selection: pd.DataFrame, routers: list[str]) -> str:
    if not routers:
        return ""
    table = selection[selection.router.isin(routers)].pivot(
        index="router", columns="model_id", values="selected_count_mean"
    ).fillna(0.0)
    model_ids = [f"fp32_p{ratio}" for ratio in range(0, 100, 10)]
    lines = [
        "## Average model selections per 2,000 images",
        "",
        "| Router | " + " | ".join(model_ids) + " |",
        "|---|" + "---:|" * len(model_ids),
    ]
    for router in routers:
        values = [float(table.loc[router, model]) for model in model_ids]
        lines.append(
            f"| {router} | " + " | ".join(f"{value:.1f}" for value in values) + " |"
        )
    lines.extend([
        "",
        "Counts are averaged across seeds. Each row sums to 2,000 apart from rounding.",
    ])
    return "\n".join(lines)


def combine(objective: str, sources: list[Path], output: Path, report: Path) -> None:
    rows = pd.concat(
        [pd.read_csv(source / "comparison_p0.csv") for source in sources],
        ignore_index=True,
    )
    counts = pd.concat(
        [pd.read_csv(source / "selection_counts.csv") for source in sources],
        ignore_index=True,
    )
    rows = rows[rows.objective == objective].copy()
    counts = counts[counts.objective == objective].copy()
    if rows.empty or counts.empty:
        raise ValueError(f"No {objective} results found in the input directories")
    if rows.duplicated(["router", "seed"]).any():
        raise ValueError(f"Duplicate router/seed result in {objective}")
    seed_values = sorted(rows.seed.unique().tolist())
    expected_runs = len(seed_values)
    per_router = rows.groupby("router").seed.nunique()
    if not (per_router == expected_runs).all():
        raise ValueError(f"Incomplete {objective} seed coverage")
    sums = counts.groupby(["router", "seed"]).selected_count.sum()
    if len(sums) != len(rows) or not (sums == 2000).all():
        raise ValueError(f"Invalid {objective} selection counts")

    output.mkdir(parents=True, exist_ok=True)
    rows.sort_values(["router", "seed"]).to_csv(output / "comparison_p0.csv", index=False)
    counts.sort_values(["router", "seed", "model_id"]).to_csv(
        output / "selection_counts.csv", index=False
    )
    result = summary(rows)
    result["pass_rate_percent"] = 100 * result.passing_seeds / result.runs
    result = result.sort_values(
        ["pass_rate_percent", "candidate_saving_mean", "accuracy_mean"],
        ascending=[False, False, False],
    )
    result.to_csv(output / "summary_p0.csv", index=False)
    selection = counts.groupby(["router", "model_id"], as_index=False).agg(
        selected_count_mean=("selected_count", "mean"),
        selected_count_std=("selected_count", "std"),
        selected_percent_mean=("selected_percent", "mean"),
    )
    selection.to_csv(output / "selection_summary.csv", index=False)

    stable = result[(result.passing_seeds == result.runs) &
                    (result.candidate_saving_mean > 0.001)]
    fast = stable[stable.router_latency_pc_mean_ms < 5]
    best_stable = stable.iloc[0] if len(stable) else None
    best_fast = fast.sort_values("candidate_saving_mean", ascending=False).iloc[0] if len(fast) else None
    unit = "energy" if objective == "energy" else "latency"
    finding = "No router saved candidate cost in every seed."
    if best_stable is not None:
        finding = (
            f"The largest candidate-{unit} saving among routers that respected the "
            f"one-point accuracy budget in all {expected_runs} seeds is "
            f"`{best_stable.router}`: {best_stable.candidate_saving_mean:.2f}% saving, "
            f"{best_stable.accuracy_mean:.2f}% accuracy, and "
            f"{best_stable.router_latency_pc_mean_ms:.2f} ms PC router overhead."
        )
    fast_finding = ""
    if best_fast is not None and best_fast.router != best_stable.router:
        fast_finding = (
            f" Restricting attention to routers below 5 ms PC overhead, "
            f"`{best_fast.router}` has the largest stable saving: "
            f"{best_fast.candidate_saving_mean:.2f}% at "
            f"{best_fast.accuracy_mean:.2f}% accuracy."
        )
    caveat = (
        "Candidate energy excludes router energy and is not total system energy."
        if objective == "energy" else
        "Candidate latency excludes router latency and is not total system latency."
    )
    recommended = []
    for candidate in (best_stable, best_fast):
        if candidate is not None and candidate.router not in recommended:
            recommended.append(candidate.router)
    selections_markdown = selection_table(selection, recommended)
    baseline_accuracy = rows.baseline_accuracy_percent.unique()
    if len(baseline_accuracy) != 1:
        raise ValueError("Baseline accuracy differs between runs")

    report.write_text(f"""# {objective.capitalize()}-only router results: {expected_runs} seeds

This is a separate **{objective}-only** experiment. It uses learner seeds
{', '.join(map(str, seed_values))}, the same fixed 2,000 evaluation images, and
the same unpruned `fp32_p0` baseline with {baseline_accuracy[0]:.2f}% accuracy. No combined
energy-latency score is used.

{markdown_table(result)}

## Main finding

{finding}{fast_finding}

{selections_markdown}

The “Passes” column counts seeds whose evaluation accuracy remained within one
percentage point of p0. {caveat}
""", encoding="utf-8")
    print(f"{objective}: {len(rows)} runs, seeds {seed_values}, {report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("energy", "latency"), required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    combine(args.objective, args.sources, args.output, args.report)


if __name__ == "__main__":
    main()
