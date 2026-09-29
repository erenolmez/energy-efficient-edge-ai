"""Convert a TensorRT benchmark summary into objective-specific model costs."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--latency-column", default="latency_trt_mean")
    parser.add_argument("--power-column", default="power_tegra_mean")
    parser.add_argument("--objective", choices=["energy", "latency"], default="energy")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_costs.csv"))
    args = parser.parse_args()
    benchmark, models = pd.read_csv(args.benchmark), pd.read_csv(args.models)
    for frame, required, name in (
        (models, {"model_id", "precision", "prune"}, "model manifest"),
        (benchmark, {"precision", "prune", args.latency_column, args.power_column}, "benchmark"),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    selected = models.merge(benchmark, on=["precision", "prune"], how="left",
                            validate="one_to_one")
    if selected[[args.latency_column, args.power_column]].isna().any().any():
        raise ValueError("One or more manifest models have no matching benchmark row")
    if args.objective == "energy":
        selected["cost"] = (selected[args.power_column].astype(float)
                            * selected[args.latency_column].astype(float))
        unit = "mJ/image"
    else:
        selected["cost"] = selected[args.latency_column].astype(float)
        unit = "ms/image"
    selected["objective"] = args.objective
    selected["unit"] = unit
    output = selected[["model_id", "cost", "objective", "unit"]].sort_values("cost")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
    print(f"Saved model costs to {args.output}")


if __name__ == "__main__":
    main()
