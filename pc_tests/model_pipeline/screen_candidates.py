"""Screen exported models against architecture-specific p0 accuracy baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_BUDGETS = (0.5, 1.0, 1.5, 2.0)


def parse_budgets(value: str) -> list[float]:
    budgets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets or any(item < 0 for item in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Accuracy-loss budgets must be unique non-negative values")
    return sorted(budgets)


def load_records(deployment_manifest: Path) -> tuple[list[dict], dict[str, dict]]:
    manifest = json.loads(deployment_manifest.read_text(encoding="utf-8"))
    records = []
    targets = {}
    for source in manifest["models"]:
        metadata = json.loads(Path(source["metadata"]).read_text(encoding="utf-8"))
        prune = int(source["prune"])
        records.append(
            {
                "architecture": source["architecture"],
                "prune": prune,
                "source_stem": source["source_stem"],
                "test_accuracy": float(source["test_accuracy"]),
                "mac_reduction_percent": float(metadata.get("mac_reduction_percent", 0.0)),
                "parameter_reduction_percent": float(
                    metadata.get("parameter_reduction_percent", 0.0)
                ),
                "onnx": source["onnx"],
            }
        )
        targets[source["source_stem"]] = source["targets"]
    return records, targets


def score_against_p0(records: list[dict]) -> list[dict]:
    baselines = {
        item["architecture"]: item["test_accuracy"]
        for item in records
        if item["prune"] == 0
    }
    architectures = {item["architecture"] for item in records}
    missing = sorted(architectures - set(baselines))
    if missing:
        raise ValueError(f"Missing p0 baseline(s): {', '.join(missing)}")
    scored = []
    for item in records:
        p0_accuracy = baselines[item["architecture"]]
        scored.append(
            {
                **item,
                "p0_test_accuracy": p0_accuracy,
                "accuracy_loss_pp": p0_accuracy - item["test_accuracy"],
            }
        )
    return scored


def choose_for_budget(records: list[dict], budget: float) -> dict:
    eligible = [item for item in records if item["accuracy_loss_pp"] <= budget + 1e-9]
    if not eligible:
        raise ValueError(f"No model satisfies a {budget:g} pp accuracy-loss budget")
    return max(
        eligible,
        key=lambda item: (
            item["mac_reduction_percent"],
            item["parameter_reduction_percent"],
            item["test_accuracy"],
            item["prune"],
        ),
    )


def screen(records: list[dict], budgets: list[float]) -> tuple[list[dict], list[dict]]:
    scored = score_against_p0(records)
    grouped = defaultdict(list)
    for item in scored:
        grouped[item["architecture"]].append(item)
    selections = []
    for architecture in sorted(grouped):
        for budget in budgets:
            choice = choose_for_budget(grouped[architecture], budget)
            selections.append(
                {
                    "architecture": architecture,
                    "budget_pp": budget,
                    **choice,
                }
            )
    return scored, selections


def _write_csv(records: list[dict], path: Path) -> None:
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_markdown(selections: list[dict], path: Path) -> None:
    lines = [
        "# Deployment candidate screening",
        "",
        "Each model is compared with the unpruned p0 model of the same architecture. "
        "Within each accuracy-loss budget, the selected model has the largest measured MAC reduction; "
        "parameter reduction and accuracy break ties.",
        "",
        "| Architecture | Maximum loss | Selected model | Test accuracy | Actual loss | MAC reduction | Parameter reduction |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in selections:
        lines.append(
            "| {architecture} | {budget:.1f} pp | {source} | {accuracy:.2f}% | "
            "{loss:.2f} pp | {mac:.2f}% | {parameters:.2f}% |".format(
                architecture=item["architecture"],
                budget=item["budget_pp"],
                source=item["source_stem"],
                accuracy=item["test_accuracy"],
                loss=item["accuracy_loss_pp"],
                mac=item["mac_reduction_percent"],
                parameters=item["parameter_reduction_percent"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidate_manifest(
    selections: list[dict], targets_by_source: dict[str, list[dict]], budgets: list[float]
) -> dict:
    grouped = {}
    for selection in selections:
        stem = selection["source_stem"]
        if stem not in grouped:
            grouped[stem] = {
                key: selection[key]
                for key in (
                    "architecture",
                    "prune",
                    "source_stem",
                    "test_accuracy",
                    "p0_test_accuracy",
                    "accuracy_loss_pp",
                    "mac_reduction_percent",
                    "parameter_reduction_percent",
                    "onnx",
                )
            }
            grouped[stem]["selected_for_budgets_pp"] = []
            grouped[stem]["targets"] = targets_by_source[stem]
        grouped[stem]["selected_for_budgets_pp"].append(selection["budget_pp"])
    candidates = sorted(
        grouped.values(), key=lambda item: (item["architecture"], item["prune"])
    )
    return {
        "schema_version": 1,
        "selection_rule": "maximum measured MAC reduction within each architecture-specific p0 accuracy-loss budget",
        "accuracy_loss_budgets_pp": budgets,
        "source_candidate_count": len(candidates),
        "precision_target_count": sum(len(item["targets"]) for item in candidates),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    args = parser.parse_args()
    budgets = parse_budgets(args.budgets)
    records, targets = load_records(args.deployment_manifest)
    scored, selections = screen(records, budgets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(scored, args.output_dir / "candidate_scores.csv")
    _write_csv(selections, args.output_dir / "candidate_selections.csv")
    _write_markdown(selections, args.output_dir / "candidate_screening.md")
    candidate_manifest = build_candidate_manifest(selections, targets, budgets)
    (args.output_dir / "jetson_candidates.json").write_text(
        json.dumps(candidate_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Screened {len(scored)} sources and selected "
        f"{candidate_manifest['source_candidate_count']} unique candidates "
        f"({candidate_manifest['precision_target_count']} precision targets)."
    )


if __name__ == "__main__":
    main()
