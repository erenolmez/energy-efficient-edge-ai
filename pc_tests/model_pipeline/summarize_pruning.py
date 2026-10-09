"""Consolidate structured-pruning metadata and produce Task 3 figures."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


ARCHITECTURE_ORDER = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "mobilenetv3_small",
    "mobilenetv3_large",
    "efficientnet_b0",
)
DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "mobilenetv3_small": "MobileNetV3-Small",
    "mobilenetv3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
}
COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#6B6B6B")
METADATA_PATTERN = re.compile(r"^(?P<architecture>.+)__p(?P<level>\d+)__fp32\.json$")
REQUIRED_FIELDS = (
    "architecture",
    "requested_prune_percent",
    "best_validation_accuracy",
    "test_accuracy",
    "fine_tuning_seconds",
    "mac_reduction_percent",
    "parameter_reduction_percent",
)


def collect_results(input_dir: Path) -> list[dict]:
    """Read completed, non-smoke-test pruning metadata sidecars."""
    results = []
    for path in input_dir.glob("*__p*__fp32.json"):
        match = METADATA_PATTERN.match(path.name)
        if not match:
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        missing = [field for field in REQUIRED_FIELDS if field not in metadata]
        if missing:
            raise ValueError(f"{path.name} is missing fields: {', '.join(missing)}")
        if metadata.get("smoke_test"):
            continue
        expected_level = int(match.group("level"))
        if metadata["architecture"] != match.group("architecture"):
            raise ValueError(f"Architecture mismatch in {path.name}")
        if metadata["requested_prune_percent"] != expected_level:
            raise ValueError(f"Pruning-level mismatch in {path.name}")
        results.append(metadata)

    order = {name: index for index, name in enumerate(ARCHITECTURE_ORDER)}
    return sorted(
        results,
        key=lambda item: (
            order.get(item["architecture"], len(order)),
            item["architecture"],
            item["requested_prune_percent"],
        ),
    )


def _write_csv(results: list[dict], path: Path) -> None:
    fields = list(results[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def _write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# Structured-pruning results",
        "",
        "All values come from the completed FP32 metadata sidecars. Training time is recovery fine-tuning time.",
        "",
        "| Architecture | Pruning | Validation accuracy | Test accuracy | MAC reduction | Parameter reduction | Time (min) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {architecture} | p{level} | {validation:.2f}% | {test:.2f}% | "
            "{mac:.2f}% | {parameters:.2f}% | {minutes:.1f} |".format(
                architecture=DISPLAY_NAMES.get(item["architecture"], item["architecture"]),
                level=item["requested_prune_percent"],
                validation=item["best_validation_accuracy"],
                test=item["test_accuracy"],
                mac=item["mac_reduction_percent"],
                parameters=item["parameter_reduction_percent"],
                minutes=item["fine_tuning_seconds"] / 60.0,
            )
        )
    total_hours = sum(item["fine_tuning_seconds"] for item in results) / 3600.0
    lines.extend(("", f"Total recovery fine-tuning time: **{total_hours:.2f} hours**.", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def _svg_line_chart(
    series: dict[str, list[tuple[float, float]]],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    y_min: float = 0.0,
    y_max: float = 100.0,
) -> None:
    width, height = 1040, 650
    left, right, top, bottom = 90, 250, 65, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_x = [x for points in series.values() for x, _ in points]
    x_min, x_max = min(all_x), max(all_x)

    def sx(value: float) -> float:
        return left + (value - x_min) * plot_width / (x_max - x_min)

    def sy(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22">{html.escape(title)}</text>',
    ]
    for step in range(11):
        tick = y_min + (y_max - y_min) * step / 10.0
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{tick:.0f}</text>')
    for tick in sorted(set(all_x)):
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Arial" font-size="13">{tick:g}</text>')
    parts.extend((
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black" stroke-width="1.5"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="16">{html.escape(x_label)}</text>',
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2})" font-family="Arial" font-size="16">{html.escape(y_label)}</text>',
    ))
    for index, (architecture, points) in enumerate(series.items()):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>')
        legend_y = top + 18 + index * 28
        parts.append(f'<line x1="{left + plot_width + 25}" y1="{legend_y}" x2="{left + plot_width + 55}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left + plot_width + 65}" y="{legend_y + 5}" font-family="Arial" font-size="14">{html.escape(DISPLAY_NAMES.get(architecture, architecture))}</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _group_series(results: list[dict], y_field: str, scale: float = 1.0) -> dict[str, list[tuple[float, float]]]:
    grouped = {}
    for item in results:
        grouped.setdefault(item["architecture"], []).append(
            (item["requested_prune_percent"], item[y_field] * scale)
        )
    return grouped


def generate_report(input_dir: Path, output_dir: Path | None = None) -> list[dict]:
    """Create the consolidated machine-readable summaries, table and figures."""
    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(input_dir)
    if not results:
        raise ValueError(f"No completed pruning metadata found in {input_dir}")
    (output_dir / "pruning_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(results, output_dir / "pruning_summary.csv")
    _write_markdown(results, output_dir / "pruning_results.md")
    _svg_line_chart(
        _group_series(results, "test_accuracy"),
        "CIFAR-100 accuracy after structured pruning",
        "Requested pruning (%)",
        "Test accuracy (%)",
        output_dir / "pruning_accuracy.svg",
    )
    _svg_line_chart(
        _group_series(results, "mac_reduction_percent"),
        "Measured MAC reduction",
        "Requested pruning (%)",
        "MAC reduction (%)",
        output_dir / "pruning_mac_reduction.svg",
    )
    _svg_line_chart(
        _group_series(results, "parameter_reduction_percent"),
        "Measured parameter reduction",
        "Requested pruning (%)",
        "Parameter reduction (%)",
        output_dir / "pruning_parameter_reduction.svg",
    )
    _svg_line_chart(
        _group_series(results, "fine_tuning_seconds", scale=1.0 / 60.0),
        "Recovery fine-tuning time",
        "Requested pruning (%)",
        "Fine-tuning time (minutes)",
        output_dir / "pruning_training_time.svg",
        y_max=max(item["fine_tuning_seconds"] for item in results) / 60.0 * 1.05,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/pruned"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    results = generate_report(args.input_dir, args.output_dir)
    print(f"Wrote consolidated outputs for {len(results)} pruning runs.")


if __name__ == "__main__":
    main()
