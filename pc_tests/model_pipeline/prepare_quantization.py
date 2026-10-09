"""Prepare trained classifiers for later TensorRT precision builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from model_pipeline.naming import model_name, parse_model_name


TARGET_PRECISIONS = ("fp32", "fp16", "int8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--pruned-dir", type=Path, default=Path("artifacts/pruned"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/quantization"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--calibration-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-models",
        default="",
        help="Comma-separated source stems to export, for example resnet18__p0__fp32.",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export and validate every discovered source checkpoint.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Replace ONNX files even when a current file already validates.",
    )
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def _metadata_for(path: Path) -> dict:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata sidecar for {path.name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("smoke_test"):
        raise ValueError(f"Smoke-test checkpoint cannot be deployed: {path.name}")
    return metadata


def discover_sources(baseline_dir: Path, pruned_dir: Path) -> list[dict]:
    """Discover completed p0 and pruned FP32 checkpoints with sidecars."""
    paths = list(baseline_dir.glob("*__p0__fp32.pth"))
    paths.extend(pruned_dir.glob("*__p*__fp32.pth"))
    sources = []
    seen = set()
    for path in paths:
        parsed = parse_model_name(path.stem)
        key = (parsed["architecture"], parsed["prune"])
        if key in seen:
            raise ValueError(f"Duplicate source checkpoint for {key}: {path}")
        seen.add(key)
        metadata = _metadata_for(path)
        metadata_prune = metadata.get(
            "requested_prune_percent", metadata.get("prune")
        )
        if metadata.get("architecture") != parsed["architecture"]:
            raise ValueError(f"Architecture mismatch in {path.with_suffix('.json').name}")
        if metadata_prune != parsed["prune"]:
            raise ValueError(f"Pruning-level mismatch in {path.with_suffix('.json').name}")
        sources.append(
            {
                "architecture": parsed["architecture"],
                "prune": parsed["prune"],
                "source_stem": path.stem,
                "checkpoint": str(path.resolve()),
                "metadata": str(path.with_suffix(".json").resolve()),
                "test_accuracy": metadata["test_accuracy"],
                "split_sha256": metadata["split_sha256"],
                "image_size": metadata["image_size"],
            }
        )
    return sorted(sources, key=lambda item: (item["architecture"], item["prune"]))


def calibration_indices(split_manifest: Path, size: int, seed: int) -> tuple[list[int], str]:
    """Select a deterministic subset of training-only CIFAR-100 indices."""
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    indices = list(split["train_indices"])
    if not 0 < size <= len(indices):
        raise ValueError(f"Calibration size must be between 1 and {len(indices)}")
    random.Random(seed).shuffle(indices)
    selected = indices[:size]
    digest = hashlib.sha256(
        ",".join(str(index) for index in selected).encode("ascii")
    ).hexdigest()
    return selected, digest


def build_manifest(sources: list[dict], output_dir: Path, calibration: dict) -> dict:
    entries = []
    onnx_dir = output_dir / "onnx"
    engine_dir = output_dir / "engines"
    for source in sources:
        architecture, prune = source["architecture"], source["prune"]
        entry = {
            **source,
            "onnx": str((onnx_dir / f"{architecture}__p{prune}.onnx").resolve()),
            "onnx_dtype": "fp32",
            "targets": [],
        }
        for precision in TARGET_PRECISIONS:
            target = {
                "precision": precision,
                "name": model_name(architecture, prune=prune, precision=precision),
                "engine": str(
                    (engine_dir / f"{model_name(architecture, prune=prune, precision=precision)}.engine").resolve()
                ),
                "requires_calibration": precision == "int8",
            }
            entry["targets"].append(target)
        entries.append(entry)
    return {
        "schema_version": 1,
        "source_count": len(entries),
        "target_count": len(entries) * len(TARGET_PRECISIONS),
        "target_precisions": list(TARGET_PRECISIONS),
        "calibration": calibration,
        "models": entries,
    }


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_onnx(onnx_path: Path) -> None:
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    input_batch = graph.graph.input[0].type.tensor_type.shape.dim[0]
    output_batch = graph.graph.output[0].type.tensor_type.shape.dim[0]
    if input_batch.dim_param != "batch" or output_batch.dim_param != "batch":
        raise RuntimeError(f"Dynamic batch axis missing from {onnx_path.name}")


def export_and_validate(
    source: dict, output_dir: Path, opset: int, force_export: bool = False
) -> dict:
    """Export one FP32 graph and validate its structure and output contract."""
    checkpoint_path = Path(source["checkpoint"])
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{source['architecture']}__p{source['prune']}.onnx"
    if (
        onnx_path.exists()
        and not force_export
        and onnx_path.stat().st_mtime >= checkpoint_path.stat().st_mtime
    ):
        validate_onnx(onnx_path)
        return {
            "source_stem": source["source_stem"],
            "onnx": str(onnx_path.resolve()),
            "bytes": onnx_path.stat().st_size,
            "opset": opset,
            "dynamic_batch": True,
            "output_classes": 100,
            "status": "validated_existing",
        }

    model = safe_torch_load(checkpoint_path)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{source['checkpoint']} did not contain torch.nn.Module")
    model.cpu().float().eval()
    dummy = torch.randn(2, 3, source["image_size"], source["image_size"])
    with torch.inference_mode():
        output = model(dummy)
    if tuple(output.shape) != (2, 100):
        raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}")

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy[:1],
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )
    validate_onnx(onnx_path)
    return {
        "source_stem": source["source_stem"],
        "onnx": str(onnx_path.resolve()),
        "bytes": onnx_path.stat().st_size,
        "opset": opset,
        "dynamic_batch": True,
        "output_classes": 100,
        "status": "validated",
    }


def main():
    args = parse_args()
    split_manifest = args.split_manifest or args.baseline_dir / "cifar100_split_seed42.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(args.baseline_dir, args.pruned_dir)
    if not sources:
        raise RuntimeError("No deployment checkpoints were discovered")
    split_hashes = {source["split_sha256"] for source in sources}
    image_sizes = {source["image_size"] for source in sources}
    if len(split_hashes) != 1 or len(image_sizes) != 1:
        raise ValueError("Source checkpoints do not share one split and image size")
    indices, indices_hash = calibration_indices(
        split_manifest, args.calibration_size, args.seed
    )
    calibration_path = args.output_dir / "calibration_indices.json"
    calibration_data = {
        "dataset": "CIFAR-100 training split",
        "seed": args.seed,
        "size": len(indices),
        "indices_sha256": indices_hash,
        "training_split_sha256": next(iter(split_hashes)),
        "indices": indices,
    }
    calibration_path.write_text(
        json.dumps(calibration_data, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(
        sources,
        args.output_dir,
        {
            key: value
            for key, value in calibration_data.items()
            if key != "indices"
        }
        | {"manifest": str(calibration_path.resolve())},
    )
    manifest_path = args.output_dir / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    requested = [value.strip() for value in args.smoke_models.split(",") if value.strip()]
    if args.export_all and requested:
        raise ValueError("Use either --export-all or --smoke-models, not both")
    if args.export_all:
        requested = [source["source_stem"] for source in sources]
    by_stem = {source["source_stem"]: source for source in sources}
    unknown = sorted(set(requested) - set(by_stem))
    if unknown:
        raise ValueError(f"Unknown smoke model(s): {', '.join(unknown)}")
    results = []
    results_path = args.output_dir / (
        "export_validation.json" if args.export_all else "smoke_validation.json"
    )
    for stem in requested:
        result = export_and_validate(
            by_stem[stem], args.output_dir, args.opset, args.force_export
        )
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{len(results)}/{len(requested)}] {stem}: {result['status']}",
            flush=True,
        )
    if not requested:
        results_path.write_text("[]\n", encoding="utf-8")
    print(
        f"Prepared {len(sources)} source models and {manifest['target_count']} precision targets; "
        f"validated {len(results)} exports."
    )


if __name__ == "__main__":
    main()
