"""Prepare a portable, integrity-checked bundle for later Jetson transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_file(source: Path, target: Path, link_mode: str) -> str:
    """Create a bundle file without duplicating model data when possible."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target) or sha256_file(source) == sha256_file(target):
            return "existing"
        target.unlink()
    if link_mode == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy_fallback"
    shutil.copy2(source, target)
    return "copy"


def prepare_bundle(
    candidate_manifest_path: Path,
    calibration_manifest_path: Path,
    output_dir: Path,
    link_mode: str = "hardlink",
) -> dict:
    candidates = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
    models_dir = output_dir / "models"
    manifests_dir = output_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    model_entries = []
    for candidate in candidates["candidates"]:
        source = Path(candidate["onnx"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing candidate ONNX model: {source}")
        target = models_dir / source.name
        method = materialize_file(source, target, link_mode)
        model_entries.append(
            {
                "architecture": candidate["architecture"],
                "prune": candidate["prune"],
                "source_stem": candidate["source_stem"],
                "path": target.relative_to(output_dir).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "materialization": method,
                "test_accuracy": candidate["test_accuracy"],
                "p0_test_accuracy": candidate["p0_test_accuracy"],
                "accuracy_loss_pp": candidate["accuracy_loss_pp"],
                "selected_for_budgets_pp": candidate["selected_for_budgets_pp"],
                "precisions": [target["precision"] for target in candidate["targets"]],
            }
        )

    copied_manifests = []
    for source, name in (
        (candidate_manifest_path, "jetson_candidates.json"),
        (calibration_manifest_path, "calibration_indices.json"),
    ):
        target = manifests_dir / name
        shutil.copy2(source, target)
        copied_manifests.append(
            {
                "path": target.relative_to(output_dir).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )

    bundle = {
        "schema_version": 1,
        "purpose": "PC-prepared inputs for later Jetson TensorRT FP32/FP16/INT8 builds",
        "source_candidate_count": len(model_entries),
        "precision_target_count": sum(len(item["precisions"]) for item in model_entries),
        "calibration_size": calibration["size"],
        "calibration_indices_sha256": calibration["indices_sha256"],
        "models": model_entries,
        "manifests": copied_manifests,
    }
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    readme = """# Jetson deployment bundle

This bundle was prepared on the PC. It contains the selected FP32 ONNX source
graphs, the candidate-selection record, fixed INT8 calibration indices and
SHA-256 checksums. No TensorRT engine is included: engines must be built on the
target Jetson because TensorRT plans depend on its hardware and software stack.

Before use, verify every file against `bundle_manifest.json`. Each ONNX model
is intended to produce FP32, FP16 and calibrated INT8 TensorRT engines.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return bundle


def verify_bundle(output_dir: Path) -> list[str]:
    manifest = json.loads((output_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in [*manifest["models"], *manifest["manifests"]]:
        path = output_dir / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
        elif path.stat().st_size != item["bytes"]:
            failures.append(f"size mismatch: {item['path']}")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"hash mismatch: {item['path']}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        bundle = prepare_bundle(
            args.candidate_manifest,
            args.calibration_manifest,
            args.output_dir,
            args.link_mode,
        )
        print(
            f"Prepared {bundle['source_candidate_count']} models and "
            f"{bundle['precision_target_count']} precision targets."
        )
    failures = verify_bundle(args.output_dir)
    if failures:
        raise RuntimeError("Bundle verification failed:\n" + "\n".join(failures))
    print("Bundle integrity verification passed.")


if __name__ == "__main__":
    main()
