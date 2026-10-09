import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from model_pipeline.data import split_fingerprint, stratified_indices  # noqa: E402
from model_pipeline.models import ARCHITECTURES, normalize_architecture  # noqa: E402
from model_pipeline.naming import model_name, parse_model_name  # noqa: E402
from pc_tests.model_pipeline.prune_finetune import parse_levels  # noqa: E402
from pc_tests.model_pipeline.summarize_pruning import collect_results  # noqa: E402
from pc_tests.model_pipeline.screen_candidates import (  # noqa: E402
    choose_for_budget,
    parse_budgets,
    score_against_p0,
)
from pc_tests.model_pipeline.prepare_deployment_bundle import (  # noqa: E402
    prepare_bundle,
    verify_bundle,
)
from pc_tests.model_pipeline.prepare_quantization import (  # noqa: E402
    build_manifest,
    calibration_indices,
    discover_sources,
    validate_onnx,
)


class ModelPipelineTests(unittest.TestCase):
    def test_selected_architectures_are_registered(self):
        self.assertEqual(
            set(ARCHITECTURES),
            {
                "resnet18", "resnet34", "resnet50", "resnet101",
                "mobilenetv3_small", "mobilenetv3_large",
                "efficientnet_b0", "shufflenetv2_x1_0",
            },
        )

    def test_aliases_are_normalized(self):
        self.assertEqual(normalize_architecture("MobileNetV3-Small"), "mobilenetv3_small")
        self.assertEqual(normalize_architecture("ShuffleNetV2"), "shufflenetv2_x1_0")

    def test_model_name_round_trip(self):
        name = model_name("EfficientNet-B0", prune=20, precision="int8")
        self.assertEqual(name, "efficientnet_b0__p20__int8")
        self.assertEqual(
            parse_model_name(name),
            {"architecture": "efficientnet_b0", "prune": 20, "precision": "int8"},
        )

    def test_stratified_split_is_stable_and_balanced(self):
        targets = [class_id for class_id in range(100) for _ in range(10)]
        train_a, validation_a = stratified_indices(targets, 200, seed=42)
        train_b, validation_b = stratified_indices(targets, 200, seed=42)
        self.assertEqual((train_a, validation_a), (train_b, validation_b))
        self.assertEqual(len(validation_a), 200)
        validation_targets = [targets[index] for index in validation_a]
        self.assertTrue(all(validation_targets.count(class_id) == 2 for class_id in range(100)))
        self.assertEqual(
            split_fingerprint(train_a, validation_a),
            split_fingerprint(train_b, validation_b),
        )

    def test_pruning_levels_are_validated(self):
        self.assertEqual(parse_levels("10,20,90"), [10, 20, 90])
        for invalid in ("", "0", "100", "10,10"):
            with self.assertRaises(ValueError):
                parse_levels(invalid)

    def test_pruning_summary_collects_and_sorts_sidecars(self):
        import json
        import tempfile

        template = {
            "best_validation_accuracy": 70.0,
            "test_accuracy": 69.0,
            "fine_tuning_seconds": 10.0,
            "mac_reduction_percent": 20.0,
            "parameter_reduction_percent": 25.0,
            "smoke_test": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for architecture, level in (("resnet34", 20), ("resnet18", 30), ("resnet18", 10)):
                metadata = {
                    **template,
                    "architecture": architecture,
                    "requested_prune_percent": level,
                }
                (root / f"{architecture}__p{level}__fp32.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
            results = collect_results(root)
        self.assertEqual(
            [(item["architecture"], item["requested_prune_percent"]) for item in results],
            [("resnet18", 10), ("resnet18", 30), ("resnet34", 20)],
        )

    def test_quantization_manifest_enumerates_three_precision_targets(self):
        source = {
            "architecture": "resnet18",
            "prune": 20,
            "source_stem": "resnet18__p20__fp32",
            "checkpoint": "checkpoint.pth",
            "metadata": "checkpoint.json",
            "test_accuracy": 70.0,
            "split_sha256": "abc",
            "image_size": 128,
        }
        manifest = build_manifest([source], Path("output"), {"size": 16})
        self.assertEqual(manifest["source_count"], 1)
        self.assertEqual(manifest["target_count"], 3)
        self.assertEqual(
            [target["precision"] for target in manifest["models"][0]["targets"]],
            ["fp32", "fp16", "int8"],
        )
        self.assertTrue(manifest["models"][0]["targets"][2]["requires_calibration"])

    def test_calibration_indices_are_deterministic_and_training_only(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(
                json.dumps({"train_indices": list(range(100)), "validation_indices": [100, 101]}),
                encoding="utf-8",
            )
            first, first_hash = calibration_indices(path, 16, seed=42)
            second, second_hash = calibration_indices(path, 16, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first_hash, second_hash)
        self.assertTrue(set(first).issubset(set(range(100))))
        self.assertEqual(len(set(first)), 16)

    def test_candidate_screening_uses_architecture_p0_and_mac_reduction(self):
        records = [
            {"architecture": "resnet18", "prune": 0, "source_stem": "p0", "test_accuracy": 80.0, "mac_reduction_percent": 0.0, "parameter_reduction_percent": 0.0},
            {"architecture": "resnet18", "prune": 10, "source_stem": "p10", "test_accuracy": 79.7, "mac_reduction_percent": 20.0, "parameter_reduction_percent": 18.0},
            {"architecture": "resnet18", "prune": 20, "source_stem": "p20", "test_accuracy": 79.1, "mac_reduction_percent": 35.0, "parameter_reduction_percent": 32.0},
        ]
        scored = score_against_p0(records)
        self.assertEqual(choose_for_budget(scored, 0.5)["source_stem"], "p10")
        self.assertEqual(choose_for_budget(scored, 1.0)["source_stem"], "p20")

    def test_candidate_budgets_are_sorted_and_validated(self):
        self.assertEqual(parse_budgets("2,0.5,1"), [0.5, 1.0, 2.0])
        for invalid in ("", "-1", "1,1"):
            with self.assertRaises(ValueError):
                parse_budgets(invalid)

    def test_deployment_bundle_is_portable_and_verifiable(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "resnet18__p0.onnx"
            model.write_bytes(b"test-onnx")
            candidate_manifest = root / "candidates.json"
            candidate_manifest.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "architecture": "resnet18",
                                "prune": 0,
                                "source_stem": "resnet18__p0__fp32",
                                "onnx": str(model),
                                "test_accuracy": 80.0,
                                "p0_test_accuracy": 80.0,
                                "accuracy_loss_pp": 0.0,
                                "selected_for_budgets_pp": [0.5, 1.0],
                                "targets": [
                                    {"precision": precision}
                                    for precision in ("fp32", "fp16", "int8")
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            calibration_manifest = root / "calibration.json"
            calibration_manifest.write_text(
                json.dumps({"size": 16, "indices_sha256": "abc", "indices": list(range(16))}),
                encoding="utf-8",
            )
            output = root / "bundle"
            result = prepare_bundle(
                candidate_manifest, calibration_manifest, output, link_mode="copy"
            )
            self.assertEqual(result["source_candidate_count"], 1)
            self.assertEqual(result["precision_target_count"], 3)
            self.assertEqual(verify_bundle(output), [])


if __name__ == "__main__":
    unittest.main()
