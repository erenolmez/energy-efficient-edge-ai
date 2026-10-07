import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from model_pipeline.data import split_fingerprint, stratified_indices  # noqa: E402
from model_pipeline.models import ARCHITECTURES, normalize_architecture  # noqa: E402
from model_pipeline.naming import model_name, parse_model_name  # noqa: E402
from pc_tests.model_pipeline.prune_finetune import parse_levels  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
