"""No downloads, GPU access, private paths or training required."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hpc.clean_protocol import (PER_CLASS, atomic_write, fingerprint,
                                load_or_create_split, make_split,
                                validate_resume, validate_split, write_json)


class CleanSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = [label for label in range(100) for _ in range(500)]
        cls.manifest = make_split(cls.targets)

    def resign(self, manifest):
        manifest["sha256"] = fingerprint({key: value for key, value in manifest.items()
                                          if key != "sha256"})

    def test_exact_sizes_balance_no_overlap_and_full_coverage(self):
        validate_split(self.manifest, self.targets)
        seen = set()
        for role, per_class in PER_CLASS.items():
            indices = self.manifest["indices"][role]
            self.assertEqual(len(indices), per_class * 100)
            self.assertFalse(seen.intersection(indices))
            seen.update(indices)
            for label in range(100):
                self.assertEqual(sum(self.targets[index] == label for index in indices), per_class)
        self.assertEqual(seen, set(range(50000)))

    def test_fixed_split_independent_of_training_seed(self):
        self.assertEqual(self.manifest, make_split(self.targets, 42))
        self.assertNotEqual(self.manifest["sha256"], make_split(self.targets, 45)["sha256"])

    def test_modified_fingerprint_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed["split_seed"] = 44
        with self.assertRaises(ValueError):
            validate_split(changed, self.targets)

    def test_overlap_rejected_even_after_fingerprint_update(self):
        changed = copy.deepcopy(self.manifest)
        changed["indices"]["router_train"][0] = changed["indices"]["classifier_train"][0]
        self.resign(changed)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split(changed, self.targets)

    def test_duplicate_and_invalid_index_rejected(self):
        for value in (self.manifest["indices"]["router_train"][1], -1, 50000, True):
            changed = copy.deepcopy(self.manifest)
            changed["indices"]["router_train"][0] = value
            self.resign(changed)
            with self.assertRaises(ValueError):
                validate_split(changed, self.targets)

    def test_label_order_change_rejected(self):
        changed_targets = self.targets.copy()
        changed_targets[0], changed_targets[500] = changed_targets[500], changed_targets[0]
        with self.assertRaisesRegex(ValueError, "label/order"):
            validate_split(self.manifest, changed_targets)

    def test_unbalanced_partition_rejected(self):
        changed = copy.deepcopy(self.manifest)
        a = changed["indices"]["classifier_train"]
        b = changed["indices"]["router_train"]
        # Disjointness and sizes remain intact, but class balance is broken.
        opposite_class = next(index for index, value in enumerate(b) if self.targets[value] == 1)
        a[0], b[opposite_class] = b[opposite_class], a[0]
        self.resign(changed)
        with self.assertRaisesRegex(ValueError, "class-balanced"):
            validate_split(changed, self.targets)

    def test_existing_split_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            original = load_or_create_split(path, self.targets)
            old_bytes = path.read_bytes()
            self.assertEqual(original, load_or_create_split(path, self.targets))
            with self.assertRaises(ValueError):
                load_or_create_split(path, self.targets, seed=43)
            self.assertEqual(old_bytes, path.read_bytes())

    def test_bad_dataset_rejected(self):
        with self.assertRaises(ValueError):
            make_split(self.targets[:-1])


class CheckpointTests(unittest.TestCase):
    def test_writer_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latest.pt"
            path.write_bytes(b"old checkpoint")
            def fail(temporary):
                temporary.write_bytes(b"incomplete")
                raise RuntimeError("simulated interruption")
            with self.assertRaises(RuntimeError):
                atomic_write(path, fail)
            self.assertEqual(path.read_bytes(), b"old checkpoint")
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_replace_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latest.pt"
            path.write_bytes(b"old checkpoint")
            with patch("hpc.clean_protocol.os.replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    atomic_write(path, lambda temporary: temporary.write_bytes(b"new"))
            self.assertEqual(path.read_bytes(), b"old checkpoint")

    def test_atomic_json_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            write_json(path, {"status": "saved"})
            self.assertEqual(json.loads(path.read_text()), {"status": "saved"})
            with self.assertRaises(ValueError):
                write_json(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"status": "saved"})

    def test_resume_recipe_split_source_and_epoch_gates(self):
        recipe = {"seed": 42, "epochs": 3}
        sources = {"pilot.py": "source-hash"}
        checkpoint = {"recipe": recipe, "split_sha256": "split-hash",
                      "source_hashes": sources, "epoch": 2,
                      "history": [{"epoch": 1}, {"epoch": 2}]}
        validate_resume(checkpoint, recipe, "split-hash", sources)
        for new_recipe, split, new_sources in (({"seed": 45}, "split-hash", sources),
                                                (recipe, "changed", sources),
                                                (recipe, "split-hash", {"pilot.py": "new"})):
            with self.assertRaises(ValueError):
                validate_resume(checkpoint, new_recipe, split, new_sources)
        checkpoint["history"] = [{"epoch": 2}]
        with self.assertRaises(ValueError):
            validate_resume(checkpoint, recipe, "split-hash", sources)


class PilotBoundaryTests(unittest.TestCase):
    def test_only_training_namespace_is_constructed(self):
        source = (Path(__file__).parents[1] / "hpc/train_classifier_pilot.py").read_text()
        tree = ast.parse(source)
        dataset_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "CIFAR100"]
        self.assertEqual(len(dataset_calls), 3)
        for call in dataset_calls:
            train = next(keyword.value for keyword in call.keywords if keyword.arg == "train")
            self.assertIsInstance(train, ast.Constant)
            self.assertIs(train.value, True)
        subset_roles = [node.slice.value for node in ast.walk(tree)
                        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                        and isinstance(node.value, ast.Subscript)
                        and isinstance(node.value.slice, ast.Constant)
                        and node.value.slice.value == "indices"]
        self.assertEqual(sorted(subset_roles), ["classifier_train", "classifier_validation"])


if __name__ == "__main__":
    unittest.main()
