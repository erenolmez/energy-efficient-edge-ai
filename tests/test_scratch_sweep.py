import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from hpc.clean_protocol import file_hash, write_json
from hpc.scratch_recipes import (check_confirmation, checked_plan, create_selection,
    freeze_plan, make_plan, ranked_search, rank_rows, trial_directory)


class SearchProtocolTests(unittest.TestCase):
    def test_bounded_balanced_search_and_no_pretraining(self):
        plan = make_plan()
        self.assertIsNone(plan["weights"])
        self.assertEqual(plan["image_size"], 32)
        self.assertEqual(len(plan["trials"]), 12)
        self.assertEqual(plan["maximum_training_epochs"], 3200)
        self.assertEqual(sum(plan["data_roles"].values()), 50000)
        self.assertEqual([trial["id"] for trial in plan["trials"]], list(range(12)))
        for optimizer in ("sgd", "adam", "adamw"):
            self.assertEqual(sum(trial["optimizer"] == optimizer for trial in plan["trials"][:9]), 3)
        self.assertEqual(plan, make_plan())

    def test_plan_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            original = freeze_plan(path)
            self.assertEqual(checked_plan(path), original)
            changed = copy.deepcopy(original)
            changed["epochs"] = 3
            write_json(path, changed)
            with self.assertRaises(ValueError):
                freeze_plan(path)
            with self.assertRaises(ValueError):
                checked_plan(path)

    def fixture(self, root):
        plan = make_plan()
        for trial in plan["trials"]:
            directory = trial_directory(root, trial["id"], 42)
            directory.mkdir(parents=True)
            checkpoint = directory / "latest.pt"
            checkpoint.write_bytes(b"synthetic checkpoint")
            result = {"status": "complete", "epochs_complete": 200,
                "plan_sha256": plan["sha256"], "trial": trial, "seed": 42,
                "split_sha256": "fixed-split", "test_evaluated": False,
                "initial_state_sha256": "same-initial-state",
                "source_hashes": {"trainer": "frozen-source"},
                "checkpoint_sha256": file_hash(checkpoint),
                "best_validation_accuracy": 70.0 + trial["id"],
                "best_validation_loss": 1.0, "training_seconds": 20.0}
            write_json(directory / "summary.json", result)
        return plan

    def test_selection_requires_all_search_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                ranked_search(folder, make_plan())
            plan = self.fixture(folder)
            path = trial_directory(folder, 5, 42) / "summary.json"
            result = json.loads(path.read_text())
            result["status"] = "paused"
            write_json(path, result)
            with self.assertRaises(ValueError):
                create_selection(folder, plan)
            self.assertFalse((Path(folder) / "selection.json").exists())

    def test_rank_then_only_two_finalists_and_fixed_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.fixture(folder)
            rows, selection = create_selection(folder, plan)
            self.assertEqual(selection["trial_ids"], [11, 10])
            self.assertEqual(rows[0]["accuracy"], 81)
            check_confirmation(folder, plan, 11, 45, "fixed-split")
            check_confirmation(folder, plan, 10, 48, "fixed-split")
            for trial, seed, split in ((0, 45, "fixed-split"), (11, 44, "fixed-split"), (11, 45, "wrong")):
                with self.assertRaises(ValueError):
                    check_confirmation(folder, plan, trial, seed, split)

    def test_bad_initialization_and_checkpoint_are_rejected(self):
        for field, value in (("initial_state_sha256", "different"), ("checkpoint_sha256", "incorrect"),
                             ("source_hashes", {"trainer": "different"}),
                             ("split_sha256", "different"), ("test_evaluated", True)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                plan = self.fixture(folder)
                path = trial_directory(folder, 1, 42) / "summary.json"
                result = json.loads(path.read_text())
                result[field] = value
                write_json(path, result)
                with self.assertRaises(ValueError):
                    create_selection(folder, plan)

    def test_ties_are_deterministic_and_nan_rejected(self):
        rows = [{"trial_id": 2, "accuracy": 80.0, "loss": 0.5},
                {"trial_id": 1, "accuracy": 80.0, "loss": 0.5},
                {"trial_id": 0, "accuracy": 80.0, "loss": 0.6}]
        self.assertEqual([row["trial_id"] for row in rank_rows(rows)], [1, 2, 0])
        with self.assertRaises(ValueError):
            rank_rows([{"trial_id": 1, "accuracy": float("nan"), "loss": 1.0}])

    def test_no_test_or_router_loaders_and_no_downloads(self):
        source = (Path(__file__).parents[1] / "hpc/train_scratch_sweep.py").read_text()
        tree = ast.parse(source)
        datasets = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "CIFAR100"]
        self.assertEqual(len(datasets), 4)
        for call in datasets:
            flags = {keyword.arg: keyword.value for keyword in call.keywords}
            self.assertIs(flags["train"].value, True)
            if "download" in flags:
                self.assertIs(flags["download"].value, False)
        roles = [node.slice.value for node in ast.walk(tree)
                 if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                 and isinstance(node.value, ast.Subscript)
                 and isinstance(node.value.slice, ast.Constant) and node.value.slice.value == "indices"]
        self.assertEqual(sorted(roles), ["classifier_train", "classifier_validation"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "optional PyTorch CPU mechanical checks")
class TorchMechanicalTests(unittest.TestCase):
    def test_cifar_model_shape_and_reproducible_initialization(self):
        import torch
        from hpc.train_scratch_sweep import build_model, state_fingerprint
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = build_model().eval()
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertIsInstance(model.maxpool, torch.nn.Identity)
        first_hash = state_fingerprint(model.state_dict())
        torch.manual_seed(42)
        self.assertEqual(first_hash, state_fingerprint(build_model().state_dict()))
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 100))
        self.assertTrue(torch.isfinite(output).all())

    def test_all_optimizer_updates_and_epoch_boundary_resume(self):
        import torch
        from hpc.train_scratch_sweep import make_optimizer, make_scheduler
        torch.set_num_threads(1)
        for trial_id in (1, 4, 7, 9):
            with self.subTest(trial_id=trial_id):
                plan = make_plan()
                trial = plan["trials"][trial_id]
                torch.manual_seed(42)
                model = torch.nn.Linear(3, 2)
                optimizer = make_optimizer(model, trial, plan)
                scheduler = make_scheduler(optimizer, trial, plan)
                def step(net, opt, sched):
                    opt.zero_grad()
                    loss = net(torch.ones(2, 3)).square().mean()
                    loss.backward()
                    opt.step()
                    sched.step()
                step(model, optimizer, scheduler)
                snapshot = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                          "scheduler": scheduler.state_dict()})
                step(model, optimizer, scheduler)
                restored = torch.nn.Linear(3, 2)
                restored_optimizer = make_optimizer(restored, trial, plan)
                restored_scheduler = make_scheduler(restored_optimizer, trial, plan)
                restored.load_state_dict(snapshot["model"])
                restored_optimizer.load_state_dict(snapshot["optimizer"])
                restored_scheduler.load_state_dict(snapshot["scheduler"])
                step(restored, restored_optimizer, restored_scheduler)
                for original, resumed in zip(model.parameters(), restored.parameters()):
                    torch.testing.assert_close(original, resumed, rtol=0, atol=0)
                self.assertEqual(scheduler.get_last_lr(), restored_scheduler.get_last_lr())

    def test_cutmix_clipped_area_and_disabled_path(self):
        import numpy as np
        import torch
        from hpc.train_scratch_sweep import cutmix
        np.random.seed(42)
        torch.manual_seed(42)
        images = torch.stack([torch.full((3, 32, 32), float(index)) for index in range(8)])
        labels = torch.arange(8)
        untouched = images.clone()
        output, a, b, lam = cutmix(images.clone(), labels, 0, 1)
        torch.testing.assert_close(output, untouched)
        self.assertEqual(lam, 1)
        for _ in range(10):
            output, a, b, lam = cutmix(images.clone(), labels, 1, 1)
            self.assertTrue(0 <= lam <= 1)
            self.assertEqual(tuple(output.shape), (8, 3, 32, 32))
            for index in range(8):
                if a[index] != b[index]:
                    changed = (output[index, 0] != float(index)).sum().item() / 1024
                    self.assertAlmostEqual(changed, 1 - lam)


if __name__ == "__main__":
    unittest.main()
