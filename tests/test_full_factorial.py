import copy
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from hpc.full_factorial_protocol import (
    ARCHITECTURES, make_plan, freeze_plan, checked_plan, build_model, trial_directory,
)
from hpc.scratch_recipes import make_plan as old_plan


class FullGridTests(unittest.TestCase):
    def test_complete_grid(self):
        plan = make_plan()
        self.assertEqual(len(plan["trials"]), 1728)
        self.assertEqual(plan["maximum_training_epochs"], 345600)
        self.assertEqual([x["id"] for x in plan["trials"]], list(range(1728)))
        for arch in ARCHITECTURES:
            for opt in ("sgd", "adam", "adamw"):
                rows = [x for x in plan["trials"] if x["architecture"] == arch and x["optimizer"] == opt]
                self.assertEqual(len(rows), 72)
                keys = {(x["lr"], x["weight_decay"], x["schedule"], x["cutmix_alpha"], x["seed"]) for x in rows}
                expected = set(product(sorted({x["lr"] for x in rows}),
                                       sorted({x["weight_decay"] for x in rows}),
                                       ("cosine", "step80_120"), (0., 1.), (42, 45, 48)))
                self.assertEqual(keys, expected)
        self.assertIsNone(plan["weights"])
        self.assertEqual(plan["data_roles"], old_plan()["data_roles"])

    def test_old_protocol_unchanged(self):
        self.assertEqual(old_plan()["sha256"], "97b5e5987abaff1887b9b82f60961bb4526615b4c60c378ec7551a3046d8802f")

    def test_frozen_plan_and_unique_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            original = freeze_plan(path)
            self.assertEqual(original, checked_plan(path))
            self.assertEqual(original, freeze_plan(path))
            paths = {trial_directory(folder, x["id"], x["seed"]) for x in original["trials"]}
            self.assertEqual(len(paths), 1728)
            changed = copy.deepcopy(original)
            changed["epochs"] = 20
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                checked_plan(path)

    def test_every_model_forward_backward_cifar32(self):
        import torch
        torch.set_num_threads(2)
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                torch.manual_seed(42)
                model = build_model(architecture)
                model.train()
                output = model(torch.randn(2, 3, 32, 32))
                self.assertEqual(tuple(output.shape), (2, 100))
                loss = torch.nn.functional.cross_entropy(output, torch.tensor([1, 2]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                del model, grads, output, loss

    def test_scheduler_and_optimizer_candidates(self):
        import torch
        from hpc.train_full_factorial import make_optimizer, make_scheduler
        plan = make_plan()
        for optimizer in ("sgd", "adam", "adamw"):
            for schedule in ("cosine", "step80_120"):
                trial = next(x for x in plan["trials"] if x["optimizer"] == optimizer and x["schedule"] == schedule)
                model = torch.nn.Linear(3, 100)
                opt = make_optimizer(model, trial, plan)
                sched = make_scheduler(opt, trial, plan)
                model(torch.ones(2, 3)).sum().backward()
                opt.step()
                sched.step()
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
