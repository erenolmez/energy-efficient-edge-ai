import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hpc import a40_crossfit as cf

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class FoldTests(unittest.TestCase):
    def test_folds_are_stratified_disjoint_and_deterministic(self):
        labels = np.repeat(np.arange(100), 500)
        rng = np.random.default_rng(1)
        pool = np.concatenate([rng.choice(np.flatnonzero(labels == c), 50, replace=False) for c in range(100)])
        fold = cf.make_folds(pool, labels, 5, 42)
        self.assertTrue(np.array_equal(fold, cf.make_folds(pool, labels, 5, 42)))
        self.assertFalse(np.array_equal(fold, cf.make_folds(pool, labels, 5, 43)))
        for f in range(5):
            self.assertEqual((fold == f).sum(), 1000)
            self.assertTrue((np.bincount(labels[pool[fold == f]], minlength=100) == 10).all())

    def test_uneven_class_is_refused(self):
        labels = np.repeat(np.arange(100), 500)
        pool = np.concatenate([np.flatnonzero(labels == c)[:7] for c in range(100)])
        with self.assertRaises(RuntimeError):
            cf.make_folds(pool, labels, 5)

    def test_table_and_minibatches(self):
        labels = np.array([0, 0, 1, 1])
        correct = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=float)
        tab = cf.class_table(correct, labels, np.arange(4), classes=2)
        self.assertTrue(np.allclose(tab, [[1, .5], [0, .5]]))
        self.assertEqual(cf.minibatches(3000), 48)
        self.assertEqual(cf.minibatches(4000), 64)
        self.assertEqual(cf.minibatches(9000), 144)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class CrossfitLoopTests(unittest.TestCase):
    def test_every_pool_image_gets_one_out_of_fold_score(self):
        import torch
        n_per = 10  # pool images per class -> 2 per fold
        labels = np.repeat(np.arange(100), 20)  # 2000 "CIFAR" images
        idx = {c: np.flatnonzero(labels == c) for c in range(100)}
        pool = np.concatenate([idx[c][:n_per] for c in range(100)])
        val = np.concatenate([idx[c][n_per:n_per + 2] for c in range(100)])
        ctrain = np.concatenate([idx[c][n_per + 2:] for c in range(100)])
        rng = np.random.default_rng(0)
        correct = (rng.random((len(labels), 8)) < 0.7).astype(np.float32)
        train_ix = np.concatenate([val, pool])
        rows = {int(i): r for r, i in enumerate(train_ix)}
        x16, x32 = torch.randn(len(train_ix), 3, 16, 16), torch.randn(len(train_ix), 3, 32, 32)
        data = dict(pool=pool, val=val, ctrain=ctrain, labels=labels, correct=correct,
                    fold=cf.make_folds(pool, labels, 5, 42), rows=rows,
                    train_x={"tiny16": x16, "wide32": x32, "res32": x32},
                    score_x={"tiny16": torch.randn(len(pool), 3, 16, 16), "wide32": torch.randn(len(pool), 3, 32, 32),
                             "res32": torch.randn(len(pool), 3, 32, 32)})
        sizes = []
        real = cf.train_net

        def spy(task, arch, x, t, y, table, seed, epochs):
            sizes.append(len(x))
            return real(task, arch, x, t, y, table, seed, epochs)

        cf.train_net = spy
        try:
            with tempfile.TemporaryDirectory() as d:
                oof, timing = cf.crossfit(data, seeds=(42,), epochs=1, log=lambda s: None, deploy_dir=d)
                self.assertEqual(len(list(Path(d).glob("*.pt"))), len(cf.NETS))
        finally:
            cf.train_net = real
        self.assertEqual(set(oof), set(cf.CF_ROUTERS))
        for fam, by_seed in oof.items():
            p = by_seed[42]
            self.assertEqual(p.shape, (len(pool), 8))
            self.assertTrue(np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1, fam)
        # fold fits: 4/5 of the pool (+ validation for the val ablation); deployment: all of it
        fold_fit, full_fit = 4 * len(pool) // 5, len(pool)
        self.assertEqual(sorted(set(sizes)), sorted({fold_fit, fold_fit + len(val), full_fit, full_fit + len(val)}))
        self.assertEqual(len(timing), 6 * len(cf.NETS))


class PolicyRolesTests(unittest.TestCase):
    def test_pool_roles_and_superset_scores(self):
        from hpc.a40_policy import load_scores, load_stage_a
        names = ["m%d" % i for i in range(7)] + ["resnet101"]
        rng = np.random.default_rng(0)
        role = np.repeat([0, 1, 2], [50, 30, 20]).astype(np.int8)
        index = rng.permutation(1000)[:100]
        label = rng.integers(0, 100, 100)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.savez(d / "predictions.npz", models=np.array(names),
                     roles=np.array(["classifier_validation", "router_train", "router_calibration"]),
                     index=index, role=role, label=label, pred_trt_fp16_gpu=np.tile(label, (8, 1)))
            rows = [dict(model=n, variant="trt_fp16", batch=b, latency_ms=1.0, energy_mj=1.0)
                    for n in names for b in (8, 16, 32, 64, 128)]
            (d / "costs_pipeline.json").write_text(json.dumps(dict(meta=dict(test_evaluated=False), rows=rows)))
            cal = load_stage_a(d)
            pool = load_stage_a(d, ("router_train", "router_calibration"))
            self.assertEqual(len(cal["index"]), 20)
            self.assertEqual(len(pool["index"]), 50)
            self.assertTrue(pool["pool"].startswith("router-train + router-calibration"))
            np.savez(d / "s.npz", index=pool["index"], models=np.array(names),
                     correct_prob=rng.random((50, 8)))
            prob, _ = load_scores(d / "s.npz", cal["names"], cal["index"])  # 50-image file, 20 requested
            self.assertEqual(prob.shape, (8, 20))
            with self.assertRaises(RuntimeError):
                load_scores(d / "s.npz", cal["names"], load_stage_a(d, ("classifier_validation",))["index"])
            with self.assertRaises(RuntimeError):
                load_stage_a(d, ("test",))


if __name__ == "__main__":
    unittest.main()
