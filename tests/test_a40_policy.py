import unittest

import numpy as np

from hpc.a40_policy import lock_router


class LockRouterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.correct = np.zeros((2, 64), dtype=bool)
        self.correct[0] = True                      # expensive reference always right
        self.correct[1, :48] = True                 # cheap model right on 3/4
        self.batchings = [rng.permutation(64).reshape(8, 8) for _ in range(5)]
        self.costs = np.array([10.0, 1.0])

    def test_informative_router_saves_within_budget(self):
        prob = self.correct.astype(float) * 0.8 + 0.1
        res = lock_router(prob, self.correct, self.batchings, self.costs, 0, target_acc=90.0)
        self.assertIsNotNone(res)
        self.assertGreaterEqual(res["accuracy"], 90.0)
        self.assertLess(res["cost"], 8 * 10.0)

    def test_infeasible_target_returns_none(self):
        prob = np.full((2, 64), 0.5)
        prob[0] = 0.4  # router always prefers the cheap model
        self.assertIsNone(lock_router(prob, self.correct, self.batchings, self.costs, 0, target_acc=99.0,
                                      lambdas=np.array([0.0])))


if __name__ == "__main__":
    unittest.main()
