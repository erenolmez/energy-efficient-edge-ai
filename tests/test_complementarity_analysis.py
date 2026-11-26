import unittest

import numpy as np

from hpc.complementarity_analysis import cost_oracle, make_batches, oracle_choice


class ComplementarityTests(unittest.TestCase):
    def test_accuracy_oracle_prefers_cheaper_model_on_ties(self):
        counts = np.array([[5., 3.], [5., 4.]])
        self.assertEqual(oracle_choice(counts, np.array([2., 1.]), 0.0).tolist(), [1, 1])

    def test_cost_oracle_meets_target_and_saves(self):
        counts = np.array([[8., 8., 8.], [8., 6., 2.]])  # model 0 expensive, model 1 cheap
        cost = np.array([10., 1.])
        pick = cost_oracle(counts, cost, target_correct=22)
        cols = np.arange(3)
        self.assertGreaterEqual(counts[pick, cols].sum(), 22)
        self.assertEqual(pick.tolist(), [1, 1, 0])
        self.assertIsNone(cost_oracle(counts, cost, target_correct=25))

    def test_batches_are_full_and_class_coherent(self):
        labels = np.repeat(np.arange(10), 20)
        rng = np.random.default_rng(0)
        shuffled = make_batches(labels, 16, rng)
        self.assertEqual(shuffled.shape, (12, 16))
        self.assertEqual(len(set(shuffled.ravel())), 192)
        coherent = make_batches(labels, 8, rng, coherent=2)
        for batch in coherent:
            self.assertEqual(len(set(labels[batch])), 2)
        self.assertIsNone(make_batches(labels, 64, rng, coherent=2))


if __name__ == '__main__':
    unittest.main()
