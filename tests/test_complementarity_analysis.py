import unittest

import numpy as np

from hpc.complementarity_analysis import cost_oracle, debatch_eval, make_batches, oracle_choice


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

    def test_debatch_costs_sub_batches_and_respects_budget(self):
        labels = np.repeat(np.arange(4), 8)
        correct = np.zeros((2, 32), dtype=bool)
        correct[0] = True             # expensive reference always right
        correct[1, :16] = True        # cheap model right on half
        sizes = np.array([1, 8, 16])
        curves = {'latency': (sizes, np.array([[10., 40., 70.], [1., 4., 7.]]))}
        costs = {'latency': {b: curves['latency'][1][:, i] / b for i, b in enumerate(sizes)}}
        r = debatch_eval(correct, labels, [0, 1], costs, curves, 16, 3,
                         np.random.default_rng(0), 0.0, ref=0)
        self.assertAlmostEqual(r['oracle']['accuracy'], 100.0)
        self.assertGreater(r['oracle']['savings_pct']['latency'], 0)
        self.assertAlmostEqual(r['budget']['accuracy'], 100.0)


if __name__ == '__main__':
    unittest.main()
