"""Batch routing uses disjoint validation partitions and explicit budgets."""
import unittest
import numpy as np
from pc_tests.model_pipeline.batch_router_policy import split_validation, make_batches, batch_targets, calibrate, select


class BatchPolicyTests(unittest.TestCase):
    def test_stratified_disjoint_split(self):
        labels = np.repeat(np.arange(100), 50)
        fit, calibration = split_validation(labels)
        self.assertEqual(len(fit), 3000)
        self.assertEqual(len(calibration), 2000)
        self.assertFalse(set(fit) & set(calibration))

    def test_batch_targets_and_partial_tail(self):
        batches = make_batches(np.arange(5), 2, 42)
        self.assertEqual(sorted(map(len, batches)), [1, 2, 2])
        targets, counts, lengths = batch_targets(batches, np.ones((5, 2), bool), [2, 1], 0)
        np.testing.assert_array_equal(targets, 0)
        self.assertEqual(int(lengths.sum()), 5)
        self.assertEqual(int(counts[:, 0].sum()), 5)

    def test_accuracy_budget_and_cheaper_candidate(self):
        counts = np.array([[100, 99], [100, 99]])
        strict = calibrate([0, 0], counts, [100, 100], [2, 1], 0, .5)
        relaxed = calibrate([0, 0], counts, [100, 100], [2, 1], 0, 1)
        np.testing.assert_array_equal(select([0, 0], strict), [0, 0])
        np.testing.assert_array_equal(select([0, 0], relaxed), [1, 1])


if __name__ == '__main__':
    unittest.main()
