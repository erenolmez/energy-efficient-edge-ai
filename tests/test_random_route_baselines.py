import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pc_tests' / 'model_pipeline'))
from random_route_baselines import selection_by_length, random_schedule, cached_accuracy


class RandomRoutingTests(unittest.TestCase):
    def test_tail_and_image_frequencies_preserved(self):
        source = dict(selected_batches=[2, 2, 0], selected_images=[256, 144, 0])
        lengths = [128, 128, 128, 16]
        counts = selection_by_length(source, lengths)
        self.assertEqual(counts[128].tolist(), [2, 1, 0])
        self.assertEqual(counts[16].tolist(), [0, 1, 0])
        for seed in range(20):
            schedule = random_schedule(lengths, 3, seed, counts)
            self.assertEqual(schedule[-1], 1)
            self.assertEqual(np.bincount(schedule, minlength=3).tolist(), [2, 2, 0])
            images = np.bincount(schedule, weights=lengths, minlength=3)
            self.assertEqual(images.tolist(), source['selected_images'])

    def test_uniform_reproducible_and_all_candidates_possible(self):
        a = random_schedule([128] * 10000, 213, 42)
        self.assertTrue(np.array_equal(a, random_schedule([128] * 10000, 213, 42)))
        self.assertEqual(len(set(a)), 213)
        self.assertFalse(np.array_equal(a, random_schedule([128] * 10000, 213, 45)))

    def test_reject_inconsistent_source(self):
        with self.assertRaises(ValueError):
            selection_by_length(dict(selected_batches=[1, 1], selected_images=[128, 8]), [128, 16])

    def test_accuracy_uses_true_batch_positions(self):
        labels = np.array([0, 1, 0, 1])
        predictions = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(cached_accuracy(predictions, labels, [np.array([0, 2]), np.array([1, 3])], [0, 1]), 100)


if __name__ == '__main__':
    unittest.main()
