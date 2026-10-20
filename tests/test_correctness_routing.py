import unittest
import numpy as np
import torch
from pc_tests.model_pipeline.correctness_router import CorrectnessRouter, BatchCorrectnessRouter
from pc_tests.model_pipeline.correctness_policy import select_correctness, calibrate_correctness


class CorrectnessRoutingTests(unittest.TestCase):
    def test_213_outputs_and_true_batch_mean(self):
        torch.manual_seed(42)
        model = CorrectnessRouter().eval()
        images = torch.randn(3, 3, 128, 128)
        with torch.no_grad():
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (3, 213))
            torch.testing.assert_close(BatchCorrectnessRouter(model)(images), logits.sigmoid().mean(0, keepdim=True))

    def test_complementary_models_not_scalar_order(self):
        q = np.array([[.99, .1], [.1, .99]])
        np.testing.assert_array_equal(select_correctness(q, [2, 1], 0), [0, 1])
        policy = calibrate_correctness(q, [[1, 0], [1, 1]], [1, 1], [2, 1], 0, 0)
        self.assertFalse(policy['fallback'])
        self.assertEqual(policy['calibration_accuracy'], 100)

    def test_fallback_and_invalid_values(self):
        policy = calibrate_correctness([[.1, .9]], [[100, 0]], [100], [2, 1], 0, .5)
        self.assertTrue(policy['fallback'])
        with self.assertRaises(ValueError):
            select_correctness([[float('nan'), 0]], [1, 2], 0)
        with self.assertRaises(ValueError):
            calibrate_correctness([[.1, .9]], [[101, 0]], [100], [2, 1], 0, .5)


if __name__ == '__main__':
    unittest.main()
