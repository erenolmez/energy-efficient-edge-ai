import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).parents[1] / "pc_tests" / "router_benchmark"
sys.path.insert(0, str(EXPERIMENT))

from routing_policy_v2 import calibrate_utility, select_utility  # noqa: E402


class UtilityPolicyTests(unittest.TestCase):
    def test_uses_explicit_p0_and_falls_back_when_predictions_are_wrong(self):
        # p0 is column zero, while cheaper models look better to the predictor.
        predictions = np.tile([0.1, 0.9, 0.8], (10, 1))
        correct = np.tile([True, False, False], (10, 1))
        costs = np.array([10.0, 1.0, 3.0])
        policy = calibrate_utility(predictions, correct, costs, baseline_index=0)
        self.assertTrue(policy["fallback"])
        self.assertEqual(policy["minimum_accuracy"], 0.99)
        np.testing.assert_array_equal(
            select_utility(predictions, costs, policy, 0), np.zeros(10, dtype=int),
        )

    def test_minimizes_actual_cost_subject_to_accuracy_budget(self):
        predictions = np.tile([0.6, 0.95, 0.8], (10, 1))
        # Model 2 is between the weakest model and p0 in both price and quality.
        correct = np.ones((10, 3), dtype=bool)
        correct[:4, 0] = False
        correct[0, 2] = False
        costs = np.array([1.0, 10.0, 3.0])
        policy = calibrate_utility(predictions, correct, costs, 1, max_drop=0.1)
        self.assertFalse(policy["fallback"])
        self.assertAlmostEqual(policy["calibration_accuracy"], 0.9)
        self.assertEqual(policy["cost"], 3.0)
        np.testing.assert_array_equal(
            select_utility(predictions, costs, policy, 1), np.full(10, 2),
        )

    def test_reordering_models_preserves_policy_and_model_selections(self):
        predictions = np.array([[0.9, 0.95, 0.6], [0.1, 0.95, 0.9],
                                [0.9, 0.95, 0.6], [0.1, 0.95, 0.9]])
        correct = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1]])
        costs = np.array([2.0, 10.0, 4.0])
        policy = calibrate_utility(predictions, correct, costs, 1, max_drop=0)
        selections = select_utility(predictions, costs, policy, 1)
        permutation = np.array([1, 2, 0])
        permuted_policy = calibrate_utility(
            predictions[:, permutation], correct[:, permutation], costs[permutation],
            0, max_drop=0,
        )
        self.assertEqual(policy, permuted_policy)
        permuted_selections = select_utility(
            predictions[:, permutation], costs[permutation], permuted_policy, 0,
        )
        np.testing.assert_array_equal(selections, permutation[permuted_selections])
        self.assertEqual(policy["cost"], 3.0)

    def test_delta_predictions_make_same_choices_as_absolute_predictions(self):
        predictions = np.array([[0.8, 0.9, 0.3], [0.2, 0.7, 0.8]])
        correct = np.array([[1, 1, 0], [0, 1, 1]])
        costs = np.array([2.0, 10.0, 4.0])
        deltas = predictions - predictions[:, [1]]
        policy = calibrate_utility(predictions, correct, costs, 1, max_drop=0)
        delta_policy = calibrate_utility(deltas, correct, costs, 1, max_drop=0)
        np.testing.assert_array_equal(
            select_utility(predictions, costs, policy, 1),
            select_utility(deltas, costs, delta_policy, 1),
        )

    def test_equal_cost_and_accuracy_prefer_simple_p0_fallback(self):
        predictions = np.tile([0.9, 0.1], (3, 1))
        correct = np.ones((3, 2), dtype=bool)
        policy = calibrate_utility(predictions, correct, [2, 2], 1)
        self.assertTrue(policy["fallback"])

    def test_higher_accuracy_wins_when_actual_cost_is_equal(self):
        predictions = np.tile([0.9, 0.1], (3, 1))
        correct = np.array([[1, 1], [1, 1], [1, 0]])
        policy = calibrate_utility(predictions, correct, [2, 2], 1)
        self.assertFalse(policy["fallback"])
        self.assertEqual(policy["calibration_accuracy"], 1.0)

    def test_rejects_invalid_inputs(self):
        predictions = np.ones((3, 2))
        correct = np.ones((3, 2), dtype=bool)
        invalid_cases = [
            (predictions.ravel(), correct, [1, 2], 1, 0.01),
            (predictions, correct[:, :1], [1, 2], 1, 0.01),
            (predictions, np.full((3, 2), 2), [1, 2], 1, 0.01),
            (predictions, correct, [1], 1, 0.01),
            (predictions, correct, [0, 2], 1, 0.01),
            (predictions, correct, [1, np.nan], 1, 0.01),
            (np.full((3, 2), np.nan), correct, [1, 2], 1, 0.01),
            (predictions, correct, [1, 2], 2, 0.01),
            (predictions, correct, [1, 2], True, 0.01),
            (predictions, correct, [1, 2], 1, np.nan),
            (predictions, correct, [1, 2], 1, -0.01),
            (np.ones((0, 2)), np.ones((0, 2)), [1, 2], 1, 0.01),
        ]
        for args in invalid_cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                calibrate_utility(*args)

    def test_rejects_invalid_policy_and_accepts_empty_inference(self):
        for policy in [{}, {"lambda": np.nan, "fallback": False},
                       {"lambda": -1, "fallback": False},
                       {"lambda": 1, "fallback": "yes"}]:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                select_utility(np.ones((3, 2)), [1, 2], policy, 1)
        result = select_utility(
            np.ones((0, 2)), [1, 2], {"lambda": 0.1, "fallback": False}, 1,
        )
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
