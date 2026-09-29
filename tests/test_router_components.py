import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROUTER_DIR = Path(__file__).resolve().parents[1] / "pc_tests" / "router_benchmark"
sys.path.insert(0, str(ROUTER_DIR))

from router_components import TinyCNN, split_positions  # noqa: E402


class RouterComponentTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_stratified(self):
        data = pd.DataFrame({"true_label": np.repeat(np.arange(10), 10)})
        first = split_positions(data, 42)
        second = split_positions(data, 42)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        fit, calibration, evaluation = first
        self.assertEqual((len(fit), len(calibration), len(evaluation)), (60, 20, 20))
        self.assertEqual(len(set(fit) | set(calibration) | set(evaluation)), 100)
        self.assertEqual(set(data.iloc[fit].true_label.value_counts()), {6})
        self.assertEqual(set(data.iloc[calibration].true_label.value_counts()), {2})
        self.assertEqual(set(data.iloc[evaluation].true_label.value_counts()), {2})

    def test_tinycnn_output_shape(self):
        model = TinyCNN().eval()
        with torch.inference_mode():
            output = model(torch.zeros(4, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (4,))


if __name__ == "__main__":
    unittest.main()
