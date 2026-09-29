import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).parents[1] / "pc_tests" / "router_benchmark"
sys.path.insert(0, str(EXPERIMENT))

from features import (FEATURE_NAMES, ENHANCED_FEATURE_NAMES, extract_features,
                      extract_enhanced_features)  # noqa: E402


class FeatureTests(unittest.TestCase):
    def test_constant_image_has_no_edges_or_texture(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = extract_features(image)
        self.assertEqual(set(result), set(FEATURE_NAMES))
        self.assertAlmostEqual(result["edge_density"], 0.0)
        self.assertAlmostEqual(result["gradient_mean"], 0.0)
        self.assertAlmostEqual(result["laplacian_variance"], 0.0)
        self.assertAlmostEqual(result["gray_entropy"], 0.0)

    def test_checkerboard_contains_edges(self):
        board = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
        image = np.repeat(board[..., None], 3, axis=2)
        result = extract_features(image)
        self.assertGreater(result["laplacian_variance"], 0.0)
        self.assertGreater(result["gray_entropy"], 0.0)

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_features(np.zeros((32, 32), dtype=np.uint8))

    def test_enhanced_features_are_complete_and_finite(self):
        image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        result = extract_enhanced_features(image)
        self.assertEqual(list(result), ENHANCED_FEATURE_NAMES)
        self.assertTrue(np.isfinite(list(result.values())).all())


if __name__ == "__main__":
    unittest.main()
