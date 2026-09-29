import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "pc_tests" / "router_benchmark"))
from router_features_v2 import FEATURE_SETS, extract_feature_vector, extract_matrix


class RouterFeatureV2Tests(unittest.TestCase):
    def test_uint8_and_normalized_float_agree_including_dark_images(self):
        rng = np.random.default_rng(3)
        for maximum in (1, 255):
            image = rng.integers(0, maximum + 1, (32, 32, 3), dtype=np.uint8)
            for feature_set in FEATURE_SETS:
                np.testing.assert_allclose(
                    extract_feature_vector(image, feature_set),
                    extract_feature_vector(image.astype(np.float32) / 255, feature_set),
                    rtol=1e-5, atol=1e-6,
                )

    def test_flat_image_and_checkerboard_capture_sharpness(self):
        flat = np.full((32, 32, 3), 128, np.uint8)
        board = np.repeat(((np.indices((32, 32)).sum(axis=0) % 2) * 255)
                          .astype(np.uint8)[..., None], 3, axis=2)
        names = FEATURE_SETS["compact"]
        flat_features = dict(zip(names, extract_feature_vector(flat)))
        sharp_features = dict(zip(names, extract_feature_vector(board)))
        self.assertEqual(flat_features["gray_entropy"], 0)
        self.assertEqual(flat_features["laplacian_variance"], 0)
        self.assertGreater(sharp_features["laplacian_variance"], 1)
        self.assertGreater(sharp_features["gray_entropy"], 0)

    def test_texture_histograms_and_hog_have_defined_normalization(self):
        ramp = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
        vector = extract_feature_vector(np.repeat(ramp[..., None], 3, axis=2), "texture")
        result = dict(zip(FEATURE_SETS["texture"], vector))
        for channel in "rgb":
            self.assertAlmostEqual(sum(result[f"rgb_{channel}_hist_{i}"]
                                       for i in range(8)), 1)
        for row in range(2):
            for col in range(2):
                self.assertAlmostEqual(sum(result[f"lbp_quadrant_{row}_{col}_bin_{i}"]
                                           for i in range(10)), 1)
        # A horizontal intensity ramp gives horizontal gradients (orientation 0).
        self.assertGreater(result["hog_block_0_0_cell_0_0_bin_0"], 0.4)
        self.assertEqual(result["hog_block_0_0_cell_0_0_bin_4"], 0)

    def test_dimensions_names_finiteness_and_batch_order(self):
        rng = np.random.default_rng(42)
        images = rng.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
        for name, names in FEATURE_SETS.items():
            matrix = extract_matrix(images, name)
            self.assertEqual(matrix.shape, (3, len(names)))
            self.assertEqual(len(set(names)), len(names))
            self.assertEqual(matrix.dtype, np.float32)
            self.assertTrue(np.isfinite(matrix).all())
            np.testing.assert_array_equal(matrix[1], extract_feature_vector(images[1], name))
            self.assertEqual(extract_matrix([], name).shape, (0, len(names)))

    def test_quadrant_color_is_spatially_sensitive(self):
        image = np.zeros((32, 32, 3), np.uint8)
        image[:16, :16, 0] = 255
        result = dict(zip(FEATURE_SETS["compact"], extract_feature_vector(image)))
        self.assertEqual(result["quadrant_0_0_r_mean"], 1)
        self.assertEqual(result["quadrant_0_1_r_mean"], 0)
        self.assertEqual(result["quadrant_0_0_g_mean"], 0)

    def test_bad_values_dimensions_and_unknown_sets_are_rejected(self):
        for bad in (np.zeros((32, 32)), np.zeros((0, 0, 3)),
                    np.zeros((31, 32, 3)), np.full((32, 32, 3), np.nan),
                    np.full((32, 32, 3), -1), np.full((32, 32, 3), 256)):
            with self.assertRaises(ValueError):
                extract_feature_vector(bad)
        with self.assertRaises(ValueError):
            extract_feature_vector(np.zeros((32, 32, 3)), "missing")


if __name__ == "__main__":
    unittest.main()
