import unittest

import numpy as np

from model_pipeline.data import CIFAR100_MEAN, CIFAR100_STD
from pc_tests.model_pipeline.prepare_router_data import rgb_from_normalized


class RouterInputCacheTests(unittest.TestCase):
    def test_normalization_roundtrip_preserves_rgb_pixels(self):
        rgb = np.arange(256, dtype=np.uint8).reshape(1,1,16,16).repeat(3, axis=1)
        mean = np.array(CIFAR100_MEAN, np.float32)[None,:,None,None]
        std = np.array(CIFAR100_STD, np.float32)[None,:,None,None]
        normalized = (rgb.astype(np.float32)/255 - mean)/std
        np.testing.assert_array_equal(rgb_from_normalized(normalized), rgb)


if __name__ == '__main__':
    unittest.main()
