import unittest

import torch

from pc_tests.model_pipeline.gpu_batch_inputs import SharedBatchInputs


class BatchInputTests(unittest.TestCase):
    def test_normalization_features_and_grid_shapes(self):
        buffers = SharedBatchInputs(8,device='cpu')
        rgb = torch.zeros(8,3,128,128,dtype=torch.uint8)
        value = buffers.prepare(rgb)
        self.assertEqual(value.shape,(8,3,128,128))
        self.assertEqual(buffers.features(value).shape,(13,))
        self.assertEqual(buffers.grid(value).shape,(1,3,32,32))
        self.assertTrue(torch.isfinite(buffers.features(value)).all())
        self.assertAlmostEqual(value[0,0,0,0].item(),-.5071/.2675,places=5)

    def test_reuses_model_buffer(self):
        buffers = SharedBatchInputs(8,device='cpu')
        rgb = torch.zeros(8,3,128,128,dtype=torch.uint8)
        first = buffers.prepare(rgb)
        second = buffers.prepare(rgb)
        self.assertEqual(first.data_ptr(),second.data_ptr())

    def test_per_image_features_match_single_image_batch_features(self):
        buffers = SharedBatchInputs(3,device='cpu')
        torch.manual_seed(42)
        rgb = torch.randint(0,256,(3,3,128,128),dtype=torch.uint8)
        normalized = buffers.prepare(rgb)
        features = buffers.image_features(normalized)
        self.assertEqual(features.shape, (3,13))
        for index in range(3):
            torch.testing.assert_close(features[index], buffers.features(normalized[index:index+1]))


if __name__ == '__main__':
    unittest.main()
