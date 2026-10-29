import unittest
import numpy as np
import torch
from pc_tests.model_pipeline.routers.correctness.device_selection import DeviceCostSelection
from pc_tests.model_pipeline.routers.correctness.correctness_policy import select_correctness


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_arithmetic_matches_historical_numpy(self):
        rng=np.random.default_rng(42)
        q=rng.random((100,213)).astype(np.float32)
        costs=rng.uniform(.01,10,213)
        for penalty in (0.,.001,.1,2.):
            expected=select_correctness(q,costs,penalty)
            actual=DeviceCostSelection(costs,penalty)(torch.from_numpy(q)).numpy()
            np.testing.assert_array_equal(actual,expected)

    def test_ties_and_fallback(self):
        q=torch.ones(1,3)*.5
        self.assertEqual(DeviceCostSelection([1,1,1],0.)(q).tolist(),[0])
        self.assertEqual(DeviceCostSelection([1,1,1],.1,baseline_index=2)(q).tolist(),[2])

    def test_invalid_constants(self):
        for costs in ([0,1],[float('nan'),1],[]):
            with self.assertRaises(ValueError):
                DeviceCostSelection(costs,.1)


if __name__=='__main__':
    unittest.main()
