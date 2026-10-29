import unittest
import numpy as np
from pc_tests.model_pipeline.runtime.grouped_dispatch import groups

class GroupTests(unittest.TestCase):
    def test_partition_reassembly(self):
        choices=np.array([2,1,2,0,1,2,0,1]);output=np.empty(8,dtype=int)
        parts=groups(choices,3)
        self.assertEqual(sorted(np.concatenate([p for _,p in parts]).tolist()),list(range(8)))
        for model,positions in parts:output[positions]=model
        self.assertTrue(np.array_equal(output,choices))
    def test_invalid(self):
        for bad in ([],[3],[.5],[-1]):
            with self.assertRaises(ValueError):groups(bad,3)

if __name__=='__main__':unittest.main()
