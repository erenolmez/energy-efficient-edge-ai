import unittest
import numpy as np
from pc_tests.model_pipeline.routers.correctness.ablation_design import nested_subsets,positive_weights

class DesignTests(unittest.TestCase):
    def test_nested_balanced_reproducible(self):
        labels=np.repeat(np.arange(100),50);positions=np.concatenate([np.arange(i*50,i*50+30) for i in range(100)])
        subsets=nested_subsets(labels,positions,[600,1000,2000,3000])
        for size,values in subsets.items():
            self.assertEqual(len(values),size);self.assertTrue(np.all(np.bincount(labels[values])==size//100))
            self.assertTrue(set(values)<=set(positions))
            self.assertTrue(np.array_equal(values,nested_subsets(labels,positions,[600,1000,2000,3000])[size]))
        self.assertTrue(set(subsets[1000])<=set(subsets[2000])<=set(subsets[3000]))
        self.assertEqual(len(set(positions)-set(subsets[600])),2400)
    def test_constant_heads_finite(self):
        weights=positive_weights([[0,1,0],[0,1,1]])
        self.assertTrue(np.isfinite(weights).all());self.assertEqual(weights[0],2);self.assertAlmostEqual(weights[1],.1,places=6);self.assertEqual(weights[2],1)
    def test_invalid_design(self):
        with self.assertRaises(ValueError):nested_subsets([0,1],[0,1],[3])
        with self.assertRaises(ValueError):positive_weights([[.5]])

if __name__=='__main__':unittest.main()
