import unittest
import numpy as np
from pc_tests.model_pipeline.routers.correctness.compact_cpu import fit,features,thumbnails
from pc_tests.model_pipeline.runtime.compact_subset import predict

class SubsetTests(unittest.TestCase):
    def test_subset_matches_full(self):
        rng=np.random.default_rng(41)
        rgb=rng.integers(0,256,size=(40,3,128,128),dtype=np.uint8)
        y=rng.integers(0,2,size=(40,5));y[:,0]=1
        for method in ('logistic','tree_d3','thumbnail_mlp16','boosted_d2'):
            x=thumbnails(rgb).reshape(40,192) if method=='thumbnail_mlp16' else features(rgb)
            model=fit(method,x,y,boost_trees=2,mlp_epochs=2)
            self.assertTrue(np.allclose(predict(model,rgb[:3],[3,0,2]),model.predict_rgb(rgb[:3])[:,[3,0,2]],atol=1e-6))

if __name__=='__main__':unittest.main()
