import hashlib
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
from pc_tests.model_pipeline.routers.correctness.compact_cpu import features,thumbnails,fit
from pc_tests.model_pipeline.runtime.compact_switching import ProfiledLRU,VerifiedHostStore,select_cache_aware,simulate_policy,calibrate_cache_aware


class CompactTests(unittest.TestCase):
    def test_descriptors_and_no_cross_image_information(self):
        rgb=np.stack((np.zeros((3,128,128),dtype=np.uint8),np.full((3,128,128),255,dtype=np.uint8)))
        self.assertEqual(thumbnails(rgb).shape,(2,3,8,8))
        self.assertEqual(features(rgb).shape,(2,12))
        np.testing.assert_allclose(features(rgb)[0],features(rgb[:1])[0])
        np.testing.assert_allclose(features(rgb)[1,:3],np.ones(3))
        self.assertTrue(np.isfinite(features(rgb)).all())

    def test_four_fit_families_return_correctness_vectors(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(80,12)).astype('float32')
        y=np.stack((x[:,0]>0,x[:,1]<0,np.ones(80),np.zeros(80)),axis=1).astype('float32')
        for method in ('logistic','tree_d3','boosted_d2','thumbnail_mlp16'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model=fit(method,x,y,boost_trees=2,mlp_epochs=2)
            q=model.predict_inputs(x[:3])
            self.assertEqual(q.shape,(3,4))
            self.assertTrue(((q>=0)&(q<=1)).all())
            if method!='thumbnail_mlp16':
                np.testing.assert_allclose(q[:,2],1)
                np.testing.assert_allclose(q[:,3],0)

    def test_memory_budget_and_lru(self):
        records=[dict(profiled_resident_bytes=10) for _ in range(3)]
        cache=ProfiledLRU(records,3,20,lambda r:object(),lambda:None,lambda:None)
        cache[0];cache[1];cache[0];cache[2]
        self.assertEqual(list(cache.resident),( [0,2] ))
        self.assertEqual(cache.loads,3)
        self.assertEqual(cache.evictions,1)
        oversized=ProfiledLRU(records,3,5,lambda r:object(),lambda:None,lambda:None)
        with self.assertRaises(RuntimeError):
            oversized[0]

    def test_cache_awareness_uses_miss_cost_not_test_labels(self):
        self.assertEqual(select_cache_aware([.90,.91],[1,1],[10,10],[0],.1),0)
        self.assertEqual(select_cache_aware([.90,.99],[1,1],[10,10],[0],.001),1)
        with self.assertRaises(ValueError):
            select_cache_aware([.9],[1],[-1],[],.1)

    def test_verified_immutable_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'engine'
            path.write_bytes(b'known trusted test bytes')
            record=dict(name='a',engine_path=str(path),engine_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            store=VerifiedHostStore([record])
            path.write_bytes(b'changed after setup')
            self.assertEqual(store.get('a'),b'known trusted test bytes')
            with self.assertRaises(RuntimeError):
                VerifiedHostStore([record])

    def test_sequential_cache_simulation_and_warm_state(self):
        q=np.array([[.99,.1],[.1,.99],[.99,.1]])
        counts=np.ones((3,2))
        cold=simulate_policy(q,counts,np.ones(3),[1.,1.],[10.,10.],1,0.)
        warm=simulate_policy(q,counts,np.ones(3),[1.,1.],[10.,10.],2,0.,initial=(0,1))
        self.assertEqual(cold['loads'],3)
        self.assertEqual(cold['estimated_total_cost'],33.)
        self.assertEqual(warm['loads'],0)
        self.assertEqual(warm['estimated_total_cost'],3.)

    def test_cache_calibration_enforces_accuracy_and_drops_schedule(self):
        q=np.array([[.99,.95],[.99,.95]])
        counts=np.ones((2,2))
        policy=calibrate_cache_aware(q,counts,np.ones(2),[10.,1.],[1.,1.],2,0,0.)
        self.assertFalse(policy['fallback'])
        self.assertEqual(policy['accuracy'],100.)
        self.assertNotIn('choices',policy)


if __name__=='__main__':
    unittest.main()
