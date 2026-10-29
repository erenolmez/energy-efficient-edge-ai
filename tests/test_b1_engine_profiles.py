import unittest
import numpy as np
from pc_tests.model_pipeline.experiments.profile_b1_engines import choose_pool


class EngineProfileTests(unittest.TestCase):
    def test_pool_uses_fit_positions_not_other_labels(self):
        records=[dict(name=str(i),timings={'1':[dict(normalized_host_input_to_logits_batch_ms=float(i+1),gpu_energy_j_per_image=float(i+1))]}) for i in range(8)]
        rng=np.random.default_rng(42)
        correct=rng.integers(0,2,(100,8)).astype(bool)
        positions=np.arange(50)
        a=choose_pool(records,correct,positions,'7')
        correct[50:]=~correct[50:]
        b=choose_pool(records,correct,positions,'7')
        self.assertEqual([r['name'] for r in a],[r['name'] for r in b])
        self.assertEqual(len(a),6)
        self.assertEqual(len({r['name'] for r in a}),6)
        self.assertEqual(a[0]['name'],'7')


if __name__=='__main__':
    unittest.main()
