import unittest
from collections import Counter
from pc_tests.model_pipeline.experiments.router_warmup_design import slots, ready, require_energy


class WarmupDesignTests(unittest.TestCase):
    def test_unique_and_balanced(self):
        rows=slots()
        self.assertEqual(len(rows),36)
        self.assertEqual(len({r['id'] for r in rows}),36)
        self.assertEqual(set(Counter((r['seed'],r['condition']) for r in rows).values()),{6})
        self.assertEqual(set(Counter((r['seed'],r['position'],r['condition']) for r in rows).values()),{2})

    def test_pair_order_balanced(self):
        rows=slots()
        self.assertEqual(Counter(r['condition'] for r in rows[::2]),{'cold':9,'warm':9})

    def test_gates(self):
        self.assertTrue(ready(60,4))
        self.assertFalse(ready(71,0))
        self.assertFalse(ready(60,32))
        self.assertFalse(ready(None,0))

    def test_energy(self):
        self.assertEqual(require_energy({'gpu_energy_j':5})['gpu_energy_j'],5)
        for value in (0,None,float('nan')):
            with self.assertRaises(RuntimeError):
                require_energy({'gpu_energy_j':value})


if __name__=='__main__':
    unittest.main()
