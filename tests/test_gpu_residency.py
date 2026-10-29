from pathlib import Path
import tempfile
import unittest
from pc_tests.model_pipeline.experiments.probe_gpu_residency import order_records,admission_estimate


class ResidencyTests(unittest.TestCase):
    def test_three_frozen_orders_preserve_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            records=[]
            for i,size in enumerate((10,20,30,40)):
                path=Path(directory)/str(i)
                path.write_bytes(b'x'*size)
                records.append(dict(name=str(i),architecture='a' if i<2 else 'b',engine_path=str(path)))
            orders=order_records(records)
            self.assertEqual([r['name'] for r in orders['smallest_first']],['0','1','2','3'])
            self.assertEqual([r['name'] for r in orders['largest_first']],['3','2','1','0'])
            self.assertEqual([r['name'] for r in orders['architecture_round_robin']],['0','2','1','3'])
            for order in orders.values():
                self.assertEqual({r['name'] for r in order},{'0','1','2','3'})

    def test_conservative_admission(self):
        self.assertEqual(admission_estimate(10,0),10+64*1024**2)
        self.assertEqual(admission_estimate(10,100*1024**2),10+100*1024**2)
        with self.assertRaises(ValueError):
            admission_estimate(0,0)


if __name__=='__main__':
    unittest.main()
