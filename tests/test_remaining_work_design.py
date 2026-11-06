import unittest
from collections import Counter
from pc_tests.model_pipeline.experiments.remaining_work_design import graph_windows,plateau_windows


class DesignTests(unittest.TestCase):
    def test_graph_balanced(self):
        rows=graph_windows()
        self.assertEqual(len(rows),72)
        self.assertEqual(len({tuple(r.items()) for r in rows}),72)
        self.assertEqual(set(Counter((r['mode'],r['residence'],r['egress']) for r in rows).values()),{6})
        for residence in ('host','gpu'):
            for egress in ('per_image_cpu','deferred_gpu'):
                orders=[[r['mode'] for r in rows if r['repeat']==rep and r['residence']==residence and r['egress']==egress] for rep in range(6)]
                for a,b in (('cpu_dispatch','cuda_graph'),('gpu_conditional','cuda_graph'),('cpu_dispatch','gpu_conditional')):
                    self.assertEqual(sum(o.index(a)<o.index(b) for o in orders),3)

    def test_plateau(self):
        rows=plateau_windows()
        self.assertEqual(len(rows),36)
        self.assertEqual(set(Counter((r['name'],r['boundary'],r['batch']) for r in rows).values()),{3})


if __name__=='__main__': unittest.main()
