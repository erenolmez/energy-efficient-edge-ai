import unittest
from pc_tests.model_pipeline.experiments.compare_dispatch_residency_v2 import checked_enqueue, variant_order


class TestDispatchRepeat(unittest.TestCase):
    def test_false_enqueue_is_failure(self):
        class Context:
            def execute_async_v3(self, stream):
                return False
        class Stream:
            cuda_stream=7
        with self.assertRaises(RuntimeError):
            checked_enqueue(Context(),Stream())

    def test_order_is_unique_balanced_and_complete(self):
        rows=variant_order()
        self.assertEqual(len(rows),48)
        self.assertEqual(len({tuple(r.values()) for r in rows}),48)
        for res in ('host','gpu'):
            for eg in ('per_image_cpu','deferred_gpu'):
                pairs=[]
                for repeat in range(6):
                    modes=[r['mode'] for r in rows if r['repeat']==repeat and r['residence']==res and r['egress']==eg]
                    self.assertEqual(set(modes),{'cpu_dispatch','gpu_conditional'})
                    pairs.append(modes[0])
                self.assertEqual(pairs.count('cpu_dispatch'),3)


if __name__=='__main__':
    unittest.main()
