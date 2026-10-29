import unittest
from pc_tests.model_pipeline.analysis.summarize_correctness_b1 import payback


class PaybackTests(unittest.TestCase):
    def test_uses_difference_not_static_energy(self):
        value=payback(200.,.05,.04,True)
        self.assertAlmostEqual(value['saved_j_per_image'],.01)
        self.assertEqual(value['valid_payback_images'],20000)

    def test_accuracy_failure_retains_crossing_not_valid_payback(self):
        value=payback(200.,.05,.04,False)
        self.assertIsNone(value['valid_payback_images'])
        self.assertEqual(value['mathematical_crossing_images'],20000)

    def test_no_savings_has_no_payback(self):
        for routed in (.05,.06):
            value=payback(200.,.05,routed,True)
            self.assertIsNone(value['mathematical_crossing_images'])
            self.assertIsNone(value['valid_payback_images'])

    def test_round_up(self):
        self.assertEqual(payback(1.,.5,.2,True)['valid_payback_images'],4)

    def test_invalid_energy(self):
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                payback(value,.05,.04,True)


if __name__=='__main__':
    unittest.main()
