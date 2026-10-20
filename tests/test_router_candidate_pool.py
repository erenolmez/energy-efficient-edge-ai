import unittest
from pc_tests.model_pipeline.router_candidate_pool import shortlist


class CandidatePoolTests(unittest.TestCase):
    def test_uses_validation_not_test_and_retains_cheap_conditional_route(self):
        rows = []
        for prune in range(0, 100, 10):
            rows.append({'architecture': 'resnet18', 'name': f'resnet18__p{prune}__fp32',
                         'validation_accuracy': 80 - prune / 10, 'test_accuracy': 100,
                         'timings': {'8': [{'normalized_host_input_to_logits_batch_ms': 100-prune,
                                           'gpu_energy_j_per_image': (100-prune)/100}]}})
        before = [row['name'] for row in shortlist(rows)]
        for row in rows:
            row['test_accuracy'] = 0
        self.assertEqual(before, [row['name'] for row in shortlist(rows)])
        self.assertIn('resnet18__p0__fp32', before)
        self.assertIn('resnet18__p90__fp32', before)
        self.assertLessEqual(len(before), 8)


if __name__ == '__main__':
    unittest.main()
