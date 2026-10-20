import unittest
from pc_tests.model_pipeline.summarize_gpu_results import comparison_rows


class GPUReportingTests(unittest.TestCase):
    def test_own_architecture_baseline_and_unavailable_energy(self):
        def record(prune, cost, accuracy):
            return {'name': f'a__p{prune}__fp32', 'architecture': 'a', 'prune': prune, 'precision': 'fp32',
                    'status': 'complete', 'test_accuracy': accuracy, 'validation_accuracy': 80,
                    'engine_bytes': 1024**2, 'integer_compute_layers': [],
                    'timings': {'32': [{'forward_batch_ms': cost, 'normalized_host_input_to_logits_batch_ms': cost+1,
                                       'gpu_energy_j_per_image': None}]}}
        report = {'results': [record(0, 4, 80), record(10, 2, 79)]}
        rows = comparison_rows(report)
        self.assertEqual(rows[1]['accuracy_loss_vs_p0_pp'], 1)
        self.assertEqual(rows[1]['forward_ms_saving_vs_p0_percent'], 50)
        self.assertIsNone(rows[1]['gpu_j_per_image_saving_vs_p0_percent'])


if __name__ == '__main__':
    unittest.main()
