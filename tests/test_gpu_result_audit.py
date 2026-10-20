import unittest
from pc_tests.model_pipeline.audit_gpu_results import validate_record


class GPUAuditTests(unittest.TestCase):
    def record(self):
        return {'status': 'complete', 'precision': 'int8', 'test_images': 10000,
                'validation_images': 5000, 'integer_compute_layers': ['conv'],
                'timings': {str(size): [{'repeat': index, 'forward_batch_ms': 1,
                            'normalized_host_input_to_logits_batch_ms': 2,
                            'gpu_energy_j_per_image': .01, 'energy_window_seconds': 3.1,
                            'timed_batches': 100, 'energy_window_images': 100*size}
                           for index in range(5)] for size in (1,8,16,32,64,128)}}

    def test_complete_record_has_thirty_repeats(self):
        self.assertEqual(validate_record(self.record()), 30)

    def test_int8_without_integer_compute_is_rejected(self):
        row = self.record()
        row['integer_compute_layers'] = []
        with self.assertRaises(ValueError):
            validate_record(row)

    def test_energy_denominator_mismatch_is_rejected(self):
        row = self.record()
        row['timings']['8'][0]['energy_window_images'] = 100
        with self.assertRaises(ValueError):
            validate_record(row)


if __name__ == '__main__':
    unittest.main()
