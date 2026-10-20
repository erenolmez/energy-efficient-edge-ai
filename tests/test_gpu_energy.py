import unittest

from pc_tests.model_pipeline.gpu_energy import counter_metrics


class GPUEnergyTests(unittest.TestCase):
    def test_counter_units_and_per_image(self):
        result = counter_metrics(1000, 31000, 3, 100)
        self.assertEqual(result['gpu_energy_j'], 30)
        self.assertEqual(result['gpu_energy_j_per_image'], .3)
        self.assertEqual(result['gpu_average_watts'], 10)

    def test_reset_counter_is_rejected(self):
        with self.assertRaises(ValueError):
            counter_metrics(2000, 1000, 3, 100)


if __name__ == '__main__':
    unittest.main()
