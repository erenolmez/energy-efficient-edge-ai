import unittest
from pc_tests.model_pipeline.analysis.count_six_model_switches import transitions, replay


class SwitchCountTests(unittest.TestCase):
    def test_direction_and_initial_load(self):
        matrix, counts, switches = transitions([1,1,2,1,3,3])
        self.assertEqual(switches, 3)
        self.assertEqual(int(matrix[1,2]), 1)
        self.assertEqual(int(matrix[2,1]), 1)
        self.assertEqual(int(matrix[1,3]), 1)
        self.assertEqual(counts.tolist(), [0,3,1,2,0,0])

    def test_constant_has_no_switches(self):
        self.assertEqual(transitions([1,1,1])[2], 0)
        self.assertEqual(replay([1,1,1], [10,10], 20)["cold_and_reload_loads"], 1)

    def test_switch_to_resident_is_hit(self):
        result = replay([0,1,0,1], [10,10], 20)
        self.assertEqual(result["cold_and_reload_loads"], 2)
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["evictions"], 0)

    def test_byte_lru_and_capacity_lru(self):
        self.assertEqual(replay([0,1,0], [10,10], 10)["evictions"], 2)
        self.assertEqual(replay([0,1,0], [10,10], 20, capacity=1)["evictions"], 2)

    def test_infeasible_not_reported_as_load(self):
        result = replay([0,1], [30,10], 20)
        self.assertEqual(result["infeasible_requests"], 1)
        self.assertEqual(result["cold_and_reload_loads"], 1)

    def test_invalid_model_id_rejected(self):
        with self.assertRaises(ValueError):
            transitions([0,6])


if __name__ == "__main__":
    unittest.main()
