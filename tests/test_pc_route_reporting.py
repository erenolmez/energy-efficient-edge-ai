import unittest
from pc_tests.model_pipeline.summarize_pc_routes import aggregate, validate_measurements


class PCRouteReportingTests(unittest.TestCase):
    def case(self, seed=42):
        return {'architecture':'a','router':'batch_tree_d3','batch_size':32,'objective':'latency',
                'budget_pp':.5,'seed':seed,'status':'complete','accuracy_loss_vs_p0_pp':1,
                'test_budget_passed':False,'latency_saving_vs_p0_fp32_percent':-10,
                'latency_saving_vs_best_eligible_static_percent':-20,
                'energy_saving_vs_p0_fp32_percent':None,'energy_saving_vs_best_eligible_static_percent':None,
                'measurements':[{'repeat':index,'test_images':10000,'selected_images':[10000,0],
                                 'selected_batches':[313,0],'rgb_input_to_prediction_seconds':1,
                                 'test_accuracy':79,'mean_gpu_decision_ms':.1} for index in range(5)]}

    def test_retains_negative_saving_accuracy_failure_and_unavailable_energy(self):
        rows=aggregate({'results':[self.case()]})
        self.assertEqual(rows[0]['test_budget_passed_seeds'],0)
        self.assertEqual(rows[0]['latency_saving_vs_p0_fp32_percent'],-10)
        self.assertIsNone(rows[0]['energy_saving_vs_p0_fp32_percent'])

    def test_duplicate_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate({'results':[self.case(),self.case()]})

    def test_both_baselines_and_matched_identity_are_retained(self):
        case = self.case()
        case['best_eligible_static'] = 'resnet101__p0__fp16'
        row = aggregate({'results':[case]})[0]
        self.assertEqual(row['accuracy_matched_static_models'], ['resnet101__p0__fp16'])
        self.assertEqual(row['latency_saving_vs_p0_fp32_percent'], -10)
        self.assertEqual(row['latency_saving_vs_best_eligible_static_percent'], -20)

    def test_incomplete_selection_counts_are_rejected(self):
        row=self.case()
        row['measurements'][0]['selected_images']=[9999,0]
        with self.assertRaises(ValueError):
            validate_measurements(row)


if __name__=='__main__':
    unittest.main()
