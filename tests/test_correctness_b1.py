import unittest
import numpy as np
from pc_tests.model_pipeline.experiments.run_correctness_b1 import expected_blocks, validate_meter
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness


class BatchOneTests(unittest.TestCase):
    def test_coverage(self):
        self.assertEqual(expected_blocks(),360)

    def test_sweep_completeness(self):
        value=dict(gpu_energy_j=10.,gpu_energy_j_per_image=.01,rgb_input_to_prediction_seconds=2.,test_images=1000,selected_images=[400,600])
        validate_meter(value,1000)
        for key,bad in [('gpu_energy_j',0),('gpu_energy_j_per_image',float('nan')),
                        ('test_images',999),('selected_images',[400,599])]:
            with self.assertRaises(RuntimeError):
                validate_meter(dict(value,**{key:bad}),1000)

    def test_one_image_calibration_can_choose_cheaper_correct_model(self):
        q=np.array([[.99,.95],[.99,.95]])
        counts=np.ones((2,2),dtype=int)
        policy=calibrate_correctness(q,counts,np.ones(2),np.array([10.,1.]),0,0.)
        self.assertFalse(policy['fallback'])
        self.assertEqual(policy['estimated_batch_cost'],1.)

    def test_one_image_calibration_keeps_accuracy_constraint(self):
        q=np.array([[.99,.1],[.99,.1]])
        counts=np.array([[1,0],[1,0]])
        policy=calibrate_correctness(q,counts,np.ones(2),np.array([10.,1.]),0,0.)
        self.assertTrue(policy['fallback'])


if __name__=='__main__':
    unittest.main()
