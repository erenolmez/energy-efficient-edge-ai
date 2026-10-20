import unittest
import numpy as np
import torch
from pc_tests.model_pipeline.correctness_router_suite import (
    fit_feature_router, SuiteTinyCNN, MeanCorrectness, fit_tune_positions)


class CorrectnessSuiteTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(160, 13)).astype('float32')
        y = np.stack((x[:, 0] > 0, x[:, 1] < 0, np.ones(160), np.zeros(160)), 1).astype('float32')
        return x, y

    def test_tree_cpu_parity(self):
        x, y = self.data()
        for method in ('correctness_tree_d1', 'correctness_tree_d3'):
            model, estimator = fit_feature_router(method, x, y)
            np.testing.assert_allclose(model(torch.from_numpy(x)).detach().numpy(), estimator.predict(x), atol=1e-6)

    def test_logistic_cpu_parity_and_constants(self):
        x, y = self.data()
        model, (scaler, estimators, constants) = fit_feature_router('correctness_logistic', x, y)
        expected = np.stack([np.full(len(x), constants[i]) if e is None else e.predict_proba(scaler.transform(x))[:, 1]
                             for i, e in enumerate(estimators)], 1)
        np.testing.assert_allclose(model(torch.from_numpy(x)).detach().numpy(), expected, atol=1e-6)

    def test_xgb_cpu_parity(self):
        x, y = self.data()
        model, estimators = fit_feature_router('correctness_xgb_d2', x, y, xgb_trees=4)
        expected = np.stack([e.predict(x) for e in estimators], 1).clip(0, 1)
        np.testing.assert_allclose(model(torch.from_numpy(x)).detach().numpy(), expected, atol=1e-6)

    def test_cnn_shapes_and_true_tail_mean(self):
        for size in (8, 16, 32):
            model = SuiteTinyCNN(size).eval()
            x = torch.randn(3, 3, 128, 128)
            with torch.no_grad():
                self.assertEqual(tuple(model(x).shape), (3, 213))
                torch.testing.assert_close(MeanCorrectness(model, logits=True)(x), model(x).sigmoid().mean(0, keepdim=True))

    def test_internal_tune_stays_within_fit(self):
        labels = np.repeat(np.arange(10), 30)
        fit = np.arange(200)
        train, tune = fit_tune_positions(fit, labels, 42)
        self.assertFalse(set(train) & set(tune))
        self.assertEqual(set(train) | set(tune), set(fit))


if __name__ == '__main__':
    unittest.main()
