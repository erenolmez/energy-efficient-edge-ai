import unittest
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pc_tests.model_pipeline.gpu_router_models import TreeRouter, LogisticRouter, XGBRouter, AggregatedTreeRouter


class GPURouterModelTests(unittest.TestCase):
    def test_tree_matches_sklearn_with_column_mapping(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(100, 13)).astype('float32')
        target = features[:, 8] ** 2 + features[:, 3]
        for depth in (1, 3):
            tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(features[:, [8, 3]], target)
            model = TreeRouter(tree.tree_, [8, 3])
            np.testing.assert_allclose(model(torch.from_numpy(features)).numpy(), tree.predict(features[:, [8, 3]]), atol=1e-6)

    def test_logistic_matches_sklearn(self):
        features = np.random.default_rng(42).normal(size=(100, 13)).astype('float32')
        scaler = StandardScaler().fit(features)
        classifier = LogisticRegression().fit(scaler.transform(features), features[:, 0] > 0)
        model = LogisticRouter(scaler, classifier)
        np.testing.assert_allclose(model(torch.from_numpy(features)).numpy(),
                                   classifier.predict_proba(scaler.transform(features))[:, 1], atol=1e-6)

    def test_xgboost_matches_original(self):
        from xgboost import XGBRegressor
        features = np.random.default_rng(42).normal(size=(100, 13)).astype('float32')
        estimator = XGBRegressor(n_estimators=12, max_depth=2, n_jobs=1).fit(features, features[:, 0] ** 2)
        model = XGBRouter(estimator.get_booster())
        np.testing.assert_allclose(model(torch.from_numpy(features)).numpy(), estimator.predict(features), atol=1e-6)

    def test_mean_and_percentile_controls(self):
        features = np.random.default_rng(42).normal(size=(32, 13)).astype('float32')
        tree = DecisionTreeRegressor(max_depth=3).fit(features, features[:, 0])
        base = TreeRouter(tree.tree_)
        scores = tree.predict(features)
        for percentile in (10, 25, 50, 70, 80, 90, None):
            model = AggregatedTreeRouter(base, 32, percentile)
            expected = scores.mean() if percentile is None else np.percentile(scores, percentile)
            self.assertAlmostEqual(float(model(torch.from_numpy(features)).item()), expected, places=6)


if __name__ == '__main__':
    unittest.main()
