"""CPU-fittable lightweight correctness families; no scalar difficulty targets.

Fit functions accept only the caller's fit partition. Export wrappers return
one mean correctness probability per candidate over the actual batch, not ranks.
This module does not open datasets, launch CUDA, or select test-time parameters.
"""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


METHODS = ('correctness_tree_d1', 'correctness_tree_d3', 'correctness_logistic',
           'correctness_xgb_d2', 'correctness_tinycnn8',
           'correctness_tinycnn16', 'correctness_tinycnn32')


def validate_fit(features, correctness):
    x, y = np.asarray(features, dtype=np.float32), np.asarray(correctness, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError('Fit image-by-feature and image-by-candidate matrices required')
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise ValueError('Finite features and actual binary correctness required')
    return x, y


class CorrectnessTree(nn.Module):
    """Shared tree splits with candidate-specific correctness-frequency leaves."""
    def __init__(self, estimator):
        super().__init__()
        tree = estimator.tree_
        for name, values, dtype in (
            ('feature', np.maximum(tree.feature, 0), torch.long),
            ('threshold', tree.threshold, torch.float32),
            ('left', tree.children_left, torch.long),
            ('right', tree.children_right, torch.long),
            ('values', tree.value[:, :, 0], torch.float32)):
            self.register_buffer(name, torch.tensor(values, dtype=dtype))
        self.depth = tree.max_depth

    def forward(self, features):
        rows = torch.arange(len(features), device=features.device)
        nodes = torch.zeros(len(features), dtype=torch.long, device=features.device)
        for _ in range(self.depth):
            leaf = self.left[nodes] < 0
            next_nodes = torch.where(features[rows, self.feature[nodes]] <= self.threshold[nodes],
                                     self.left[nodes], self.right[nodes])
            nodes = torch.where(leaf, nodes, next_nodes)
        return self.values[nodes]


class CorrectnessLogistic(nn.Module):
    """Independent binary logistic heads, executed as one dense GPU operation."""
    def __init__(self, mean, scale, weights, biases, constants):
        super().__init__()
        for name, value in (('mean', mean), ('scale', scale), ('weights', weights),
                            ('biases', biases), ('constants', constants)):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def forward(self, features):
        probabilities = torch.sigmoid(((features - self.mean) / self.scale) @ self.weights.T + self.biases)
        # All-wrong/all-correct heads remain exact constants, not fabricated fits.
        return torch.where(self.constants[None, :] >= 0, self.constants[None, :], probabilities)


class CorrectnessXGB(nn.Module):
    """Shallow independent binary-target regressors; clip estimates to [0,1]."""
    def __init__(self, estimators):
        super().__init__()
        try:
            from .gpu_router_models import XGBRouter
        except ImportError:
            from pc_tests.model_pipeline.runtime.gpu_router_models import XGBRouter
        self.heads = nn.ModuleList(XGBRouter(e.get_booster()) for e in estimators)

    def forward(self, features):
        return torch.stack([head(features) for head in self.heads], dim=1).clamp(0, 1)


def fit_feature_router(method, features, correctness, seed=42, xgb_trees=32):
    """Return exportable model + fitted CPU estimator for independent parity QA.

    Tree depths, logistic C, and XGB size are fixed a priori. XGB is CPU fit to
    avoid accidental competing GPU jobs; real fitting must not overlap timing.
    """
    x, y = validate_fit(features, correctness)
    if method in ('correctness_tree_d1', 'correctness_tree_d3'):
        from sklearn.tree import DecisionTreeRegressor
        estimator = DecisionTreeRegressor(max_depth=1 if method.endswith('d1') else 3,
                                          min_samples_leaf=20, random_state=seed).fit(x, y)
        return CorrectnessTree(estimator), estimator
    if method == 'correctness_logistic':
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        scaler = StandardScaler().fit(x)
        normalized = scaler.transform(x)
        weights, biases, constants, estimators = [], [], [], []
        for column in y.T:
            unique = np.unique(column)
            if len(unique) == 1:
                weights.append(np.zeros(x.shape[1])); biases.append(0.)
                constants.append(float(unique[0])); estimators.append(None)
            else:
                estimator = LogisticRegression(C=.25, max_iter=2000, random_state=seed).fit(normalized, column)
                weights.append(estimator.coef_[0]); biases.append(estimator.intercept_[0])
                constants.append(-1.); estimators.append(estimator)
        return CorrectnessLogistic(scaler.mean_, scaler.scale_, np.asarray(weights),
                                   np.asarray(biases), np.asarray(constants)), (scaler, estimators, constants)
    if method == 'correctness_xgb_d2':
        from xgboost import XGBRegressor
        estimators = [XGBRegressor(n_estimators=xgb_trees, max_depth=2, learning_rate=.04,
                                  min_child_weight=20, reg_lambda=15, reg_alpha=.2,
                                  subsample=.85, colsample_bytree=.85, objective='reg:squarederror',
                                  tree_method='hist', device='cpu', n_jobs=1, random_state=seed).fit(x, column)
                      for column in y.T]
        return CorrectnessXGB(estimators), estimators
    raise ValueError('Unsupported feature correctness family: ' + method)


class SuiteTinyCNN(nn.Module):
    """Restore deeper historical TinyCNN, BN and 8/16/32 thumbnail alternatives."""
    def __init__(self, size, candidates=213):
        super().__init__()
        if size not in (8, 16, 32):
            raise ValueError('Thumbnail size must be 8,16,32')
        self.size = size
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 24, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24), nn.Conv2d(24, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32), nn.Conv2d(32, 48, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, candidates))

    def forward(self, normalized_images):
        return self.net(F.adaptive_avg_pool2d(normalized_images, (self.size, self.size)))


class MeanCorrectness(nn.Module):
    """True batch mean; same calibration/inference arithmetic, no tail padding."""
    def __init__(self, image_model, logits=False):
        super().__init__()
        self.image_model, self.logits = image_model, logits

    def forward(self, values):
        output = self.image_model(values)
        return (output.sigmoid() if self.logits else output).mean(0, keepdim=True)


def fit_tune_positions(fit_positions, labels, seed):
    """Only partition the original fit subset; policy calibration remains unseen."""
    from sklearn.model_selection import train_test_split
    fit_positions = np.asarray(fit_positions)
    return train_test_split(fit_positions, test_size=.2, random_state=seed,
                            stratify=np.asarray(labels)[fit_positions])
