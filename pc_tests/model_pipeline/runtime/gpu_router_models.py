"""Small learned routers evaluated entirely with CUDA tensor operations."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import numpy as np
import json
import torch
from torch import nn


class TreeRouter(nn.Module):
    """Convert a fitted sklearn regression tree without a CPU feature copy."""
    def __init__(self, tree, feature_columns=None):
        super().__init__()
        feature = tree.feature.copy()
        if feature_columns is not None:
            columns = np.asarray(feature_columns)
            feature[feature >= 0] = columns[feature[feature >= 0]]
        self.register_buffer('feature', torch.tensor(np.maximum(feature, 0), dtype=torch.long))
        self.register_buffer('threshold', torch.tensor(tree.threshold, dtype=torch.float32))
        self.register_buffer('left', torch.tensor(tree.children_left, dtype=torch.long))
        self.register_buffer('right', torch.tensor(tree.children_right, dtype=torch.long))
        self.register_buffer('values', torch.tensor(tree.value[:, 0, 0], dtype=torch.float32))
        self.depth = int(tree.max_depth)

    def forward(self, features):
        nodes = torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
        rows = torch.arange(features.shape[0], device=features.device)
        for _ in range(self.depth):
            leaf = self.left[nodes] < 0
            branch = features[rows, self.feature[nodes]] <= self.threshold[nodes]
            next_nodes = torch.where(branch, self.left[nodes], self.right[nodes])
            nodes = torch.where(leaf, nodes, next_nodes)
        return self.values[nodes]


class LogisticRouter(nn.Module):
    def __init__(self, scaler, classifier):
        super().__init__()
        self.register_buffer('mean', torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scaler.scale_, dtype=torch.float32))
        self.register_buffer('weight', torch.tensor(classifier.coef_[0], dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(classifier.intercept_[0], dtype=torch.float32))

    def forward(self, features):
        return torch.sigmoid(((features - self.mean) / self.scale * self.weight).sum(1) + self.bias)


class ConstantRouter(nn.Module):
    """Explicit fallback when the fit labels have no variation."""
    def __init__(self, value):
        super().__init__()
        self.register_buffer('value', torch.tensor(float(value)))

    def forward(self, features):
        return self.value.expand(features.shape[0])


class XGBRouter(nn.Module):
    """Vectorized shallow XGBoost regression trees on the same GPU features."""
    def __init__(self, booster):
        super().__init__()
        trees = [json.loads(value) for value in booster.get_dump(dump_format='json')]
        nodes, maximum_depth = [], 0
        for tree in trees:
            mapping = {}
            def visit(node, depth=0):
                nonlocal maximum_depth
                mapping[node['nodeid']] = node
                maximum_depth = max(maximum_depth, depth)
                for child in node.get('children', []):
                    visit(child, depth + 1)
            visit(tree)
            nodes.append(mapping)
        width = max(max(tree) + 1 for tree in nodes)
        feature = np.zeros((len(trees), width), dtype=np.int64)
        threshold = np.zeros((len(trees), width), dtype=np.float32)
        left = np.full((len(trees), width), -1, dtype=np.int64)
        right = left.copy()
        value = threshold.copy()
        for index, tree in enumerate(nodes):
            for node_id, node in tree.items():
                if 'leaf' in node:
                    value[index, node_id] = node['leaf']
                else:
                    feature[index, node_id] = int(node['split'].removeprefix('f'))
                    threshold[index, node_id] = node['split_condition']
                    left[index, node_id], right[index, node_id] = node['yes'], node['no']
        for name, array in [('feature', feature), ('threshold', threshold),
                            ('left', left), ('right', right), ('values', value)]:
            self.register_buffer(name, torch.from_numpy(array))
        config = json.loads(booster.save_config())
        parameter = config['learner']['learner_model_param']['base_score']
        self.register_buffer('base', torch.tensor(float(str(parameter).strip('[]'))))
        self.depth = maximum_depth

    def forward(self, features):
        # Inputs are finite; XGBoost missing-value branches therefore do not apply.
        trees = torch.arange(self.feature.shape[0], device=features.device)[None, :]
        rows = torch.arange(features.shape[0], device=features.device)[:, None]
        nodes = torch.zeros((features.shape[0], self.feature.shape[0]), device=features.device, dtype=torch.long)
        for _ in range(self.depth):
            leaf = self.left[trees, nodes] < 0
            branch = features[rows, self.feature[trees, nodes]] < self.threshold[trees, nodes]
            next_nodes = torch.where(branch, self.left[trees, nodes], self.right[trees, nodes])
            nodes = torch.where(leaf, nodes, next_nodes)
        return self.base + self.values[trees, nodes].sum(1)


class AggregatedTreeRouter(nn.Module):
    """Per-image difficulty control with explicit mean/percentile aggregation."""
    def __init__(self, tree, batch_size, percentile=None):
        super().__init__()
        self.tree = tree
        self.batch_size = int(batch_size)
        self.percentile = percentile
        position = 0 if percentile is None else (batch_size - 1) * percentile / 100
        self.lower, self.upper = int(np.floor(position)), int(np.ceil(position))
        self.weight = position - self.lower

    def forward(self, features):
        scores = self.tree(features)
        if self.percentile is None:
            return scores.mean().reshape(1)
        sorted_scores = torch.sort(scores).values
        return ((1-self.weight)*sorted_scores[self.lower] + self.weight*sorted_scores[self.upper]).reshape(1)
