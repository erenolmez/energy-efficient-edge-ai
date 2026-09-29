"""Shared neural components and deterministic data splitting for routers."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def split_positions(data, seed: int):
    """Return stratified 60/20/20 fit, calibration, and evaluation positions."""
    positions = np.arange(len(data))
    fit, rest = train_test_split(
        positions,
        train_size=0.6,
        random_state=seed,
        stratify=data["true_label"],
    )
    calibration, evaluation = train_test_split(
        rest,
        train_size=0.5,
        random_state=seed + 1,
        stratify=data.iloc[rest]["true_label"],
    )
    return fit, calibration, evaluation


class MobileNetEmbedder(nn.Module):
    """Frozen ImageNet MobileNetV3-Small feature extractor."""

    def __init__(self):
        super().__init__()
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = model.features
        self.pool = model.avgpool
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


class TinyCNN(nn.Module):
    """Small depthwise-separable CNN used by direct-image routers."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 24, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24),
            nn.Conv2d(24, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32),
            nn.Conv2d(32, 48, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
