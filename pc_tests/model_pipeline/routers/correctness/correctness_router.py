"""A shared tiny image encoder with one correctness logit per candidate."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import torch
from torch import nn
from torch.nn import functional as F


class CorrectnessRouter(nn.Module):
    def __init__(self, candidates=213):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16), nn.Conv2d(16, 24, 1), nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24), nn.Conv2d(24, 32, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(32, candidates)

    def forward(self, normalized_images):
        return self.head(self.encoder(F.adaptive_avg_pool2d(normalized_images, (16, 16))))


class BatchCorrectnessRouter(nn.Module):
    """Average candidate probabilities, never average scalar candidate ranks."""
    def __init__(self, image_router):
        super().__init__()
        self.image_router = image_router

    def forward(self, normalized_images):
        return self.image_router(normalized_images).sigmoid().mean(0, keepdim=True)
