"""Low-cost batch descriptors and thumbnail-grid construction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


FEATURE_NAMES = [
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
    "rgb_std_r", "rgb_std_g", "rgb_std_b",
    "gray_mean", "gray_std", "gradient_mean", "gradient_std",
    "edge_density", "saturation_mean", "batch_brightness_std",
]


def batch_feature_vector(images):
    rgb = np.asarray(images, dtype=np.float32) / np.float32(255)
    gray = rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    gradient = np.concatenate((np.abs(dx).ravel(), np.abs(dy).ravel()))
    saturation = rgb.max(axis=-1) - rgb.min(axis=-1)
    per_image_brightness = gray.mean(axis=(1, 2))
    return np.concatenate((
        rgb.mean(axis=(0, 1, 2)), rgb.std(axis=(0, 1, 2)),
        np.asarray((
            gray.mean(), gray.std(), gradient.mean(), gradient.std(),
            np.mean(gradient > 0.1), saturation.mean(),
            per_image_brightness.std(),
        )),
    )).astype(np.float32)


def thumbnail_grid(images):
    source = torch.from_numpy(np.asarray(images).copy()).permute(0, 3, 1, 2).float() / 255
    small = F.interpolate(source, size=(8, 8), mode="bilinear", align_corners=False,
                          antialias=True)
    selected = np.linspace(0, len(images) - 1, 16).round().astype(int)
    tiles = small[selected]
    grid = tiles.reshape(4, 4, 3, 8, 8).permute(2, 0, 3, 1, 4).reshape(1, 3, 32, 32)
    mean = torch.tensor([.5071, .4867, .4408])[None, :, None, None]
    std = torch.tensor([.2675, .2565, .2761])[None, :, None, None]
    return (grid - mean) / std
