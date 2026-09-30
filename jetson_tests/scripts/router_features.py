"""Input-only descriptors shared with the PC router experiments.

This is kept dependency-light so the trained XGBoost routers can run on the
Jetson without importing the PC training pipeline.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _spatial_indices(height: int, width: int, divisions: int) -> np.ndarray:
    rows = np.minimum(np.arange(height) * divisions // height, divisions - 1)
    cols = np.minimum(np.arange(width) * divisions // width, divisions - 1)
    result = (rows[:, None] * divisions + cols[None, :]).ravel()
    result.setflags(write=False)
    return result


def _histogram(values: np.ndarray, bins: int) -> np.ndarray:
    indices = np.minimum((values.ravel() * bins).astype(np.int32), bins - 1)
    return np.bincount(indices, minlength=bins).astype(np.float32) / values.size


def _hog(dx: np.ndarray, dy: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    orientation = np.mod(np.arctan2(dy, dx), np.pi).ravel() * (9 / np.pi)
    lower = np.floor(orientation).astype(np.int32)
    fraction = orientation - lower
    cells = _spatial_indices(*dx.shape, 4) * 9
    mag = magnitude.ravel()
    histogram = (
        np.bincount(cells + lower % 9, weights=mag * (1 - fraction), minlength=144)
        + np.bincount(cells + (lower + 1) % 9, weights=mag * fraction, minlength=144)
    ).reshape(4, 4, 9)
    blocks = np.stack((histogram[:-1, :-1], histogram[:-1, 1:],
                       histogram[1:, :-1], histogram[1:, 1:]), axis=2)
    blocks = blocks.reshape(3, 3, 36)
    blocks /= np.sqrt(np.sum(blocks * blocks, axis=2, keepdims=True) + 1e-10)
    np.minimum(blocks, 0.2, out=blocks)
    blocks /= np.sqrt(np.sum(blocks * blocks, axis=2, keepdims=True) + 1e-10)
    return blocks.ravel()


def _lbp(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbours = np.stack((gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
                           gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
                           gray[2:, :-2], gray[1:-1, :-2]))
    bits = neighbours >= center
    transitions = np.count_nonzero(bits != np.roll(bits, 1, axis=0), axis=0)
    codes = np.where(transitions <= 2, bits.sum(axis=0), 9).ravel()
    quadrants = _spatial_indices(*center.shape, 2)
    counts = np.bincount(quadrants * 10 + codes, minlength=40).reshape(4, 10)
    return (counts / counts.sum(axis=1, keepdims=True)).ravel()


def extract_feature_vector(image: np.ndarray, feature_set: str) -> np.ndarray:
    original = np.asarray(image)
    rgb = original.astype(np.float32, copy=False) / np.float32(255)
    h, w = rgb.shape[:2]
    gray = rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
    gray_hist = _histogram(gray, 32)
    nonzero = gray_hist[gray_hist > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    p10, p90 = np.percentile(gray, (10, 90))
    dy, dx = np.gradient(gray)
    magnitude = np.hypot(dx, dy)
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                 + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * gray)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    quadrants = rgb.reshape(2, h // 2, 2, w // 2, 3)
    spatial = np.stack((quadrants.mean(axis=(1, 3)),
                        quadrants.std(axis=(1, 3))), axis=2).ravel()
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    center_mean = center.mean()
    border_mean = (gray.sum() - center.sum()) / (gray.size - center.size)
    values = np.concatenate((
        rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)),
        np.asarray((gray.mean(), gray.std(), entropy, p10, p90, p90 - p10,
                    magnitude.mean(), magnitude.std(), np.mean(magnitude > 0.1),
                    np.abs(laplacian).mean(), laplacian.var(),
                    saturation.mean(), saturation.std())),
        spatial, np.asarray((center_mean, border_mean, center_mean - border_mean)),
    ))
    if feature_set == "texture":
        values = np.concatenate((values,
            *(_histogram(rgb[..., i], 8) for i in range(3)),
            _histogram(gray, 16), _hog(dx, dy, magnitude), _lbp(gray)))
    elif feature_set != "compact":
        raise ValueError(f"Unsupported Jetson feature set: {feature_set}")
    return values.astype(np.float32, copy=False)
