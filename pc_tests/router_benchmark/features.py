"""Cheap, deterministic image features for input-difficulty routing."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "red_mean", "green_mean", "blue_mean",
    "red_std", "green_std", "blue_std",
    "gray_mean", "gray_std", "gray_entropy",
    "gray_p10", "gray_p90", "gray_dynamic_range",
    "gradient_mean", "gradient_std", "edge_density",
    "laplacian_abs_mean", "laplacian_variance",
    "saturation_mean", "saturation_std",
]

ENHANCED_FEATURE_NAMES = (
    FEATURE_NAMES
    + [f"rgb_{channel}_hist_{i}" for channel in "rgb" for i in range(8)]
    + [f"gray_hist_{i}" for i in range(16)]
    + [f"grid_{row}_{col}_{stat}" for row in range(4) for col in range(4)
       for stat in ("mean", "std")]
    + [f"gradient_orientation_{i}" for i in range(8)]
    + [f"fft_low_{row}_{col}" for row in range(4) for col in range(4)]
    + ["center_mean", "border_mean", "center_border_difference"]
)


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_features(image: np.ndarray) -> dict[str, float]:
    """Extract low-cost features from an HxWx3 RGB image.

    Integer arrays are interpreted as [0, 255]. Floating arrays must be in
    [0, 1] or [0, 255]. The implementation uses only NumPy so that the exact
    feature pipeline can later be benchmarked on a Jetson.
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, received shape {rgb.shape}")
    rgb = rgb.astype(np.float32, copy=False)
    if rgb.size == 0 or not np.isfinite(rgb).all():
        raise ValueError("Image must be non-empty and contain only finite values")
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    if rgb.min() < 0.0 or rgb.max() > 1.0:
        raise ValueError("Image values must fall in [0, 1] or [0, 255]")

    channel_mean = rgb.mean(axis=(0, 1))
    channel_std = rgb.std(axis=(0, 1))
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    p10, p90 = np.percentile(gray, [10, 90])

    dy, dx = np.gradient(gray)
    gradient = np.hypot(dx, dy)
    # A fixed threshold keeps the feature comparable between images.
    edge_density = np.mean(gradient > 0.10)

    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4.0 * gray
    )
    saturation = rgb.max(axis=2) - rgb.min(axis=2)

    values = [
        *channel_mean, *channel_std,
        gray.mean(), gray.std(), _entropy(gray),
        p10, p90, p90 - p10,
        gradient.mean(), gradient.std(), edge_density,
        np.abs(laplacian).mean(), laplacian.var(),
        saturation.mean(), saturation.std(),
    ]
    return dict(zip(FEATURE_NAMES, map(float, values), strict=True))


def extract_enhanced_features(image: np.ndarray) -> dict[str, float]:
    """Extract global, spatial, histogram, gradient, and frequency features."""
    basic = extract_features(image)
    rgb = np.asarray(image).astype(np.float32, copy=False)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    values = list(basic.values())

    for channel in range(3):
        counts, _ = np.histogram(rgb[..., channel], bins=8, range=(0.0, 1.0))
        values.extend((counts / counts.sum()).tolist())
    counts, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
    values.extend((counts / counts.sum()).tolist())

    for row_indices in np.array_split(np.arange(gray.shape[0]), 4):
        for col_indices in np.array_split(np.arange(gray.shape[1]), 4):
            cell = gray[np.ix_(row_indices, col_indices)]
            values.extend([float(cell.mean()), float(cell.std())])

    dy, dx = np.gradient(gray)
    magnitude = np.hypot(dx, dy)
    orientation = (np.arctan2(dy, dx) + np.pi) % (2.0 * np.pi)
    orientation_hist, _ = np.histogram(
        orientation, bins=8, range=(0.0, 2.0 * np.pi), weights=magnitude
    )
    orientation_hist = orientation_hist / max(float(orientation_hist.sum()), 1e-12)
    values.extend(orientation_hist.tolist())

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    center_y, center_x = np.array(spectrum.shape) // 2
    low = spectrum[center_y - 2:center_y + 2, center_x - 2:center_x + 2]
    low = low / max(float(spectrum.sum()), 1e-12)
    values.extend(low.ravel().tolist())

    h, w = gray.shape
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = False
    center_mean, border_mean = float(center.mean()), float(gray[border_mask].mean())
    values.extend([center_mean, border_mean, center_mean - border_mean])
    return dict(zip(ENHANCED_FEATURE_NAMES, map(float, values), strict=True))
