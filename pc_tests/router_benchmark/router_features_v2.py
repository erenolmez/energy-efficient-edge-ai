"""Vectorized, input-only image descriptors for the refined routing experiments.

The compact descriptor adds spatial color to global image-quality statistics.
The texture descriptor adds HOG and local binary patterns without invoking a
classification model. HOG uses unsigned orientation voting, 4x4 cells, and
overlapping 2x2-cell L2-Hys blocks; spatial votes are assigned to one cell.
LBP uses eight radius-one neighbours and rotation-invariant uniform bins.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


_GLOBAL_NAMES = (
    "red_mean", "green_mean", "blue_mean",
    "red_std", "green_std", "blue_std",
    "gray_mean", "gray_std", "gray_entropy",
    "gray_p10", "gray_p90", "gray_dynamic_range",
    "gradient_mean", "gradient_std", "edge_density",
    "laplacian_abs_mean", "laplacian_variance",
    "saturation_mean", "saturation_std",
)
_QUADRANT_NAMES = tuple(
    f"quadrant_{row}_{col}_{channel}_{stat}"
    for row in range(2) for col in range(2)
    for stat in ("mean", "std") for channel in "rgb"
)
COMPACT_FEATURE_NAMES = _GLOBAL_NAMES + _QUADRANT_NAMES + (
    "center_mean", "border_mean", "center_border_difference",
)
TEXTURE_FEATURE_NAMES = (
    COMPACT_FEATURE_NAMES
    + tuple(f"rgb_{channel}_hist_{i}" for channel in "rgb" for i in range(8))
    + tuple(f"gray_hist_{i}" for i in range(16))
    + tuple(
        f"hog_block_{row}_{col}_cell_{cy}_{cx}_bin_{b}"
        for row in range(3) for col in range(3)
        for cy in range(2) for cx in range(2) for b in range(9)
    )
    + tuple(f"lbp_quadrant_{row}_{col}_bin_{b}"
            for row in range(2) for col in range(2) for b in range(10))
)
FEATURE_SETS = {"compact": COMPACT_FEATURE_NAMES, "texture": TEXTURE_FEATURE_NAMES}


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    original = np.asarray(image)
    if (original.ndim != 3 or original.shape[2] != 3
            or min(original.shape[:2]) < 4
            or original.shape[0] % 4 or original.shape[1] % 4):
        raise ValueError("Expected HxWx3 RGB image with H,W positive multiples of 4")
    if not np.issubdtype(original.dtype, np.number):
        raise ValueError("Image must contain numeric RGB values")
    if np.issubdtype(original.dtype, np.complexfloating):
        raise ValueError("Image must contain real RGB values")
    rgb = original.astype(np.float32, copy=False)
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError("Image must contain finite values in [0,1] or [0,255]")
    # Integer images always use the documented 0..255 convention, even when
    # the observed maximum is 1 (e.g. a very dark uint8 image).
    if np.issubdtype(original.dtype, np.integer) or rgb.max() > 1:
        rgb = rgb / np.float32(255)
    return rgb


@lru_cache(maxsize=8)
def _spatial_indices(height: int, width: int, divisions: int) -> np.ndarray:
    rows = np.minimum(np.arange(height) * divisions // height, divisions - 1)
    cols = np.minimum(np.arange(width) * divisions // width, divisions - 1)
    indices = (rows[:, None] * divisions + cols[None, :]).ravel()
    indices.setflags(write=False)
    return indices


def _histogram(values: np.ndarray, bins: int) -> np.ndarray:
    indices = np.minimum((values.ravel() * bins).astype(np.int32), bins - 1)
    return np.bincount(indices, minlength=bins).astype(np.float32) / values.size


def _hog(dx: np.ndarray, dy: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    # Linear voting across adjacent circular unsigned orientation bins.
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
    # Uniform patterns have at most two transitions; non-uniforms use bin 9.
    codes = np.where(transitions <= 2, bits.sum(axis=0), 9).ravel()
    quadrants = _spatial_indices(*center.shape, 2)
    counts = np.bincount(quadrants * 10 + codes, minlength=40).reshape(4, 10)
    return (counts / counts.sum(axis=1, keepdims=True)).ravel()


def extract_feature_vector(image: np.ndarray, feature_set: str = "compact") -> np.ndarray:
    """Return an ordered float32 descriptor of an RGB image.

    Inputs are integer [0,255] or floating [0,1]/[0,255] arrays. The intended
    input is native CIFAR-100 resolution, 32x32, without image resizing.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose {tuple(FEATURE_SETS)}")
    rgb = _normalize_rgb(image)
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
    cell_means = quadrants.mean(axis=(1, 3))
    cell_stds = quadrants.std(axis=(1, 3))
    spatial = np.stack((cell_means, cell_stds), axis=2).ravel()
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
        values = np.concatenate((values, *(_histogram(rgb[..., i], 8) for i in range(3)),
                                 _histogram(gray, 16), _hog(dx, dy, magnitude), _lbp(gray)))
    return values.astype(np.float32, copy=False)


def extract_matrix(images: np.ndarray, feature_set: str = "compact") -> np.ndarray:
    """Extract rows independently; output order exactly matches image order."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}")
    return np.asarray([extract_feature_vector(image, feature_set) for image in images],
                      dtype=np.float32).reshape(-1, len(FEATURE_SETS[feature_set]))
