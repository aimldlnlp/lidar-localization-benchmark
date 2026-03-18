"""LiDAR noise helpers."""

from __future__ import annotations

import numpy as np


def apply_range_noise(
    ranges_m: np.ndarray,
    noise_std_m: float,
    dropout_prob: float,
    outlier_prob: float,
    min_range_m: float,
    max_range_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy = np.asarray(ranges_m, dtype=float).copy()
    if noise_std_m > 0.0:
        noisy += rng.normal(0.0, noise_std_m, size=noisy.shape)
    if dropout_prob > 0.0:
        drop_mask = rng.random(noisy.shape) < dropout_prob
        noisy[drop_mask] = max_range_m
    if outlier_prob > 0.0:
        outlier_mask = rng.random(noisy.shape) < outlier_prob
        noisy[outlier_mask] = rng.uniform(min_range_m, max_range_m, size=int(np.count_nonzero(outlier_mask)))
    return np.clip(noisy, min_range_m, max_range_m)

