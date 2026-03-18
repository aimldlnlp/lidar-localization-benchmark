"""Measurement-model utilities."""

from __future__ import annotations

import numpy as np


def select_beam_indices(total_beams: int, desired_beams: int) -> np.ndarray:
    if desired_beams <= 0:
        raise ValueError("desired_beams must be positive.")
    if desired_beams >= total_beams:
        return np.arange(total_beams, dtype=int)
    return np.linspace(0, total_beams - 1, desired_beams, dtype=int)


def extract_sparse_measurement(
    scan_ranges_m: np.ndarray,
    beam_angles_rad: np.ndarray,
    desired_beams: int,
    max_range_m: float,
    ray_step_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    selected = select_beam_indices(len(scan_ranges_m), desired_beams)
    selected_ranges = scan_ranges_m[selected]
    selected_angles = beam_angles_rad[selected]
    valid_mask = selected_ranges < (max_range_m - 0.5 * ray_step_m)
    valid_fraction = float(np.mean(valid_mask)) if len(valid_mask) > 0 else 0.0
    return selected_ranges[valid_mask], selected_angles[valid_mask], selected[valid_mask], valid_fraction
