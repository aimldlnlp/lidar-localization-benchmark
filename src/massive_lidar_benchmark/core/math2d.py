"""Small 2D math helpers."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle_rad: float | np.ndarray) -> float | np.ndarray:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def heading_error(theta_est: float, theta_gt: float) -> float:
    return float(wrap_angle(theta_est - theta_gt))


def rotation_matrix(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)

