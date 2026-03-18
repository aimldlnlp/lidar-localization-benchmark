"""Motion model helpers."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.core.math2d import wrap_angle


def unicycle_step(mean: np.ndarray, control: np.ndarray, dt_s: float) -> np.ndarray:
    x, y, theta = mean
    v, omega = control
    next_state = np.array(
        [
            x + dt_s * v * np.cos(theta),
            y + dt_s * v * np.sin(theta),
            wrap_angle(theta + dt_s * omega),
        ],
        dtype=float,
    )
    return next_state

