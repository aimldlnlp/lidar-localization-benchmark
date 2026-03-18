"""Grid/world coordinate transforms."""

from __future__ import annotations

import numpy as np


def world_to_grid(x_m: float, y_m: float, resolution_m: float, origin_xy: tuple[float, float]) -> tuple[int, int]:
    origin_x, origin_y = origin_xy
    gx = int(np.floor((x_m - origin_x) / resolution_m))
    gy = int(np.floor((y_m - origin_y) / resolution_m))
    return gx, gy


def grid_to_world(gx: int, gy: int, resolution_m: float, origin_xy: tuple[float, float]) -> tuple[float, float]:
    origin_x, origin_y = origin_xy
    x_m = origin_x + (gx + 0.5) * resolution_m
    y_m = origin_y + (gy + 0.5) * resolution_m
    return x_m, y_m

