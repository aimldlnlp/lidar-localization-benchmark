"""Map validation helpers used by generation and tests."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.core.types import OccupancyMap
from massive_lidar_benchmark.maps.grid import world_to_grid


def free_ratio(occupancy: np.ndarray) -> float:
    return float(np.mean(~occupancy.astype(bool)))


def in_bounds(map_data: OccupancyMap, gx: int, gy: int) -> bool:
    height, width = map_data.occupancy.shape
    return 0 <= gx < width and 0 <= gy < height


def is_world_point_free(map_data: OccupancyMap, x_m: float, y_m: float) -> bool:
    gx, gy = world_to_grid(x_m, y_m, map_data.resolution_m, map_data.origin_xy)
    if not in_bounds(map_data, gx, gy):
        return False
    return not bool(map_data.occupancy[gy, gx])


def is_segment_free(
    map_data: OccupancyMap,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    step_m: float = 0.05,
) -> bool:
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance == 0.0:
        return is_world_point_free(map_data, float(start[0]), float(start[1]))
    steps = max(2, int(np.ceil(distance / step_m)))
    for alpha in np.linspace(0.0, 1.0, steps):
        x_m, y_m = start + alpha * delta
        if not is_world_point_free(map_data, float(x_m), float(y_m)):
            return False
    return True


def sample_free_points(map_data: OccupancyMap, count: int, rng: np.random.Generator) -> np.ndarray:
    free_indices = np.argwhere(~map_data.occupancy.astype(bool))
    if free_indices.size == 0:
        raise ValueError("Map contains no free cells.")
    chosen = free_indices[rng.integers(0, len(free_indices), size=count)]
    points = np.empty((count, 2), dtype=float)
    for index, (gy, gx) in enumerate(chosen):
        points[index, 0] = map_data.origin_xy[0] + (gx + 0.5) * map_data.resolution_m
        points[index, 1] = map_data.origin_xy[1] + (gy + 0.5) * map_data.resolution_m
    return points

