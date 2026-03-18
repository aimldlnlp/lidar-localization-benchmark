"""Waypoint pattern helpers."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.core.types import OccupancyMap
from massive_lidar_benchmark.maps.validate import sample_free_points


def sample_waypoints(
    map_data: OccupancyMap,
    pattern: str,
    num_waypoints: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if pattern in {"explore", "waypoint"}:
        return sample_free_points(map_data, num_waypoints, rng)

    points = sample_free_points(map_data, max(num_waypoints, 4), rng)
    center = np.mean(points, axis=0)
    radius = max(map_data.resolution_m * 10.0, 0.15 * min(map_data.shape) * map_data.resolution_m)

    if pattern == "loop":
        angles = np.linspace(0.0, 2.0 * np.pi, num_waypoints, endpoint=False)
        return np.stack(
            [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)],
            axis=1,
        )
    if pattern == "zigzag":
        xs = np.linspace(center[0] - radius, center[0] + radius, num_waypoints)
        ys = np.where(np.arange(num_waypoints) % 2 == 0, center[1] - 0.5 * radius, center[1] + 0.5 * radius)
        return np.stack([xs, ys], axis=1)
    if pattern == "figure8":
        angles = np.linspace(0.0, 2.0 * np.pi, num_waypoints, endpoint=False)
        xs = center[0] + radius * np.sin(angles)
        ys = center[1] + radius * np.sin(angles) * np.cos(angles)
        return np.stack([xs, ys], axis=1)

    raise ValueError(f"Unsupported trajectory pattern: {pattern}")

