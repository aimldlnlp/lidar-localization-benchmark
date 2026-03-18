"""Simple straight-line trajectory planning helpers."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.core.types import OccupancyMap
from massive_lidar_benchmark.maps.validate import is_segment_free


def filter_collision_free_waypoints(
    map_data: OccupancyMap,
    waypoints: np.ndarray,
    step_m: float = 0.05,
) -> np.ndarray:
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required.")

    accepted = [waypoints[0]]
    for waypoint in waypoints[1:]:
        if is_segment_free(map_data, tuple(accepted[-1]), tuple(waypoint), step_m=step_m):
            accepted.append(waypoint)
    if len(accepted) < 2:
        raise ValueError("Unable to find a collision-free waypoint chain.")
    return np.asarray(accepted, dtype=float)


def interpolate_polyline(waypoints: np.ndarray, step_m: float) -> np.ndarray:
    points: list[np.ndarray] = [waypoints[0]]
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        count = max(2, int(np.ceil(distance / step_m)))
        segment = np.linspace(start, end, count, endpoint=False)[1:]
        points.extend(segment)
    points.append(waypoints[-1])
    return np.asarray(points, dtype=float)

