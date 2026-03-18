"""Reference CPU LiDAR ray casting."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.core.types import OccupancyMap, Pose2D
from massive_lidar_benchmark.maps.grid import world_to_grid


def make_beam_angles(num_beams: int, fov_deg: float) -> np.ndarray:
    if num_beams <= 0:
        raise ValueError("num_beams must be positive.")
    if num_beams == 1:
        return np.array([0.0], dtype=float)
    half_fov = np.deg2rad(fov_deg) * 0.5
    return np.linspace(-half_fov, half_fov, num_beams, dtype=float)


def ray_cast_scan(
    map_data: OccupancyMap,
    pose: Pose2D,
    beam_angles_rad: np.ndarray,
    min_range_m: float,
    max_range_m: float,
    ray_step_m: float,
) -> np.ndarray:
    ranges = np.full(len(beam_angles_rad), max_range_m, dtype=float)
    for beam_index, rel_angle in enumerate(beam_angles_rad):
        angle = pose.theta + float(rel_angle)
        for distance in np.arange(min_range_m, max_range_m + ray_step_m, ray_step_m):
            x_m = pose.x + distance * np.cos(angle)
            y_m = pose.y + distance * np.sin(angle)
            gx, gy = world_to_grid(x_m, y_m, map_data.resolution_m, map_data.origin_xy)
            height, width = map_data.occupancy.shape
            if not (0 <= gx < width and 0 <= gy < height):
                ranges[beam_index] = min(distance, max_range_m)
                break
            if map_data.occupancy[gy, gx]:
                ranges[beam_index] = distance
                break
    return ranges
