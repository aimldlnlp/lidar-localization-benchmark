import numpy as np

from massive_lidar_benchmark.core.types import OccupancyMap, Pose2D
from massive_lidar_benchmark.sensors.lidar_cpu import ray_cast_scan


def test_ray_cast_scan_hits_wall() -> None:
    occupancy = np.zeros((12, 12), dtype=bool)
    occupancy[:, 8] = True
    map_data = OccupancyMap(occupancy=occupancy, resolution_m=1.0, origin_xy=(0.0, 0.0), family="toy", seed=0)
    pose = Pose2D(x=5.5, y=5.5, theta=0.0)
    ranges = ray_cast_scan(map_data, pose, np.array([0.0]), min_range_m=0.5, max_range_m=10.0, ray_step_m=0.5)
    assert np.isclose(ranges[0], 2.5)

