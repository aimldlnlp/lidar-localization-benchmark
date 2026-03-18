import numpy as np

from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.maps.families import generate_open_map
from massive_lidar_benchmark.maps.validate import is_world_point_free
from massive_lidar_benchmark.traj.generate import generate_trajectory


def test_generate_trajectory_stays_in_free_space() -> None:
    config = ProjectConfig()
    config.map.family = "open"
    config.map.width_m = 20.0
    config.map.height_m = 20.0
    config.map.resolution_m = 0.1
    config.trajectory.pattern = "explore"
    config.trajectory.num_waypoints = 5
    config.trajectory.dt_s = 0.1
    map_data = generate_open_map(config.map, seed=101)
    trajectory = generate_trajectory(map_data, config, seed=202, map_id="test_map")
    assert trajectory.states.shape[1] == 6
    assert trajectory.states.shape[0] >= 2
    for x_m, y_m in trajectory.states[:, 1:3]:
        assert is_world_point_free(map_data, float(x_m), float(y_m))

