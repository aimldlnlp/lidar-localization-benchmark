import numpy as np

from massive_lidar_benchmark.config.schema import MapConfig
from massive_lidar_benchmark.maps.families import generate_open_map
from massive_lidar_benchmark.maps.validate import free_ratio


def test_generate_open_map_has_reasonable_free_ratio() -> None:
    config = MapConfig(family="open", width_m=20.0, height_m=20.0, resolution_m=0.1, obstacle_density=0.08)
    map_data = generate_open_map(config, seed=123)
    assert map_data.occupancy.dtype == np.bool_
    assert map_data.shape[0] > 0 and map_data.shape[1] > 0
    assert 0.40 < free_ratio(map_data.occupancy) < 0.95

