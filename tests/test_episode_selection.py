import pandas as pd

from massive_lidar_benchmark.benchmarks.summary import episode_identifier, select_representative_episode_group


def test_select_representative_episode_group_is_deterministic() -> None:
    catalog = pd.DataFrame(
        [
            {"experiment": "portfolio_fast", "method": "mcl", "map_family": "office", "map_id": "map_b", "traj_id": "traj_00", "seed": 1, "episode_index": 1, "range_noise_std_m": 0.02, "position_rmse_m": 0.21, "failed": False},
            {"experiment": "portfolio_fast", "method": "mcl", "map_family": "open", "map_id": "map_a", "traj_id": "traj_00", "seed": 0, "episode_index": 0, "range_noise_std_m": 0.02, "position_rmse_m": 0.21, "failed": False},
            {"experiment": "portfolio_fast", "method": "ekf", "map_family": "office", "map_id": "map_b", "traj_id": "traj_00", "seed": 1, "episode_index": 1, "range_noise_std_m": 0.02, "position_rmse_m": 0.31, "failed": False},
            {"experiment": "portfolio_fast", "method": "ekf", "map_family": "open", "map_id": "map_a", "traj_id": "traj_00", "seed": 0, "episode_index": 0, "range_noise_std_m": 0.02, "position_rmse_m": 0.28, "failed": False},
        ]
    )

    first = select_representative_episode_group(catalog)
    second = select_representative_episode_group(catalog)

    assert [episode_identifier(row) for _, row in first.iterrows()] == [episode_identifier(row) for _, row in second.iterrows()]
    assert set(first["map_id"]) == {"map_b"}
    assert set(first["seed"]) == {1}
