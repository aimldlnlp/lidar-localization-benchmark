from massive_lidar_benchmark.benchmarks.runner import run_benchmark_experiment
from massive_lidar_benchmark.config.schema import ProjectConfig


def test_portfolio_fast_like_benchmark_writes_expected_run_root(tmp_path) -> None:
    config = ProjectConfig()
    config.output_root = str(tmp_path)
    config.seed = 11
    config.device = "cuda"
    config.map.family = "open"
    config.map.width_m = 8.0
    config.map.height_m = 8.0
    config.map.resolution_m = 0.5
    config.map.obstacle_density = 0.02
    config.trajectory.pattern = "explore"
    config.trajectory.horizon_s = 2.0
    config.trajectory.num_waypoints = 3
    config.lidar.num_beams = 8
    config.lidar.max_range_m = 6.0
    config.lidar.ray_step_m = 0.25
    config.ekf.sparse_beams = 4
    config.mcl.particle_count = 16
    config.mcl.measurement_beams = 4
    config.benchmark.maps_per_family = 1
    config.benchmark.trajectories_per_map = 1
    config.benchmark.map_families = ["open"]
    config.benchmark.seeds = [0]
    config.benchmark.step_limit = 10
    config.experiment.name = "portfolio_fast"
    config.experiment.sweep_key = "lidar.range_noise_std_m"
    config.experiment.values = [0.02]

    episode_metrics, summary = run_benchmark_experiment(config, tmp_path)

    assert not episode_metrics.empty
    assert not summary.empty
    assert (tmp_path / "runs" / "portfolio_fast").exists()
