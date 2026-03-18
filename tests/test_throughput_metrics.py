import numpy as np

from massive_lidar_benchmark.benchmarks.runner import run_benchmark_experiment
from massive_lidar_benchmark.config.schema import ProjectConfig


def test_throughput_metrics_include_each_device_batch_combination(tmp_path) -> None:
    config = ProjectConfig()
    config.output_root = str(tmp_path)
    config.seed = 7
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
    config.mcl.particle_count = 8
    config.mcl.measurement_beams = 4
    config.benchmark.maps_per_family = 1
    config.benchmark.trajectories_per_map = 1
    config.benchmark.map_families = ["open"]
    config.benchmark.seeds = [0]
    config.benchmark.step_limit = 10
    config.throughput.devices = ["cpu", "cuda"]
    config.throughput.warmup_iters = 1
    config.throughput.timed_iters = 1
    config.experiment.name = "throughput_gpu_fast"
    config.experiment.sweep_key = "benchmark.batch_size"
    config.experiment.values = [1, 2]

    metrics, _ = run_benchmark_experiment(config, tmp_path)

    assert len(metrics) == 4
    assert set(metrics["device"]) == {"cpu", "cuda"}
    assert set(metrics["batch_size"]) == {1, 2}
    numeric = metrics[["scan_batches_per_second", "scans_per_second", "particle_likelihoods_per_second", "runtime_s"]].to_numpy(dtype=float)
    assert np.all(np.isfinite(numeric))
    assert np.all(numeric >= 0.0)
