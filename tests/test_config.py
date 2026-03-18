from pathlib import Path

from massive_lidar_benchmark.config.load import load_config


def test_load_config_with_inheritance() -> None:
    config = load_config(Path("configs/debug/smoke.yaml"))
    assert config.project_name == "MASSIVE-PARALLEL LIDAR LOCALIZATION BENCHMARK"
    assert config.map.family == "open"
    assert config.mcl.particle_count == 64
    assert config.render.frame_size == [960, 540]


def test_load_portfolio_fast_config() -> None:
    config = load_config(Path("configs/benchmarks/portfolio_fast.yaml"))
    assert config.experiment.name == "portfolio_fast"
    assert config.map.width_m == 24.0
    assert config.lidar.num_beams == 90
    assert config.ekf.sparse_beams == 8
    assert config.mcl.particle_count == 256


def test_load_throughput_gpu_fast_config() -> None:
    config = load_config(Path("configs/benchmarks/throughput_gpu_fast.yaml"))
    assert config.experiment.name == "throughput_gpu_fast"
    assert config.throughput.devices == ["cuda"]
    assert config.throughput.warmup_iters == 5
    assert config.throughput.timed_iters == 20
