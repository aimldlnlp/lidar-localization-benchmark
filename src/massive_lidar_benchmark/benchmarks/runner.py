"""Benchmark execution and artifact writing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from massive_lidar_benchmark.benchmarks.metrics import summarize_episode_metrics, translation_errors
from massive_lidar_benchmark.benchmarks.summary import summarize_metrics_frame
from massive_lidar_benchmark.config.load import save_resolved_config
from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.core.io import ensure_dir
from massive_lidar_benchmark.core.logging import get_logger
from massive_lidar_benchmark.core.math2d import wrap_angle
from massive_lidar_benchmark.core.types import Pose2D
from massive_lidar_benchmark.localization.ekf import run_ekf_localization
from massive_lidar_benchmark.localization.mcl import run_mcl_localization, sample_initial_particles, sample_kidnap_pose
from massive_lidar_benchmark.maps.generate import generate_maps_from_config, load_map_artifact
from massive_lidar_benchmark.maps.validate import is_world_point_free
from massive_lidar_benchmark.sensors.lidar_cpu import make_beam_angles, ray_cast_scan
from massive_lidar_benchmark.sensors.lidar_torch import ray_cast_scan_batched, torch_available
from massive_lidar_benchmark.sensors.noise import apply_range_noise
from massive_lidar_benchmark.traj.generate import generate_trajectories_from_config, load_trajectory_artifact


def _slug_value(value: Any) -> str:
    text = str(value).lower().replace(" ", "_").replace(".", "p").replace("/", "_")
    return "".join(char for char in text if char.isalnum() or char in {"_", "-"})


def _namespace_path(base_experiment: str, sweep_value: Any | None = None) -> Path:
    namespace = Path(base_experiment)
    if sweep_value is not None:
        namespace = namespace / _slug_value(sweep_value)
    return namespace


def _file_slug(namespace: Path) -> str:
    return "_".join(part for part in namespace.parts if part)


def _set_nested_value(config: ProjectConfig, dotted_key: str, value: Any) -> ProjectConfig:
    updated = deepcopy(config)
    target: Any = updated
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    return updated


def _episode_seed(config: ProjectConfig, benchmark_seed: int, episode_index: int) -> int:
    return int(np.random.SeedSequence([config.seed, benchmark_seed, episode_index]).generate_state(1)[0])


def _active_methods(config: ProjectConfig) -> list[str]:
    methods: list[str] = []
    if config.ekf.enabled:
        methods.append("ekf")
    if config.mcl.enabled:
        methods.append("mcl")
    return methods


def _is_throughput_benchmark(config: ProjectConfig) -> bool:
    return bool(config.throughput.devices) and config.experiment.sweep_key == "benchmark.batch_size"


def _sample_initial_mean(gt_pose: np.ndarray, map_data, episode_seed: int) -> np.ndarray:
    rng = np.random.default_rng(episode_seed)
    reference = np.asarray(gt_pose, dtype=float).copy()
    for _ in range(64):
        candidate = reference.copy()
        candidate[0] += rng.normal(0.0, 0.5)
        candidate[1] += rng.normal(0.0, 0.5)
        candidate[2] = wrap_angle(candidate[2] + rng.normal(0.0, 0.2))
        if is_world_point_free(map_data, float(candidate[0]), float(candidate[1])):
            return candidate
    reference[2] = wrap_angle(reference[2])
    return reference


def _simulate_observed_scans(
    map_data,
    gt_states: np.ndarray,
    config: ProjectConfig,
    episode_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    beam_angles = make_beam_angles(config.lidar.num_beams, config.lidar.fov_deg)
    rng = np.random.default_rng(episode_seed)
    scans = np.empty((len(gt_states), len(beam_angles)), dtype=float)
    for step_index, state in enumerate(gt_states):
        pose = Pose2D(x=float(state[1]), y=float(state[2]), theta=float(state[3]))
        clean = ray_cast_scan(
            map_data,
            pose,
            beam_angles,
            min_range_m=config.lidar.min_range_m,
            max_range_m=config.lidar.max_range_m,
            ray_step_m=config.lidar.ray_step_m,
        )
        scans[step_index] = apply_range_noise(
            clean,
            noise_std_m=config.lidar.range_noise_std_m,
            dropout_prob=config.lidar.dropout_prob,
            outlier_prob=config.lidar.outlier_prob,
            min_range_m=config.lidar.min_range_m,
            max_range_m=config.lidar.max_range_m,
            rng=rng,
        )
    return scans, beam_angles


def _run_single_episode(
    config: ProjectConfig,
    output_root: Path,
    method: str,
    benchmark_seed: int,
    episode_index: int,
    map_path: Path,
    trajectory_path: Path,
    run_namespace: Path,
    sweep_value: Any | None = None,
) -> dict[str, Any]:
    map_data = load_map_artifact(map_path)
    trajectory = load_trajectory_artifact(trajectory_path)
    gt_states = trajectory.states[: config.benchmark.step_limit].copy()
    episode_seed = _episode_seed(config, benchmark_seed, episode_index)
    observed_scans, beam_angles = _simulate_observed_scans(map_data, gt_states, config, episode_seed)
    initial_mean = _sample_initial_mean(gt_states[0, 1:4], map_data, episode_seed)

    kidnap_step: int | None = None
    kidnap_pose: np.ndarray | None = None
    if config.benchmark.kidnapped_enabled and len(gt_states) >= 20:
        kidnap_step = len(gt_states) // 2
        kidnap_pose = sample_kidnap_pose(map_data, gt_states[kidnap_step, 1:4], np.random.default_rng(episode_seed + 17))

    start_time = time.perf_counter()
    if method == "ekf":
        result = run_ekf_localization(
            map_data=map_data,
            gt_states=gt_states,
            observed_scans=observed_scans,
            beam_angles_rad=beam_angles,
            config=config,
            initial_mean=initial_mean,
            kidnap_step=kidnap_step,
            kidnap_pose=kidnap_pose,
        )
        ess_values = np.full(len(gt_states), np.nan, dtype=float)
        particle_snapshots: dict[str, np.ndarray] = {}
    elif method == "mcl":
        result = run_mcl_localization(
            map_data=map_data,
            gt_states=gt_states,
            observed_scans=observed_scans,
            beam_angles_rad=beam_angles,
            config=config,
            initial_mean=initial_mean,
            rng=np.random.default_rng(episode_seed + 33),
            kidnap_step=kidnap_step,
            kidnap_pose=kidnap_pose,
        )
        ess_values = np.asarray(result["ess"], dtype=float)
        particle_snapshots = dict(result["particle_snapshots"])
    else:
        raise ValueError(f"Unsupported method: {method}")

    runtime_s = float(time.perf_counter() - start_time)
    estimates = np.asarray(result["estimates"], dtype=float)
    valid_beam_fraction = np.asarray(result["valid_beam_fraction"], dtype=float)
    summary = summarize_episode_metrics(
        gt_states=gt_states,
        est_states=estimates,
        dt_s=trajectory.dt_s,
        valid_beam_fraction=valid_beam_fraction,
        ess_values=None if method == "ekf" else ess_values,
        kidnap_step=kidnap_step,
    )

    run_dir = ensure_dir(
        output_root
        / "runs"
        / run_namespace
        / method
        / f"seed_{benchmark_seed:03d}"
        / f"episode_{episode_index:03d}"
    )
    save_resolved_config(config, run_dir / "resolved_config.yaml")

    heading_errors_deg = np.degrees(np.abs(wrap_angle(gt_states[:, 3] - estimates[:, 2])))
    step_frame = pd.DataFrame(
        {
            "step": np.arange(len(gt_states), dtype=int),
            "t": gt_states[:, 0],
            "x_gt": gt_states[:, 1],
            "y_gt": gt_states[:, 2],
            "theta_gt": gt_states[:, 3],
            "x_est": estimates[:, 0],
            "y_est": estimates[:, 1],
            "theta_est": estimates[:, 2],
            "translation_error_m": translation_errors(gt_states[:, 1:3], estimates[:, :2]),
            "heading_error_deg": heading_errors_deg,
            "valid_beam_fraction": valid_beam_fraction,
            "ess": ess_values,
        }
    )
    step_frame.to_csv(run_dir / "step_metrics.csv", index=False)

    estimate_frame = step_frame[
        [
            "t",
            "x_gt",
            "y_gt",
            "theta_gt",
            "x_est",
            "y_est",
            "theta_est",
            "translation_error_m",
            "heading_error_deg",
        ]
    ]
    estimate_frame.to_csv(run_dir / "estimates.csv", index=False)

    np.savez_compressed(
        run_dir / "episode_data.npz",
        gt_states=gt_states,
        estimated_states=estimates,
        observed_scans=observed_scans,
        beam_angles_rad=beam_angles,
        initial_mean=initial_mean,
        kidnap_step=-1 if kidnap_step is None else kidnap_step,
        kidnap_pose=np.zeros(3, dtype=float) if kidnap_pose is None else kidnap_pose,
    )
    if particle_snapshots:
        np.savez_compressed(run_dir / "particle_snapshots.npz", **particle_snapshots)

    episode_row: dict[str, Any] = {
        "experiment": config.experiment.name,
        "method": method,
        "map_family": map_data.family,
        "map_id": map_path.stem,
        "traj_id": trajectory_path.stem,
        "seed": benchmark_seed,
        "episode_index": episode_index,
        "runtime_s": runtime_s,
        "scans_per_second": float(len(gt_states) / max(runtime_s, 1e-9)),
        "episodes_per_second": float(1.0 / max(runtime_s, 1e-9)),
        "kidnap_step": kidnap_step,
        "sweep_value": sweep_value,
        "range_noise_std_m": float(config.lidar.range_noise_std_m),
        "dropout_prob": float(config.lidar.dropout_prob),
        "outlier_prob": float(config.lidar.outlier_prob),
        "batch_size": int(config.benchmark.batch_size),
    }
    episode_row.update(summary)
    pd.DataFrame([episode_row]).to_csv(run_dir / "episode_metrics.csv", index=False)
    return episode_row


def run_method_experiment(
    config: ProjectConfig,
    output_root: str | Path,
    method: str,
    run_namespace: str | Path | None = None,
    sweep_value: Any | None = None,
    map_artifacts: list[Path] | None = None,
    trajectory_artifacts: list[Path] | None = None,
) -> pd.DataFrame:
    logger = get_logger(__name__)
    root = Path(output_root)
    namespace = Path(run_namespace) if run_namespace is not None else Path(config.experiment.name)

    if map_artifacts is None:
        map_artifacts = generate_maps_from_config(config, root)
    if trajectory_artifacts is None:
        trajectory_artifacts = generate_trajectories_from_config(config, root, map_artifacts)

    rows: list[dict[str, Any]] = []
    episode_index = 0
    for benchmark_seed in config.benchmark.seeds:
        for trajectory_path in trajectory_artifacts:
            trajectory = load_trajectory_artifact(trajectory_path)
            map_path = root / "maps" / f"{trajectory.map_id}.npz"
            row = _run_single_episode(
                config=config,
                output_root=root,
                method=method,
                benchmark_seed=benchmark_seed,
                episode_index=episode_index,
                map_path=map_path,
                trajectory_path=trajectory_path,
                run_namespace=namespace,
                sweep_value=sweep_value,
            )
            rows.append(row)
            episode_index += 1

    frame = pd.DataFrame(rows)
    metrics_dir = ensure_dir(root / "metrics")
    metric_path = metrics_dir / f"{_file_slug(namespace)}_{method}_episode_metrics.csv"
    frame.to_csv(metric_path, index=False)

    summary_frame = summarize_metrics_frame(frame)
    summary_path = metrics_dir / f"{_file_slug(namespace)}_{method}_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    logger.info("Completed %d %s episode(s). Metrics written to %s.", len(frame), method, metric_path)
    return frame


def _select_pose_batch(gt_states: np.ndarray, batch_size: int) -> np.ndarray:
    if len(gt_states) == 0:
        raise ValueError("Ground-truth trajectory must contain at least one state.")
    indices = np.arange(batch_size, dtype=int) % len(gt_states)
    return gt_states[indices, 1:4].astype(float)


def _ray_cast_cpu_batch(map_data, poses: np.ndarray, beam_angles: np.ndarray, config: ProjectConfig) -> np.ndarray:
    scans = np.empty((len(poses), len(beam_angles)), dtype=float)
    for index, pose_values in enumerate(poses):
        scans[index] = ray_cast_scan(
            map_data,
            Pose2D(x=float(pose_values[0]), y=float(pose_values[1]), theta=float(pose_values[2])),
            beam_angles,
            min_range_m=config.lidar.min_range_m,
            max_range_m=config.lidar.max_range_m,
            ray_step_m=config.lidar.ray_step_m,
        )
    return scans


def _ray_cast_torch_batch(map_data, poses: np.ndarray, beam_angles: np.ndarray, device_name: str, config: ProjectConfig) -> np.ndarray:
    if not torch_available():
        raise RuntimeError("PyTorch is unavailable.")
    import torch

    device = torch.device(device_name)
    beam_tensor = torch.as_tensor(beam_angles, dtype=torch.float32, device=device)
    chunk_size = 256 if device_name.startswith("cuda") else 64
    outputs: list[np.ndarray] = []
    for start in range(0, len(poses), chunk_size):
        pose_tensor = torch.as_tensor(poses[start : start + chunk_size], dtype=torch.float32, device=device)
        ranges = ray_cast_scan_batched(
            occupancy=map_data.occupancy,
            resolution_m=map_data.resolution_m,
            origin_xy=map_data.origin_xy,
            poses=pose_tensor,
            beam_angles_rad=beam_tensor,
            min_range_m=config.lidar.min_range_m,
            max_range_m=config.lidar.max_range_m,
            ray_step_m=config.lidar.ray_step_m,
        )
        outputs.append(ranges.detach().cpu().numpy())
    return np.vstack(outputs)


def _ray_cast_device_batch(
    map_data,
    poses: np.ndarray,
    beam_angles: np.ndarray,
    config: ProjectConfig,
    device_name: str,
) -> np.ndarray:
    if torch_available():
        return _ray_cast_torch_batch(map_data, poses, beam_angles, device_name, config)
    return _ray_cast_cpu_batch(map_data, poses, beam_angles, config)


def _synchronize_device(device_name: str) -> None:
    if not device_name.startswith("cuda") or not torch_available():
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_scan_throughput(
    map_data,
    poses: np.ndarray,
    beam_angles: np.ndarray,
    config: ProjectConfig,
    device_name: str,
) -> tuple[float, float, float]:
    kernel = lambda: _ray_cast_device_batch(map_data, poses, beam_angles, config, device_name=device_name)

    for _ in range(config.throughput.warmup_iters):
        kernel()
    _synchronize_device(device_name)

    start = time.perf_counter()
    for _ in range(config.throughput.timed_iters):
        kernel()
    _synchronize_device(device_name)
    runtime_s = float(time.perf_counter() - start)
    scan_batches_per_second = float(config.throughput.timed_iters / max(runtime_s, 1e-9))
    scans_per_second = float(config.throughput.timed_iters * len(poses) / max(runtime_s, 1e-9))
    return scan_batches_per_second, scans_per_second, runtime_s


def _measure_particle_likelihood_throughput(
    map_data,
    reference_pose: np.ndarray,
    beam_angles: np.ndarray,
    config: ProjectConfig,
    device_name: str,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    particle_count = int(config.mcl.particle_count) * int(batch_size)
    particles = sample_initial_particles(map_data, reference_pose, particle_count, rng)
    measurement_beams = beam_angles[np.linspace(0, len(beam_angles) - 1, min(config.mcl.measurement_beams, len(beam_angles)), dtype=int)]
    observed_ranges = _ray_cast_cpu_batch(map_data, reference_pose[None, :], measurement_beams, config)[0]
    sigma = max(config.lidar.range_noise_std_m, 0.05)

    kernel = lambda: _ray_cast_device_batch(map_data, particles, measurement_beams, config, device_name=device_name)

    for _ in range(config.throughput.warmup_iters):
        expected = kernel()
        _ = -0.5 * np.sum(np.square((expected - observed_ranges[None, :]) / sigma), axis=1)
    _synchronize_device(device_name)

    start = time.perf_counter()
    for _ in range(config.throughput.timed_iters):
        expected = kernel()
        _ = -0.5 * np.sum(np.square((expected - observed_ranges[None, :]) / sigma), axis=1)
    _synchronize_device(device_name)
    runtime_s = float(time.perf_counter() - start)
    return float(config.throughput.timed_iters * particle_count / max(runtime_s, 1e-9))


def _device_available(device_name: str) -> bool:
    if device_name == "cpu":
        return True
    if not device_name.startswith("cuda"):
        return False
    if not torch_available():
        return False
    import torch

    return bool(torch.cuda.is_available())


def _run_throughput_benchmark(config: ProjectConfig, output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_logger(__name__)
    root = Path(output_root)
    run_root = ensure_dir(root / "runs" / config.experiment.name)
    save_resolved_config(config, run_root / "resolved_config.yaml")

    map_artifacts = generate_maps_from_config(config, root)
    trajectory_artifacts = generate_trajectories_from_config(config, root, map_artifacts)
    map_data = load_map_artifact(map_artifacts[0])
    trajectory = load_trajectory_artifact(trajectory_artifacts[0])
    gt_states = trajectory.states[: config.benchmark.step_limit].copy()
    beam_angles = make_beam_angles(config.lidar.num_beams, config.lidar.fov_deg)
    batch_sizes = [int(value) for value in (config.experiment.values if config.experiment.values else [config.benchmark.batch_size])]
    throughput_metrics_path = run_root / "throughput_metrics.csv"
    throughput_summary_path = run_root / "throughput_summary.csv"
    metrics_dir = ensure_dir(root / "metrics")
    aggregate_metrics_path = metrics_dir / f"{config.experiment.name}_throughput_metrics.csv"
    aggregate_summary_path = metrics_dir / f"{config.experiment.name}_summary.csv"

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 7000)
    for device_name in config.throughput.devices:
        available = _device_available(device_name)
        for batch_size in batch_sizes:
            row: dict[str, Any] = {
                "experiment": config.experiment.name,
                "device": device_name,
                "batch_size": int(batch_size),
                "available": bool(available),
                "scan_batches_per_second": 0.0,
                "scans_per_second": 0.0,
                "particle_likelihoods_per_second": 0.0,
                "runtime_s": 0.0,
            }
            if available:
                poses = _select_pose_batch(gt_states, batch_size)
                scan_batches_per_second, scans_per_second, scan_runtime_s = _measure_scan_throughput(
                    map_data=map_data,
                    poses=poses,
                    beam_angles=beam_angles,
                    config=config,
                    device_name=device_name,
                )
                particle_likelihoods_per_second = _measure_particle_likelihood_throughput(
                    map_data=map_data,
                    reference_pose=poses[0],
                    beam_angles=beam_angles,
                    config=config,
                    device_name=device_name,
                    batch_size=batch_size,
                    rng=rng,
                )
                row.update(
                    {
                        "scan_batches_per_second": scan_batches_per_second,
                        "scans_per_second": scans_per_second,
                        "particle_likelihoods_per_second": particle_likelihoods_per_second,
                        "runtime_s": scan_runtime_s,
                    }
                )
            rows.append(row)
            partial_metrics = pd.DataFrame(rows)
            partial_summary = summarize_metrics_frame(partial_metrics)
            partial_metrics.to_csv(throughput_metrics_path, index=False)
            partial_summary.to_csv(throughput_summary_path, index=False)
            partial_metrics.to_csv(aggregate_metrics_path, index=False)
            partial_summary.to_csv(aggregate_summary_path, index=False)

    metrics = pd.DataFrame(rows)
    summary = summarize_metrics_frame(metrics)
    metrics.to_csv(throughput_metrics_path, index=False)
    summary.to_csv(throughput_summary_path, index=False)
    metrics.to_csv(aggregate_metrics_path, index=False)
    summary.to_csv(aggregate_summary_path, index=False)
    logger.info("Throughput benchmark written to %s.", throughput_metrics_path)
    return metrics, summary


def run_benchmark_experiment(config: ProjectConfig, output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_logger(__name__)
    if _is_throughput_benchmark(config):
        return _run_throughput_benchmark(config, output_root)

    methods = _active_methods(config)
    if not methods:
        raise ValueError("At least one method must be enabled for benchmark execution.")

    sweep_values = config.experiment.values if config.experiment.values else [None]
    all_rows: list[pd.DataFrame] = []
    for sweep_value in sweep_values:
        run_config = deepcopy(config)
        if sweep_value is not None and config.experiment.sweep_key:
            run_config = _set_nested_value(run_config, config.experiment.sweep_key, sweep_value)
        namespace = _namespace_path(config.experiment.name, sweep_value)
        map_artifacts = generate_maps_from_config(run_config, output_root)
        trajectory_artifacts = generate_trajectories_from_config(run_config, output_root, map_artifacts)
        for method in methods:
            frame = run_method_experiment(
                config=run_config,
                output_root=output_root,
                method=method,
                run_namespace=namespace,
                sweep_value=sweep_value,
                map_artifacts=map_artifacts,
                trajectory_artifacts=trajectory_artifacts,
            )
            all_rows.append(frame)

    non_empty_rows = [frame for frame in all_rows if not frame.empty]
    if non_empty_rows:
        row_records = [record for frame in non_empty_rows for record in frame.to_dict(orient="records")]
        episode_metrics = pd.DataFrame(row_records)
    else:
        episode_metrics = pd.DataFrame()
    summary_frame = summarize_metrics_frame(episode_metrics)

    metrics_dir = ensure_dir(Path(output_root) / "metrics")
    episode_path = metrics_dir / f"{config.experiment.name}_episode_metrics.csv"
    summary_path = metrics_dir / f"{config.experiment.name}_summary.csv"
    episode_metrics.to_csv(episode_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    logger.info("Benchmark summary written to %s.", summary_path)
    return episode_metrics, summary_frame
