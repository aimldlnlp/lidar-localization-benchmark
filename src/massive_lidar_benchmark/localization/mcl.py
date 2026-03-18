"""Particle-filter helpers used by tests and future implementation."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.core.math2d import wrap_angle
from massive_lidar_benchmark.core.types import OccupancyMap, Pose2D
from massive_lidar_benchmark.localization.measurement import extract_sparse_measurement
from massive_lidar_benchmark.maps.validate import is_world_point_free, sample_free_points
from massive_lidar_benchmark.sensors.lidar_cpu import ray_cast_scan
from massive_lidar_benchmark.sensors.lidar_torch import ray_cast_scan_batched, torch_available


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    weights_sum = np.sum(weights)
    if weights_sum <= 0.0:
        raise ValueError("Particle weights sum to zero.")
    return weights / weights_sum


def effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    indexes = np.zeros(n, dtype=int)
    i = 0
    j = 0
    while i < n:
        if positions[i] < cumulative[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def weighted_pose_mean(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean_x = float(np.sum(weights * particles[:, 0]))
    mean_y = float(np.sum(weights * particles[:, 1]))
    sin_sum = float(np.sum(weights * np.sin(particles[:, 2])))
    cos_sum = float(np.sum(weights * np.cos(particles[:, 2])))
    mean_theta = float(np.arctan2(sin_sum, cos_sum))
    return np.array([mean_x, mean_y, mean_theta], dtype=float)


def _sample_pose_near(
    map_data: OccupancyMap,
    center_pose: np.ndarray,
    rng: np.random.Generator,
    std_xy_m: float,
    std_theta_rad: float,
) -> np.ndarray:
    for _ in range(64):
        candidate = np.asarray(center_pose, dtype=float).copy()
        candidate[0] += rng.normal(0.0, std_xy_m)
        candidate[1] += rng.normal(0.0, std_xy_m)
        candidate[2] = wrap_angle(candidate[2] + rng.normal(0.0, std_theta_rad))
        if is_world_point_free(map_data, float(candidate[0]), float(candidate[1])):
            return candidate
    fallback = np.asarray(center_pose, dtype=float).copy()
    fallback[2] = wrap_angle(fallback[2])
    return fallback


def sample_initial_particles(
    map_data: OccupancyMap,
    center_pose: np.ndarray,
    particle_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    particles = np.zeros((particle_count, 3), dtype=float)
    for index in range(particle_count):
        particles[index] = _sample_pose_near(map_data, center_pose, rng, std_xy_m=0.7, std_theta_rad=0.35)
    return particles


def sample_kidnap_pose(
    map_data: OccupancyMap,
    reference_pose: np.ndarray,
    rng: np.random.Generator,
    min_distance_m: float = 3.0,
) -> np.ndarray:
    for _ in range(64):
        point = sample_free_points(map_data, 1, rng)[0]
        distance = float(np.linalg.norm(point - reference_pose[:2]))
        if distance >= min_distance_m:
            return np.array([point[0], point[1], rng.uniform(-np.pi, np.pi)], dtype=float)
    point = sample_free_points(map_data, 1, rng)[0]
    return np.array([point[0], point[1], rng.uniform(-np.pi, np.pi)], dtype=float)


def _predict_particles(
    particles: np.ndarray,
    control: np.ndarray,
    dt_s: float,
    noise_std: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    predicted = particles.copy()
    v, omega = control
    predicted[:, 0] += dt_s * v * np.cos(predicted[:, 2]) + rng.normal(0.0, noise_std[0], size=len(predicted))
    predicted[:, 1] += dt_s * v * np.sin(predicted[:, 2]) + rng.normal(0.0, noise_std[1], size=len(predicted))
    predicted[:, 2] = wrap_angle(predicted[:, 2] + dt_s * omega + rng.normal(0.0, noise_std[2], size=len(predicted)))
    return predicted


def _expected_ranges_cpu(
    map_data: OccupancyMap,
    particles: np.ndarray,
    beam_angles_rad: np.ndarray,
    config: ProjectConfig,
) -> np.ndarray:
    expected = np.empty((len(particles), len(beam_angles_rad)), dtype=float)
    for index, particle in enumerate(particles):
        pose = Pose2D(x=float(particle[0]), y=float(particle[1]), theta=float(particle[2]))
        expected[index] = ray_cast_scan(
            map_data,
            pose,
            beam_angles_rad,
            min_range_m=config.lidar.min_range_m,
            max_range_m=config.lidar.max_range_m,
            ray_step_m=config.lidar.ray_step_m,
        )
    return expected


def _expected_ranges_torch(
    map_data: OccupancyMap,
    particles: np.ndarray,
    beam_angles_rad: np.ndarray,
    config: ProjectConfig,
) -> np.ndarray:
    if not torch_available():
        raise RuntimeError("PyTorch is unavailable.")
    import torch

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    beam_tensor = torch.as_tensor(beam_angles_rad, dtype=torch.float32, device=device)
    chunk_size = min(256, len(particles))
    outputs: list[np.ndarray] = []
    for start in range(0, len(particles), chunk_size):
        chunk = torch.as_tensor(particles[start : start + chunk_size], dtype=torch.float32, device=device)
        ranges = ray_cast_scan_batched(
            occupancy=map_data.occupancy,
            resolution_m=map_data.resolution_m,
            origin_xy=map_data.origin_xy,
            poses=chunk,
            beam_angles_rad=beam_tensor,
            min_range_m=config.lidar.min_range_m,
            max_range_m=config.lidar.max_range_m,
            ray_step_m=config.lidar.ray_step_m,
        )
        outputs.append(ranges.detach().cpu().numpy())
    return np.vstack(outputs)


def run_mcl_localization(
    map_data: OccupancyMap,
    gt_states: np.ndarray,
    observed_scans: np.ndarray,
    beam_angles_rad: np.ndarray,
    config: ProjectConfig,
    initial_mean: np.ndarray,
    rng: np.random.Generator,
    kidnap_step: int | None = None,
    kidnap_pose: np.ndarray | None = None,
    record_history: bool = False,
    history_particle_limit: int = 128,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    particle_count = int(config.mcl.particle_count)
    particles = sample_initial_particles(map_data, initial_mean, particle_count, rng)
    weights = np.full(particle_count, 1.0 / particle_count, dtype=float)

    num_steps = len(gt_states)
    estimates = np.zeros((num_steps, 3), dtype=float)
    ess_values = np.zeros(num_steps, dtype=float)
    valid_beam_fraction = np.zeros(num_steps, dtype=float)
    snapshot_steps = sorted(set([0, max(0, num_steps // 2), max(0, num_steps - 1)]))
    snapshots: dict[str, np.ndarray] = {}
    particle_history: dict[str, np.ndarray] = {}
    motion_noise = np.asarray(config.mcl.motion_noise_std, dtype=float)

    for step_index in range(num_steps):
        if step_index == kidnap_step and kidnap_pose is not None:
            particles = sample_initial_particles(map_data, kidnap_pose, particle_count, rng)
            weights.fill(1.0 / particle_count)

        if step_index > 0:
            dt_s = float(gt_states[step_index, 0] - gt_states[step_index - 1, 0])
            control = gt_states[step_index, 4:6]
            particles = _predict_particles(particles, control, dt_s, motion_noise, rng)

        observed_ranges, observed_angles, _, fraction = extract_sparse_measurement(
            observed_scans[step_index],
            beam_angles_rad,
            config.mcl.measurement_beams,
            config.lidar.max_range_m,
            config.lidar.ray_step_m,
        )
        valid_beam_fraction[step_index] = fraction

        if len(observed_ranges) >= 3:
            use_torch = False
            if config.device.startswith("cuda") and torch_available():
                import torch

                use_torch = torch.cuda.is_available()
            if use_torch:
                expected = _expected_ranges_torch(map_data, particles, observed_angles, config)
            else:
                expected = _expected_ranges_cpu(map_data, particles, observed_angles, config)
            residual = np.clip(expected - observed_ranges[None, :], -2.0, 2.0)
            sigma = max(config.lidar.range_noise_std_m, 0.05)
            log_weights = -0.5 * np.sum(np.square(residual / sigma), axis=1)
            weights = normalize_log_weights(log_weights)

        estimates[step_index] = weighted_pose_mean(particles, weights)
        ess_values[step_index] = effective_sample_size(weights)

        if step_index in snapshot_steps:
            sample_count = min(256, particle_count)
            sample_indices = rng.choice(particle_count, size=sample_count, replace=False)
            snapshots[str(step_index)] = particles[sample_indices].copy()
        if record_history:
            sample_count = min(max(1, history_particle_limit), particle_count)
            sample_indices = rng.choice(particle_count, size=sample_count, replace=False)
            particle_history[str(step_index)] = particles[sample_indices].copy()

        if ess_values[step_index] < config.mcl.resample_ess_ratio * particle_count:
            indices = systematic_resample(weights, rng)
            particles = particles[indices]
            weights.fill(1.0 / particle_count)

    return {
        "estimates": estimates,
        "ess": ess_values,
        "valid_beam_fraction": valid_beam_fraction,
        "particle_snapshots": snapshots,
        "particle_history": particle_history,
    }
