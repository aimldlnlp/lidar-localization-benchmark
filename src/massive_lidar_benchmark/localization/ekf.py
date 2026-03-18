"""Minimal EKF math utilities for the baseline implementation."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.core.math2d import wrap_angle
from massive_lidar_benchmark.core.types import OccupancyMap, Pose2D
from massive_lidar_benchmark.localization.measurement import extract_sparse_measurement
from massive_lidar_benchmark.localization.motion import unicycle_step
from massive_lidar_benchmark.sensors.lidar_cpu import ray_cast_scan


def predict_step(
    mean: np.ndarray,
    covariance: np.ndarray,
    control: np.ndarray,
    dt_s: float,
    process_noise_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, theta = mean
    v, _ = control
    predicted_mean = unicycle_step(mean, control, dt_s)
    jacobian_f = np.array(
        [
            [1.0, 0.0, -dt_s * v * np.sin(theta)],
            [0.0, 1.0, dt_s * v * np.cos(theta)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    process_noise = np.diag(process_noise_diag)
    predicted_cov = jacobian_f @ covariance @ jacobian_f.T + process_noise
    return predicted_mean, predicted_cov


def update_step(
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement_residual: np.ndarray,
    measurement_jacobian: np.ndarray,
    measurement_noise_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    measurement_noise = np.diag(measurement_noise_diag)
    innovation_cov = measurement_jacobian @ covariance @ measurement_jacobian.T + measurement_noise
    kalman_gain = covariance @ measurement_jacobian.T @ np.linalg.pinv(innovation_cov)
    updated_mean = mean + kalman_gain @ measurement_residual
    identity = np.eye(covariance.shape[0])
    updated_cov = (identity - kalman_gain @ measurement_jacobian) @ covariance
    updated_cov = 0.5 * (updated_cov + updated_cov.T)
    return updated_mean, updated_cov


def _expected_ranges(
    map_data: OccupancyMap,
    pose_state: np.ndarray,
    beam_angles_rad: np.ndarray,
    config: ProjectConfig,
) -> np.ndarray:
    pose = Pose2D(x=float(pose_state[0]), y=float(pose_state[1]), theta=float(pose_state[2]))
    return ray_cast_scan(
        map_data,
        pose,
        beam_angles_rad,
        min_range_m=config.lidar.min_range_m,
        max_range_m=config.lidar.max_range_m,
        ray_step_m=config.lidar.ray_step_m,
    )


def numerical_measurement_jacobian(
    map_data: OccupancyMap,
    mean: np.ndarray,
    beam_angles_rad: np.ndarray,
    config: ProjectConfig,
) -> np.ndarray:
    eps = np.asarray(config.ekf.jacobian_eps, dtype=float)
    jacobian = np.zeros((len(beam_angles_rad), 3), dtype=float)
    for axis in range(3):
        perturb = np.zeros(3, dtype=float)
        perturb[axis] = eps[axis]
        plus_state = mean + perturb
        minus_state = mean - perturb
        plus_state[2] = wrap_angle(plus_state[2])
        minus_state[2] = wrap_angle(minus_state[2])
        scan_plus = _expected_ranges(map_data, plus_state, beam_angles_rad, config)
        scan_minus = _expected_ranges(map_data, minus_state, beam_angles_rad, config)
        jacobian[:, axis] = (scan_plus - scan_minus) / max(2.0 * eps[axis], 1e-6)
    return jacobian


def run_ekf_localization(
    map_data: OccupancyMap,
    gt_states: np.ndarray,
    observed_scans: np.ndarray,
    beam_angles_rad: np.ndarray,
    config: ProjectConfig,
    initial_mean: np.ndarray,
    kidnap_step: int | None = None,
    kidnap_pose: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    num_steps = len(gt_states)
    estimates = np.zeros((num_steps, 3), dtype=float)
    valid_beam_fraction = np.zeros(num_steps, dtype=float)

    mean = np.asarray(initial_mean, dtype=float).copy()
    mean[2] = wrap_angle(mean[2])
    covariance = np.diag(np.array([0.5, 0.5, 0.25], dtype=float))
    process_noise_diag = np.asarray(config.ekf.process_noise_diag, dtype=float)
    measurement_noise_diag_value = float(config.ekf.measurement_noise_std_m**2)

    for step_index in range(num_steps):
        if step_index == kidnap_step and kidnap_pose is not None:
            mean = np.asarray(kidnap_pose, dtype=float).copy()
            mean[2] = wrap_angle(mean[2])
            covariance = np.diag(np.array([1.0, 1.0, 0.5], dtype=float))

        if step_index > 0:
            dt_s = float(gt_states[step_index, 0] - gt_states[step_index - 1, 0])
            control = gt_states[step_index, 4:6]
            mean, covariance = predict_step(mean, covariance, control, dt_s, process_noise_diag)
            mean[2] = wrap_angle(mean[2])

        observed_ranges, observed_angles, _, fraction = extract_sparse_measurement(
            observed_scans[step_index],
            beam_angles_rad,
            config.ekf.sparse_beams,
            config.lidar.max_range_m,
            config.lidar.ray_step_m,
        )
        valid_beam_fraction[step_index] = fraction

        if len(observed_ranges) >= 3:
            expected_ranges = _expected_ranges(map_data, mean, observed_angles, config)
            jacobian = numerical_measurement_jacobian(map_data, mean, observed_angles, config)
            residual = observed_ranges - expected_ranges
            mean, covariance = update_step(
                mean,
                covariance,
                residual,
                jacobian,
                np.full(len(observed_ranges), measurement_noise_diag_value, dtype=float),
            )
            mean[2] = wrap_angle(mean[2])

        estimates[step_index] = mean

    return {
        "estimates": estimates,
        "valid_beam_fraction": valid_beam_fraction,
        "covariance_diag": np.diag(covariance),
    }
