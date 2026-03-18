"""Metrics used by the benchmark and tests."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.core.math2d import wrap_angle


def position_rmse(gt_xy: np.ndarray, est_xy: np.ndarray) -> float:
    delta = gt_xy - est_xy
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def heading_rmse_deg(gt_theta: np.ndarray, est_theta: np.ndarray) -> float:
    delta = wrap_angle(gt_theta - est_theta)
    return float(np.degrees(np.sqrt(np.mean(delta * delta))))


def translation_errors(gt_xy: np.ndarray, est_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(gt_xy - est_xy, axis=1)


def convergence_time_s(errors_m: np.ndarray, dt_s: float, threshold_m: float = 0.5, window: int = 20) -> float | None:
    below = errors_m < threshold_m
    if len(errors_m) < window:
        return None
    for start in range(0, len(errors_m) - window + 1):
        if np.all(below[start : start + window]):
            return float(start * dt_s)
    return None


def kidnapped_recovery_success(
    errors_m: np.ndarray,
    dt_s: float,
    kidnap_step: int | None,
    threshold_m: float = 0.75,
    max_recovery_time_s: float = 5.0,
    sustain_time_s: float = 2.0,
) -> bool:
    if kidnap_step is None:
        return False
    max_recovery_steps = max(1, int(np.ceil(max_recovery_time_s / max(dt_s, 1e-6))))
    sustain_steps = max(1, int(np.ceil(sustain_time_s / max(dt_s, 1e-6))))
    end_limit = min(len(errors_m), kidnap_step + max_recovery_steps + sustain_steps)
    for start in range(kidnap_step, max(kidnap_step, end_limit - sustain_steps + 1)):
        if np.all(errors_m[start : start + sustain_steps] < threshold_m):
            return True
    return False


def summarize_episode_metrics(
    gt_states: np.ndarray,
    est_states: np.ndarray,
    dt_s: float,
    valid_beam_fraction: np.ndarray,
    ess_values: np.ndarray | None = None,
    kidnap_step: int | None = None,
) -> dict[str, float | int | bool | None]:
    gt_xy = gt_states[:, 1:3]
    est_xy = est_states[:, :2]
    gt_theta = gt_states[:, 3]
    est_theta = est_states[:, 2]

    trans_errors = translation_errors(gt_xy, est_xy)
    heading_errors_deg = np.degrees(np.abs(wrap_angle(gt_theta - est_theta)))
    convergence = convergence_time_s(trans_errors, dt_s=dt_s)
    final_translation = float(trans_errors[-1])
    final_heading = float(heading_errors_deg[-1])
    failed = bool(
        (final_translation > 1.5)
        or (np.mean(trans_errors > 2.0) > 0.25)
    )

    metrics: dict[str, float | int | bool | None] = {
        "position_rmse_m": position_rmse(gt_xy, est_xy),
        "heading_rmse_deg": heading_rmse_deg(gt_theta, est_theta),
        "median_translation_error_m": float(np.median(trans_errors)),
        "final_translation_error_m": final_translation,
        "final_heading_error_deg": final_heading,
        "convergence_time_s": convergence,
        "failed": failed,
        "mean_valid_beam_fraction": float(np.mean(valid_beam_fraction)),
        "kidnapped_recovered": kidnapped_recovery_success(trans_errors, dt_s, kidnap_step),
    }
    if ess_values is not None:
        metrics["ess_mean"] = float(np.mean(ess_values))
        metrics["ess_min"] = float(np.min(ess_values))
    else:
        metrics["ess_mean"] = None
        metrics["ess_min"] = None
    return metrics
