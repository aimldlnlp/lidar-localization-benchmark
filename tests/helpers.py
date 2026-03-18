from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from massive_lidar_benchmark.core.io import ensure_dir


def write_episode_artifact(
    output_root: Path,
    run_name: str,
    method: str,
    noise_value: float,
    seed: int,
    episode_index: int,
    position_rmse_m: float,
    failed: bool,
    map_family: str = "open",
    map_id: str = "open_seed0001_map00",
    traj_id: str = "open_seed0001_map00_traj00",
) -> Path:
    noise_slug = f"noise_{noise_value:.2f}".replace(".", "p")
    run_dir = ensure_dir(output_root / "runs" / run_name / noise_slug / method / f"seed_{seed:03d}" / f"episode_{episode_index:03d}")
    step_frame = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "t": [0.0, 0.1, 0.2],
            "x_gt": [0.0, 0.1, 0.2],
            "y_gt": [0.0, 0.0, 0.0],
            "theta_gt": [0.0, 0.0, 0.0],
            "x_est": [0.0, 0.1, 0.2],
            "y_est": [0.0, 0.01, 0.01],
            "theta_est": [0.0, 0.0, 0.0],
            "translation_error_m": [position_rmse_m] * 3,
            "heading_error_deg": [1.0, 1.0, 1.0],
            "valid_beam_fraction": [1.0, 1.0, 1.0],
            "ess": [np.nan, np.nan, np.nan] if method == "ekf" else [32.0, 28.0, 24.0],
        }
    )
    step_frame.to_csv(run_dir / "step_metrics.csv", index=False)
    step_frame[
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
    ].to_csv(run_dir / "estimates.csv", index=False)

    episode_metrics = pd.DataFrame(
        [
            {
                "experiment": run_name,
                "method": method,
                "map_family": map_family,
                "map_id": map_id,
                "traj_id": traj_id,
                "seed": seed,
                "episode_index": episode_index,
                "runtime_s": 0.5,
                "scans_per_second": 6.0,
                "episodes_per_second": 2.0,
                "kidnap_step": np.nan,
                "sweep_value": noise_value,
                "range_noise_std_m": noise_value,
                "dropout_prob": 0.0,
                "outlier_prob": 0.0,
                "batch_size": 1,
                "position_rmse_m": position_rmse_m,
                "heading_rmse_deg": 1.0,
                "median_translation_error_m": position_rmse_m,
                "convergence_time_s": 0.5,
                "failed": failed,
                "kidnapped_recovered": False,
                "ess_mean": np.nan if method == "ekf" else 28.0,
                "ess_min": np.nan if method == "ekf" else 24.0,
            }
        ]
    )
    episode_metrics.to_csv(run_dir / "episode_metrics.csv", index=False)

    gt_states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.6, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.6, 0.0],
            [0.2, 0.2, 0.0, 0.0, 0.6, 0.0],
        ],
        dtype=float,
    )
    estimated_states = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.01, 0.0],
            [0.2, 0.01, 0.0],
        ],
        dtype=float,
    )
    observed_scans = np.ones((3, 4), dtype=float)
    beam_angles_rad = np.array([-0.5, -0.1, 0.1, 0.5], dtype=float)
    np.savez_compressed(
        run_dir / "episode_data.npz",
        gt_states=gt_states,
        estimated_states=estimated_states,
        observed_scans=observed_scans,
        beam_angles_rad=beam_angles_rad,
        initial_mean=np.array([0.0, 0.0, 0.0], dtype=float),
        kidnap_step=-1,
        kidnap_pose=np.zeros(3, dtype=float),
    )
    if method == "mcl":
        np.savez_compressed(run_dir / "particle_snapshots.npz", **{"0": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)})
    return run_dir
