import numpy as np

from massive_lidar_benchmark.benchmarks.metrics import convergence_time_s, heading_rmse_deg, position_rmse


def test_metrics_are_computed_correctly() -> None:
    gt_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    est_xy = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert np.isclose(position_rmse(gt_xy, est_xy), np.sqrt(0.5))

    gt_theta = np.array([0.0, 0.0])
    est_theta = np.array([0.0, np.pi / 2.0])
    assert np.isclose(heading_rmse_deg(gt_theta, est_theta), 63.63961030678928)

    errors = np.array([1.0] * 10 + [0.1] * 25)
    assert np.isclose(convergence_time_s(errors, dt_s=0.1, threshold_m=0.5, window=20), 1.0)

