import numpy as np

from massive_lidar_benchmark.localization.ekf import predict_step, update_step


def test_ekf_predict_update_shapes() -> None:
    mean = np.array([1.0, 2.0, 0.1])
    covariance = np.eye(3) * 0.1
    control = np.array([0.5, 0.05])
    predicted_mean, predicted_cov = predict_step(mean, covariance, control, dt_s=0.1, process_noise_diag=np.array([0.01, 0.01, 0.01]))

    jacobian_h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    residual = np.array([0.1, -0.1])
    updated_mean, updated_cov = update_step(predicted_mean, predicted_cov, residual, jacobian_h, np.array([0.05, 0.05]))

    assert updated_mean.shape == (3,)
    assert updated_cov.shape == (3, 3)
    assert np.allclose(updated_cov, updated_cov.T)

