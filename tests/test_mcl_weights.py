import numpy as np

from massive_lidar_benchmark.localization.mcl import effective_sample_size, normalize_log_weights, systematic_resample


def test_weight_normalization_and_resampling() -> None:
    rng = np.random.default_rng(123)
    weights = normalize_log_weights(np.array([-2.0, -1.0, -0.5, -3.0]))
    assert np.isclose(np.sum(weights), 1.0)
    ess = effective_sample_size(weights)
    assert 1.0 <= ess <= 4.0
    indices = systematic_resample(weights, rng)
    assert indices.shape == (4,)
    assert np.all(indices >= 0)
    assert np.all(indices < 4)

