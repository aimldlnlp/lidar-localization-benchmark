import numpy as np
import pytest

from massive_lidar_benchmark.sensors.lidar_torch import ray_cast_scan_batched, torch_available


def test_torch_ray_cast_matches_simple_case() -> None:
    if not torch_available():
        pytest.skip("PyTorch is not installed.")

    import torch

    occupancy = np.zeros((12, 12), dtype=bool)
    occupancy[:, 8] = True
    poses = torch.tensor([[5.5, 5.5, 0.0]], dtype=torch.float32)
    beam_angles = torch.tensor([0.0], dtype=torch.float32)
    ranges = ray_cast_scan_batched(
        occupancy=occupancy,
        resolution_m=1.0,
        origin_xy=(0.0, 0.0),
        poses=poses,
        beam_angles_rad=beam_angles,
        min_range_m=0.5,
        max_range_m=10.0,
        ray_step_m=0.5,
    )
    assert torch.allclose(ranges, torch.tensor([[2.5]], dtype=torch.float32))

