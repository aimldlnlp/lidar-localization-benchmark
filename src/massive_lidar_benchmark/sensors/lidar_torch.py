"""Optional batched PyTorch LiDAR ray casting."""

from __future__ import annotations

import numpy as np

_TORCH_MODULE = None
_TORCH_ATTEMPTED = False


def _get_torch():
    global _TORCH_MODULE, _TORCH_ATTEMPTED
    if _TORCH_ATTEMPTED:
        return _TORCH_MODULE
    _TORCH_ATTEMPTED = True
    try:
        import torch as torch_module
    except ImportError:  # pragma: no cover - exercised when torch is absent
        _TORCH_MODULE = None
    else:
        _TORCH_MODULE = torch_module
    return _TORCH_MODULE


def torch_available() -> bool:
    return _get_torch() is not None


def ray_cast_scan_batched(
    occupancy: np.ndarray,
    resolution_m: float,
    origin_xy: tuple[float, float],
    poses,
    beam_angles_rad,
    min_range_m: float,
    max_range_m: float,
    ray_step_m: float,
):
    torch = _get_torch()
    if torch is None:
        raise ImportError("PyTorch is required for lidar_torch.ray_cast_scan_batched.")

    device = poses.device
    occ = torch.as_tensor(occupancy.astype(np.bool_), device=device)
    steps = torch.arange(min_range_m, max_range_m + ray_step_m, ray_step_m, device=device)
    angles = poses[:, 2:3] + beam_angles_rad[None, :]
    cosines = torch.cos(angles)[:, :, None]
    sines = torch.sin(angles)[:, :, None]

    xs = poses[:, 0:1, None] + cosines * steps[None, None, :]
    ys = poses[:, 1:2, None] + sines * steps[None, None, :]

    gx = torch.floor((xs - origin_xy[0]) / resolution_m).long()
    gy = torch.floor((ys - origin_xy[1]) / resolution_m).long()

    height, width = occupancy.shape
    in_bounds = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    hits = torch.zeros_like(in_bounds)
    valid_gx = torch.where(in_bounds, gx, torch.zeros_like(gx))
    valid_gy = torch.where(in_bounds, gy, torch.zeros_like(gy))
    hits[in_bounds] = occ[valid_gy[in_bounds], valid_gx[in_bounds]]
    hits = hits | (~in_bounds)

    first_hit_index = torch.argmax(hits.to(torch.int64), dim=-1)
    no_hit_mask = ~hits.any(dim=-1)
    ranges = steps[first_hit_index]
    ranges = torch.where(no_hit_mask, torch.full_like(ranges, max_range_m), ranges)
    return ranges
