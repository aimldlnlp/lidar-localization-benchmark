"""Shared dataclasses for maps, trajectories, and scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass(slots=True)
class Control2D:
    v: float
    omega: float


@dataclass(slots=True)
class OccupancyMap:
    occupancy: np.ndarray
    resolution_m: float
    origin_xy: tuple[float, float]
    family: str
    seed: int

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.occupancy.shape)


@dataclass(slots=True)
class Trajectory:
    states: np.ndarray
    dt_s: float
    pattern: str
    map_id: str
    seed: int


@dataclass(slots=True)
class LidarScan:
    ranges_m: np.ndarray
    beam_angles_rad: np.ndarray
    pose: Pose2D

