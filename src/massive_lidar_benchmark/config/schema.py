"""Typed config schema for the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MapConfig:
    family: str = "room"
    width_m: float = 40.0
    height_m: float = 40.0
    resolution_m: float = 0.05
    robot_radius_m: float = 0.20
    obstacle_density: float = 0.10
    wall_thickness_cells: int = 4


@dataclass(slots=True)
class TrajectoryConfig:
    pattern: str = "explore"
    dt_s: float = 0.1
    horizon_s: float = 50.0
    target_speed_mps: float = 0.7
    max_turn_rate_rps: float = 0.8
    num_waypoints: int = 8


@dataclass(slots=True)
class LidarConfig:
    num_beams: int = 180
    fov_deg: float = 270.0
    min_range_m: float = 0.10
    max_range_m: float = 20.0
    ray_step_m: float = 0.025
    range_noise_std_m: float = 0.02
    dropout_prob: float = 0.0
    outlier_prob: float = 0.0


@dataclass(slots=True)
class EKFConfig:
    enabled: bool = True
    sparse_beams: int = 16
    process_noise_diag: list[float] = field(default_factory=lambda: [0.03, 0.03, 0.02])
    measurement_noise_std_m: float = 0.08
    jacobian_eps: list[float] = field(default_factory=lambda: [0.02, 0.02, 0.01])


@dataclass(slots=True)
class MCLConfig:
    enabled: bool = True
    particle_count: int = 1024
    measurement_beams: int = 64
    motion_noise_std: list[float] = field(default_factory=lambda: [0.04, 0.04, 0.03])
    resample_ess_ratio: float = 0.5


@dataclass(slots=True)
class BenchmarkConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    maps_per_family: int = 3
    trajectories_per_map: int = 2
    map_families: list[str] = field(default_factory=list)
    kidnapped_enabled: bool = False
    step_limit: int = 500
    batch_size: int = 1


@dataclass(slots=True)
class RenderConfig:
    dpi: int = 180
    fps: int = 20
    frame_size: list[int] = field(default_factory=lambda: [1280, 720])
    frame_stride: int = 2
    max_frames: int | None = None
    font_family: str = "DejaVu Serif"
    cleanup_frames: bool = False
    target_video_name: str = "demo_main_localization"
    target_run_dir: str = "outputs/runs"
    strict_run_root: bool = False
    allow_fallback: bool = True
    source_experiment: str | None = None
    figure_names: list[str] = field(default_factory=list)
    gif_scale_width: int = 960


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "baseline"
    sweep_key: str | None = None
    values: list[Any] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class ThroughputConfig:
    devices: list[str] = field(default_factory=list)
    warmup_iters: int = 10
    timed_iters: int = 50


@dataclass(slots=True)
class ProjectConfig:
    project_name: str = "MASSIVE-PARALLEL LIDAR LOCALIZATION BENCHMARK"
    seed: int = 42
    device: str = "cpu"
    output_root: str = "outputs"
    map: MapConfig = field(default_factory=MapConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    mcl: MCLConfig = field(default_factory=MCLConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    throughput: ThroughputConfig = field(default_factory=ThroughputConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ProjectConfig":
        return cls(
            project_name=str(mapping.get("project_name", "MASSIVE-PARALLEL LIDAR LOCALIZATION BENCHMARK")),
            seed=int(mapping.get("seed", 42)),
            device=str(mapping.get("device", "cpu")),
            output_root=str(mapping.get("output_root", "outputs")),
            map=MapConfig(**mapping.get("map", {})),
            trajectory=TrajectoryConfig(**mapping.get("trajectory", {})),
            lidar=LidarConfig(**mapping.get("lidar", {})),
            ekf=EKFConfig(**mapping.get("ekf", {})),
            mcl=MCLConfig(**mapping.get("mcl", {})),
            benchmark=BenchmarkConfig(**mapping.get("benchmark", {})),
            throughput=ThroughputConfig(**mapping.get("throughput", {})),
            render=RenderConfig(**mapping.get("render", {})),
            experiment=ExperimentConfig(**mapping.get("experiment", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)
