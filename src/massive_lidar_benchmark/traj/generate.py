"""Trajectory artifact generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.core.io import ensure_dir, write_yaml
from massive_lidar_benchmark.core.logging import get_logger
from massive_lidar_benchmark.core.seeding import spawn_seed_integers
from massive_lidar_benchmark.core.types import OccupancyMap, Trajectory
from massive_lidar_benchmark.maps.generate import load_map_artifact
from massive_lidar_benchmark.maps.validate import is_world_point_free
from massive_lidar_benchmark.traj.patterns import sample_waypoints
from massive_lidar_benchmark.traj.planner import filter_collision_free_waypoints, interpolate_polyline
from massive_lidar_benchmark.viz.style import apply_paper_style


def load_trajectory_artifact(npz_path: str | Path) -> Trajectory:
    payload = np.load(npz_path, allow_pickle=False)
    pattern = str(payload["pattern"]) if payload["pattern"].ndim == 0 else str(payload["pattern"].item())
    map_id = str(payload["map_id"]) if payload["map_id"].ndim == 0 else str(payload["map_id"].item())
    return Trajectory(
        states=payload["states"].astype(float),
        dt_s=float(payload["dt_s"]),
        pattern=pattern,
        map_id=map_id,
        seed=int(payload["seed"]),
    )


def _trajectory_stem(map_stem: str, index: int) -> str:
    return f"{map_stem}_traj{index:02d}"


def generate_trajectory(map_data: OccupancyMap, config: ProjectConfig, seed: int, map_id: str) -> Trajectory:
    rng = np.random.default_rng(seed)
    waypoints = sample_waypoints(map_data, config.trajectory.pattern, config.trajectory.num_waypoints, rng)
    waypoints = np.asarray(
        [point for point in waypoints if is_world_point_free(map_data, float(point[0]), float(point[1]))],
        dtype=float,
    )
    if len(waypoints) < 2:
        raise ValueError("Not enough free waypoints were sampled for trajectory generation.")
    filtered = filter_collision_free_waypoints(map_data, waypoints)
    interp = interpolate_polyline(filtered, step_m=max(config.map.resolution_m, config.trajectory.target_speed_mps * config.trajectory.dt_s))
    max_steps = max(2, int(np.floor(config.trajectory.horizon_s / max(config.trajectory.dt_s, 1e-6))) + 1)
    if len(interp) > max_steps:
        interp = interp[:max_steps]
    headings = np.zeros(len(interp), dtype=float)
    deltas = np.diff(interp, axis=0, prepend=interp[:1])
    headings[1:] = np.arctan2(deltas[1:, 1], deltas[1:, 0])
    speeds = np.linalg.norm(deltas, axis=1) / max(config.trajectory.dt_s, 1e-6)
    omegas = np.diff(headings, prepend=headings[:1]) / max(config.trajectory.dt_s, 1e-6)
    times = np.arange(len(interp), dtype=float) * config.trajectory.dt_s
    states = np.column_stack([times, interp[:, 0], interp[:, 1], headings, speeds, omegas])
    return Trajectory(states=states, dt_s=config.trajectory.dt_s, pattern=config.trajectory.pattern, map_id=map_id, seed=seed)


def save_trajectory_artifacts(
    map_data: OccupancyMap,
    trajectory: Trajectory,
    output_root: str | Path,
    map_stem: str,
    index: int,
) -> Path:
    traj_dir = ensure_dir(Path(output_root) / "trajectories")
    stem = _trajectory_stem(map_stem, index)
    npz_path = traj_dir / f"{stem}.npz"
    csv_path = traj_dir / f"{stem}.csv"
    png_path = traj_dir / f"{stem}.png"
    yaml_path = traj_dir / f"{stem}.yaml"

    np.savez_compressed(
        npz_path,
        states=trajectory.states,
        dt_s=trajectory.dt_s,
        pattern=trajectory.pattern,
        map_id=trajectory.map_id,
        seed=trajectory.seed,
    )

    np.savetxt(
        csv_path,
        trajectory.states,
        delimiter=",",
        header="t,x,y,theta,v,omega",
        comments="",
    )

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(map_data.occupancy, cmap="gray_r", origin="lower")
    xs = trajectory.states[:, 1] / map_data.resolution_m
    ys = trajectory.states[:, 2] / map_data.resolution_m
    ax.plot(xs, ys, color="#D95F02", linewidth=2.0)
    ax.set_title(f"{trajectory.pattern} | seed {trajectory.seed}")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.savefig(png_path, dpi=180, facecolor="white")
    plt.close(fig)

    write_yaml(
        yaml_path,
        {
            "pattern": trajectory.pattern,
            "seed": trajectory.seed,
            "dt_s": trajectory.dt_s,
            "num_states": int(len(trajectory.states)),
            "map_id": trajectory.map_id,
        },
    )
    return npz_path


def generate_trajectories_from_config(
    config: ProjectConfig,
    output_root: str | Path,
    map_artifacts: list[Path],
) -> list[Path]:
    logger = get_logger(__name__)
    seeds = spawn_seed_integers(
        config.seed + 1000,
        len(map_artifacts) * config.benchmark.trajectories_per_map * 8,
    )
    trajectories: list[Path] = []
    seed_index = 0
    for map_path in map_artifacts:
        map_data = load_map_artifact(map_path)
        map_stem = map_path.stem
        for traj_index in range(config.benchmark.trajectories_per_map):
            trajectory_path: Path | None = None
            for _ in range(8):
                trajectory_seed = seeds[seed_index]
                seed_index += 1
                try:
                    trajectory = generate_trajectory(map_data, config, trajectory_seed, map_id=map_stem)
                except ValueError:
                    continue
                trajectory_path = save_trajectory_artifacts(map_data, trajectory, output_root, map_stem, traj_index)
                trajectories.append(trajectory_path)
                break
            if trajectory_path is None:
                raise RuntimeError(f"Failed to generate a valid trajectory for map artifact: {map_path}")
    logger.info(
        "Generated %d trajectory artifact(s) in %s.",
        len(trajectories),
        Path(output_root) / "trajectories",
    )
    return trajectories
