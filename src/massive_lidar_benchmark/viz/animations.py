"""Video frame rendering for benchmark demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from massive_lidar_benchmark.benchmarks.summary import (
    PORTFOLIO_VIDEO_NAMES,
    episode_identifier,
    load_episode_catalog,
    preferred_noise_levels,
    select_noise_comparison_catalog,
    select_parallel_grid_catalog,
    select_representative_episode_group,
)
from massive_lidar_benchmark.config.load import load_config
from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.core.io import ensure_dir, write_json
from massive_lidar_benchmark.core.types import Pose2D
from massive_lidar_benchmark.localization.mcl import run_mcl_localization, sample_kidnap_pose
from massive_lidar_benchmark.maps.generate import load_map_artifact
from massive_lidar_benchmark.sensors.lidar_cpu import ray_cast_scan
from massive_lidar_benchmark.viz.style import METHOD_COLORS, apply_paper_style
from massive_lidar_benchmark.viz.video_export import cleanup_frame_dir, export_video_bundle


def _resolve_catalog(config: ProjectConfig, output_root: Path) -> tuple[Path, bool, pd.DataFrame]:
    requested_root = Path(config.render.target_run_dir)
    catalog = load_episode_catalog(requested_root, source_experiment=config.render.source_experiment)
    if not catalog.empty:
        return requested_root, False, catalog

    if config.render.strict_run_root or not config.render.allow_fallback:
        raise FileNotFoundError(
            f"No episode outputs were found under strict run root: {requested_root}"
        )

    fallback_root = output_root / "runs"
    fallback_catalog = load_episode_catalog(fallback_root, source_experiment=config.render.source_experiment)
    return fallback_root, True, fallback_catalog


def _load_step_frame(episode_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(episode_dir) / "step_metrics.csv")


def _load_episode_data(episode_dir: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(episode_dir) / "episode_data.npz", allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def _load_context(row: pd.Series, output_root: Path) -> dict[str, object]:
    return {
        "row": row,
        "step_frame": _load_step_frame(row["episode_dir"]),
        "episode_data": _load_episode_data(row["episode_dir"]),
        "run_config": load_config(Path(row["episode_dir"]) / "resolved_config.yaml"),
        "map_data": load_map_artifact(output_root / "maps" / f"{row['map_id']}.npz"),
    }


def _frame_geometry(config: ProjectConfig) -> tuple[tuple[float, float], int]:
    width_px, height_px = config.render.frame_size
    dpi = 100
    return (width_px / dpi, height_px / dpi), dpi


def _prepare_frame_dir(output_root: Path, video_name: str) -> Path:
    frame_dir = ensure_dir(output_root / "frames" / video_name)
    cleanup_frame_dir(frame_dir)
    return frame_dir


def _frame_indices(total_steps: int, config: ProjectConfig) -> list[int]:
    stride = max(1, int(config.render.frame_stride))
    indices = list(range(0, total_steps, stride))
    if config.render.max_frames is not None:
        indices = indices[: max(1, int(config.render.max_frames))]
    elif indices and indices[-1] != total_steps - 1:
        indices.append(total_steps - 1)
    return indices


def _save_frame(fig: plt.Figure, frame_dir: Path, frame_index: int, dpi: int) -> None:
    fig.savefig(frame_dir / f"frame_{frame_index:05d}.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_map(ax: plt.Axes, occupancy: np.ndarray) -> None:
    ax.imshow(occupancy, cmap="gray_r", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])


def _world_to_grid(values: np.ndarray, resolution_m: float) -> np.ndarray:
    return np.asarray(values, dtype=float) / max(resolution_m, 1e-9)


def _episode_seed(config: ProjectConfig, benchmark_seed: int, episode_index: int) -> int:
    return int(np.random.SeedSequence([config.seed, benchmark_seed, episode_index]).generate_state(1)[0])


def _kidnap_step_from_episode_data(episode_data: dict[str, np.ndarray]) -> int | None:
    raw = int(np.asarray(episode_data["kidnap_step"]).item())
    return None if raw < 0 else raw


def _notice_frames(config: ProjectConfig, frame_dir: Path, title: str, message: str, frame_count: int | None = None) -> int:
    figsize, dpi = _frame_geometry(config)
    total_frames = frame_count or max(12, config.render.fps)
    apply_paper_style()
    for frame_index in range(total_frames):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=20, color="#111111")
        ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=12, color="#444444")
        _save_frame(fig, frame_dir, frame_index, dpi=dpi)
    return total_frames


def _replay_mcl_context(context: dict[str, object], kidnap_override: bool = False) -> dict[str, object]:
    row = context["row"]
    run_config = context["run_config"]
    episode_data = context["episode_data"]
    map_data = context["map_data"]
    gt_states = episode_data["gt_states"]
    kidnap_step = _kidnap_step_from_episode_data(episode_data)
    kidnap_pose = np.asarray(episode_data["kidnap_pose"], dtype=float)
    if kidnap_step is None or kidnap_override:
        kidnap_step = len(gt_states) // 2
        kidnap_pose = sample_kidnap_pose(
            map_data,
            gt_states[kidnap_step, 1:4],
            np.random.default_rng(_episode_seed(run_config, int(row["seed"]), int(row["episode_index"])) + 17),
        )
    result = run_mcl_localization(
        map_data=map_data,
        gt_states=gt_states,
        observed_scans=episode_data["observed_scans"],
        beam_angles_rad=episode_data["beam_angles_rad"],
        config=run_config,
        initial_mean=np.asarray(episode_data["initial_mean"], dtype=float),
        rng=np.random.default_rng(_episode_seed(run_config, int(row["seed"]), int(row["episode_index"])) + 33),
        kidnap_step=kidnap_step,
        kidnap_pose=kidnap_pose,
        record_history=True,
        history_particle_limit=128,
    )
    return {
        "gt_states": gt_states,
        "estimates": np.asarray(result["estimates"], dtype=float),
        "ess": np.asarray(result["ess"], dtype=float),
        "particle_history": dict(result["particle_history"]),
        "kidnap_step": kidnap_step,
        "kidnap_pose": kidnap_pose,
        "map_data": map_data,
        "beam_angles_rad": episode_data["beam_angles_rad"],
        "observed_scans": episode_data["observed_scans"],
    }


def _render_demo_main_localization(config: ProjectConfig, output_root: Path, catalog: pd.DataFrame, frame_dir: Path) -> tuple[int, dict[str, object]]:
    if catalog.empty:
        return _notice_frames(config, frame_dir, "demo_main_localization", "No run data found for video rendering."), {"source_episode_ids": []}

    group = select_representative_episode_group(catalog, anchor_method="mcl", noise_value=0.02)
    contexts = {str(row["method"]): _load_context(row, output_root) for _, row in group.iterrows()}
    source_episode_ids = [episode_identifier(row) for _, row in group.iterrows()]
    primary = next(iter(contexts.values()))
    episode_data = primary["episode_data"]
    map_data = primary["map_data"]
    gt_states = episode_data["gt_states"]
    beam_angles_deg = np.degrees(episode_data["beam_angles_rad"])

    figsize, dpi = _frame_geometry(config)
    apply_paper_style()
    frame_steps = _frame_indices(len(gt_states), config)
    for frame_index, step_index in enumerate(frame_steps):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
        ax_map = fig.add_subplot(grid[:, 0])
        ax_scan = fig.add_subplot(grid[0, 1])
        ax_err = fig.add_subplot(grid[1, 1])

        _plot_map(ax_map, map_data.occupancy)
        ax_map.plot(_world_to_grid(gt_states[: step_index + 1, 1], map_data.resolution_m), _world_to_grid(gt_states[: step_index + 1, 2], map_data.resolution_m), color=METHOD_COLORS["gt"], linewidth=2.2, label="GT")
        ax_map.scatter(_world_to_grid(np.array([gt_states[step_index, 1]]), map_data.resolution_m), _world_to_grid(np.array([gt_states[step_index, 2]]), map_data.resolution_m), color=METHOD_COLORS["gt"], s=40)
        for method, context in contexts.items():
            est = context["episode_data"]["estimated_states"]
            step_frame = context["step_frame"]
            ax_map.plot(_world_to_grid(est[: step_index + 1, 0], map_data.resolution_m), _world_to_grid(est[: step_index + 1, 1], map_data.resolution_m), color=METHOD_COLORS.get(method, "#444444"), linewidth=2.0, label=method.upper())
            ax_map.scatter(_world_to_grid(np.array([est[step_index, 0]]), map_data.resolution_m), _world_to_grid(np.array([est[step_index, 1]]), map_data.resolution_m), color=METHOD_COLORS.get(method, "#444444"), s=35)
            ax_err.plot(step_frame["t"].iloc[: step_index + 1], step_frame["translation_error_m"].iloc[: step_index + 1], color=METHOD_COLORS.get(method, "#444444"), linewidth=2.0, label=method.upper())
        ax_map.set_title("Representative Episode")
        ax_map.legend(frameon=False, loc="upper right")

        current_scan = episode_data["observed_scans"][step_index]
        ax_scan.plot(beam_angles_deg, current_scan, color=METHOD_COLORS["scan"], linewidth=1.8)
        ax_scan.set_title("Observed LiDAR Scan")
        ax_scan.set_xlabel("beam angle [deg]")
        ax_scan.set_ylabel("range [m]")

        ax_err.set_title("Translation Error")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("error [m]")
        ax_err.legend(frameon=False, loc="upper right")
        fig.suptitle(f"Representative episode | t = {gt_states[step_index, 0]:.1f} s", fontsize=14)
        _save_frame(fig, frame_dir, frame_index, dpi=dpi)

    return len(frame_steps), {"selected_methods": sorted(contexts), "source_episode_ids": source_episode_ids}


def _render_demo_particle_convergence(config: ProjectConfig, output_root: Path, catalog: pd.DataFrame, frame_dir: Path) -> tuple[int, dict[str, object]]:
    if catalog.empty:
        return _notice_frames(config, frame_dir, "demo_particle_convergence", "No MCL runs were found for particle convergence rendering."), {"source_episode_ids": []}

    representative = select_representative_episode_group(catalog, anchor_method="mcl", noise_value=0.02)
    mcl_rows = representative[representative["method"] == "mcl"]
    if mcl_rows.empty:
        return _notice_frames(config, frame_dir, "demo_particle_convergence", "No representative MCL run was found."), {"source_episode_ids": []}

    row = mcl_rows.iloc[0]
    context = _load_context(row, output_root)
    replay = _replay_mcl_context(context)
    map_data = replay["map_data"]
    gt_states = replay["gt_states"]
    estimates = replay["estimates"]
    ess_values = replay["ess"]
    particle_history = replay["particle_history"]
    translation_error = np.linalg.norm(gt_states[:, 1:3] - estimates[:, :2], axis=1)
    source_episode_ids = [episode_identifier(row)]

    figsize, dpi = _frame_geometry(config)
    apply_paper_style()
    frame_steps = _frame_indices(len(gt_states), config)
    for frame_index, step_index in enumerate(frame_steps):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1.0, 1.0])
        ax_map = fig.add_subplot(grid[:, 0])
        ax_ess = fig.add_subplot(grid[0, 1])
        ax_err = fig.add_subplot(grid[1, 1])

        _plot_map(ax_map, map_data.occupancy)
        particles = particle_history.get(str(step_index), particle_history.get(str(max(0, step_index - 1)), np.empty((0, 3))))
        if len(particles) > 0:
            ax_map.scatter(_world_to_grid(particles[:, 0], map_data.resolution_m), _world_to_grid(particles[:, 1], map_data.resolution_m), s=8, alpha=0.18, color=METHOD_COLORS["particle"])
        ax_map.plot(_world_to_grid(gt_states[: step_index + 1, 1], map_data.resolution_m), _world_to_grid(gt_states[: step_index + 1, 2], map_data.resolution_m), color=METHOD_COLORS["gt"], linewidth=2.2, label="GT")
        ax_map.plot(_world_to_grid(estimates[: step_index + 1, 0], map_data.resolution_m), _world_to_grid(estimates[: step_index + 1, 1], map_data.resolution_m), color=METHOD_COLORS["mcl"], linewidth=2.0, label="MCL")
        ax_map.scatter(_world_to_grid(np.array([gt_states[step_index, 1]]), map_data.resolution_m), _world_to_grid(np.array([gt_states[step_index, 2]]), map_data.resolution_m), color=METHOD_COLORS["gt"], s=40)
        ax_map.scatter(_world_to_grid(np.array([estimates[step_index, 0]]), map_data.resolution_m), _world_to_grid(np.array([estimates[step_index, 1]]), map_data.resolution_m), color=METHOD_COLORS["mcl"], s=40)
        ax_map.set_title("Particle Cloud")
        ax_map.legend(frameon=False, loc="upper right")

        ax_ess.plot(gt_states[: step_index + 1, 0], ess_values[: step_index + 1], color=METHOD_COLORS["mcl"], linewidth=2.0)
        ax_ess.axhline(context["run_config"].mcl.resample_ess_ratio * context["run_config"].mcl.particle_count, color="#777777", linestyle="--", linewidth=1.2)
        ax_ess.set_title("Effective Sample Size")
        ax_ess.set_xlabel("time [s]")
        ax_ess.set_ylabel("ESS")

        ax_err.plot(gt_states[: step_index + 1, 0], translation_error[: step_index + 1], color=METHOD_COLORS["mcl"], linewidth=2.0)
        ax_err.set_title("Translation Error")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("error [m]")
        fig.suptitle(f"Representative MCL convergence | t = {gt_states[step_index, 0]:.1f} s", fontsize=14)
        _save_frame(fig, frame_dir, frame_index, dpi=dpi)

    return len(frame_steps), {"source_episode_ids": source_episode_ids}


def _render_demo_noise_robustness(config: ProjectConfig, output_root: Path, catalog: pd.DataFrame, frame_dir: Path) -> tuple[int, dict[str, object]]:
    if catalog.empty:
        return _notice_frames(config, frame_dir, "demo_noise_robustness", "No run data found for noise robustness rendering."), {"source_episode_ids": []}

    low_noise, high_noise = preferred_noise_levels(catalog)
    comparison = select_noise_comparison_catalog(catalog, low_noise=low_noise, high_noise=high_noise, anchor_method="mcl")
    source_episode_ids = [episode_identifier(row) for _, row in comparison.iterrows()]
    low_rows = comparison[np.isclose(pd.to_numeric(comparison["range_noise_std_m"], errors="coerce"), low_noise, atol=1e-9)]
    high_rows = comparison[np.isclose(pd.to_numeric(comparison["range_noise_std_m"], errors="coerce"), high_noise, atol=1e-9)]
    low_contexts = {str(row["method"]): _load_context(row, output_root) for _, row in low_rows.iterrows()}
    high_contexts = {str(row["method"]): _load_context(row, output_root) for _, row in high_rows.iterrows()}
    primary = next(iter(high_contexts.values()))
    episode_data = primary["episode_data"]
    gt_states = episode_data["gt_states"]
    map_data = primary["map_data"]
    beam_angles_deg = np.degrees(episode_data["beam_angles_rad"])

    figsize, dpi = _frame_geometry(config)
    apply_paper_style()
    frame_steps = _frame_indices(len(gt_states), config)
    for frame_index, step_index in enumerate(frame_steps):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.3], height_ratios=[1.0, 1.0])
        ax_clean = fig.add_subplot(grid[0, 0])
        ax_noisy = fig.add_subplot(grid[0, 1])
        ax_map = fig.add_subplot(grid[1, 0])
        ax_err = fig.add_subplot(grid[1, 1])

        clean_scan = next(iter(low_contexts.values()))["episode_data"]["observed_scans"][step_index]
        noisy_scan = next(iter(high_contexts.values()))["episode_data"]["observed_scans"][step_index]
        ax_clean.plot(beam_angles_deg, clean_scan, color="#3B6EA8", linewidth=1.8)
        ax_clean.set_title(f"Low Noise Scan ({low_noise:.2f} m)")
        ax_clean.set_xlabel("beam angle [deg]")
        ax_clean.set_ylabel("range [m]")

        ax_noisy.plot(beam_angles_deg, noisy_scan, color="#B24A3A", linewidth=1.8)
        ax_noisy.set_title(f"High Noise Scan ({high_noise:.2f} m)")
        ax_noisy.set_xlabel("beam angle [deg]")
        ax_noisy.set_ylabel("range [m]")

        _plot_map(ax_map, map_data.occupancy)
        ax_map.plot(_world_to_grid(gt_states[: step_index + 1, 1], map_data.resolution_m), _world_to_grid(gt_states[: step_index + 1, 2], map_data.resolution_m), color=METHOD_COLORS["gt"], linewidth=2.2, label="GT")
        for method, context in high_contexts.items():
            est = context["episode_data"]["estimated_states"]
            ax_map.plot(_world_to_grid(est[: step_index + 1, 0], map_data.resolution_m), _world_to_grid(est[: step_index + 1, 1], map_data.resolution_m), color=METHOD_COLORS.get(method, "#444444"), linewidth=2.0, label=f"{method.upper()} high")
        ax_map.set_title("Higher-Noise Trajectories")
        ax_map.legend(frameon=False, loc="upper right")

        for method, context in low_contexts.items():
            ax_err.plot(context["step_frame"]["t"].iloc[: step_index + 1], context["step_frame"]["translation_error_m"].iloc[: step_index + 1], color=METHOD_COLORS.get(method, "#444444"), linewidth=2.0, linestyle="-", label=f"{method.upper()} low")
        for method, context in high_contexts.items():
            ax_err.plot(context["step_frame"]["t"].iloc[: step_index + 1], context["step_frame"]["translation_error_m"].iloc[: step_index + 1], color=METHOD_COLORS.get(method, "#444444"), linewidth=2.0, linestyle="--", label=f"{method.upper()} high")
        ax_err.set_title("Translation Error")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("error [m]")
        ax_err.legend(frameon=False, loc="upper right", fontsize=9)
        fig.suptitle(f"Noise robustness comparison | t = {gt_states[step_index, 0]:.1f} s", fontsize=14)
        _save_frame(fig, frame_dir, frame_index, dpi=dpi)

    return len(frame_steps), {"source_episode_ids": source_episode_ids}


def _render_demo_kidnapped_recovery(config: ProjectConfig, output_root: Path, catalog: pd.DataFrame, frame_dir: Path) -> tuple[int, dict[str, object]]:
    mcl_rows = catalog[catalog["method"] == "mcl"]
    if mcl_rows.empty:
        return _notice_frames(config, frame_dir, "demo_kidnapped_recovery", "No MCL runs were found for kidnapped recovery rendering."), {"source_episode_ids": []}

    kidnapped_rows = mcl_rows[mcl_rows["kidnap_step"].fillna(-1) >= 0]
    if kidnapped_rows.empty:
        row = mcl_rows.sort_values("position_rmse_m").iloc[0]
        context = _load_context(row, output_root)
        replay = _replay_mcl_context(context, kidnap_override=True)
    else:
        row = kidnapped_rows.sort_values("position_rmse_m").iloc[0]
        context = _load_context(row, output_root)
        replay = _replay_mcl_context(context, kidnap_override=False)

    map_data = replay["map_data"]
    gt_states = replay["gt_states"]
    estimates = replay["estimates"]
    ess_values = replay["ess"]
    particle_history = replay["particle_history"]
    kidnap_step = int(replay["kidnap_step"])
    kidnap_pose = np.asarray(replay["kidnap_pose"], dtype=float)
    translation_error = np.linalg.norm(gt_states[:, 1:3] - estimates[:, :2], axis=1)

    figsize, dpi = _frame_geometry(config)
    apply_paper_style()
    frame_steps = _frame_indices(len(gt_states), config)
    for frame_index, step_index in enumerate(frame_steps):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1.0, 1.0])
        ax_map = fig.add_subplot(grid[:, 0])
        ax_err = fig.add_subplot(grid[0, 1])
        ax_ess = fig.add_subplot(grid[1, 1])

        _plot_map(ax_map, map_data.occupancy)
        particles = particle_history.get(str(step_index), particle_history.get(str(max(0, step_index - 1)), np.empty((0, 3))))
        if len(particles) > 0:
            ax_map.scatter(_world_to_grid(particles[:, 0], map_data.resolution_m), _world_to_grid(particles[:, 1], map_data.resolution_m), s=8, alpha=0.18, color=METHOD_COLORS["particle"])
        ax_map.plot(_world_to_grid(gt_states[: step_index + 1, 1], map_data.resolution_m), _world_to_grid(gt_states[: step_index + 1, 2], map_data.resolution_m), color=METHOD_COLORS["gt"], linewidth=2.2, label="GT")
        ax_map.plot(_world_to_grid(estimates[: step_index + 1, 0], map_data.resolution_m), _world_to_grid(estimates[: step_index + 1, 1], map_data.resolution_m), color=METHOD_COLORS["mcl"], linewidth=2.0, label="MCL")
        ax_map.scatter(_world_to_grid(np.array([kidnap_pose[0]]), map_data.resolution_m), _world_to_grid(np.array([kidnap_pose[1]]), map_data.resolution_m), color="#B22222", s=50, label="kidnap pose")
        ax_map.set_title("Particle Recovery")
        ax_map.legend(frameon=False, loc="upper right")

        ax_err.plot(gt_states[: step_index + 1, 0], translation_error[: step_index + 1], color=METHOD_COLORS["mcl"], linewidth=2.0)
        ax_err.axvline(gt_states[kidnap_step, 0], color="#B22222", linestyle="--", linewidth=1.6)
        ax_err.set_title("Recovery Error")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("error [m]")

        ax_ess.plot(gt_states[: step_index + 1, 0], ess_values[: step_index + 1], color=METHOD_COLORS["mcl"], linewidth=2.0)
        ax_ess.axvline(gt_states[kidnap_step, 0], color="#B22222", linestyle="--", linewidth=1.6)
        ax_ess.set_title("ESS After Kidnap")
        ax_ess.set_xlabel("time [s]")
        ax_ess.set_ylabel("ESS")
        fig.suptitle(f"Kidnapped recovery | t = {gt_states[step_index, 0]:.1f} s", fontsize=14)
        _save_frame(fig, frame_dir, frame_index, dpi=dpi)

    return len(frame_steps), {"source_episode_ids": [episode_identifier(row)]}


def _render_demo_parallel_grid(config: ProjectConfig, output_root: Path, catalog: pd.DataFrame, frame_dir: Path) -> tuple[int, dict[str, object]]:
    if catalog.empty:
        return _notice_frames(config, frame_dir, "demo_parallel_grid", "No run data found for parallel grid rendering."), {"source_episode_ids": []}

    selected = select_parallel_grid_catalog(catalog, noise_value=0.05, limit=16)
    contexts = [_load_context(row, output_root) for _, row in selected.iterrows()]
    source_episode_ids = [episode_identifier(row) for _, row in selected.iterrows()]
    max_steps = max(len(context["episode_data"]["gt_states"]) for context in contexts)
    figsize, dpi = _frame_geometry(config)
    apply_paper_style()

    frame_steps = _frame_indices(max_steps, config)
    for frame_index, step_index in enumerate(frame_steps):
        fig, axes = plt.subplots(4, 4, figsize=figsize, constrained_layout=True)
        axes_flat = axes.ravel()
        for ax in axes_flat:
            ax.axis("off")

        for ax, context in zip(axes_flat, contexts):
            episode_data = context["episode_data"]
            map_data = context["map_data"]
            local_step_index = min(step_index, len(episode_data["gt_states"]) - 1)
            gt_states = episode_data["gt_states"]
            estimates = episode_data["estimated_states"]
            method = str(context["row"]["method"])

            ax.axis("on")
            _plot_map(ax, map_data.occupancy)
            ax.plot(_world_to_grid(gt_states[: local_step_index + 1, 1], map_data.resolution_m), _world_to_grid(gt_states[: local_step_index + 1, 2], map_data.resolution_m), color=METHOD_COLORS["gt"], linewidth=1.3)
            ax.plot(_world_to_grid(estimates[: local_step_index + 1, 0], map_data.resolution_m), _world_to_grid(estimates[: local_step_index + 1, 1], map_data.resolution_m), color=METHOD_COLORS.get(method, "#444444"), linewidth=1.2)
            ax.scatter(_world_to_grid(np.array([gt_states[local_step_index, 1]]), map_data.resolution_m), _world_to_grid(np.array([gt_states[local_step_index, 2]]), map_data.resolution_m), color=METHOD_COLORS["gt"], s=12)
            ax.scatter(_world_to_grid(np.array([estimates[local_step_index, 0]]), map_data.resolution_m), _world_to_grid(np.array([estimates[local_step_index, 1]]), map_data.resolution_m), color=METHOD_COLORS.get(method, "#444444"), s=12)
            ax.set_title(f"{method.upper()} | seed {int(context['row']['seed'])}", fontsize=8)

        fig.suptitle(f"Parallel episode grid | frame {frame_index + 1}/{len(frame_steps)}", fontsize=14)
        _save_frame(fig, frame_dir, frame_index, dpi=dpi)

    return len(frame_steps), {"source_episode_ids": source_episode_ids}


def render_video_demo(config: ProjectConfig, output_root: str | Path, export_media: bool = True) -> dict[str, object]:
    output_path = Path(output_root)
    frame_dir = _prepare_frame_dir(output_path, config.render.target_video_name)
    requested_run_root = Path(config.render.target_run_dir)
    resolved_run_root, fallback_used, catalog = _resolve_catalog(config, output_path)

    renderers = {
        "demo_main_localization": _render_demo_main_localization,
        "demo_particle_convergence": _render_demo_particle_convergence,
        "demo_noise_robustness": _render_demo_noise_robustness,
        "demo_kidnapped_recovery": _render_demo_kidnapped_recovery,
        "demo_parallel_grid": _render_demo_parallel_grid,
    }
    renderer = renderers.get(config.render.target_video_name)
    if renderer is None:
        frame_count = _notice_frames(
            config,
            frame_dir,
            config.render.target_video_name,
            "No renderer is implemented for this video name.",
        )
        scene_meta: dict[str, object] = {"source_episode_ids": []}
    else:
        frame_count, scene_meta = renderer(config, output_path, catalog, frame_dir)

    export_paths: dict[str, object] = {"frame_dir": frame_dir, "mp4": None, "gif": None}
    if export_media:
        export_paths = export_video_bundle(
            frame_dir=frame_dir,
            output_root=output_path,
            video_name=config.render.target_video_name,
            fps=config.render.fps,
            gif_scale_width=config.render.gif_scale_width,
        )
        if config.render.cleanup_frames and export_paths["mp4"] is not None and export_paths["gif"] is not None:
            cleanup_frame_dir(frame_dir)

    validated = (
        config.render.target_video_name in PORTFOLIO_VIDEO_NAMES
        and not fallback_used
        and export_paths["mp4"] is not None
        and export_paths["gif"] is not None
    )
    manifest = {
        "video_name": config.render.target_video_name,
        "requested_run_root": str(requested_run_root),
        "resolved_run_root": str(resolved_run_root),
        "fallback_used": bool(fallback_used),
        "source_experiment": config.render.source_experiment,
        "source_run_root": str(resolved_run_root),
        "source_episode_ids": list(scene_meta.get("source_episode_ids", [])),
        "required_for_portfolio": bool(config.render.target_video_name in PORTFOLIO_VIDEO_NAMES),
        "portfolio_profile": config.render.source_experiment,
        "validated_for_portfolio": bool(validated),
        "frame_count": frame_count,
        "exported_mp4": None if export_paths["mp4"] is None else str(export_paths["mp4"]),
        "exported_gif": None if export_paths["gif"] is None else str(export_paths["gif"]),
        "scene_meta": scene_meta,
    }
    write_json(frame_dir / "render_manifest.json", manifest)
    return manifest
