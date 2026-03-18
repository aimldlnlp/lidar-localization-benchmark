"""Publication-style figure rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from massive_lidar_benchmark.benchmarks.summary import (
    PORTFOLIO_FIGURE_FILENAMES,
    aggregate_step_metrics,
    load_episode_catalog,
    select_representative_episode_group,
    throughput_experiment_name,
)
from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.core.io import ensure_dir
from massive_lidar_benchmark.maps.generate import load_map_artifact
from massive_lidar_benchmark.viz.style import METHOD_COLORS, apply_paper_style


FIGURE_FILENAMES = [
    "fig_localization_overview.png",
    "fig_error_vs_time.png",
    "fig_robustness_noise_sweep.png",
    "fig_error_distribution.png",
    "fig_particle_evolution.png",
    "fig_kidnapped_recovery.png",
    "fig_gpu_scalability.png",
    "fig_failure_cases.png",
]


def _figure_path(output_root: str | Path, filename: str) -> Path:
    return ensure_dir(Path(output_root) / "figures") / filename


def _load_step_frame(episode_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(episode_dir) / "step_metrics.csv")


def _load_episode_data(episode_dir: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(episode_dir) / "episode_data.npz", allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def _load_particle_snapshots(episode_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(episode_dir) / "particle_snapshots.npz"
    if not path.exists():
        return {}
    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


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


def _selected_figure_names(config: ProjectConfig) -> list[str]:
    if config.render.figure_names:
        return list(config.render.figure_names)
    return list(FIGURE_FILENAMES)


def _save_notice_figure(path: Path, title: str, message: str) -> Path:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, color="#111111")
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=11, color="#444444")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def _plot_map(ax: plt.Axes, occupancy: np.ndarray) -> None:
    ax.imshow(occupancy, cmap="gray_r", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])


def _world_to_grid_series(values: np.ndarray, resolution_m: float) -> np.ndarray:
    return values / max(resolution_m, 1e-9)


def _representative_group(catalog: pd.DataFrame) -> pd.DataFrame:
    return select_representative_episode_group(catalog, anchor_method="mcl", noise_value=0.02)


def _plot_error_band(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
) -> None:
    for method, method_frame in frame.groupby("method"):
        stats = (
            method_frame.groupby("t")[value_column]
            .agg(
                median="median",
                low=lambda x: np.quantile(x, 0.2),
                high=lambda x: np.quantile(x, 0.8),
            )
            .reset_index()
        )
        ax.plot(stats["t"], stats["median"], linewidth=2.0, color=METHOD_COLORS.get(method, "#444444"), label=method.upper())
        ax.fill_between(stats["t"], stats["low"], stats["high"], alpha=0.18, color=METHOD_COLORS.get(method, "#444444"))
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)


def _render_localization_overview(figures_dir: Path, catalog: pd.DataFrame, output_root: Path) -> Path:
    path = figures_dir / "fig_localization_overview.png"
    if catalog.empty:
        return _save_notice_figure(path, "Localization Overview", "No episode outputs found under the requested run root.")

    group = _representative_group(catalog)
    primary = group.iloc[0]
    map_data = load_map_artifact(output_root / "maps" / f"{primary['map_id']}.npz")
    episode_data = {row["method"]: _load_episode_data(row["episode_dir"]) for _, row in group.iterrows()}
    step_frames = {row["method"]: _load_step_frame(row["episode_dir"]) for _, row in group.iterrows()}

    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    ax = axes[0, 0]
    _plot_map(ax, map_data.occupancy)
    gt = next(iter(episode_data.values()))["gt_states"]
    ax.plot(_world_to_grid_series(gt[:, 1], map_data.resolution_m), _world_to_grid_series(gt[:, 2], map_data.resolution_m), color=METHOD_COLORS["gt"], linewidth=2.2, label="GT")
    for method, payload in episode_data.items():
        est = payload["estimated_states"]
        ax.plot(
            _world_to_grid_series(est[:, 0], map_data.resolution_m),
            _world_to_grid_series(est[:, 1], map_data.resolution_m),
            color=METHOD_COLORS.get(method, "#444444"),
            linewidth=2.0,
            label=method.upper(),
        )
    ax.set_title("Representative Episode")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    payload = next(iter(episode_data.values()))
    beam_angles = np.degrees(payload["beam_angles_rad"])
    final_scan = payload["observed_scans"][-1]
    ax.plot(beam_angles, final_scan, color=METHOD_COLORS["scan"], linewidth=1.8)
    ax.set_title("Final LiDAR Scan")
    ax.set_xlabel("beam angle [deg]")
    ax.set_ylabel("range [m]")

    ax = axes[1, 0]
    for method, step_frame in step_frames.items():
        ax.plot(step_frame["t"], step_frame["translation_error_m"], linewidth=2.0, color=METHOD_COLORS.get(method, "#444444"), label=method.upper())
    ax.set_title("Translation Error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [m]")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for method, step_frame in step_frames.items():
        ax.plot(step_frame["t"], step_frame["heading_error_deg"], linewidth=2.0, color=METHOD_COLORS.get(method, "#444444"), label=method.upper())
    ax.set_title("Heading Error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [deg]")
    ax.legend(frameon=False)

    fig.suptitle("Representative localization episode", fontsize=14)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def _render_error_vs_time(figures_dir: Path, catalog: pd.DataFrame) -> Path:
    path = figures_dir / "fig_error_vs_time.png"
    if catalog.empty:
        return _save_notice_figure(path, "Error vs Time", "No run data is available for plotting.")

    step_catalog = aggregate_step_metrics(catalog)
    if step_catalog.empty:
        return _save_notice_figure(path, "Error vs Time", "No step metrics were found.")

    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    _plot_error_band(axes[0], step_catalog, "translation_error_m", "translation error [m]")
    axes[0].set_title("Translation Error Over Time")
    _plot_error_band(axes[1], step_catalog, "heading_error_deg", "heading error [deg]")
    axes[1].set_title("Heading Error Over Time")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def _render_robustness_noise_sweep(figures_dir: Path, catalog: pd.DataFrame) -> Path:
    path = figures_dir / "fig_robustness_noise_sweep.png"
    if catalog.empty:
        return _save_notice_figure(path, "Robustness Sweep", "No run data is available for robustness plotting.")

    noise_frame = catalog.copy()
    if "range_noise_std_m" not in noise_frame.columns or noise_frame["range_noise_std_m"].isna().all():
        return _save_notice_figure(path, "Robustness Sweep", "No numeric noise sweep values were found.")

    noise_frame["range_noise_std_m"] = pd.to_numeric(noise_frame["range_noise_std_m"], errors="coerce")
    summary = (
        noise_frame.groupby(["method", "range_noise_std_m"], dropna=False)
        .agg(position_rmse_m=("position_rmse_m", "mean"), failure_rate=("failed", "mean"))
        .reset_index()
        .sort_values("range_noise_std_m")
    )

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for method, method_frame in summary.groupby("method"):
        axes[0].plot(method_frame["range_noise_std_m"], method_frame["position_rmse_m"], marker="o", linewidth=2.0, color=METHOD_COLORS.get(method, "#444444"), label=method.upper())
        axes[1].plot(method_frame["range_noise_std_m"], method_frame["failure_rate"], marker="o", linewidth=2.0, color=METHOD_COLORS.get(method, "#444444"), label=method.upper())

    axes[0].set_title("Position RMSE")
    axes[0].set_xlabel("range noise std [m]")
    axes[0].set_ylabel("RMSE [m]")
    axes[1].set_title("Failure Rate")
    axes[1].set_xlabel("range noise std [m]")
    axes[1].set_ylabel("rate")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def _render_error_distribution(figures_dir: Path, catalog: pd.DataFrame) -> Path:
    path = figures_dir / "fig_error_distribution.png"
    if catalog.empty:
        return _save_notice_figure(path, "Error Distribution", "No episode outputs found.")

    step_catalog = aggregate_step_metrics(catalog)
    if step_catalog.empty:
        return _save_notice_figure(path, "Error Distribution", "No step metrics were found.")

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for method, method_frame in step_catalog.groupby("method"):
        for ax, column, label in [
            (axes[0], "translation_error_m", "translation error [m]"),
            (axes[1], "heading_error_deg", "heading error [deg]"),
        ]:
            values = np.sort(method_frame[column].to_numpy())
            probs = np.linspace(0.0, 1.0, len(values), endpoint=True)
            ax.plot(values, probs, linewidth=2.0, color=METHOD_COLORS.get(method, "#444444"), label=method.upper())
            ax.set_xlabel(label)
            ax.set_ylabel("ECDF")
            ax.legend(frameon=False)
    axes[0].set_title("Translation Error ECDF")
    axes[1].set_title("Heading Error ECDF")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def _render_particle_evolution(figures_dir: Path, catalog: pd.DataFrame, output_root: Path) -> Path:
    path = figures_dir / "fig_particle_evolution.png"
    if catalog.empty:
        return _save_notice_figure(path, "Particle Evolution", "No MCL runs were found.")

    representative = _representative_group(catalog)
    mcl_rows = representative[representative["method"] == "mcl"]
    if mcl_rows.empty:
        return _save_notice_figure(path, "Particle Evolution", "No representative MCL run was found.")

    row = mcl_rows.iloc[0]
    snapshots = _load_particle_snapshots(row["episode_dir"])
    if not snapshots:
        return _save_notice_figure(path, "Particle Evolution", "No particle snapshots were stored for the selected run.")

    episode_data = _load_episode_data(row["episode_dir"])
    map_data = load_map_artifact(output_root / "maps" / f"{row['map_id']}.npz")
    ordered_keys = sorted(snapshots.keys(), key=lambda value: int(value))
    if len(ordered_keys) > 3:
        ordered_keys = [ordered_keys[0], ordered_keys[len(ordered_keys) // 2], ordered_keys[-1]]

    apply_paper_style()
    fig, axes = plt.subplots(1, len(ordered_keys), figsize=(14, 4.6), constrained_layout=True)
    if len(ordered_keys) == 1:
        axes = np.asarray([axes])
    for ax, key in zip(axes, ordered_keys):
        _plot_map(ax, map_data.occupancy)
        particles = snapshots[key]
        step_index = int(key)
        gt = episode_data["gt_states"][step_index]
        est = episode_data["estimated_states"][step_index]
        ax.scatter(_world_to_grid_series(particles[:, 0], map_data.resolution_m), _world_to_grid_series(particles[:, 1], map_data.resolution_m), s=8, alpha=0.18, color=METHOD_COLORS["particle"])
        ax.scatter(_world_to_grid_series(np.array([gt[1]]), map_data.resolution_m), _world_to_grid_series(np.array([gt[2]]), map_data.resolution_m), s=40, color=METHOD_COLORS["gt"], label="GT")
        ax.scatter(_world_to_grid_series(np.array([est[0]]), map_data.resolution_m), _world_to_grid_series(np.array([est[1]]), map_data.resolution_m), s=40, color=METHOD_COLORS["mcl"], label="MCL")
        ax.set_title(f"step {step_index}")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Representative Episode: Particle Evolution", fontsize=14)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def _render_kidnapped_recovery(figures_dir: Path, catalog: pd.DataFrame, output_root: Path) -> Path:
    path = figures_dir / "fig_kidnapped_recovery.png"
    if catalog.empty:
        return _save_notice_figure(path, "Kidnapped Recovery", "Run a kidnapped_robot benchmark to populate this figure.")
    kidnapped = catalog[catalog["kidnap_step"].fillna(-1) >= 0]
    if kidnapped.empty:
        return _save_notice_figure(path, "Kidnapped Recovery", "Run a kidnapped_robot benchmark to populate this figure.")

    row = kidnapped.sort_values("position_rmse_m").iloc[0]
    step_frame = _load_step_frame(row["episode_dir"])
    episode_data = _load_episode_data(row["episode_dir"])
    map_data = load_map_artifact(output_root / "maps" / f"{row['map_id']}.npz")
    kidnap_step = int(row["kidnap_step"])
    kidnap_pose = episode_data["kidnap_pose"]

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(step_frame["t"], step_frame["translation_error_m"], color=METHOD_COLORS.get(row["method"], "#444444"), linewidth=2.0)
    axes[0].axvline(step_frame["t"].iloc[kidnap_step], color="#B22222", linestyle="--", linewidth=1.8)
    axes[0].set_title("Recovery Error Trace")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("translation error [m]")

    _plot_map(axes[1], map_data.occupancy)
    gt = episode_data["gt_states"]
    est = episode_data["estimated_states"]
    axes[1].plot(_world_to_grid_series(gt[:, 1], map_data.resolution_m), _world_to_grid_series(gt[:, 2], map_data.resolution_m), color=METHOD_COLORS["gt"], linewidth=2.0, label="GT")
    axes[1].plot(_world_to_grid_series(est[:, 0], map_data.resolution_m), _world_to_grid_series(est[:, 1], map_data.resolution_m), color=METHOD_COLORS.get(row["method"], "#444444"), linewidth=2.0, label=row["method"].upper())
    axes[1].scatter(_world_to_grid_series(np.array([kidnap_pose[0]]), map_data.resolution_m), _world_to_grid_series(np.array([kidnap_pose[1]]), map_data.resolution_m), color="#B22222", s=50, label="kidnap pose")
    axes[1].set_title("Trajectory and Kidnap Pose")
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def _render_gpu_scalability(figures_dir: Path, output_root: Path, strict: bool, source_experiment: str | None) -> Path:
    path = figures_dir / "fig_gpu_scalability.png"
    throughput_name = throughput_experiment_name(source_experiment)
    throughput_path = output_root / "metrics" / f"{throughput_name}_summary.csv"
    if not throughput_path.exists():
        if strict:
            raise FileNotFoundError(f"Expected throughput summary at {throughput_path}")
        return _save_notice_figure(path, "GPU Scalability", "No throughput summary is available yet.")

    summary = pd.read_csv(throughput_path)
    if summary.empty:
        if strict:
            raise ValueError(f"Throughput summary is empty: {throughput_path}")
        return _save_notice_figure(path, "GPU Scalability", "Throughput summary is empty.")

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for device, device_frame in summary.groupby("device"):
        label = str(device).upper()
        axes[0].plot(device_frame["batch_size"], device_frame["scans_per_second"], marker="o", linewidth=2.0, label=label)
        axes[1].plot(device_frame["batch_size"], device_frame["particle_likelihoods_per_second"], marker="o", linewidth=2.0, label=label)

    axes[0].set_title("Kernel Throughput: Scan Simulation")
    axes[0].set_xlabel("batch size")
    axes[0].set_ylabel("scans / s")
    axes[1].set_title("Kernel Throughput: Particle Likelihoods")
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("particles / s")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def _render_failure_cases(figures_dir: Path, catalog: pd.DataFrame, output_root: Path) -> Path:
    path = figures_dir / "fig_failure_cases.png"
    if catalog.empty:
        return _save_notice_figure(path, "Failure Cases", "No episode outputs were found.")

    failed_rows = catalog[catalog["failed"].astype(bool)] if "failed" in catalog.columns else pd.DataFrame()
    if failed_rows.empty:
        return _save_notice_figure(path, "Failure Cases", "No failed runs were found in the requested benchmark output.")

    selected = failed_rows.sort_values("position_rmse_m", ascending=False).head(4)
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax in axes_flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes_flat, selected.iterrows()):
        ax.axis("on")
        map_data = load_map_artifact(output_root / "maps" / f"{row['map_id']}.npz")
        episode_data = _load_episode_data(row["episode_dir"])
        _plot_map(ax, map_data.occupancy)
        gt = episode_data["gt_states"]
        est = episode_data["estimated_states"]
        ax.plot(_world_to_grid_series(gt[:, 1], map_data.resolution_m), _world_to_grid_series(gt[:, 2], map_data.resolution_m), color=METHOD_COLORS["gt"], linewidth=2.0, label="GT")
        ax.plot(_world_to_grid_series(est[:, 0], map_data.resolution_m), _world_to_grid_series(est[:, 1], map_data.resolution_m), color=METHOD_COLORS.get(row["method"], "#444444"), linewidth=2.0, label=row["method"].upper())
        ax.set_title(f"{row['method'].upper()} | RMSE {row['position_rmse_m']:.3f} m")
    if len(selected) > 0:
        axes_flat[0].legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def render_figure_suite(config: ProjectConfig, output_root: str | Path) -> list[Path]:
    apply_paper_style()
    figures_dir = ensure_dir(Path(output_root) / "figures")
    output_path = Path(output_root)
    _, _, catalog = _resolve_catalog(config, output_path)
    selected = _selected_figure_names(config)

    renderers = {
        "fig_localization_overview.png": lambda: _render_localization_overview(figures_dir, catalog, output_path),
        "fig_error_vs_time.png": lambda: _render_error_vs_time(figures_dir, catalog),
        "fig_robustness_noise_sweep.png": lambda: _render_robustness_noise_sweep(figures_dir, catalog),
        "fig_error_distribution.png": lambda: _render_error_distribution(figures_dir, catalog),
        "fig_particle_evolution.png": lambda: _render_particle_evolution(figures_dir, catalog, output_path),
        "fig_kidnapped_recovery.png": lambda: _render_kidnapped_recovery(figures_dir, catalog, output_path),
        "fig_gpu_scalability.png": lambda: _render_gpu_scalability(
            figures_dir,
            output_path,
            strict=config.render.strict_run_root,
            source_experiment=config.render.source_experiment,
        ),
        "fig_failure_cases.png": lambda: _render_failure_cases(figures_dir, catalog, output_path),
    }

    rendered: list[Path] = []
    for filename in selected:
        try:
            renderer = renderers[filename]
        except KeyError as exc:
            raise ValueError(f"Unknown figure filename requested: {filename}") from exc
        rendered.append(renderer())
    return rendered
