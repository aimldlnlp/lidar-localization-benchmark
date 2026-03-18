"""Summary helpers, episode selection, and portfolio provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PORTFOLIO_PROFILE_FAST = "portfolio_fast"
THROUGHPUT_PROFILE_FAST = "throughput_gpu_fast"

PORTFOLIO_REQUIRED_FIGURES = [
    "fig_localization_overview.png",
    "fig_error_vs_time.png",
    "fig_robustness_noise_sweep.png",
    "fig_particle_evolution.png",
]

PORTFOLIO_OPTIONAL_FIGURES = [
    "fig_gpu_scalability.png",
]

PORTFOLIO_FIGURE_FILENAMES = PORTFOLIO_REQUIRED_FIGURES + PORTFOLIO_OPTIONAL_FIGURES

PORTFOLIO_VIDEO_NAMES = [
    "demo_main_localization",
    "demo_particle_convergence",
    "demo_noise_robustness",
]


def collect_episode_metric_files(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    if not root.exists():
        return []
    return sorted(root.rglob("episode_metrics.csv"))


def collect_throughput_metric_files(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    if not root.exists():
        return []
    direct = root / "throughput_metrics.csv"
    if direct.exists():
        return [direct]
    return sorted(root.rglob("throughput_metrics.csv"))


def _filter_by_experiment(frame: pd.DataFrame, source_experiment: str | None) -> pd.DataFrame:
    if frame.empty or source_experiment is None or "experiment" not in frame.columns:
        return frame
    return frame[frame["experiment"].astype(str) == source_experiment].copy()


def load_episode_catalog(run_root: str | Path, source_experiment: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in collect_episode_metric_files(run_root):
        frame = pd.read_csv(metrics_path)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        row["episode_dir"] = str(metrics_path.parent)
        rows.append(row)
    catalog = pd.DataFrame(rows)
    return _filter_by_experiment(catalog, source_experiment)


def aggregate_episode_metrics(run_root: str | Path, source_experiment: str | None = None) -> pd.DataFrame:
    csv_paths = collect_episode_metric_files(run_root)
    if not csv_paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in csv_paths]
    metrics = pd.concat(frames, ignore_index=True)
    return _filter_by_experiment(metrics, source_experiment)


def aggregate_step_metrics(catalog: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in catalog.iterrows():
        step_path = Path(str(row["episode_dir"])) / "step_metrics.csv"
        if not step_path.exists():
            continue
        frame = pd.read_csv(step_path)
        for column in [
            "experiment",
            "method",
            "map_family",
            "map_id",
            "traj_id",
            "seed",
            "episode_index",
            "range_noise_std_m",
        ]:
            if column in row.index:
                frame[column] = row[column]
        frame["episode_dir"] = row["episode_dir"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_throughput_metrics(run_root: str | Path) -> pd.DataFrame:
    csv_paths = collect_throughput_metric_files(run_root)
    if not csv_paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in csv_paths]
    return pd.concat(frames, ignore_index=True)


def summarize_metrics_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    if {"device", "batch_size", "scan_batches_per_second", "particle_likelihoods_per_second"}.issubset(metrics.columns):
        group_cols = [column for column in ["experiment", "device", "batch_size", "available"] if column in metrics.columns]
        return (
            metrics.groupby(group_cols, dropna=False)
            .agg(
                scan_batches_per_second=("scan_batches_per_second", "mean"),
                scans_per_second=("scans_per_second", "mean"),
                particle_likelihoods_per_second=("particle_likelihoods_per_second", "mean"),
                runtime_s=("runtime_s", "mean"),
            )
            .reset_index()
            .sort_values(group_cols)
        )

    group_cols = [column for column in ["experiment", "method", "range_noise_std_m", "map_family", "seed"] if column in metrics.columns]
    if not group_cols:
        group_cols = [column for column in ["experiment", "method"] if column in metrics.columns]
    if not group_cols:
        return metrics.copy()

    aggregations: dict[str, tuple[str, str]] = {}
    for column, reducer in [
        ("position_rmse_m", "mean"),
        ("heading_rmse_deg", "mean"),
        ("median_translation_error_m", "mean"),
        ("convergence_time_s", "mean"),
        ("failed", "mean"),
        ("kidnapped_recovered", "mean"),
        ("scans_per_second", "mean"),
        ("episodes_per_second", "mean"),
        ("runtime_s", "mean"),
        ("ess_mean", "mean"),
        ("ess_min", "mean"),
    ]:
        if column in metrics.columns:
            aggregations[column] = (column, reducer)

    if "episode_index" in metrics.columns:
        aggregations["episodes"] = ("episode_index", "count")

    summary = metrics.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
    return summary.sort_values(group_cols).reset_index(drop=True)


def episode_identifier(row: pd.Series | dict[str, Any]) -> str:
    def _value(key: str, default: Any = "na") -> Any:
        if isinstance(row, pd.Series):
            return row.get(key, default)
        return row.get(key, default)

    noise_value = _value("range_noise_std_m", _value("sweep_value", "na"))
    if pd.isna(noise_value):
        noise_str = "na"
    else:
        try:
            noise_str = f"{float(noise_value):.3f}"
        except (TypeError, ValueError):
            noise_str = str(noise_value)

    return (
        f"{_value('experiment')}|{_value('method')}|{_value('map_family')}|{_value('map_id')}|"
        f"{_value('traj_id')}|seed={int(_value('seed', -1))}|episode={int(_value('episode_index', -1))}|"
        f"noise={noise_str}"
    )


def _numeric_filter(frame: pd.DataFrame, column: str, target: float) -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    numeric = pd.to_numeric(frame[column], errors="coerce")
    mask = numeric.notna() & np.isclose(numeric.to_numpy(dtype=float), float(target), atol=1e-9)
    return frame.loc[mask].copy()


def _numeric_noise_values(catalog: pd.DataFrame) -> list[float]:
    if "range_noise_std_m" not in catalog.columns:
        return []
    values = pd.to_numeric(catalog["range_noise_std_m"], errors="coerce")
    return sorted(float(value) for value in values.dropna().unique())


def preferred_noise_levels(catalog: pd.DataFrame) -> tuple[float, float]:
    values = _numeric_noise_values(catalog)
    if not values:
        return 0.02, 0.10

    high_noise = 0.10 if any(np.isclose(value, 0.10, atol=1e-9) for value in values) else values[-1]
    if any(np.isclose(value, 0.02, atol=1e-9) for value in values):
        low_noise = 0.02
    elif any(np.isclose(value, 0.0, atol=1e-9) for value in values):
        low_noise = 0.0
    else:
        low_noise = values[0]
    return float(low_noise), float(high_noise)


def throughput_experiment_name(source_experiment: str | None) -> str:
    if source_experiment in {None, PORTFOLIO_PROFILE_FAST, THROUGHPUT_PROFILE_FAST}:
        return THROUGHPUT_PROFILE_FAST
    name = str(source_experiment)
    if "medium" in name:
        return "throughput_gpu_medium"
    if "throughput" in name:
        return name
    return THROUGHPUT_PROFILE_FAST


def _scenario_columns(catalog: pd.DataFrame) -> list[str]:
    return [column for column in ["map_family", "map_id", "traj_id", "seed"] if column in catalog.columns]


def _require_catalog(catalog: pd.DataFrame, message: str) -> pd.DataFrame:
    if catalog.empty:
        raise ValueError(message)
    return catalog.copy()


def select_representative_episode_group(
    catalog: pd.DataFrame,
    anchor_method: str = "mcl",
    noise_value: float = 0.02,
) -> pd.DataFrame:
    catalog = _require_catalog(catalog, "Episode catalog is empty.")
    anchor_rows = catalog[catalog["method"].astype(str) == anchor_method].copy()
    anchor_rows = _numeric_filter(anchor_rows, "range_noise_std_m", noise_value)
    if "failed" in anchor_rows.columns:
        successful = anchor_rows[~anchor_rows["failed"].astype(bool)].copy()
        if not successful.empty:
            anchor_rows = successful
    anchor_rows = _require_catalog(anchor_rows, f"No {anchor_method} episodes were found for noise={noise_value}.")

    median_rmse = float(anchor_rows["position_rmse_m"].median())
    ranking = anchor_rows.assign(_distance=(anchor_rows["position_rmse_m"] - median_rmse).abs())
    sort_cols = ["_distance", "map_family", "map_id", "traj_id", "seed", "episode_index"]
    ranking = ranking.sort_values([column for column in sort_cols if column in ranking.columns]).reset_index(drop=True)
    selected = ranking.iloc[0]

    scenario_mask = pd.Series(True, index=catalog.index)
    for column in _scenario_columns(catalog):
        scenario_mask &= catalog[column] == selected[column]
    if "range_noise_std_m" in catalog.columns:
        scenario_mask &= np.isclose(pd.to_numeric(catalog["range_noise_std_m"], errors="coerce"), float(selected["range_noise_std_m"]), atol=1e-9)
    return catalog.loc[scenario_mask].sort_values(["method", "map_family", "map_id", "traj_id", "seed", "episode_index"]).reset_index(drop=True)


def select_noise_comparison_catalog(
    catalog: pd.DataFrame,
    low_noise: float | None = None,
    high_noise: float = 0.10,
    anchor_method: str = "mcl",
) -> pd.DataFrame:
    catalog = _require_catalog(catalog, "Episode catalog is empty.")
    if low_noise is None:
        low_noise, detected_high = preferred_noise_levels(catalog)
        high_noise = detected_high

    anchor_rows = catalog[catalog["method"].astype(str) == anchor_method].copy()
    if "failed" in anchor_rows.columns:
        successful = anchor_rows[~anchor_rows["failed"].astype(bool)].copy()
        if not successful.empty:
            anchor_rows = successful

    low_rows = _numeric_filter(anchor_rows, "range_noise_std_m", low_noise)
    high_rows = _numeric_filter(anchor_rows, "range_noise_std_m", high_noise)
    low_rows = _require_catalog(low_rows, f"No {anchor_method} episodes were found for noise={low_noise}.")
    high_rows = _require_catalog(high_rows, f"No {anchor_method} episodes were found for noise={high_noise}.")

    scenario_cols = _scenario_columns(catalog)
    low_keys = low_rows[scenario_cols].drop_duplicates()
    high_keys = high_rows[scenario_cols].drop_duplicates()
    shared = low_keys.merge(high_keys, on=scenario_cols, how="inner")
    if shared.empty:
        raise ValueError("No shared scenario exists between the requested low/high noise conditions.")

    high_shared = high_rows.merge(shared, on=scenario_cols, how="inner")
    median_rmse = float(high_shared["position_rmse_m"].median())
    high_shared = high_shared.assign(_distance=(high_shared["position_rmse_m"] - median_rmse).abs())
    high_shared = high_shared.sort_values(["_distance", "map_family", "map_id", "traj_id", "seed", "episode_index"]).reset_index(drop=True)
    selected = high_shared.iloc[0]

    scenario_mask = pd.Series(True, index=catalog.index)
    for column in scenario_cols:
        scenario_mask &= catalog[column] == selected[column]
    low_mask = np.isclose(pd.to_numeric(catalog["range_noise_std_m"], errors="coerce"), float(low_noise), atol=1e-9)
    high_mask = np.isclose(pd.to_numeric(catalog["range_noise_std_m"], errors="coerce"), float(high_noise), atol=1e-9)
    return catalog.loc[scenario_mask & (low_mask | high_mask)].sort_values(
        ["range_noise_std_m", "method", "map_family", "map_id", "traj_id", "seed", "episode_index"]
    ).reset_index(drop=True)


def select_parallel_grid_catalog(catalog: pd.DataFrame, noise_value: float = 0.05, limit: int = 16) -> pd.DataFrame:
    catalog = _require_catalog(catalog, "Episode catalog is empty.")
    rows = _numeric_filter(catalog, "range_noise_std_m", noise_value)
    if "failed" in rows.columns:
        successful = rows[~rows["failed"].astype(bool)].copy()
        if not successful.empty:
            rows = successful
    rows = _require_catalog(rows, f"No non-failed episodes were found for noise={noise_value}.")
    return rows.sort_values(["method", "map_family", "map_id", "traj_id", "seed", "episode_index"]).head(limit).reset_index(drop=True)


def _episode_ids(frame: pd.DataFrame) -> list[str]:
    return [episode_identifier(row) for _, row in frame.iterrows()]


def _safe_episode_group_ids(catalog: pd.DataFrame, selector) -> list[str]:
    try:
        return _episode_ids(selector(catalog))
    except ValueError:
        return []


def build_portfolio_asset_entries(
    output_root: str | Path,
    noise_catalog: pd.DataFrame,
    throughput_metrics: pd.DataFrame,
    validated: bool,
    portfolio_profile: str = PORTFOLIO_PROFILE_FAST,
    throughput_profile: str = THROUGHPUT_PROFILE_FAST,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    noise_run_root = root / "runs" / portfolio_profile
    throughput_run_root = root / "runs" / throughput_profile

    all_noise_ids = _episode_ids(noise_catalog) if not noise_catalog.empty else []
    representative_ids = _safe_episode_group_ids(
        noise_catalog,
        lambda catalog: select_representative_episode_group(catalog, anchor_method="mcl", noise_value=0.02),
    )

    particle_ids: list[str]
    try:
        representative_group = select_representative_episode_group(noise_catalog, anchor_method="mcl", noise_value=0.02)
        representative_mcl = representative_group[representative_group["method"].astype(str) == "mcl"].reset_index(drop=True)
        particle_ids = _episode_ids(representative_mcl.iloc[[0]]) if not representative_mcl.empty else representative_ids[:1]
    except ValueError:
        particle_ids = representative_ids[:1]

    noise_compare_ids = _safe_episode_group_ids(noise_catalog, lambda catalog: select_noise_comparison_catalog(catalog, low_noise=None, high_noise=0.10))
    throughput_available = not throughput_metrics.empty

    asset_specs = [
        ("fig_localization_overview.png", "figure", portfolio_profile, noise_run_root, representative_ids, True, [root / "figures" / "fig_localization_overview.png"]),
        ("fig_error_vs_time.png", "figure", portfolio_profile, noise_run_root, all_noise_ids, True, [root / "figures" / "fig_error_vs_time.png"]),
        ("fig_robustness_noise_sweep.png", "figure", portfolio_profile, noise_run_root, all_noise_ids, True, [root / "figures" / "fig_robustness_noise_sweep.png"]),
        ("fig_particle_evolution.png", "figure", portfolio_profile, noise_run_root, particle_ids, True, [root / "figures" / "fig_particle_evolution.png"]),
        ("fig_gpu_scalability.png", "figure", throughput_profile, throughput_run_root, [], False, [root / "figures" / "fig_gpu_scalability.png"]),
        ("demo_main_localization", "video_bundle", portfolio_profile, noise_run_root, representative_ids, True, [root / "videos" / "demo_main_localization.mp4", root / "gifs" / "demo_main_localization.gif"]),
        ("demo_particle_convergence", "video_bundle", portfolio_profile, noise_run_root, particle_ids, True, [root / "videos" / "demo_particle_convergence.mp4", root / "gifs" / "demo_particle_convergence.gif"]),
        ("demo_noise_robustness", "video_bundle", portfolio_profile, noise_run_root, noise_compare_ids, True, [root / "videos" / "demo_noise_robustness.mp4", root / "gifs" / "demo_noise_robustness.gif"]),
    ]

    entries: list[dict[str, Any]] = []
    for asset_name, asset_type, source_experiment, run_root_path, episode_ids, required_for_portfolio, artifact_paths in asset_specs:
        artifact_exists = all(path.exists() for path in artifact_paths)
        source_available = run_root_path.exists() and (
            bool(episode_ids) or (source_experiment == throughput_profile and throughput_available)
        )
        validated_for_portfolio = bool(validated and artifact_exists and source_available)
        entries.append(
            {
                "asset_name": asset_name,
                "asset_type": asset_type,
                "artifact_paths": [str(path) for path in artifact_paths],
                "source_experiment": source_experiment,
                "source_run_root": str(run_root_path),
                "source_episode_ids": list(episode_ids),
                "fallback_used": False,
                "artifact_exists": artifact_exists,
                "source_available": source_available,
                "required_for_portfolio": bool(required_for_portfolio),
                "portfolio_profile": portfolio_profile,
                "validated_for_portfolio": validated_for_portfolio,
            }
        )
    return entries
