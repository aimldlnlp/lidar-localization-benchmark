from pathlib import Path
import json

import pytest

from massive_lidar_benchmark.benchmarks.summary import build_portfolio_asset_entries, load_episode_catalog
from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.viz.animations import render_video_demo
from massive_lidar_benchmark.viz.plots import render_figure_suite
from scripts.validate_portfolio_assets import validate_portfolio_assets
from tests.helpers import write_episode_artifact


def test_render_video_demo_strict_mode_rejects_missing_run_root(tmp_path: Path) -> None:
    config = ProjectConfig()
    config.output_root = str(tmp_path)
    config.render.target_run_dir = str(tmp_path / "runs" / "missing")
    config.render.target_video_name = "demo_main_localization"
    config.render.strict_run_root = True
    config.render.allow_fallback = False
    config.render.source_experiment = "portfolio_fast"

    with pytest.raises(FileNotFoundError):
        render_video_demo(config, tmp_path, export_media=False)


def test_render_figure_suite_strict_mode_rejects_missing_run_root(tmp_path: Path) -> None:
    config = ProjectConfig()
    config.output_root = str(tmp_path)
    config.render.target_run_dir = str(tmp_path / "runs" / "missing")
    config.render.strict_run_root = True
    config.render.allow_fallback = False
    config.render.source_experiment = "portfolio_fast"

    with pytest.raises(FileNotFoundError):
        render_figure_suite(config, tmp_path)


def _write_portfolio_fast_assets(output_root: Path) -> None:
    for noise_value in [0.02, 0.10]:
        write_episode_artifact(output_root, "portfolio_fast", "mcl", noise_value, seed=0, episode_index=int(noise_value * 100) + 1, position_rmse_m=0.05 + noise_value, failed=False)
        write_episode_artifact(output_root, "portfolio_fast", "ekf", noise_value, seed=0, episode_index=int(noise_value * 100) + 1, position_rmse_m=0.08 + noise_value, failed=False)


def _materialize_required_portfolio_outputs(output_root: Path) -> None:
    catalog = load_episode_catalog(output_root / "runs" / "portfolio_fast", source_experiment="portfolio_fast")
    entries = build_portfolio_asset_entries(output_root, catalog, throughput_metrics=catalog.iloc[0:0], validated=False)

    for entry in entries:
        if not entry["required_for_portfolio"]:
            continue
        for artifact_path in entry["artifact_paths"]:
            path = Path(artifact_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        if entry["asset_type"] == "video_bundle":
            frame_dir = output_root / "frames" / entry["asset_name"]
            frame_dir.mkdir(parents=True, exist_ok=True)
            (frame_dir / "render_manifest.json").write_text(
                json.dumps(
                    {
                        "video_name": entry["asset_name"],
                        "fallback_used": False,
                        "source_experiment": "portfolio_fast",
                        "source_run_root": str(output_root / "runs" / "portfolio_fast"),
                        "source_episode_ids": entry["source_episode_ids"],
                        "validated_for_portfolio": True,
                    }
                ),
                encoding="utf-8",
            )


def test_validate_portfolio_assets_allows_missing_optional_gpu_bonus(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_portfolio_fast_assets(output_root)
    _materialize_required_portfolio_outputs(output_root)

    manifest_path = validate_portfolio_assets(output_root=output_root)

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    gpu_entry = next(entry for entry in payload["assets"] if entry["asset_name"] == "fig_gpu_scalability.png")
    assert gpu_entry["required_for_portfolio"] is False


def test_validate_portfolio_assets_fails_when_core_asset_is_missing(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_portfolio_fast_assets(output_root)
    _materialize_required_portfolio_outputs(output_root)
    (output_root / "figures" / "fig_error_vs_time.png").unlink()

    with pytest.raises(RuntimeError):
        validate_portfolio_assets(output_root=output_root)
