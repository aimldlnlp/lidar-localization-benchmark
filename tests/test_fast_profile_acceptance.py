import pandas as pd

from massive_lidar_benchmark.benchmarks.summary import (
    PORTFOLIO_REQUIRED_FIGURES,
    PORTFOLIO_VIDEO_NAMES,
    build_portfolio_asset_entries,
    load_episode_catalog,
)
from tests.helpers import write_episode_artifact


def test_portfolio_fast_manifest_requires_only_core_assets(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    for noise_value in [0.02, 0.10]:
        write_episode_artifact(output_root, "portfolio_fast", "mcl", noise_value, seed=0, episode_index=int(noise_value * 100) + 1, position_rmse_m=0.05 + noise_value, failed=False)
        write_episode_artifact(output_root, "portfolio_fast", "ekf", noise_value, seed=0, episode_index=int(noise_value * 100) + 1, position_rmse_m=0.08 + noise_value, failed=False)

    catalog = load_episode_catalog(output_root / "runs" / "portfolio_fast", source_experiment="portfolio_fast")
    entries = build_portfolio_asset_entries(
        output_root=output_root,
        noise_catalog=catalog,
        throughput_metrics=pd.DataFrame(),
        validated=False,
    )

    required_assets = {entry["asset_name"] for entry in entries if entry["required_for_portfolio"]}
    optional_assets = {entry["asset_name"] for entry in entries if not entry["required_for_portfolio"]}

    assert required_assets == set(PORTFOLIO_REQUIRED_FIGURES + PORTFOLIO_VIDEO_NAMES)
    assert optional_assets == {"fig_gpu_scalability.png"}
    assert all(entry["portfolio_profile"] == "portfolio_fast" for entry in entries)
