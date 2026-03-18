from pathlib import Path
import json

from scripts.build_portfolio_report import build_portfolio_report
from tests.helpers import write_episode_artifact


def test_build_portfolio_report_succeeds_without_optional_throughput(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Test README\n\n<!-- PORTFOLIO_RESULTS_CARD:START -->\nplaceholder\n<!-- PORTFOLIO_RESULTS_CARD:END -->\n",
        encoding="utf-8",
    )

    for noise_value, position_mcl, position_ekf in [
        (0.02, 0.06, 0.10),
        (0.10, 0.16, 0.24),
    ]:
        write_episode_artifact(output_root, "portfolio_fast", "mcl", noise_value, seed=0, episode_index=int(noise_value * 100) + 1, position_rmse_m=position_mcl, failed=False)
        write_episode_artifact(output_root, "portfolio_fast", "ekf", noise_value, seed=0, episode_index=int(noise_value * 100) + 1, position_rmse_m=position_ekf, failed=False)

    paths = build_portfolio_report(output_root=output_root, readme_path=readme_path)

    assert paths["noise_summary"].exists()
    assert paths["throughput_summary"].exists()
    assert paths["portfolio_summary"].exists()
    assert paths["results_card_md"].exists()
    assert paths["manifest"].exists()

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    gpu_entry = next(entry for entry in manifest["assets"] if entry["asset_name"] == "fig_gpu_scalability.png")
    assert gpu_entry["required_for_portfolio"] is False
    assert "Results Snapshot" in paths["results_card_md"].read_text(encoding="utf-8")
    assert "Optional CUDA throughput snapshot:" in paths["results_card_md"].read_text(encoding="utf-8")
    assert "Not generated in this run." in paths["results_card_md"].read_text(encoding="utf-8")
    assert "Method | Episodes" in readme_path.read_text(encoding="utf-8")
