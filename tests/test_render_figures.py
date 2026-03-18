from pathlib import Path

from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.viz.plots import FIGURE_FILENAMES, render_figure_suite


def test_render_figure_suite_writes_all_expected_files_without_run_data(tmp_path: Path) -> None:
    config = ProjectConfig()
    config.output_root = str(tmp_path)
    config.render.target_run_dir = str(tmp_path / "runs")

    rendered = render_figure_suite(config, tmp_path)

    assert len(rendered) == len(FIGURE_FILENAMES)
    for filename in FIGURE_FILENAMES:
        assert (tmp_path / "figures" / filename).exists()

