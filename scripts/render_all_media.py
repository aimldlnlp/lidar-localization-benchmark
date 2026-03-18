"""Render media for the available render configs."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for bootstrap_path in (REPO_ROOT, SRC_ROOT):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

from massive_lidar_benchmark.cli import handle_render_figures, handle_render_video


def main() -> int:
    render_dir = Path("configs/render")
    for figure_config in sorted(render_dir.glob("*figures*.yaml")):
        handle_render_figures(figure_config)
    render_dir = Path("configs/render")
    for config_path in sorted(render_dir.glob("*demo*.yaml")):
        handle_render_video(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
