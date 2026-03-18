"""Run benchmark entrypoints for all benchmark configs."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for bootstrap_path in (REPO_ROOT, SRC_ROOT):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

from massive_lidar_benchmark.cli import handle_run_benchmark


def main() -> int:
    benchmark_dir = Path("configs/benchmarks")
    for config_path in sorted(benchmark_dir.glob("*.yaml")):
        handle_run_benchmark(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
