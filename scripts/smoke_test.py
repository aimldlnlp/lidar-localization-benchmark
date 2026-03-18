"""Run the local smoke command."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for bootstrap_path in (REPO_ROOT, SRC_ROOT):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

from massive_lidar_benchmark.cli import handle_smoke


def main() -> int:
    return handle_smoke(Path("configs/debug/smoke.yaml"))


if __name__ == "__main__":
    raise SystemExit(main())
