"""Bootstrap package for running the project directly from the repository root."""

from __future__ import annotations

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE_ROOT = _PACKAGE_ROOT.parent / "src" / "massive_lidar_benchmark"

if _SRC_PACKAGE_ROOT.exists():
    __path__.append(str(_SRC_PACKAGE_ROOT))

__all__ = ["__version__"]
__version__ = "0.1.0"
