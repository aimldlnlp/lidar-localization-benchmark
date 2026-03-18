"""Map artifact generation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.core.io import ensure_dir, write_yaml
from massive_lidar_benchmark.core.logging import get_logger
from massive_lidar_benchmark.core.seeding import spawn_seed_integers
from massive_lidar_benchmark.core.types import OccupancyMap
from massive_lidar_benchmark.maps.families import generate_map_family
from massive_lidar_benchmark.maps.validate import free_ratio
from massive_lidar_benchmark.viz.style import apply_paper_style


def _map_stem(map_data: OccupancyMap, index: int) -> str:
    return f"{map_data.family}_seed{map_data.seed:010d}_map{index:02d}"


def load_map_artifact(npz_path: str | Path) -> OccupancyMap:
    payload = np.load(npz_path, allow_pickle=False)
    family = str(payload["family"]) if payload["family"].ndim == 0 else str(payload["family"].item())
    return OccupancyMap(
        occupancy=payload["occupancy"].astype(bool),
        resolution_m=float(payload["resolution_m"]),
        origin_xy=tuple(float(v) for v in payload["origin_xy"]),
        family=family,
        seed=int(payload["seed"]),
    )


def save_map_artifacts(map_data: OccupancyMap, output_root: str | Path, index: int) -> Path:
    maps_dir = ensure_dir(Path(output_root) / "maps")
    stem = _map_stem(map_data, index)
    npz_path = maps_dir / f"{stem}.npz"
    png_path = maps_dir / f"{stem}.png"
    yaml_path = maps_dir / f"{stem}.yaml"

    np.savez_compressed(
        npz_path,
        occupancy=map_data.occupancy.astype(np.uint8),
        resolution_m=map_data.resolution_m,
        origin_xy=np.asarray(map_data.origin_xy, dtype=float),
        family=map_data.family,
        seed=map_data.seed,
    )

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(map_data.occupancy, cmap="gray_r", origin="lower")
    ax.set_title(f"{map_data.family} | seed {map_data.seed}")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.savefig(png_path, dpi=180, facecolor="white")
    plt.close(fig)

    write_yaml(
        yaml_path,
        {
            "family": map_data.family,
            "seed": map_data.seed,
            "resolution_m": map_data.resolution_m,
            "origin_xy": list(map_data.origin_xy),
            "shape": list(map_data.shape),
            "free_ratio": free_ratio(map_data.occupancy),
        },
    )
    return npz_path


def generate_maps_from_config(config: ProjectConfig, output_root: str | Path) -> list[Path]:
    logger = get_logger(__name__)
    artifacts: list[Path] = []
    families = list(config.benchmark.map_families) if config.benchmark.map_families else [config.map.family]
    for family_index, family in enumerate(families):
        family_config = deepcopy(config.map)
        family_config.family = family
        seeds = spawn_seed_integers(config.seed + family_index, config.benchmark.maps_per_family)
        for map_index, map_seed in enumerate(seeds):
            map_data = generate_map_family(family_config, map_seed)
            artifact_path = save_map_artifacts(map_data, output_root, index=map_index)
            artifacts.append(artifact_path)
    logger.info("Generated %d map artifact(s) in %s.", len(artifacts), Path(output_root) / "maps")
    return artifacts
