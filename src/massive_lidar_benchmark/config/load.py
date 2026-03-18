"""Load YAML configs with shallow inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.core.io import ensure_dir, write_yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "inherit_from":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_mapping(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inherit_from = raw.get("inherit_from")
    if inherit_from is None:
        return raw
    parent_path = (path.parent / inherit_from).resolve()
    parent = load_mapping(parent_path)
    return _deep_merge(parent, raw)


def load_config(config_path: str | Path) -> ProjectConfig:
    config = ProjectConfig.from_mapping(load_mapping(config_path))
    if not config.render.allow_fallback:
        config.render.strict_run_root = True
    return config


def save_resolved_config(config: ProjectConfig, output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)
    write_yaml(path, config.to_mapping())
