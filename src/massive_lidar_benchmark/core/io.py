"""Core file-system helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(content, encoding="utf-8")

