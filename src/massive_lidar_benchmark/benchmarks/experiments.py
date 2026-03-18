"""Experiment metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from massive_lidar_benchmark.config.schema import ProjectConfig


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    sweep_key: str | None
    values: list[Any]
    notes: str

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "ExperimentSpec":
        return cls(
            name=config.experiment.name,
            sweep_key=config.experiment.sweep_key,
            values=list(config.experiment.values),
            notes=config.experiment.notes,
        )

