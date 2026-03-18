"""Synthetic map family generators."""

from __future__ import annotations

import numpy as np

from massive_lidar_benchmark.config.schema import MapConfig
from massive_lidar_benchmark.core.types import OccupancyMap


def _base_grid(config: MapConfig) -> np.ndarray:
    width = max(10, int(round(config.width_m / config.resolution_m)))
    height = max(10, int(round(config.height_m / config.resolution_m)))
    grid = np.zeros((height, width), dtype=bool)
    thickness = max(1, int(config.wall_thickness_cells))
    grid[:thickness, :] = True
    grid[-thickness:, :] = True
    grid[:, :thickness] = True
    grid[:, -thickness:] = True
    return grid


def _clip_rect(
    grid: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> None:
    height, width = grid.shape
    grid[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True


def _add_random_rectangles(grid: np.ndarray, rng: np.random.Generator, count: int) -> None:
    height, width = grid.shape
    for _ in range(count):
        rect_w = int(rng.integers(max(4, width // 30), max(6, width // 10)))
        rect_h = int(rng.integers(max(4, height // 30), max(6, height // 10)))
        x0 = int(rng.integers(2, max(3, width - rect_w - 2)))
        y0 = int(rng.integers(2, max(3, height - rect_h - 2)))
        _clip_rect(grid, x0, y0, x0 + rect_w, y0 + rect_h)


def generate_open_map(config: MapConfig, seed: int) -> OccupancyMap:
    rng = np.random.default_rng(seed)
    grid = _base_grid(config)
    obstacle_count = max(4, int(config.obstacle_density * 40))
    _add_random_rectangles(grid, rng, obstacle_count)
    return OccupancyMap(grid, config.resolution_m, (0.0, 0.0), "open", seed)


def generate_room_map(config: MapConfig, seed: int) -> OccupancyMap:
    rng = np.random.default_rng(seed)
    grid = _base_grid(config)
    height, width = grid.shape
    wall = max(2, config.wall_thickness_cells)
    split_x = width // 2
    split_y = height // 2
    door_w = max(6, width // 16)
    door_h = max(6, height // 16)

    _clip_rect(grid, split_x - wall // 2, wall, split_x + wall // 2 + 1, height - wall)
    _clip_rect(grid, wall, split_y - wall // 2, width - wall, split_y + wall // 2 + 1)

    grid[split_y - door_h // 2 : split_y + door_h // 2, split_x - wall : split_x + wall + 1] = False
    grid[split_y - wall : split_y + wall + 1, split_x - door_w // 2 : split_x + door_w // 2] = False

    _add_random_rectangles(grid, rng, max(3, int(config.obstacle_density * 20)))
    return OccupancyMap(grid, config.resolution_m, (0.0, 0.0), "room", seed)


def generate_corridor_map(config: MapConfig, seed: int) -> OccupancyMap:
    grid = _base_grid(config)
    height, width = grid.shape
    wall = max(2, config.wall_thickness_cells)
    for split_x in (width // 3, (2 * width) // 3):
        _clip_rect(grid, split_x - wall // 2, wall, split_x + wall // 2 + 1, height - wall)
    for split_y in (height // 3, (2 * height) // 3):
        _clip_rect(grid, wall, split_y - wall // 2, width - wall, split_y + wall // 2 + 1)

    gap = max(8, min(width, height) // 10)
    grid[height // 6 : height // 6 + gap, width // 3 - wall : width // 3 + wall + 1] = False
    grid[(5 * height) // 6 - gap : (5 * height) // 6, (2 * width) // 3 - wall : (2 * width) // 3 + wall + 1] = False
    grid[height // 3 - wall : height // 3 + wall + 1, width // 6 : width // 6 + gap] = False
    grid[(2 * height) // 3 - wall : (2 * height) // 3 + wall + 1, (5 * width) // 6 - gap : (5 * width) // 6] = False
    return OccupancyMap(grid, config.resolution_m, (0.0, 0.0), "corridor", seed)


def generate_office_map(config: MapConfig, seed: int) -> OccupancyMap:
    room_map = generate_room_map(config, seed)
    rng = np.random.default_rng(seed + 17)
    grid = room_map.occupancy.copy()
    _add_random_rectangles(grid, rng, max(8, int(config.obstacle_density * 60)))
    return OccupancyMap(grid, config.resolution_m, (0.0, 0.0), "office", seed)


def generate_maze_lite_map(config: MapConfig, seed: int) -> OccupancyMap:
    rng = np.random.default_rng(seed)
    grid = _base_grid(config)
    height, width = grid.shape
    wall = max(2, config.wall_thickness_cells)
    spacing = max(18, min(width, height) // 8)
    for x0 in range(spacing, width - spacing, spacing):
        _clip_rect(grid, x0, wall, x0 + wall, height - wall)
        gap_y = int(rng.integers(wall + spacing // 2, height - wall - spacing // 2))
        grid[max(wall, gap_y - spacing // 3) : min(height - wall, gap_y + spacing // 3), x0 : x0 + wall] = False
    for y0 in range(spacing, height - spacing, spacing):
        _clip_rect(grid, wall, y0, width - wall, y0 + wall)
        gap_x = int(rng.integers(wall + spacing // 2, width - wall - spacing // 2))
        grid[y0 : y0 + wall, max(wall, gap_x - spacing // 3) : min(width - wall, gap_x + spacing // 3)] = False
    return OccupancyMap(grid, config.resolution_m, (0.0, 0.0), "maze_lite", seed)


def generate_map_family(config: MapConfig, seed: int) -> OccupancyMap:
    generators = {
        "open": generate_open_map,
        "room": generate_room_map,
        "corridor": generate_corridor_map,
        "office": generate_office_map,
        "maze_lite": generate_maze_lite_map,
    }
    try:
        generator = generators[config.family]
    except KeyError as exc:
        raise ValueError(f"Unsupported map family: {config.family}") from exc
    return generator(config, seed)

