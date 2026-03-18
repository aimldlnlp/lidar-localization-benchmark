"""Command-line interface for the synthetic localization benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from massive_lidar_benchmark.benchmarks.runner import run_benchmark_experiment, run_method_experiment
from massive_lidar_benchmark.benchmarks.summary import aggregate_episode_metrics, load_throughput_metrics, summarize_metrics_frame
from massive_lidar_benchmark.config.load import load_config, save_resolved_config
from massive_lidar_benchmark.core.io import ensure_dir
from massive_lidar_benchmark.core.logging import configure_logging, get_logger
from massive_lidar_benchmark.maps.generate import generate_maps_from_config
from massive_lidar_benchmark.traj.generate import generate_trajectories_from_config
from massive_lidar_benchmark.viz.animations import render_video_demo
from massive_lidar_benchmark.viz.plots import render_figure_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="massive-lidar-benchmark",
        description="Synthetic LiDAR localization benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", required=True, help="Path to YAML config file.")

    subparsers.add_parser("smoke", parents=[config_parser], help="Run a local smoke path.")
    subparsers.add_parser("generate-maps", parents=[config_parser], help="Generate synthetic maps.")
    subparsers.add_parser(
        "generate-trajectories",
        parents=[config_parser],
        help="Generate synthetic trajectories.",
    )

    method_parser = subparsers.add_parser(
        "run-method",
        parents=[config_parser],
        help="Run a single localization method.",
    )
    method_parser.add_argument("--method", choices=["ekf", "mcl"], required=True)

    subparsers.add_parser(
        "run-benchmark",
        parents=[config_parser],
        help="Run a benchmark suite.",
    )
    subparsers.add_parser(
        "render-figures",
        parents=[config_parser],
        help="Render figure outputs.",
    )
    subparsers.add_parser(
        "render-video",
        parents=[config_parser],
        help="Render one video output.",
    )

    summary_parser = subparsers.add_parser(
        "export-summary",
        help="Export an aggregate CSV summary.",
    )
    summary_parser.add_argument("--run-root", required=True)
    summary_parser.add_argument("--out", required=True)

    return parser


def _prepare_run(config_path: str | Path):
    config = load_config(config_path)
    output_root = ensure_dir(config.output_root)
    log_path = output_root / "logs" / f"{config.experiment.name}.log"
    configure_logging(log_path)
    save_resolved_config(config, output_root / "resolved_configs" / f"{config.experiment.name}.yaml")
    return config, output_root


def handle_smoke(config_path: str | Path) -> int:
    config, output_root = _prepare_run(config_path)
    logger = get_logger(__name__)
    maps = generate_maps_from_config(config, output_root)
    trajectories = generate_trajectories_from_config(config, output_root, maps)
    methods_run = []
    if config.ekf.enabled:
        run_method_experiment(config, output_root, method="ekf", map_artifacts=maps, trajectory_artifacts=trajectories)
        methods_run.append("ekf")
    if config.mcl.enabled:
        run_method_experiment(config, output_root, method="mcl", map_artifacts=maps, trajectory_artifacts=trajectories)
        methods_run.append("mcl")
    if not methods_run:
        logger.info(
            "Smoke path completed with %d map artifact(s) and %d trajectory artifact(s).",
            len(maps),
            len(trajectories),
        )
    else:
        logger.info("Smoke path completed with method runs: %s.", ", ".join(methods_run))
    return 0


def handle_generate_maps(config_path: str | Path) -> int:
    config, output_root = _prepare_run(config_path)
    generate_maps_from_config(config, output_root)
    return 0


def handle_generate_trajectories(config_path: str | Path) -> int:
    config, output_root = _prepare_run(config_path)
    maps = generate_maps_from_config(config, output_root)
    generate_trajectories_from_config(config, output_root, maps)
    return 0


def handle_run_method(config_path: str | Path, method: str) -> int:
    config, output_root = _prepare_run(config_path)
    run_method_experiment(config, output_root, method=method)
    return 0


def handle_run_benchmark(config_path: str | Path) -> int:
    config, output_root = _prepare_run(config_path)
    run_benchmark_experiment(config, output_root)
    return 0


def handle_render_figures(config_path: str | Path) -> int:
    config, output_root = _prepare_run(config_path)
    render_figure_suite(config, output_root)
    return 0


def handle_render_video(config_path: str | Path) -> int:
    config, output_root = _prepare_run(config_path)
    render_video_demo(config, output_root)
    return 0


def handle_export_summary(run_root: str | Path, out_path: str | Path) -> int:
    import pandas as pd

    output_path = Path(out_path)
    ensure_dir(output_path.parent)
    metrics = aggregate_episode_metrics(run_root)
    if metrics.empty:
        metrics = load_throughput_metrics(run_root)
    summary = summarize_metrics_frame(metrics)
    if summary.empty:
        summary = pd.DataFrame([{"status": "empty", "message": "No episode_metrics.csv files were found."}])
    summary.to_csv(output_path, index=False)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "smoke":
        return handle_smoke(args.config)
    if args.command == "generate-maps":
        return handle_generate_maps(args.config)
    if args.command == "generate-trajectories":
        return handle_generate_trajectories(args.config)
    if args.command == "run-method":
        return handle_run_method(args.config, args.method)
    if args.command == "run-benchmark":
        return handle_run_benchmark(args.config)
    if args.command == "render-figures":
        return handle_render_figures(args.config)
    if args.command == "render-video":
        return handle_render_video(args.config)
    if args.command == "export-summary":
        return handle_export_summary(args.run_root, args.out)

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
