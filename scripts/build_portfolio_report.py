"""Build fast-portfolio summaries, tables, provenance, and README inserts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for bootstrap_path in (REPO_ROOT, SRC_ROOT):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

import pandas as pd

from massive_lidar_benchmark.benchmarks.summary import (
    PORTFOLIO_PROFILE_FAST,
    THROUGHPUT_PROFILE_FAST,
    aggregate_episode_metrics,
    build_portfolio_asset_entries,
    load_episode_catalog,
    load_throughput_metrics,
    summarize_metrics_frame,
)
from massive_lidar_benchmark.core.io import ensure_dir, write_json, write_text


README_RESULTS_START = "<!-- PORTFOLIO_RESULTS_CARD:START -->"
README_RESULTS_END = "<!-- PORTFOLIO_RESULTS_CARD:END -->"


def _empty_throughput_summary(experiment_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "experiment",
            "device",
            "batch_size",
            "available",
            "scan_batches_per_second",
            "scans_per_second",
            "particle_likelihoods_per_second",
            "runtime_s",
        ]
    ).assign(experiment=experiment_name)


def _build_localization_summary(noise_metrics: pd.DataFrame) -> pd.DataFrame:
    if noise_metrics.empty:
        return pd.DataFrame()
    return (
        noise_metrics.groupby("method", dropna=False)
        .agg(
            episodes=("episode_index", "count"),
            position_rmse_m=("position_rmse_m", "mean"),
            median_translation_error_m=("median_translation_error_m", "mean"),
            heading_rmse_deg=("heading_rmse_deg", "mean"),
            failure_rate=("failed", "mean"),
        )
        .reset_index()
        .sort_values("method")
        .reset_index(drop=True)
    )


def _build_portfolio_summary(
    localization_summary: pd.DataFrame,
    throughput_summary: pd.DataFrame,
    portfolio_profile: str,
    throughput_profile: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in localization_summary.iterrows():
        rows.append(
            {
                "row_type": "localization",
                "label": f"{portfolio_profile}_mean",
                "method": row["method"],
                "device": None,
                "batch_size": None,
                "episodes": int(row["episodes"]),
                "position_rmse_m": float(row["position_rmse_m"]),
                "median_translation_error_m": float(row["median_translation_error_m"]),
                "heading_rmse_deg": float(row["heading_rmse_deg"]),
                "failure_rate": float(row["failure_rate"]),
                "scans_per_second": None,
                "particle_likelihoods_per_second": None,
            }
        )

    if throughput_summary.empty:
        rows.append(
            {
                "row_type": "throughput_status",
                "label": f"{throughput_profile}_optional_bonus_not_generated",
                "method": None,
                "device": None,
                "batch_size": None,
                "episodes": None,
                "position_rmse_m": None,
                "median_translation_error_m": None,
                "heading_rmse_deg": None,
                "failure_rate": None,
                "scans_per_second": None,
                "particle_likelihoods_per_second": None,
            }
        )
    else:
        for _, row in throughput_summary.iterrows():
            rows.append(
                {
                    "row_type": "throughput",
                    "label": "kernel_throughput",
                    "method": None,
                    "device": row.get("device"),
                    "batch_size": int(row.get("batch_size", 0)),
                    "episodes": None,
                    "position_rmse_m": None,
                    "median_translation_error_m": None,
                    "heading_rmse_deg": None,
                    "failure_rate": None,
                    "scans_per_second": float(row.get("scans_per_second", 0.0)),
                    "particle_likelihoods_per_second": float(row.get("particle_likelihoods_per_second", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _markdown_results_card(
    localization_summary: pd.DataFrame,
    throughput_summary: pd.DataFrame,
    portfolio_profile: str,
) -> str:
    lines = [
        "### Results Snapshot",
        "",
        "| Method | Episodes | Mean position RMSE [m] | Mean median translation error [m] | Mean heading RMSE [deg] | Failure rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in localization_summary.iterrows():
        lines.append(
            f"| {str(row['method']).upper()} | {int(row['episodes'])} | {row['position_rmse_m']:.4f} | "
            f"{row['median_translation_error_m']:.4f} | {row['heading_rmse_deg']:.3f} | {row['failure_rate']:.3f} |"
        )

    lines.extend(["", "Optional CUDA throughput snapshot:"])
    if throughput_summary.empty:
        lines.append("")
        lines.append("Not generated in this run.")
    else:
        lines.extend(
            [
                "",
                "| Device | Batch size | Scans / s | Particle likelihoods / s |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in throughput_summary.sort_values(["device", "batch_size"]).iterrows():
            lines.append(
                f"| {str(row['device']).upper()} | {int(row['batch_size'])} | {row['scans_per_second']:.2f} | "
                f"{row['particle_likelihoods_per_second']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def _latex_results_card(localization_summary: pd.DataFrame, throughput_summary: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Method & Episodes & Pos. RMSE [m] & Median trans. err. [m] & Heading RMSE [deg] & Failure rate \\\\",
        "\\hline",
    ]
    for _, row in localization_summary.iterrows():
        lines.append(
            f"{str(row['method']).upper()} & {int(row['episodes'])} & {row['position_rmse_m']:.4f} & "
            f"{row['median_translation_error_m']:.4f} & {row['heading_rmse_deg']:.3f} & {row['failure_rate']:.3f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    if throughput_summary.empty:
        lines.append("% Optional GPU throughput bonus not generated.")
    else:
        lines.extend(
            [
                "\\begin{tabular}{lrrr}",
                "\\hline",
                "Device & Batch size & Scans / s & Particle likelihoods / s \\\\",
                "\\hline",
            ]
        )
        for _, row in throughput_summary.sort_values(["device", "batch_size"]).iterrows():
            lines.append(
                f"{str(row['device']).upper()} & {int(row['batch_size'])} & {row['scans_per_second']:.2f} & "
                f"{row['particle_likelihoods_per_second']:.2f} \\\\"
            )
        lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _replace_readme_results_block(readme_path: Path, replacement: str) -> None:
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    block = f"{README_RESULTS_START}\n{replacement.rstrip()}\n{README_RESULTS_END}"
    if README_RESULTS_START in text and README_RESULTS_END in text:
        start = text.index(README_RESULTS_START)
        end = text.index(README_RESULTS_END) + len(README_RESULTS_END)
        text = text[:start] + block + text[end:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    readme_path.write_text(text, encoding="utf-8")


def build_portfolio_report(
    output_root: str | Path = "outputs",
    readme_path: str | Path = "README.md",
    portfolio_profile: str = PORTFOLIO_PROFILE_FAST,
    throughput_profile: str = THROUGHPUT_PROFILE_FAST,
) -> dict[str, Path]:
    root = Path(output_root)
    noise_root = root / "runs" / portfolio_profile
    throughput_root = root / "runs" / throughput_profile

    noise_metrics = aggregate_episode_metrics(noise_root, source_experiment=portfolio_profile)
    if noise_metrics.empty:
        raise FileNotFoundError(f"No episode metrics were found under {noise_root}")

    throughput_metrics = load_throughput_metrics(throughput_root)
    if not throughput_metrics.empty and "experiment" in throughput_metrics.columns:
        throughput_metrics = throughput_metrics[throughput_metrics["experiment"].astype(str) == throughput_profile].copy()

    noise_summary = summarize_metrics_frame(noise_metrics)
    throughput_summary = summarize_metrics_frame(throughput_metrics)
    if throughput_summary.empty:
        throughput_summary = _empty_throughput_summary(throughput_profile)
    localization_summary = _build_localization_summary(noise_metrics)
    portfolio_summary = _build_portfolio_summary(localization_summary, throughput_summary, portfolio_profile, throughput_profile)

    metrics_dir = ensure_dir(root / "metrics")
    tables_dir = ensure_dir(root / "tables")
    reports_dir = ensure_dir(root / "reports")

    noise_summary_path = metrics_dir / f"{portfolio_profile}_localization_summary.csv"
    throughput_summary_path = metrics_dir / f"{throughput_profile}_summary.csv"
    portfolio_summary_path = metrics_dir / f"{portfolio_profile}_summary.csv"
    results_card_md_path = tables_dir / f"{portfolio_profile}_results_card.md"
    results_card_tex_path = tables_dir / f"{portfolio_profile}_results_card.tex"
    manifest_path = reports_dir / f"{portfolio_profile}_evidence_manifest.json"

    noise_summary.to_csv(noise_summary_path, index=False)
    throughput_summary.to_csv(throughput_summary_path, index=False)
    portfolio_summary.to_csv(portfolio_summary_path, index=False)

    results_card_md = _markdown_results_card(localization_summary, throughput_summary, portfolio_profile)
    results_card_tex = _latex_results_card(localization_summary, throughput_summary)
    write_text(results_card_md_path, results_card_md)
    write_text(results_card_tex_path, results_card_tex)
    _replace_readme_results_block(Path(readme_path), results_card_md)

    noise_catalog = load_episode_catalog(noise_root, source_experiment=portfolio_profile)
    manifest_entries = build_portfolio_asset_entries(
        output_root=root,
        noise_catalog=noise_catalog,
        throughput_metrics=throughput_metrics,
        validated=False,
        portfolio_profile=portfolio_profile,
        throughput_profile=throughput_profile,
    )
    write_json(manifest_path, {"assets": manifest_entries})

    return {
        "noise_summary": noise_summary_path,
        "throughput_summary": throughput_summary_path,
        "portfolio_summary": portfolio_summary_path,
        "results_card_md": results_card_md_path,
        "results_card_tex": results_card_tex_path,
        "manifest": manifest_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portfolio tables and evidence manifests.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--portfolio-profile", default=PORTFOLIO_PROFILE_FAST)
    parser.add_argument("--throughput-profile", default=THROUGHPUT_PROFILE_FAST)
    args = parser.parse_args()
    build_portfolio_report(
        output_root=args.output_root,
        readme_path=args.readme,
        portfolio_profile=args.portfolio_profile,
        throughput_profile=args.throughput_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
