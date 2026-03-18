"""Validate that fast-portfolio assets exist and are sourced from real benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for bootstrap_path in (REPO_ROOT, SRC_ROOT):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

from massive_lidar_benchmark.benchmarks.summary import (
    PORTFOLIO_PROFILE_FAST,
    PORTFOLIO_REQUIRED_FIGURES,
    PORTFOLIO_VIDEO_NAMES,
    THROUGHPUT_PROFILE_FAST,
    build_portfolio_asset_entries,
    load_episode_catalog,
    load_throughput_metrics,
)
from massive_lidar_benchmark.core.io import ensure_dir, write_json


def validate_portfolio_assets(
    output_root: str | Path = "outputs",
    portfolio_profile: str = PORTFOLIO_PROFILE_FAST,
    throughput_profile: str = THROUGHPUT_PROFILE_FAST,
) -> Path:
    root = Path(output_root)
    noise_root = root / "runs" / portfolio_profile
    throughput_root = root / "runs" / throughput_profile
    noise_catalog = load_episode_catalog(noise_root, source_experiment=portfolio_profile)
    throughput_metrics = load_throughput_metrics(throughput_root)
    manifest_entries = build_portfolio_asset_entries(
        output_root=root,
        noise_catalog=noise_catalog,
        throughput_metrics=throughput_metrics,
        validated=True,
        portfolio_profile=portfolio_profile,
        throughput_profile=throughput_profile,
    )

    errors: list[str] = []
    expected_run_root = str(noise_root)

    for filename in PORTFOLIO_REQUIRED_FIGURES:
        if not (root / "figures" / filename).exists():
            errors.append(f"Missing portfolio figure: {root / 'figures' / filename}")

    for video_name in PORTFOLIO_VIDEO_NAMES:
        mp4_path = root / "videos" / f"{video_name}.mp4"
        gif_path = root / "gifs" / f"{video_name}.gif"
        render_manifest_path = root / "frames" / video_name / "render_manifest.json"
        if not mp4_path.exists():
            errors.append(f"Missing MP4 asset: {mp4_path}")
        if not gif_path.exists():
            errors.append(f"Missing GIF asset: {gif_path}")
        if not render_manifest_path.exists():
            errors.append(f"Missing render manifest: {render_manifest_path}")
            continue

        payload = json.loads(render_manifest_path.read_text(encoding="utf-8"))
        if payload.get("fallback_used", True):
            errors.append(f"Fallback was used for portfolio video: {video_name}")
        if payload.get("source_experiment") != portfolio_profile:
            errors.append(f"Unexpected source experiment for {video_name}: {payload.get('source_experiment')}")
        if payload.get("source_run_root") != expected_run_root:
            errors.append(f"Unexpected source run root for {video_name}: {payload.get('source_run_root')}")

        expected_entry = next((entry for entry in manifest_entries if entry["asset_name"] == video_name), None)
        if expected_entry is None:
            errors.append(f"No expected manifest entry was generated for {video_name}")
            continue
        expected_ids = list(expected_entry["source_episode_ids"])
        actual_ids = list(payload.get("source_episode_ids", []))
        if expected_ids != actual_ids:
            errors.append(f"Source episode mismatch for {video_name}")

    finalized_entries = []
    for entry in manifest_entries:
        finalized = dict(entry)
        if finalized["asset_type"] == "video_bundle":
            render_manifest_path = root / "frames" / finalized["asset_name"] / "render_manifest.json"
            payload = json.loads(render_manifest_path.read_text(encoding="utf-8")) if render_manifest_path.exists() else {}
            finalized["fallback_used"] = bool(payload.get("fallback_used", False))
            finalized["validated_for_portfolio"] = bool(
                finalized["artifact_exists"]
                and finalized["source_available"]
                and not finalized["fallback_used"]
            )
        else:
            finalized["validated_for_portfolio"] = bool(finalized["artifact_exists"] and finalized["source_available"])

        if finalized["required_for_portfolio"] and not finalized["validated_for_portfolio"]:
            errors.append(f"Asset not validated for portfolio: {finalized['asset_name']}")
        finalized_entries.append(finalized)

    manifest_path = ensure_dir(root / "reports") / f"{portfolio_profile}_evidence_manifest.json"
    write_json(manifest_path, {"assets": finalized_entries})
    if errors:
        raise RuntimeError("\n".join(errors))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate evidence-backed portfolio assets.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--portfolio-profile", default=PORTFOLIO_PROFILE_FAST)
    parser.add_argument("--throughput-profile", default=THROUGHPUT_PROFILE_FAST)
    args = parser.parse_args()
    validate_portfolio_assets(
        output_root=args.output_root,
        portfolio_profile=args.portfolio_profile,
        throughput_profile=args.throughput_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
