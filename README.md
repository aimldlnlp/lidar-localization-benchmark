# MASSIVE-PARALLEL LIDAR LOCALIZATION BENCHMARK

2D LiDAR localization benchmark with EKF and Monte Carlo Localization, batched evaluation, publication-style figures, and MP4/GIF demos.

![Localization Overview](outputs/figures/fig_localization_overview.png)

Overview of the benchmark: occupancy map, reference trajectory, localization estimates, scan geometry, and error traces.

## Results Snapshot

<!-- PORTFOLIO_RESULTS_CARD:START -->
### Results Snapshot

| Method | Episodes | Mean position RMSE [m] | Mean median translation error [m] | Mean heading RMSE [deg] | Failure rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| EKF | 8 | 0.7852 | 0.1704 | 7.669 | 0.375 |
| MCL | 8 | 0.0533 | 0.0179 | 0.302 | 0.000 |

Optional CUDA throughput snapshot:

Not generated in this run.
<!-- PORTFOLIO_RESULTS_CARD:END -->

## Visual Walkthrough

### Main Localization Demo

<p align="center">
  <img src="outputs/gifs/demo_main_localization.gif" alt="Main localization demo" width="90%">
</p>

This demo shows EKF and MCL tracking the same episode with the occupancy map, live LiDAR scan, and translation error evolving in sync.

### Noise Robustness Demo

<p align="center">
  <img src="outputs/gifs/demo_noise_robustness.gif" alt="Noise robustness demo" width="90%">
</p>

This comparison shows how the same trajectory looks under lower-noise and higher-noise LiDAR scans, and how the localization error changes with measurement quality.

### Particle Convergence Demo

<p align="center">
  <img src="outputs/gifs/demo_particle_convergence.gif" alt="Particle convergence demo" width="90%">
</p>

This sequence visualizes particle collapse and pose convergence in Monte Carlo Localization.

### Static Figure Gallery

<p align="center">
  <img src="outputs/figures/fig_robustness_noise_sweep.png" alt="Robustness noise sweep" width="48%">
  <img src="outputs/figures/fig_error_vs_time.png" alt="Error over time" width="48%">
</p>

<p align="center">
  <img src="outputs/figures/fig_particle_evolution.png" alt="Particle evolution" width="48%">
  <img src="outputs/figures/fig_localization_overview.png" alt="Localization overview" width="48%">
</p>

## What This Project Shows

- Occupancy-grid map generation with multiple layout families
- Trajectory generation with collision-aware waypoint sampling
- 2D LiDAR scan simulation with noise and dropout controls
- EKF localization as a fast baseline
- Particle-filter localization as a robust baseline
- Figure, video, GIF, metrics, and report export from the same run outputs

## Methods

### EKF

Pose state `(x, y, theta)` is propagated with a motion model and corrected using a sparse range observation against the map. It is fast and simple, but it becomes fragile when the scan is noisy or the scene is ambiguous.

### Monte Carlo Localization

The particle filter maintains a set of pose hypotheses, scores them with scan likelihoods against the map, and resamples when confidence collapses. It is more computationally involved, but substantially more robust in this implementation.

## Key Results

- MCL achieves much lower mean position RMSE than EKF in the current run: `0.0533 m` vs `0.7852 m`.
- MCL records zero failures across 8 episodes, while EKF reaches a failure rate of `0.375`.
- EKF degrades sharply at the higher noise level, while MCL remains stable across both configured noise settings.

## Quickstart

```bash
python -m massive_lidar_benchmark.cli run-benchmark --config configs/benchmarks/portfolio_fast.yaml
python scripts/build_portfolio_report.py
python -m massive_lidar_benchmark.cli render-figures --config configs/render/portfolio_fast_figures.yaml
python -m massive_lidar_benchmark.cli render-video --config configs/render/portfolio_fast_demo_main_localization.yaml
python -m massive_lidar_benchmark.cli render-video --config configs/render/portfolio_fast_demo_particle_convergence.yaml
python -m massive_lidar_benchmark.cli render-video --config configs/render/portfolio_fast_demo_noise_robustness.yaml
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

For CUDA-enabled runs and media export:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Repository Layout

```text
configs/   YAML configs for smoke runs, fast main runs, broader benchmark runs, and rendering
src/       Python package with maps, trajectories, sensors, localization, benchmarks, and viz
tests/     Lightweight correctness and regression tests
scripts/   Helper scripts for report generation and result validation
docs/      Notes and checklists
outputs/   Generated maps, trajectories, runs, metrics, figures, videos, gifs, and reports
```

## Outputs

- `outputs/metrics/portfolio_fast_summary.csv`
- `outputs/tables/portfolio_fast_results_card.md`
- `outputs/reports/portfolio_fast_evidence_manifest.json`
- `outputs/figures/`
- `outputs/videos/`
- `outputs/gifs/`

## Optional CUDA Throughput

This extra run is optional and is not required for the main localization workflow.

```bash
python -m massive_lidar_benchmark.cli run-benchmark --config configs/benchmarks/throughput_gpu_fast.yaml
python scripts/build_portfolio_report.py
```