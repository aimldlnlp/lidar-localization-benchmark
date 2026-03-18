# Results Checklist

- [ ] `python -m pytest -q` passes
- [ ] `python -m massive_lidar_benchmark.cli smoke --config configs/debug/smoke.yaml` passes
- [ ] `python -m massive_lidar_benchmark.cli run-benchmark --config configs/benchmarks/portfolio_fast.yaml` passes
- [ ] `python scripts/build_portfolio_report.py` writes the fast summary CSVs and results snapshot
- [ ] `python -m massive_lidar_benchmark.cli render-figures --config configs/render/portfolio_fast_figures.yaml` writes the four core PNGs
- [ ] portfolio videos are written to `outputs/videos/`
- [ ] matching GIFs are written to `outputs/gifs/`
- [ ] `python scripts/validate_portfolio_assets.py` passes without fallback errors
- [ ] `outputs/reports/portfolio_fast_evidence_manifest.json` exists and only marks core assets as validated
- [ ] optional GPU bonus can be generated with `python -m massive_lidar_benchmark.cli run-benchmark --config configs/benchmarks/throughput_gpu_fast.yaml`
