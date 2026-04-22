# scripts/analysis — Results Analysis and Plotting

These scripts process the benchmark and diagnostic JSON outputs produced by
`scripts/benchmarking/` and `scripts/diagnostics/` and generate the
dissertation figures, summary tables, and weight-sensitivity reports.
Run them only after all upstream data-generation steps have completed.

---

## Script Reference

| Script | Experiment | What it does |
|---|---|---|
| `run_weight_sensitivity_sweep.py` | EXP-04 | Runs the full w\_p × w\_f × w\_c parameter sweep (~80 optimisation runs), identifies Pareto-optimal weight configurations across the planarity–closeness trade-off, and writes a JSON report to `data/output/weight_sensitivity/`. |
| `summarise_and_export_results.py` | EXP-01 to EXP-05 | Reads all benchmark and sensitivity JSON files, prints per-experiment statistical summaries, fits the empirical complexity exponent O(n^α) via log-log regression, and writes a LaTeX table and CSV to `data/output/tables/`. |
| `plot_convergence_and_scaling.py` | EXP-01, EXP-03 | Generates (1) energy-vs-iteration convergence curves for all tested mesh sizes and (2) the log-log complexity scaling plot with the fitted O(n^1.30) reference line. Saves PNGs to `data/output/benchmarks/`. |
| `plot_scalability_loglog_overlay.py` | EXP-01 | Produces the combined log-log overlay of Series A (synthetic noisy grids) and Series B (oloid resolution series) to confirm the complexity exponent holds across both synthetic and real-world geometry. Saves PNG to `data/output/benchmarks/`. |
| `plot_realworld_planarity_histograms.py` | EXP-05 | Produces per-face planarity deviation histograms (before vs. after optimisation) for Spot, Blub, Oloid-256, and Bob. Saves one PNG per mesh to `data/output/benchmarks/`. |
| `plot_weight_sensitivity_pareto.py` | EXP-04 | Generates the Pareto frontier (planarity energy vs. closeness energy) and a w\_p × w\_f weight heatmap from the sensitivity sweep JSON. Saves PNGs to `data/output/weight_sensitivity/plots/`. |
| `plot_style_config.py` | (shared) | Shared Matplotlib style configuration (fonts, colours, figure sizes, DPI) imported by all plot scripts. Not intended to be run directly. |

---

## Full Evaluation Pipeline — Execution Order

The eight steps below constitute the complete, reproducible evaluation
pipeline from raw mesh inputs to final dissertation figures.

```bash
# ── Step 1 ── Generate synthetic noisy grid meshes
python scripts/mesh_generation/generate_test_meshes.py

# ── Step 2 ── EXP-01 Series A: scalability sweep on synthetic grids
python scripts/benchmarking/benchmark_optimisation.py

# ── Step 3 ── EXP-01 Series B: scalability sweep on oloid resolution series
python scripts/benchmarking/benchmark_optimisation.py \
    --mesh data/input/reference_datasets/oloid/oloid64_quad.obj \
    --mesh data/input/reference_datasets/oloid/oloid256_quad.obj \
    --mesh data/input/reference_datasets/oloid/oloid1024_quad.obj \
    --mesh data/input/reference_datasets/oloid/oloid4096_quad.obj

# ── Step 4 ── EXP-02: backend comparison (repeat for each backend)
PQ_BACKEND=numpy  python scripts/benchmarking/benchmark_optimisation.py
PQ_BACKEND=numba  python scripts/benchmarking/benchmark_optimisation.py
PQ_BACKEND=cupy   python scripts/benchmarking/benchmark_optimisation.py

# ── Step 5 ── EXP-03: gradient verification on synthetic and reference meshes
python scripts/diagnostics/gradient_verification.py
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/reference_datasets/oloid/oloid256_quad.obj
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/reference_datasets/spot/spot_quadrangulated.obj
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/reference_datasets/blub/blub_quadrangulated.obj
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/reference_datasets/bob/bob_quad.obj

# ── Step 6 ── EXP-04: weight sensitivity sweep
python scripts/analysis/run_weight_sensitivity_sweep.py \
    --mesh data/input/generated/plane_10x10_noisy.obj

# ── Step 7 ── EXP-05: real-world dataset evaluation
python scripts/analysis/summarise_and_export_results.py \
    --mesh data/input/reference_datasets/spot/spot_quadrangulated.obj \
    --mesh data/input/reference_datasets/blub/blub_quadrangulated.obj \
    --mesh data/input/reference_datasets/oloid/oloid256_quad.obj \
    --mesh data/input/reference_datasets/bob/bob_quad.obj

# ── Step 8 ── Generate all dissertation figures
python scripts/analysis/plot_convergence_and_scaling.py
python scripts/analysis/plot_scalability_loglog_overlay.py
python scripts/analysis/plot_realworld_planarity_histograms.py
python scripts/analysis/plot_weight_sensitivity_pareto.py
```

All outputs are written to subdirectories under `data/output/`.
See `docs/results/RESULTS_INDEX.md` for the corresponding experiment
documents and expected result summaries.
