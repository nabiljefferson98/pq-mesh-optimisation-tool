# scripts/ — Scripts Overview

This directory contains standalone command-line scripts for mesh generation, analysis, benchmarking, and diagnostics. None of these scripts are part of the importable library in `src/`; they are research and evaluation tools intended to be run from the project root.

All scripts are organised into four subdirectories, each with its own `README.md` that documents the command-line interface. This file provides a unified reference across all subdirectories.

---

## Directory map

```
scripts/
├── SCRIPTS_OVERVIEW.md           # This file
├── analysis/
│   ├── README.md
│   ├── analyse_results.py        # Batch results analysis with planarity statistics
│   ├── plot_convergence.py       # Convergence curve plotting (energy vs. iteration)
│   ├── plot_sensitivity.py       # Weight sensitivity visualisation
│   └── sensitivity_sweep.py     # Grid search over weight combinations
├── benchmarking/
│   ├── README.md
│   ├── benchmark_optimisation.py # Timed optimisation runs across mesh sizes
│   └── stress_test.py           # High-iteration and large-mesh stress tests
├── diagnostics/
│   ├── README.md
│   ├── energy_analysis.py        # Per-component energy breakdown diagnostics
│   └── gradient_verification.py  # Finite-difference gradient checker
└── mesh_generation/
    ├── README.md
    └── generate_test_meshes.py   # Procedural quad mesh generator for test surfaces
```

---

## scripts/mesh_generation/

### `generate_test_meshes.py`

Generates all procedural quad meshes used as inputs to the optimisation pipeline. Writes OBJ files to `data/input/generated/`.

```bash
python scripts/mesh_generation/generate_test_meshes.py
```

Generated surface types and their geometric properties:

| Surface | Description | Developability |
| ------- | ----------- | -------------- |
| Flat plane (5x5) | Axis-aligned flat plane, optionally perturbed with Gaussian noise | Fully developable |
| Cylinder (10x8) | Parametric cylinder, exactly developable | Singly curved |
| Saddle (8x8) | Hyperbolic paraboloid `z = x^2 - y^2` | Doubly curved, non-developable |
| Scherk (8x8) | Scherk's first minimal surface | Non-developable |
| Sphere cap (8x8) | Spherical cap | Non-developable |
| Torus patch (6x6) | Patch of torus | Non-developable |
| Cone (8x6) | Parametric cone | Singly curved |

The script also applies a controllable noise perturbation (`noise_scale` parameter) to otherwise planar meshes to create test cases where planarity optimisation is non-trivial.

---

## scripts/analysis/

### `analyse_results.py`

Loads one or more optimised OBJ meshes from `data/output/optimised_meshes/` and prints a structured statistical report.

```bash
# Analyse all optimised meshes
python scripts/analysis/analyse_results.py

# Analyse a specific result
python scripts/analysis/analyse_results.py data/output/optimised_meshes/saddle_optimised.obj
```

The report includes:
- Per-face planarity deviation statistics (min, max, mean, standard deviation)
- Panel counts at each quality threshold: buildable (`< 1e-3`), near-flat (`< 1e-2`), acceptable (`< 0.1`)
- Vertex displacement statistics relative to `vertices_original`
- Energy reduction percentage (requires a paired original mesh to be resolvable)

### `sensitivity_sweep.py`

Performs a grid search over a configurable range of `w_planarity`, `w_fairness`, and `w_closeness` values, running the optimiser for each combination and recording the final planarity statistics. Results are written to `data/output/sensitivity/sensitivity_results.json`.

```bash
python scripts/analysis/sensitivity_sweep.py \
    --mesh data/input/generated/saddle_8x8.obj \
    --planarity-range 1 100 5 \
    --fairness-range 0.1 10 3 \
    --closeness-range 1 20 3
```

This script is the primary tool for the weight sensitivity analysis reported in the dissertation evaluation chapter.

### `plot_convergence.py`

Plots the energy-vs-iteration convergence curve from a previously saved optimisation log. Requires that `verbose=True` was set during optimisation so that per-iteration energy values were printed to stdout and redirected to a log file.

```bash
python scripts/analysis/plot_convergence.py data/output/logs/saddle_run.log
```

Outputs a PNG figure to `data/output/figures/`.

### `plot_sensitivity.py`

Visualises the sensitivity sweep results from `sensitivity_results.json` as a heatmap of final mean planarity deviation over the `(w_planarity, w_closeness)` grid.

```bash
python scripts/analysis/plot_sensitivity.py data/output/sensitivity/sensitivity_results.json
```

Outputs a PNG heatmap to `data/output/figures/`.

---

## scripts/benchmarking/

### `benchmark_optimisation.py`

Runs timed optimisation trials across a range of mesh sizes and records wall-clock time, peak memory usage, and final energy. Designed to produce the scalability data reported in the evaluation chapter.

```bash
python scripts/benchmarking/benchmark_optimisation.py
```

Mesh sizes tested: `5x5` (25 faces), `8x8` (64 faces), `10x10` (100 faces), `16x16` (256 faces), `24x24` (576 faces). Results are printed as a formatted table and written to `data/output/benchmarks/benchmark_results.json`.

### `stress_test.py`

Runs the optimiser with extreme parameters to test robustness: very high iteration counts, large mesh sizes, degenerate weight configurations (`w_planarity = 0`, all weights equal), and meshes with near-flat geometry where gradients are near-zero.

```bash
python scripts/benchmarking/stress_test.py
```

Outputs pass/fail results for each stress scenario to stdout. Used to verify that the `1e300` non-finite energy fallback and `nan_to_num` gradient guard in `optimiser.py` function correctly under pathological inputs.

---

## scripts/diagnostics/

### `energy_analysis.py`

Prints a per-component energy breakdown for a given mesh and weight configuration, showing how much of the total energy is contributed by each term.

```bash
python scripts/diagnostics/energy_analysis.py \
    --mesh data/input/generated/saddle_8x8.obj \
    --planarity 10.0 --fairness 1.0 --closeness 5.0
```

Also calls `suggest_weight_scaling(mesh)` from `src.optimisation.gradients` and prints the recommended weight values. This is useful for diagnosing cases where one energy term dominates the gradient and preventing meaningful progress on the others.

### `gradient_verification.py`

Runs `verify_gradient` from `src.optimisation.gradients` on a specified mesh and weight configuration, comparing the analytic gradient against a finite-difference approximation for a random sample of coordinate components.

```bash
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/generated/plane_5x5_noisy.obj \
    --planarity 10.0 --fairness 1.0 --closeness 5.0 \
    --sample-size 20 --eps 1e-7
```

Outputs the maximum relative error, per-component errors, and a `PASSED` / `FAILED` verdict. A relative error below `1e-4` is required for the gradient to be considered correct. This script is used to validate any changes to `gradients.py` or `backends.py`.

---

## Running all scripts from the project root

All scripts use `sys.path` insertion to import from `src/` without requiring `pip install -e .`. They must be run from the project root:

```bash
# From the project root
python scripts/mesh_generation/generate_test_meshes.py
python scripts/analysis/sensitivity_sweep.py --mesh data/input/generated/saddle_8x8.obj
python scripts/diagnostics/gradient_verification.py --mesh data/input/generated/plane_5x5_noisy.obj
```

Running a script from within its own subdirectory will cause import errors because the `src/` package will not be on the path.
