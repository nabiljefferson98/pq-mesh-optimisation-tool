# docs/results/ — Experimental Results Index

This directory contains the five core experiments that constitute the evaluation chapter of the dissertation. Each experiment is documented in a self-contained Markdown file with hardware configuration, setup, results tables, plots, interpretation, and references.

---

## Quick-reference summary

| ID | Experiment | Mesh sizes / datasets | Key finding |
|----|------------|----------------------|-------------|
| EXP-01 | Scalability Benchmark | Series A: 3×3 to 75×75 (9–5,625 faces); Series B: oloid64 to oloid4096 (64–4,096 faces) | Empirical complexity O(n^1.27); interactive threshold (<2 s) met for meshes up to 400 faces; oloid series validates complexity on real-world developable surface |
| EXP-02 | Backend Comparison | 10×10 to 75×75 | Numba: 2.4–2.8× speedup; CuPy: 2.2–3.1× speedup; max numerical difference < 4.1×10⁻⁷ |
| EXP-03 | Convergence Analysis | 5×5, 10×10, 20×20 | Convergence in 9–13 L-BFGS-B iterations regardless of mesh size; mean planarity deviation below 3×10⁻⁴ by iteration 5 |
| EXP-04 | Weight Sensitivity | 10×10 (fixed seed 42) | Default (w_p=10, w_f=1, w_c=5) achieves 99.99% planarity improvement; factor-of-two weight change produces <0.1% change in result |
| EXP-05 | Real-World Dataset Evaluation | Spot (2,928 faces), Blub (7,104 faces), Oloid-256 (256 faces), Bob (5,344 faces) | All Crane CC0 native quad meshes; oloid provides theoretically motivated developable surface test case; Bob serves as large-scale stress test |

---

## Experiment file map

```
docs/results/
├── RESULTS_INDEX.md              ← This file
├── EXP-01_scalability.md
├── EXP-02_backend_comparison.md
├── EXP-03_convergence.md
├── EXP-04_weight_sensitivity.md
└── EXP-05_realworld.md
```

---

## Reference Dataset Map

```
data/input/reference_datasets/
├── spot/
│   ├── spot_quadrangulated.obj      <- EXP-05 primary benchmark (2,930 vertices, 2,928 faces)
│   └── spot_control_mesh.obj        <- visual reference only
├── blub/
│   ├── blub_quadrangulated.obj      <- EXP-05 secondary benchmark (7,106 vertices, 7,104 faces)
│   └── blub_control_mesh.obj        <- visual reference only
├── oloid/
│   ├── oloid64_quad.obj             <- EXP-01 Series B (smallest)
│   ├── oloid256_quad.obj            <- EXP-01 Series B + EXP-05 developable test (258 vertices, 256 faces)
│   ├── oloid1024_quad.obj           <- EXP-01 Series B
│   └── oloid4096_quad.obj           <- EXP-01 Series B (stress test)
└── bob/
    └── bob_quad.obj                 <- EXP-05 large-scale stress test (5,344 vertices, 5,344 faces)
```

All meshes: Keenan Crane, *3D Model Repository*, CMU. Licence: CC0.
https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/

---

## Hardware reference

All experiments on Machine A (primary):

| Field | Value |
|-------|-------|
| CPU | AMD Ryzen 7 5800X (8-core, 3.8 GHz) |
| GPU | NVIDIA RTX 3070 (8 GB VRAM, CUDA 12) |
| RAM | 16 GB DDR4-3600 |
| OS | Windows 11 (22H2) / Ubuntu 22.04 WSL |
| Python | 3.10.14 |
| NumPy | 1.26.x |
| SciPy | 1.12.x |
| Numba | 0.59.x |
| CuPy | 13.x |

EXP-01 additionally benchmarked on Machine B (Apple M3 8-core, macOS Sequoia 15.x,
Python 3.11, 8 GB unified memory, Numba/NumPy only, meshes ≤ 20×20).
EXP-05 run exclusively on Machine B (NumPy backend, same spec as above).

---

## Reproducibility

All experiments can be reproduced from the project root by running the
following eight steps in order. See `scripts/analysis/README.md` for
further detail on each script.

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

# ── Step 5 ── EXP-03: convergence analysis
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

All random seeds are fixed (`numpy.random.seed(42)`) in EXP-04. EXP-01
through EXP-03 use deterministic synthetic meshes with fixed noise
generation in `generate_test_meshes.py`.

---

## Cross-file verification checklist

| Check | Status |
|-------|--------|
| All five EXP-*.md files present | ✓ |
| Each file contains: hardware config, setup, results table, plot references, interpretation, discussion, references | ✓ |
| Complexity exponent O(n^1.27) consistent across EXP-01, EXP-03 discussion, and `methodology.md` §4.3 | ✓ |
| Default weights (w_p=10, w_f=1, w_c=5) consistent across EXP-03, EXP-04, `methodology.md`, and `architecture.md` | ✓ |
| Backend numerical equivalence (max diff < 10⁻⁶) consistent between EXP-02 results and `architecture.md` §backends | ✓ |
| 229 tests, 1 skipped — consistent between `architecture.md` test coverage section and `TESTING_GUIDE.md` (C3) | ✓ |
| EXP-05 mesh face counts: Spot 2,928; Blub 7,104; Oloid-256 256; Bob 5,344 — consistent with EXP-05_realworld.md | ✓ |
| EXP-05 mesh sizes within scalability bounds established by EXP-01 (Blub 7,104 faces < 75×75 = 5,625 faces — see note) | ⚠ Note: Blub (7,104 faces) exceeds EXP-01 Series A upper bound of 5,625 faces. This is documented in EXP-05 as expected given the larger mesh; runtimes are consistent with O(n^1.27) extrapolation. |
| EXP-03 uses gtol=1e-6 (tighter than `OptimisationConfig` default gtol=1e-5) — intentional for academic rigour | ✓ |
| Numba `except Exception` broadening documented consistently in `architecture.md`, `methodology.md` §2.1, and robustness audit table | ✓ |
| `scatter_matrix` shape described as `(n, m·4)` in `architecture.md` and `(n_vertices, n_faces × 4)` in `methodology.md` §3.1 — equivalent formulations | ✓ |
| EXP-05 reference datasets: all four meshes confirmed native quad, CC0, sourced from Crane model repository | ✓ |
| Oloid resolution series (EXP-01 Series B) uses same four files as oloid/ subfolder in reference_datasets/ | ✓ |
| All reproducibility commands reference current script names (no stale old names) | ✓ |
| Machine B spec: Apple M3, 8 GB unified memory, macOS Sequoia 15.x, Python 3.11 — consistent across EXP-01 (updated commit e7be15b), EXP-05, and RESULTS_INDEX | ✓ |
| EXP-02 figure: Figure 4.4 (updated commit 2deffab); EXP-03 figures: 4.5/4.6; EXP-04 figure: 4.7 — all consistent with dissertation sequential numbering from Figure 4.1 (pipeline overview) | ✓ |
