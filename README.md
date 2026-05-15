# PQ Mesh Optimisation Tool

**Real-Time Planar Quad (PQ) Mesh Optimisation and Visualisation Tool for Designing Developable Surfaces**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/nabiljefferson98/pq-mesh-optimisation-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/nabiljefferson98/pq-mesh-optimisation-tool/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nabiljefferson98/pq-mesh-optimisation-tool/branch/main/graph/badge.svg)](https://codecov.io/gh/nabiljefferson98/pq-mesh-optimisation-tool)
[![Wiki](https://img.shields.io/badge/docs-Wiki-blue)](https://github.com/nabiljefferson98/pq-mesh-optimisation-tool/wiki)

A computational geometry tool for optimising quad meshes to achieve planar faces, enabling fabrication of architectural surfaces from flat panels. Implements SVD-based planarity formulation with L-BFGS-B optimisation.

> 📖 **Full documentation is available in the [GitHub Wiki](https://github.com/nabiljefferson98/pq-mesh-optimisation-tool/wiki).**
> The Wiki covers mathematical foundations, API reference, architecture, experiments, testing, reproducibility, and a complete FAQ.

---

## 🧭 What Is This? (Beginner-Friendly Overview)

If you have never worked with computational geometry before, this section explains the core idea in plain language.

### The Problem: Wobbly Panels

Imagine you are an architect designing a curved glass roof — like a wave-shaped canopy over a railway station. You model the shape as a 3D mesh made up of many small quadrilateral (four-sided) panels. In theory the design looks beautiful. In practice, when you send those panels to a fabricator, they discover a problem: **the four corners of each panel do not lie on the same flat plane**. A slightly twisted panel cannot be cut from a flat sheet of glass or metal. The result is expensive custom bending of every single panel, or the whole design has to be abandoned.

This tool solves that problem. It takes your quad mesh, and **nudges the vertices just enough** so that every face becomes flat (planar), while keeping the overall shape as close to your original design as possible.

### The Solution: PQ Meshes

A **Planar Quad (PQ) mesh** is simply a quad mesh where every face is exactly flat. The concept was formalised by Liu et al. (2006), who showed that PQ meshes allow freeform glass structures to be built from flat panels, which dramatically reducing fabrication cost. This tool automates the conversion of any quad mesh into a PQ mesh.

### How It Works (High-Level)

The optimisation works in four conceptual steps:

```
1. Load mesh        →   Read your .obj quad mesh into memory
        ↓
2. Measure flatness →   For each face, compute how "twisted" it is
                        (using a mathematical technique called SVD)
        ↓
3. Fix the mesh     →   Move vertices to reduce twisting, while
                        keeping the shape smooth and close to original
                        (L-BFGS-B optimiser, typically 9–13 steps)
        ↓
4. Visualise        →   Show the before/after in a 3D viewer with a
                        colour heatmap (green = flat, red = twisted)
```

### Key Concepts Explained

| Term | Plain English |
|:--|:--|
| **Quad mesh** | A 3D surface made of four-sided patches (like a grid draped over a shape) |
| **Planar face** | A face where all four corners lie exactly on the same flat plane |
| **SVD** | Singular Value Decomposition — a mathematical tool to measure how much a set of points deviates from a plane |
| **L-BFGS-B** | A well-known algorithm for finding the best solution to a smooth mathematical problem efficiently |
| **Energy** | A score that measures how "bad" the mesh is — lower is better; the optimiser drives this towards zero |
| **Conical mesh** | A special PQ mesh where the faces around each vertex all touch a common cone; useful for offsetting panels at a constant distance |
| **Developable surface** | A surface that can be unrolled flat without stretching — think of unrolling a paper towel tube |
| **Heatmap** | A colour overlay on the mesh showing planarity deviation: green = very flat, red = very twisted |

### Why Does It Matter?

In architecture, **conical and PQ meshes** are the gold standard for glass-and-steel freeform structures (think the British Museum Great Court roof, or Zaha Hadid's designs). They allow:
- Flat panels that can be cut from standard sheet material
- Offset layers (insulation, framing) that maintain the same geometry
- A structural support system that is orthogonal to the surface

This tool makes that technology accessible as an open-source Python library.

---

## 📸 Screenshot

![Banner](docs/images/banner_comparison.png)
*Before (left) and after (right) optimisation with planarity heatmap*

---

## In Action

![Interactive optimisation demo](docs/images/demo.gif)

---

## 🎯 Features

- **✨ Planar Quad Optimisation**: Reduces face planarity deviation by >96× (average 96.4% improvement)
- **🚀 Hardware Acceleration**: Automatic backend detection for NVIDIA GPU (`cupy`), Parallel CPU (`numba`), and Baseline CPU (`numpy`)
- **⚡ Interactive Performance**: Optimises typical meshes (100–400 faces) in <2 seconds
- **🎨 Real-Time Visualisation**: Side-by-side comparison with planarity heatmaps using Polyscope
- **🎛️ Interactive UI**: Adjustable weight parameters with live optimisation
- **📊 Comprehensive Analysis**: Built-in benchmarking, convergence tracking, and sensitivity analysis
- **🧪 Robust Testing**: 321 tests, 0 failures, 1 skipped (GUI-only); ≥81% coverage (excl. GUI); 0 flake8/bandit/mypy violations
- **🔒 Security & Robustness**: Path-traversal protection, NaN/Inf guards in energy/gradient, atomic DXF/SVG writes, input sanitisation, active return-type contract assertions in optimiser
- **📈 Scalable**: O(n^1.27) complexity; practical ceiling ~5,625 faces (75×75 grid, ~80 s)
- **🏗️ CI/CD Pipeline**: GitHub Actions matrix (Ubuntu + macOS, Python 3.10–3.12) with type checking, security scanning, and Codecov
- **🔧 Two-Stage Optimisation**: Rapid planarity pass followed by balanced refinement for improved convergence on complex meshes

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/nabiljefferson98/pq-mesh-optimisation-tool.git
cd pq-mesh-optimisation-tool

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (includes optional hardware-acceleration packages, see requirements.txt)
pip install -r requirements.txt

# Note: GPU/Numba packages in requirements.txt may not be available on all platforms.
# For CPU-only builds, install requirements_without_CUDA.txt instead.
pip install -r requirements_without_CUDA.txt

# Verify installation
python -m pytest tests/ -v  # Expect: 321 passed or skipped, depending on your setup
```

> **New to virtual environments?** A virtual environment is simply an isolated Python installation for this project so it does not interfere with other Python software on your machine. The commands above create one in a folder called `.venv`, then activate it. You only need to activate it (the `source` line) each time you open a new terminal.

### Interactive Optimisation

```bash
# Launch interactive viewer with parameter sliders
# (runs with cylinder_10x8.obj by default if no argument given)
python src/visualisation/interactive_optimisation.py

# With a specific mesh
python src/visualisation/interactive_optimisation.py data/input/generated/plane_5x5_noisy.obj

# With conical constraint and custom iteration budget
python src/visualisation/interactive_optimisation.py data/input/generated/sphere_cap_10x8.obj

# Force a specific backend (runs with cylinder_10x8.obj by default if no argument given):

# ── NumPy baseline (your Mac, or force on PC) ────────────────────────────────
PQ_BACKEND=numpy python src/visualisation/interactive_optimisation.py

# ── Numba parallel CPU (PC — triggers warmup on first run) ───────────────────
PQ_BACKEND=numba python src/visualisation/interactive_optimisation.py

# ── CuPy GPU (PC RTX 3070 — skip warmup, no Numba used) ─────────────────────
PQ_BACKEND=cupy python src/visualisation/interactive_optimisation.py

# ── With a specific mesh file ────────────────────────────────────────────────
PQ_BACKEND=numba python src/visualisation/interactive_optimisation.py data/input/generated/plane_5x5_noisy.obj
```

> **What are backends?** The tool can run the heavy maths on three different pieces of hardware. `numpy` is the safe default (works everywhere). `numba` uses all your CPU cores in parallel (faster on most laptops/desktops). `cupy` uses an NVIDIA GPU (fastest, but requires a CUDA-capable GPU). The tool picks the best available option automatically if you do not set `PQ_BACKEND`.

---

## 📂 Project Structure

```
pq-mesh-optimisation/
├── README.md                         # Project overview and quick-start
├── check.py                          # Ad-hoc diagnostic script
├── src/                              # Core library — import only, do not run directly
│   ├── OVERVIEW.md                   # Internal library documentation
│   ├── backends.py                   # Hardware backend detection (GPU/Parallel CPU/NumPy)
│   ├── core/
│   │   └── mesh.py                   # QuadMesh data structure; lazy scatter matrix
│   ├── optimisation/
│   │   ├── energy_terms.py           # Planarity (SVD), fairness, closeness, angle-balance energies
│   │   ├── mesh_geometry.py          # Geometric utilities (face planarity, conical imbalance)
│   │   ├── gradients.py              # Analytical gradient ∂E/∂V for all energy terms
│   │   ├── optimiser.py              # L-BFGS-B wrapper; OptimisationConfig/Result dataclasses
│   │   └── OPTIMISATION_PIPELINE.md  # Full optimiser pipeline documentation
│   ├── io/
│   │   ├── obj_handler.py            # Wavefront OBJ reader/writer (preserves quads)
│   │   └── panel_exporter.py         # Export flat panels to DXF and SVG
│   ├── preprocessing/
│   │   └── preprocessor.py           # Normalisation, degenerate-face detection, weight suggestions
│   └── visualisation/
│       ├── interactive_optimisation.py  # Polyscope 3D viewer; planarity & conical heatmaps
│       └── VISUALISATION_GUIDE.md    # Visualisation usage guide
├── scripts/                          # Research and analysis pipeline
│   ├── SCRIPTS_OVERVIEW.md           # Overview of all scripts and their purpose
│   ├── mesh_generation/
│   │   ├── README.md
│   │   └── generate_test_meshes.py   # Curved surface test mesh generation
│   ├── benchmarking/
│   │   ├── README.md
│   │   ├── benchmark_optimisation.py # Performance timing across mesh sizes
│   │   └── stress_test.py            # Upper mesh-size limit with RAM profiling
│   ├── diagnostics/
│   │   ├── README.md
│   │   ├── energy_analysis.py        # Energy component breakdown and weight recommendations
│   │   └── gradient_verification.py  # Analytical vs numerical gradient check
│   ├── analysis/
│   │   ├── README.md
│   │   ├── run_weight_sensitivity_sweep.py    # 80-config weight sweep, Pareto analysis
│   │   ├── summarise_and_export_results.py    # Statistics, complexity, LaTeX/CSV tables
│   │   ├── plot_convergence_and_scaling.py    # Convergence and scaling plots
│   │   ├── plot_scalability_loglog_overlay.py # Log-log scalability overlay plot
│   │   ├── plot_realworld_planarity_histograms.py  # Planarity histograms for real-world meshes
│   │   ├── plot_weight_sensitivity_pareto.py  # Pareto frontier and weight heatmaps
│   │   └── plot_style_config.py               # Shared Matplotlib style configuration
│   └── plotting/
│       └── plot_benchmarks.py        # Benchmark timing and performance plots
├── tests/                            # 321 tests, 0 failures (≥81% coverage excl. GUI)
│   ├── TESTING_GUIDE.md              # Test module catalogue and running instructions
│   ├── test_mesh.py
│   ├── test_quad_topology_preservation.py
│   ├── test_geometry.py
│   ├── test_energy_terms.py
│   ├── test_gradients.py
│   ├── test_gradients_extended.py
│   ├── test_optimiser.py
│   ├── test_obj_handler.py
│   ├── test_obj_handler_extended.py
│   ├── test_panel_exporter.py
│   ├── test_preprocessor.py
│   ├── test_scalability.py
│   ├── test_error_handling.py
│   ├── test_coverage_extended.py
│   ├── test_coverage_gaps.py         # Additional branch coverage for backends/preprocessor/energy
│   ├── test_robustness.py
│   ├── test_backends.py              # Backend detection and fallback logic
│   ├── test_numerical_equivalence.py # Numerical consistency across backends
│   ├── test_vertex_face_ids.py       # Vertex-to-face topology cache correctness
│   └── test_quad_loading.py          # Quad-vs-triangle loading comparison
├── data/
│   ├── input/
│   │   ├── INPUT_MESHES.md           # Input mesh documentation and provenance
│   │   ├── generated/                # Synthetic noisy plane grids + curved surfaces (OBJ)
│   │   └── reference_datasets/       # Real-world benchmark meshes
│   │       ├── blub/                 # Blub model
│   │       ├── bob/                  # Bob model
│   │       ├── oloid/                # Oloid model
│   │       └── spot/                 # Spot (Keenan Crane) model
│   └── output/
│       ├── benchmarks/               # JSON performance data + analysis plots
│       ├── diagnostics/              # Energy and gradient diagnostic outputs
│       ├── figures/                  # Generated dissertation figures
│       ├── tables/                   # LaTeX and CSV dissertation tables
│       └── weight_sensitivity/       # Weight sweep JSON, plots, report
├── docs/
│   ├── README.md                     # Documentation overview
│   ├── architecture.md               # System architecture: modules, data flow, design decisions
│   ├── methodology.md                # Mathematical methodology: energy formulation, gradients, algorithm
│   ├── results/                      # Results and analysis documents
│   └── images/                       # Figures and comparison screenshots
├── .github/
│   └── workflows/ci.yml              # GitHub Actions: lint, test matrix, type-check, security
├── .pre-commit-config.yaml           # Pre-commit hooks (black, isort, flake8, bandit)
├── .pylintrc                         # Pylint configuration
├── Makefile                          # Developer quality pipeline
├── pyproject.toml                    # Project config (isort, flake8, coverage)
├── requirements.txt                  # Python dependencies (includes CUDA/Numba)
├── requirements_without_CUDA.txt     # Python dependencies without CUDA packages
├── conftest.py                       # Pytest configuration / shared fixtures
└── LICENSE
```

---

## 🔧 Technical Details

### How the Optimiser Works (Plain English)

The tool frames mesh optimisation as a mathematical minimisation problem. Think of it like adjusting the positions of hundreds of tiny ball bearings connected by springs, where you want to make every face flat but without pulling the shape too far from what you originally designed. The optimiser solves this by computing a *score* (called **energy**) and repeatedly moving vertices in the direction that reduces the score fastest.

There are **four components** to the score, each measuring something different:

| Component | What it measures | Symbol |
|:--|:--|:--|
| **Planarity energy** | How twisted (non-flat) each face is | `E_p` |
| **Fairness energy** | How bumpy or irregular the surface is | `E_f` |
| **Closeness energy** | How far vertices have moved from the original design | `E_c` |
| **Angle-balance energy** | How well the faces around each vertex form a manufacturable cone joint | `E_a` |

You control the trade-off between these goals using **weight sliders** in the interactive viewer. For example, setting planarity weight very high forces near-perfect flatness, even if the surface smoothness suffers slightly.

### Algorithm Overview

The tool implements a **constrained nonlinear optimisation** approach:

1. **Energy Formulation** (Chapter 3 of dissertation)
    - **Planarity Energy**: SVD-based measure of face non-planarity (`w_p * sum_f sigma_min(M_f)^2`)
    - **Fairness Energy**: Discrete Laplacian for mesh smoothness (`w_f * ||LV||^2_F`)
    - **Closeness Energy**: Deviation from original design (`w_c * ||V - V_0||^2_F`)
    - **Angle-Balance Energy**: Conical constraint for manufacturable panel joints (`w_a * sum_v (sum_f alpha_{v,f})^2`)
2. **Optimisation Method** (Chapter 3 of dissertation)
    - **Algorithm**: L-BFGS-B (Limited-memory BFGS with bounds)
    - **Gradients**: Analytical (closed-form derivation for all four energy terms)
    - **Two-Stage**: Stage 1 rapid planarity pass, Stage 2 balanced refinement
    - **Convergence**: ftol=10⁻⁹, gtol=10⁻⁵, maxcor=20, maxls=40
3. **Complexity**
    - **Time**: O(n^1.27) where n = number of vertices (NumPy baseline)
    - **Speedup**: Up to 10× with CUDA GPU acceleration (`cupy`) on 10k+ vertex meshes
    - **Memory**: O(n) via limited-memory Hessian approximation
    - **Iterations**: Typically 9–13 iterations to convergence

### Using the Tool as a Python Library

You do not need the interactive viewer to use this tool. You can import it directly into your own Python scripts:

```python
from src.io.obj_handler import load_obj, save_obj
from src.preprocessing.preprocessor import preprocess_mesh
from src.optimisation.optimiser import optimise_mesh_simple

# Load any quad mesh in OBJ format
mesh = load_obj("data/input/generated/plane_5x5_noisy.obj")

# Clean up: remove duplicate vertices and normalise scale
mesh = preprocess_mesh(mesh, normalise=True, verbose=True)

# Run the optimisation — adjust weights to taste
result = optimise_mesh_simple(
    mesh,
    weights={"planarity": 10.0, "fairness": 1.0, "closeness": 5.0},
    max_iterations=200,
    verbose=True,
)

print(result.summary())           # Human-readable results
save_obj(mesh, "result.obj")      # Save the optimised mesh
```

The `weights` dictionary controls the balance between the four energy goals described above. A higher `planarity` weight means flatter faces at the potential cost of moving further from the original shape. See [`src/OVERVIEW.md`](src/OVERVIEW.md) for the full API reference.

### CLI Output

When `verbose=True` (the default), the optimiser prints three sections:

```
======================================================================
MESH OPTIMISATION — STARTING
======================================================================
Mesh loaded: 121 corner points, 100 panels
Priority settings — Flatness: 100.0, Smoothness: 1.0, Shape fidelity: 10.0

Starting scores (lower is better — the optimiser will reduce these):
  Overall combined score:     9823.1234
  Panel flatness score:       98.1234  (how uneven the panels are)
  Surface smoothness score:   0.1234   (how bumpy the surface is)
  Shape fidelity score:       0.0000   (how far vertices have moved from the original design)
======================================================================
Progress will be printed every 10 improvement steps.
Each line shows: step number, combined score, rate of change (lower = nearly done), and time elapsed.

Step   10: score = 141.03,  rate of change = 0.6722,  time elapsed = 1.75s
           (technical: iteration 10,  energy E = 1.410300e+02,  || gradient E || = 6.7220e-04)
...

======================================================================
OPTIMISATION COMPLETE — RESULTS SUMMARY
======================================================================
Result: FINISHED SUCCESSFULLY — the optimiser found the best solution it could
(Technical message from solver: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL)

Overall score (lower = flatter, smoother, closer to original):
  Score at the start:            9823.1234
  Score at the end:              1.0234
  Total improvement:             99.99%

How hard did the optimiser work?
  Improvement steps taken:       12
  Times it checked the score:    36
  Times it checked the direction:12
  Total time:                    1.09 seconds

Per-goal score breakdown (how each individual goal changed):
  Panel flatness    : 98.1234 to 0.0001  (99.9% better)
  Surface smoothness: 0.1234  to 0.1187  (3.8% better)
  Shape fidelity    : 0.0000  -> still 0  (nothing to improve here)
======================================================================
```

### Dependencies

- **NumPy** (≥1.24): Numerical computation and linear algebra
- **SciPy** (≥1.11): L-BFGS-B optimiser
- **Polyscope** (≥2.2): 3D mesh visualisation
- **Matplotlib** (≥3.7): Plotting and convergence analysis
- **pytest** (≥7.4): Testing framework

Full dependency list: [`requirements.txt`](requirements.txt)

---

## 📊 Performance Benchmarks

The table below shows how long the optimiser takes on synthetic noisy-plane grids of increasing size, measured on a single CPU core (NumPy backend). Energy reduction figures above ~9,000% on smaller meshes reflect the starting mesh being far from planar — the optimiser achieves near-zero residual planarity energy.

| Mesh Size | Vertices | Faces | Time (s) | Energy Reduction | Status |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 3×3 | 16 | 9 | 0.08 | 9629.7% | ✓ |
| 5×5 | 36 | 25 | 0.36 | 9668.1% | ✓ |
| 10×10 | 121 | 100 | 1.09 | 9675.6% | ✓ |
| 20×20 | 441 | 400 | 5.31 | 9542.9% | ✓ |
| 30×30 | 961 | 900 | 11.37 | 82.9% | ✓ |
| 40×40 | 1,681 | 1,600 | 21.70 | 82.9% | ✓ |
| 50×50 | 2,601 | 2,500 | 27.17 | 83.0% | ✓ |
| 75×75 | 5,776 | 5,625 | 79.63 | 83.2% | ✓ |

**Scalability**: T(n) ≈ O(n^1.27) with R² = 1.000
**Speedup**: 2.3–2.4× over original Python loop (vectorised SVD + sparse scatter matrix)

> **What do these numbers mean?** A 10×10 mesh has 100 panels and takes just over 1 second. A 75×75 mesh has 5,625 panels (a large architectural roof) and takes about 80 seconds — still feasible for design iteration. For meshes larger than this, the GPU (`cupy`) backend is the recommended path.

See [`data/output/benchmarks/`](data/output/benchmarks/) for detailed performance data.

---

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific test categories
pytest tests/test_optimiser.py -v              # Optimisation tests
pytest tests/test_gradients.py -v              # Gradient verification
pytest tests/test_scalability.py -v            # Scalability tests
pytest tests/test_robustness.py -v             # Robustness and security regression tests
pytest tests/test_backends.py -v               # Hardware backend tests
pytest tests/test_numerical_equivalence.py -v  # Numba vs NumPy numerical equivalence
pytest tests/test_coverage_gaps.py -v          # Branch coverage for backends/preprocessor/energy

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Full quality pipeline (format, lint, type, security, test)
make check

# Pre-commit hooks (black, isort, flake8, bandit)
pre-commit run --all-files
```

**Test Results**: 321 passed, 1 skipped | **Coverage**: ≥81% (excluding GUI module) | **0 flake8/bandit/mypy violations**

The CI environment (GitHub Actions) does not install Numba; all `test_numerical_equivalence.py` tests are therefore automatically skipped in CI and the NumPy backend is used exclusively. To replicate the CI environment locally:

```bash
pip install pytest numpy scipy
pytest tests/ -m "not slow" -v
```

> **New to testing?** Tests are small scripts that automatically check that the code still works correctly after any change. Running `pytest tests/ -v` executes all 321 checks and prints a pass/fail result for each one. If you make a change and a test fails, it tells you exactly what broke.

---

## 📖 Documentation

> **Comprehensive documentation — including mathematical foundations, full API reference, architecture diagrams, experimental results, testing guidance, and a complete FAQ — is available in the [GitHub Wiki](https://github.com/nabiljefferson98/pq-mesh-optimisation-tool/wiki).**

In-repository documentation:

- **[Architecture](docs/architecture.md)**: System design — module reference, data flow, design decisions
- **[Methodology](docs/methodology.md)**: Mathematical derivations — energy formulation, analytical gradients, optimisation algorithm
- **[src/OVERVIEW.md](src/OVERVIEW.md)**: Internal library documentation with full module reference
- **[tests/TESTING_GUIDE.md](tests/TESTING_GUIDE.md)**: Test module catalogue and running instructions
- **[scripts/SCRIPTS_OVERVIEW.md](scripts/SCRIPTS_OVERVIEW.md)**: Overview of all research and analysis scripts

---

## 🎓 Academic Context

This tool was developed as the primary software artefact of an undergraduate dissertation submitted to the **University of Leeds, School of Computing**, as part of the **COMP3931 Individual Project** module in the third year of the BSc/MEng Computer Science degree programme (2025/26).

The dissertation — titled *Real-Time Planar Quad (PQ) Mesh Optimisation and Visualisation Tool for Developable Surfaces* — investigates planar quad mesh optimisation through a four-term energy model, hardware-accelerated backends, interactive visualisation, and a structured experimental evaluation across five experiments. It was submitted in May 2026.

- **Module**: COMP3931 Individual Project (Level 3, University of Leeds)
- **Supervisor**: Professor Hamish Carr
- **Academic Assessor**: Dr Sebastian Ordyniak
- **Student**: Muhammad Nabil Bin Muhammad Saiful Wong (sc23mnbm@leeds.ac.uk)
- **Project Title**: Real-Time Planar Quad Mesh Optimisation and Visualisation Tool for Developable Surfaces
- **Submission**: May 2026

**Key References** (see [Wiki — Citation](https://github.com/nabiljefferson98/pq-mesh-optimisation-tool/wiki/Citation) for the full list of 28 references):

1. Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L. and Wang, W. (2006). *Geometric modeling with conical meshes and developable surfaces*. ACM Transactions on Graphics, 25(3), pp. 681–689.
2. Pottmann, H., Asperl, A., Hofer, M. and Kilian, A. (2007). *Architectural Geometry*. Bentley Institute Press.
3. Nocedal, J. and Wright, S.J. (2006). *Numerical Optimization*. 2nd edn. Springer.
4. Zhu, C., Byrd, R.H., Lu, P. and Nocedal, J. (1997). *Algorithm 778: L-BFGS-B*. ACM TOMS, 23(4), pp. 550–560.

---

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Follow code style (PEP 8, use `black` formatter)
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Commit with clear messages (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Professor Hamish Carr** — Academic supervisor and guidance
- **Dr Sebastian Ordyniak** — Academic Assessor and guidance
- **Dr Samuel Wilson** — Director of Student Education and his support and guidance
- **Keenan Crane** — Benchmark mesh (`spot_quadrangulated.obj`) used in experimental evaluation
- **Helmut Pottmann et al.** — Foundational research on architectural geometry
- **SciPy Contributors** — Excellent optimisation library
- **Polyscope Team** — Beautiful 3D visualisation framework

---

## 📧 Contact

**Author**: Muhammad Nabil Bin Muhammad Saiful Wong
**Email**: [sc23mnbm@leeds.ac.uk](mailto:sc23mnbm@leeds.ac.uk)
**GitHub**: [@nabiljefferson98](https://github.com/nabiljefferson98)

**University of Leeds**
School of Computing
Leeds LS2 9JT, United Kingdom

---

## 📈 Roadmap

Future enhancements under consideration:

- [x] Numba-parallel planarity SVD energy kernel (CPU acceleration) ✓
- [x] Numba-parallel planarity gradient kernel `_planarity_gradient_contributions_numba` ✓
- [ ] Full GPU acceleration for large meshes (>10,000 vertices)
- [x] Conical mesh optimisation — two-stage with angle-balance constraint ✓
- [ ] Interactive mesh editing with constraint preservation
- [x] Export to fabrication formats (DXF + SVG) ✓
- [ ] Web-based version using WebAssembly
- [ ] Integration with Rhino/Grasshopper

---

## 🐛 Known Issues

- **macOS OpenGL**: Polyscope requires OpenGL 3.3+; update drivers if issues occur
- **Windows WSL**: GUI requires X server (e.g., VcXsrv)
- **Very large meshes (>10,000 vertices)**: Runtime exceeds ~120 s; both the Numba planarity energy kernel and gradient kernel (`_planarity_gradient_contributions_numba`) are now implemented (15 Mar 2026). CuPy GPU path remains the target for very large meshes

Report bugs: [GitHub Issues](https://github.com/nabiljefferson98/pq-mesh-optimisation-tool/issues)

---

*Last updated: 15 May 2026*
