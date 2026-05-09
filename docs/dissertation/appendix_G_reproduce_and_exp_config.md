# Appendix G: Reproducibility and Experimental Configuration

This appendix provides complete configuration specifications,
hardware environment records, and step-by-step instructions
sufficient to reproduce all five experiments (EXP-01 to EXP-05)
reported in Chapter 4. It is intended to satisfy the
reproducibility standards of computational geometry research
(Pottmann et al., 2015) and to complement the experimental
overview in Chapter 4, Section 4.1.

---

## G.1 Hardware and Software Environments

All experiments were conducted on two machines. Machine A
provided primary results; Machine B provided cross-platform
validation for EXP-02 and EXP-05.

**Table G.1 — Hardware and Software Environment**

| Component            | Machine A (Primary)                        | Machine B (Validation)         |
|:---------------------|:-------------------------------------------|:-------------------------------|
| CPU                  | AMD Ryzen 7 5800X (8-core, 3.8 GHz)       | Apple M3 (8-core, 3.7 GHz)    |
| RAM                  | 16 GB DDR4-3600                            | 8 GB unified memory            |
| GPU                  | NVIDIA RTX 3070 8 GB VRAM (CUDA 12)       | None (no CUDA)                 |
| OS                   | Windows 11 22H2                            | macOS Sonoma 14.x              |
| Python               | 3.10.14                                    | 3.11.x                         |
| NumPy                | 1.26.x                                     | 1.26.x                         |
| SciPy                | 1.11.x                                     | 1.11.x                         |
| Numba                | 0.59.x                                     | 0.59.x                         |
| CuPy                 | 13.x (CUDA 12)                             | Not installed                  |
| Polyscope            | 2.x                                        | 2.x                            |

All Python package versions are recorded in `requirements.txt`
(CUDA-capable) and `requirements_without_CUDA.txt`
(CPU-only) in the repository root.

---

## G.2 Repository Checkout and Environment Setup

All experiments are reproducible from the repository at
`https://github.com/nabiljefferson98/pq-mesh-optimisation-tool`.

**Step 1 — Clone the repository:**

```bash
git clone https://github.com/nabiljefferson98/pq-mesh-optimisation-tool.git
cd pq-mesh-optimisation-tool
```

**Step 2 — Create and activate a virtual environment:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**Step 3 — Install dependencies:**

For CUDA-capable machines (Machine A):
```bash
pip install -r requirements.txt
```

For CPU-only machines (Machine B):
```bash
pip install -r requirements_without_CUDA.txt
```

**Step 4 — Verify installation:**

```bash
python -c "from src.optimisation.optimiser import optimise_mesh_simple; print('OK')"
pytest tests/ -q
```

All 321 tests (1 skipped) should pass. Overall source
coverage should be at or above 79 per cent of `src/`,
excluding `interactive_optimisation.py`.

---

## G.3 Backend Selection

The computational backend is selected automatically at runtime
via the dispatch logic in `src/backends.py`. It can be
overridden explicitly by setting the `PQBACKEND` environment
variable before running any experiment script:

```bash
# Force NumPy baseline
export PQBACKEND=numpy      # macOS / Linux
set PQBACKEND=numpy         # Windows

# Force Numba CPU-parallel
export PQBACKEND=numba

# Force CuPy GPU (requires CUDA)
export PQBACKEND=cupy
```

For EXP-02, backends were isolated using this mechanism and
speedup factors were normalised against the NumPy baseline on
the same machine to prevent cross-machine confounds
(Mytkowicz et al., 2009). Numba JIT warm-up invocations were
excluded from all reported timings.

---

## G.4 Experiment Scripts

All experiment scripts reside in `scripts/benchmarking/` and
`scripts/analysis/`. The table below maps each experiment to
its entry-point script and primary output artefacts.

**Table G.2 — Experiment Script Mapping**

| Experiment | Script                                        | Output Path                                        |
|:-----------|:----------------------------------------------|:---------------------------------------------------|
| EXP-01     | `scripts/benchmarking/run_exp01_scalability.py` | `data/output/results/EXP-01_scalability.csv`     |
| EXP-02     | `scripts/benchmarking/run_exp02_backends.py`    | `data/output/results/EXP-02_backends.csv`        |
| EXP-03     | `scripts/benchmarking/run_exp03_convergence.py` | `data/output/results/EXP-03_convergence.csv`     |
| EXP-04     | `scripts/benchmarking/run_exp04_weights.py`     | `data/output/results/EXP-04_weights.csv`         |
| EXP-05     | `scripts/benchmarking/run_exp05_realworld.py`   | `data/output/results/EXP-05_realworld.csv`       |
| Figures    | `scripts/analysis/generate_figures.py`          | `data/output/figures/`                            |

To reproduce all five experiments in sequence:

```bash
python scripts/benchmarking/run_exp01_scalability.py
python scripts/benchmarking/run_exp02_backends.py
python scripts/benchmarking/run_exp03_convergence.py
python scripts/benchmarking/run_exp04_weights.py
python scripts/benchmarking/run_exp05_realworld.py
python scripts/analysis/generate_figures.py
```

Each script prints a progress summary to stdout and writes
a timestamped CSV to `data/output/results/`. Runtime on
Machine A for the full suite is approximately 45 minutes.

---

## G.5 Optimisation Configuration Parameters

The L-BFGS-B hyperparameters used in all experiments are
specified via `OptimisationConfig` (Appendix E, Section E.2.6).
The values below reproduce the exact configuration used in
Chapter 4.

**Table G.3 — L-BFGS-B Configuration Used in All Experiments**

| Parameter                     | Stage 1 Value    | Stage 2 Value    |
|:------------------------------|:-----------------|:-----------------|
| `ftol`                        | 10⁻⁷             | 10⁻⁹             |
| `gtol`                        | 10⁻⁴             | 10⁻⁵             |
| `maxcor`                      | 10               | 20               |
| `maxls`                       | 20               | 40               |
| `maxiter` (synthetic, EXP-01–04) | min(200, N // 3) | 200 − Stage 1 used |
| `maxiter` (real-world, EXP-05) | min(200, N // 3) | 800 (at default max_iterations = 1,000) |
| `stage1_planarity_multiplier` | 5.0              | N/A              |

**Experimental weight configuration (EXP-01 to EXP-05):**

| Weight | Value |
|:-------|:------|
| w_p    | 10.0  |
| w_f    | 1.0   |
| w_c    | 5.0   |
| w_a    | 0.0   |

These are the experimentally calibrated values established
in EXP-04 (Chapter 4, Section 4.4.5). They are distinct from
the heuristic preprocessor defaults (w_p = 100.0, w_f = 1.0,
w_c = 10.0) computed by `suggest_weights_for_mesh`.

---

## G.6 Input Mesh Locations

All input meshes are stored in `data/input/` under the
following subdirectories:

```
data/input/
├── generated/ # Synthetic regular grid meshes (EXP-01 to EXP-04)
│ ├── plane_3x3_noisy.obj
│ ├── plane_5x5_noisy.obj
│ ├── plane_10x10_noisy.obj
│ ├── plane_20x20_noisy.obj
│ ├── plane_30x30_noisy.obj
│ ├── plane_40x40_noisy.obj
│ ├── plane_50x50_noisy.obj
│ └── plane_75x75_noisy.obj
└── benchmark/ # Real-world quad meshes (EXP-05)
├── spot.obj # Keenan Crane repository (CC0 1.0)
├── blub.obj # Keenan Crane repository (CC0 1.0)
├── oloid.obj # Original to this project (MIT)
└── bob.obj # Thingi10K re-meshed (CC BY 4.0)
```


Dataset provenance, licence information, and preprocessing
decisions for the EXP-05 benchmark meshes are documented in
full in Appendix F.

---

## G.7 Gradient Verification

Before running any optimisation experiment, the analytic
gradients should be verified against central finite differences:

```bash
pytest tests/test_gradients.py tests/test_gradients_extended.py -v
```

Expected outcome: all tests pass with maximum relative error
below 10⁻⁴ for all four energy terms across flat,
sinusoidally perturbed, and cylindrically curved mesh
geometries. This verification is a prerequisite for trusting
reported planarity figures, as undetected gradient errors
would silently degrade convergence quality without triggering
any runtime exception.

---

## G.8 Numerical Equivalence Across Backends

To confirm that all three backends (NumPy, Numba, CuPy)
produce numerically equivalent results before running EXP-02:

```bash
pytest tests/test_numerical_equivalence.py -v
```

Expected outcome: Numba-versus-NumPy gradient differences
below 10⁻¹⁰ for 10 × 10 meshes and below 10⁻⁸ for
20 × 20 meshes, confirming platform-invariant convergence
behaviour (Chapter 4, Section 4.2.2; Higham, 2002).

---

## G.9 Expected Output Values for Validation

The table below provides reference output values drawn
directly from Chapter 4 against which reproduced runs can
be validated. Minor numerical differences due to
floating-point non-determinism across hardware platforms
are expected; differences exceeding the stated tolerances
indicate a configuration error.

**Table G.4 — Reference Output Values for Validation**

| Experiment | Mesh         | Expected Runtime (Machine A, NumPy) | Expected E_p Reduction (%) | Tolerance    |
|:-----------|:-------------|:------------------------------------|:---------------------------|:-------------|
| EXP-01     | 3 × 3        | 0.08 ± 0.02 s                       | > 9,000                    | ± 5%         |
| EXP-01     | 10 × 10      | 1.09 ± 0.10 s                       | > 9,000                    | ± 5%         |
| EXP-01     | 75 × 75      | 79.63 ± 3.00 s                      | 83.2                       | ± 2%         |
| EXP-02     | 10 × 10      | Numba 2.79× over NumPy              | Equivalent to NumPy        | ± 0.1×       |
| EXP-03     | 5 × 5        | 9 iterations to convergence         | Grad norm < 10⁻⁵           | ± 2 iters    |
| EXP-05     | Oloid        | 0.80 ± 0.15 s                       | 66.03                      | ± 3%         |
| EXP-05     | Spot         | 18.47 ± 1.50 s                      | 7.55                       | ± 2%         |

---

## References

Higham, N. J. (2002). *Accuracy and Stability of Numerical
Algorithms*. 2nd ed. Philadelphia: SIAM.

Mytkowicz, T., Diwan, A., Hauswirth, M. and Sweeney, P. F.
(2009). Producing wrong data without doing anything obviously
wrong! In *Proceedings of ASPLOS XIV*. New York: ACM,
pp. 265–276.

Pottmann, H., Eigensatz, M., Vaxman, A. and Wallner, J.
(2015). Architectural geometry. *Computers and Graphics*,
47, pp. 145–164.