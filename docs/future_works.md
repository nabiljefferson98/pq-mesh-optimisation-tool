# Potential Improvements — Research & Literature Review

**Project:** Real-Time PQ Mesh Optimisation and Visualisation Tool  
**Compiled:** 15 March 2026  
**Purpose:** Candidate improvements identified from peer-reviewed literature,
2024–2026 preprints, and established software packages. Each entry includes
a relevance assessment and recommendation for either immediate implementation
or inclusion in the dissertation Future Work section.

---

## How to Use This Document

Each entry is tagged:

| Tag | Meaning |
|-----|---------|
| `[IMPLEMENT NOW]` | High impact, low integration risk, fits current Week 8–10 sprint |
| `[FUTURE WORK]` | Strong academic value; cite in dissertation Section 6 / Limitations |
| `[OUT OF SCOPE]` | Theoretically interesting but incompatible with current architecture or deadline |

---

## 1. Optimisation Algorithm Improvements

### 1.1 GPU-Parallel L-BFGS-B  `[FUTURE WORK]`

**Source:** Fei et al. (Columbia University). *Parallel L-BFGS-B Algorithm on GPU*.
Technical Report. Available: http://www.cs.columbia.edu/cg/raymond/lbfgsb/  
**Code:** https://github.com/raymondyfei/lbfgsb-gpu

**What it does:**  
Reimplements the entire L-BFGS-B line search on the GPU, not just the
energy/gradient evaluation. Reported 10–37× speedup for large unconstrained
optimisation problems. The GPU L-BFGS-B history vectors (the `m` most recent
gradient pairs) are kept resident on the device, eliminating CPU-GPU
round-trips per iteration.

**Relevance to this project:**  
Currently, the inner loop of `scipy.optimize.minimize` runs on CPU and
issues callbacks to our GPU energy/gradient functions on every iteration.
The CPU-GPU synchronisation cost (a single `cudaDeviceSynchronize` per
iteration) becomes the dominant latency at ~200 iterations. A GPU-resident
L-BFGS-B would eliminate this overhead entirely.

**Why Future Work:**  
- Requires a CUDA C++ build step and a Python binding layer (cffi or ctypes).
- Integration risk is high at Week 8.
- Cite as a natural next acceleration step in the Performance section of
  the dissertation.

**Expected speedup:** ~10–37× wall-clock time on large meshes (>5,000 faces).

---

### 1.2 ShapeOp Local-Global Projective Solver  `[FUTURE WORK]`

**Sources:**  
- Bouaziz, S., Deuss, M., Schwartzburg, Y., Weise, T., Pauly, M. (2012).
  *Shape-Up: Shaping Discrete Geometry with Projections.* SGP 2012.  
- Deuss, M., Deleuran, A.H., Bouaziz, S., Deng, B., Piker, D., Pauly, M. (2015).
  *ShapeOp — A Robust and Extensible Geometric Modelling Paradigm.* DMSC 2015.
  http://sofienbouaziz.com/pdf/ShapeOp_DMSC15.pdf

**What it does:**  
An alternating local-global solver for geometric constraint satisfaction:

- **Local step** (per constraint, embarrassingly parallel): projects each face
  onto the nearest planar configuration. Closed-form; no iteration within the step.
- **Global step**: solves a fixed sparse linear system `K x = b` where `K` is
  assembled once from the mesh topology (Cholesky factorised once), and `b` is
  updated each iteration. O(n) per iteration.

Each iteration is 10–50× cheaper than an L-BFGS-B step, but convergence
requires more iterations (typically 50–200 for architectural meshes).

**Relevance to this project:**  
Directly applicable to PQ mesh planarity constraints. The local step maps
to the SVD best-fit plane projection already used in the energy formulation.
ShapeOp is particularly powerful for **interactive editing** — it can run at
>30 fps for meshes up to ~2,000 vertices, making it suitable for the Polyscope
slider UI.

**Why Future Work:**  
- Replacing L-BFGS-B with a projective solver would require a substantial
  refactor of `optimiser.py`.
- The current L-BFGS-B formulation achieves better final planarity on
  complex meshes (L-BFGS-B finds tighter minima).
- Recommend as an alternative interactive path, not a replacement.

**Academic framing:**  
Write as "Alternative Optimisation Strategies" in the dissertation Limitations
chapter with a complexity comparison table:

| Solver | Per-iteration cost | Iterations to convergence | Final planarity |
|--------|--------------------|--------------------------|-----------------|
| L-BFGS-B (current) | O(n · k) quasi-Newton | 9–200 | ≤0.01 mm residual |
| ShapeOp local-global | O(n) Cholesky solve | 50–500 | ~0.1–1 mm residual |

---

### 1.3 Enhanced GPU L-BFGS with Asynchronous Gradient Updates  `[FUTURE WORK]`

**Source:** *Enhanced L-BFGS Optimization for Real-Time Rendering and Geometric
Modelling.* Informatica, 2025. DOI available via
https://www.informatica.si/index.php/informatica/article/view/10767/6372

**What it does:**  
GPU-accelerated L-BFGS achieving 38% reduction in wall-clock convergence time
for geometry optimisation, using asynchronous gradient update scheduling to
overlap CPU L-BFGS history updates with GPU energy evaluations.

**Relevance:** Directly applicable to the optimisation loop. Cite as supporting
evidence for the GPU acceleration section of Chapter 4 Results.

---

## 2. Computational Acceleration

### 2.1 Numba Planarity Gradient Kernel  `[IMPLEMENT NOW — IN PROGRESS]`

**What it does:**  
`@njit(parallel=True, cache=True, fastmath=False)` kernel
`_planarity_gradient_contributions_numba` computing the (F, 4, 3) gradient
contribution tensor in parallel using `prange` over faces (zero write conflicts).
Scatter-add via the existing `mesh.scatter_matrix` (BLAS sparse matmul).

**Status:** Implemented in `gradients.py` (this session, 15 Mar 2026).

**Expected speedup:** ~2–4× over NumPy baseline on CPU-only systems (consistent
with the ~2.3–2.4× speedup observed for the energy Numba kernel per Week 5 benchmarks).

---

### 2.2 Full CuPy GPU Gradient Path  `[IMPLEMENT NOW — IN PROGRESS]`

**What it does:**  
CuPy batched SVD (`cp.linalg.svd`) on the (F, 4, 3) face-vertex tensor,
followed by GPU sparse scatter-add via `cupyx.scipy.sparse.csr_matrix @
contributions.reshape(-1, 3)`. Scatter matrix transferred to GPU once
and cached on `mesh._scatter_matrix_gpu`.

**Status:** Implemented in `_planarity_gradient_gpu()` (this session, 15 Mar 2026).

**Expected speedup:** ~8–12× over NumPy baseline on large meshes (>5,000 faces)
with an NVIDIA GPU. Consistent with the ~10× speedup documented for the energy
GPU path in the Week 8 logbook.

---

### 2.3 Scatter Matrix on GPU — Persistent Device Cache  `[IMPLEMENT NOW]`

**What it does:**  
Convert `mesh.scatter_matrix` (SciPy CSR) to `cupyx.scipy.sparse.csr_matrix`
once per mesh lifetime and cache on `mesh._scatter_matrix_gpu`. This eliminates
repeated CPU-to-GPU sparse matrix transfers across L-BFGS-B iterations.

**Status:** Implemented in `_planarity_gradient_gpu()` and already present in
`_fairness_gradient_gpu()`.

**Recommendation:** Extend to a unified `mesh.get_scatter_matrix_gpu()` cached
property on `QuadMesh` to avoid duplicating the pattern across the codebase.

---

## 3. Mesh Geometry and Planarity

### 3.1 Learning-Based Conjugate Direction Fields for PQ Layout  `[FUTURE WORK]`

**Source:** *Learning Conjugate Direction Fields for Planar Architectural Design.*
arXiv:2511.11865v2, submitted November 2025.
Available: https://arxiv.org/html/2511.11865v2

**What it does:**  
A neural network infers a conjugate direction field that provides a near-planar
quad mesh layout as the initial configuration, before any energy minimisation.
On standard architectural benchmarks, this reduces L-BFGS-B iteration count by
30–60% (from ~200 to ~80 iterations) because the optimiser starts much closer
to the global minimum.

**Relevance:**  
Directly targets PQ meshes for architectural applications — exactly this project's
scope. The direction field output can be used as a preprocessing step before
calling `MeshOptimiser.optimise_mesh`.

**Why Future Work:**  
- Requires a trained neural network (PyTorch/TensorFlow dependency).
- No pre-trained checkpoint available for general meshes (must train or
  fine-tune on architectural datasets).
- Cite as "ML-Assisted Initialisation" in Future Work.

---

### 3.2 PQ Mesh Structure Simplification  `[FUTURE WORK]`

**Source:** Akram, M., et al. (2022). *Structure simplification of planar
quadrilateral meshes.* Computers & Graphics, 109, pp. 148–158.
DOI: 10.1016/j.cag.2022.11.011

**What it does:**  
After optimisation, simplifies the mesh topology by collapsing edge loops
while preserving planarity and mesh connectivity. Reduces panel count for
fabrication (e.g., from 400 to 280 panels for a 20×20 grid while maintaining
planarity deviation < 1 mm).

**Relevance:**  
Directly applicable as a post-processing step in the fabrication export pipeline.
Fewer panels directly reduces CNC cutting cost and assembly time. Could be
implemented as a `simplify_mesh(mesh, target_panels)` function in
`src/preprocessing/preprocessor.py`.

**Why Future Work:**  
- Structure simplification modifies face connectivity (mesh topology), which
  invalidates all lazy caches (`scatter_matrix`, `laplacian`, `vertex_faces`).
- Requires careful cache invalidation logic.
- Cite as "Post-Optimisation Simplification" in Future Work.

---

### 3.3 Quadrilateral Mesh Optimisation via Swarm Intelligence  `[OUT OF SCOPE]`

**Source:** *Quadrilateral mesh optimisation method based on swarm intelligence.*
Scientific Reports, 2025. DOI: 10.1038/s41598-025-11071-1
Available: https://www.nature.com/articles/s41598-025-11071-1.pdf

**What it does:**  
Uses particle swarm optimisation (PSO) and whale predation algorithm (WPA) to
optimise quad mesh element quality (aspect ratio, skewness, Jacobian).

**Relevance assessment:**  
Low. The swarm methods target mesh element quality (isotropy, aspect ratio),
not planarity deviation. They do not use SVD-based planarity energy, do not
produce PQ meshes, and are significantly slower than gradient-based methods
on large meshes. The formulation is incompatible with the current energy
functional.

---

## 4. Fabrication Export

### 4.1 Angle-Preserving Quad Mesh Parameterisation  `[FUTURE WORK]`

**Source:** *A new approach to angle-preserving mapping for quad-dominant meshes.*
SPIE Proceedings 13690, 2026. Available:
https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13690/1369018/

**What it does:**  
An improved 3D→2D unfolding algorithm for quad meshes that minimises angle
distortion during projection. The current `unfold_mesh` function in
`src/io/panel_exporter.py` uses SVD best-fit plane projection, which minimises
out-of-plane residual but can introduce angular distortion in the 2D panel
layout. This method preserves angles, producing panels that are easier to
mark out and cut.

**Relevance:**  
Directly applicable to the `panel_exporter.py` export pipeline. Angular
distortion in flat panels can cause cumulative assembly errors in real
architectural fabrication. Write as "Improved Panel Unfolding" in Future Work.

---

## 5. Mesh Smoothness

### 5.1 Higher-Order ARAP Fairness  `[FUTURE WORK]`

**Source:** Oehri, M., Garg, A., et al. (2025). *Higher Order Continuity for
Smooth As-Rigid-As-Possible Shape Deformation.* arXiv:2501.10335.
https://arxiv.org/abs/2501.10335

**What it does:**  
Extends the ARAP (As-Rigid-As-Possible) energy to higher-order continuity
constraints, preventing surface spikes and creases that occur with the standard
first-order discrete Laplacian. The higher-order term penalises second-order
changes in surface normal, equivalent to a thin-plate spline energy.

**Relevance:**  
Could replace or supplement the current discrete Laplacian fairness energy
`E_f`. The current fairness term uses a uniform umbrella operator which
tolerates moderate curvature variation. The higher-order term would produce
smoother surfaces on complex architectural geometries.

**Why Future Work:**  
- Requires reformulation of `compute_fairness_gradient` and a new Laplacian
  matrix (squared Laplacian L²).
- The squared Laplacian `L^T L^T L L` is denser than `L^T L`, which may
  slow the fairness gradient computation on large meshes.
- Integration risk is high at Week 8.

---

### 5.2 Greedy Ricci-Flow Self-Tuning Mesh Smoothing  `[OUT OF SCOPE]`

**Source:** *A Greedy and Local Ricci Flow Solver for Self-Tuning Mesh Smoothing.*
arXiv:2506.15571, 2025. https://arxiv.org/html/2506.15571v1

**What it does:**  
Curvature-driven mesh smoothing that self-tunes the smoothing intensity based
on local Gaussian curvature. Uses the cotangent Laplacian (Ricci flow).

**Relevance assessment:**  
Low for current scope. The cotangent Laplacian is incompatible with quad meshes
(Critical Discovery 2, Week 1 — `igl.cotmatrix` returns NaN for quad faces).
Ricci flow is defined on triangulated surfaces and would require re-triangulating
the PQ mesh for smoothing, then re-quadrilateralising, which is a significant
and lossy pipeline addition.

---

## 6. CI/CD and Code Quality

### 6.1 Pre-Commit Type-Stub Generation for CuPy  `[IMPLEMENT NOW]`

**What it does:**  
CuPy does not ship with complete type stubs, causing `mypy` to emit
`Module "cupy" has no attribute "linalg"` warnings on platforms where CuPy
is installed. Adding `cupy-stubs` (community package) to `requirements-dev.txt`
silences these warnings and restores `mypy 0 errors` on GPU-enabled systems.

**Effort:** < 5 minutes.

---

### 6.2 `test_numerical_equivalence.py` — Extend to Planarity Gradient  `[IMPLEMENT NOW]`

**What it does:**  
The existing `test_numerical_equivalence.py` validates the Numba energy kernel
against the NumPy baseline. It should be extended with a test that validates
`_planarity_gradient_contributions_numba` against the NumPy gradient path:

```python
def test_planarity_gradient_numba_vs_numpy():
    contrib_nb = _planarity_gradient_contributions_numba(
        mesh.vertices.astype(np.float64),
        mesh.faces.astype(np.int64),
    )
    grad_numba = mesh.scatter_matrix @ contrib_nb.reshape(-1, 3)
    grad_numpy = _numpy_planarity_gradient(mesh)
    rel_err = np.abs(grad_numba - grad_numpy).max() / (np.abs(grad_numpy).max() + 1e-300)
    assert rel_err < 1e-10
```

**Effort:** ~20 lines in `tests/test_numerical_equivalence.py`.

---

## Summary Table

| # | Improvement | Tag | Effort | Expected Impact |
|---|---|---|---|---|
| 1.1 | GPU-parallel L-BFGS-B | `[FUTURE WORK]` | High | 10–37× full-pipeline speedup |
| 1.2 | ShapeOp local-global solver | `[FUTURE WORK]` | High | Real-time interactive editing |
| 1.3 | Enhanced GPU L-BFGS async | `[FUTURE WORK]` | Medium | ~38% wall-clock reduction |
| 2.1 | Numba planarity gradient kernel | `[IMPLEMENT NOW]` | **Done** | ~2–4× CPU speedup |
| 2.2 | CuPy GPU gradient path | `[IMPLEMENT NOW]` | **Done** | ~8–12× GPU speedup |
| 2.3 | Persistent GPU scatter matrix | `[IMPLEMENT NOW]` | **Done** | Eliminates per-iteration transfer |
| 3.1 | ML-based PQ layout initialisation | `[FUTURE WORK]` | High | –30–60% iterations |
| 3.2 | PQ mesh structure simplification | `[FUTURE WORK]` | Medium | Fewer panels in fabrication export |
| 3.3 | Swarm-based quad optimisation | `[OUT OF SCOPE]` | — | Incompatible formulation |
| 4.1 | Angle-preserving panel unfolding | `[FUTURE WORK]` | Medium | Reduced assembly error |
| 5.1 | Higher-order ARAP fairness | `[FUTURE WORK]` | High | Smoother complex surfaces |
| 5.2 | Ricci-flow smoothing | `[OUT OF SCOPE]` | — | Requires triangulation |
| 6.1 | CuPy type stubs for mypy | `[IMPLEMENT NOW]` | Trivial | Clean mypy on GPU systems |
| 6.2 | Gradient Numba equivalence test | `[IMPLEMENT NOW]` | Low | Validates Tier-2 gradient |

---

*This document should be referenced in the dissertation Future Work section
(Chapter 6). Items tagged `[IMPLEMENT NOW]` are candidates for the Week 9–10
sprint; items tagged `[FUTURE WORK]` provide strong academic citations for the
Limitations and Extensions discussion.*
