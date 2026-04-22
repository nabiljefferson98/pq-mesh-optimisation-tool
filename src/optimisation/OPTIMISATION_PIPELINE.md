# Optimisation Pipeline

`src/optimisation/`

This document is the definitive technical reference for the complete
optimisation pipeline implemented in `src/optimisation/`. It covers the
energy functional and its mathematical derivation, the analytical gradient
formulae, the two-stage L-BFGS-B strategy, the three-tier backend dispatch
hierarchy, all configuration parameters, and the data structures that carry
results out of the optimiser. The document is derived directly from the
production source code and is intended to be self-contained for dissertation
documentation purposes.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Module Structure](#module-structure)
3. [Energy Functional](#energy-functional)
4. [Gradient Computation](#gradient-computation)
5. [Two-Stage Optimisation Strategy](#two-stage-optimisation-strategy)
6. [Backend Dispatch Hierarchy](#backend-dispatch-hierarchy)
7. [Configuration Reference](#configuration-reference)
8. [Data Structures](#data-structures)
9. [SciPy Interface Layer](#scipy-interface-layer)
10. [Diagnostic and Verification Utilities](#diagnostic-and-verification-utilities)
11. [References](#references)

---

## Quick Start

For most users the convenience function `optimise_mesh_simple` is the
simplest entry point:

```python
from src.io.obj_handler import load_obj
from src.preprocessing.preprocessor import preprocess_mesh
from src.optimisation.optimiser import optimise_mesh_simple

raw   = load_obj("data/input/generated/saddle_8x8.obj")
mesh, _ = preprocess_mesh(raw)
result  = optimise_mesh_simple(mesh, max_iter=500)
print(result.summary())
```

For fine-grained control, use `MeshOptimiser` with an `OptimisationConfig`
directly:

```python
from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig

config = OptimisationConfig(
    weights={"planarity": 100.0, "fairness": 1.0, "closeness": 10.0},
    max_iterations=1000,
    two_stage=True,
)
result = MeshOptimiser(config).optimise(mesh)
```

---

## Module Structure

```
src/optimisation/
├── OPTIMISATION_PIPELINE.md   <- this file
├── energy_terms.py            <- four energy functions + combined energy
├── gradients.py               <- four analytical gradient functions + SciPy interface
├── mesh_geometry.py           <- geometric helpers (conical angle imbalance, etc.)
└── optimiser.py               <- MeshOptimiser, OptimisationConfig, OptimisationResult
```

---

## Energy Functional

The optimiser minimises a weighted sum of four geometric penalty functions.
Each function measures how far the current mesh deviates from one desired
geometric property. The total energy is:

```
E_total = w_p * E_planarity
        + w_f * E_fairness
        + w_c * E_closeness
        + w_a * E_angle_balance
```

where `w_p`, `w_f`, `w_c`, `w_a` are scalar weights from `OptimisationConfig.weights`.
The `angle_balance` weight defaults to `0.0`, meaning that term is disabled
unless explicitly enabled.

---

### E_planarity — Panel Flatness

**Physical meaning.** A quad face is fabricable from flat sheet material
(glass, steel, UHPC) only if its four corners are coplanar. This term
penalises non-coplanar quads.

**Formula.** For each face f with vertices v_0, v_1, v_2, v_3, form the
centred matrix M_f in R^(4x3):

```
M_f[k] = v_k - centroid(f),   k = 0..3
centroid(f) = (v_0 + v_1 + v_2 + v_3) / 4
```

Decompose M_f by SVD:

```
M_f = U * S * V^T
```

The smallest singular value sigma_min is the half-thickness of the face in
the direction perpendicular to its best-fit plane. The per-face planarity
energy is its square, and the total is the sum across all faces:

```
E_planarity = sum_f  sigma_min(M_f)^2
```

A face with all four vertices exactly coplanar has sigma_min = 0 and
contributes zero energy.

**Implementation note.** In the NumPy Tier-3 path, the computation is
vectorised across all faces simultaneously:

```python
face_verts = mesh.vertices[mesh.faces]          # (F, 4, 3)
centroids  = face_verts.mean(axis=1, keepdims=True)
centered   = face_verts - centroids
_, _, Vt   = np.linalg.svd(centered, full_matrices=False)
normals    = Vt[:, -1, :]                        # last row = smallest singular vector
signed_dists = np.einsum("fvd,fd->fv", centered, normals)
E_planarity  = float(np.sum(signed_dists**2))
```

Note that `np.sum(signed_dists**2)` is equivalent to summing `sigma_min^2`
because `signed_dists` contains the projections of each vertex onto the
face normal, whose squared sum equals the squared smallest singular value.

**Reference.** Liu et al. (2006).

---

### E_fairness — Surface Smoothness

**Physical meaning.** The fairness term penalises irregular curvature by
measuring how much each vertex deviates from the average position of its
direct neighbours. A mesh with low fairness energy has smooth, evenly
distributed curvature.

**Formula.** Let L be the combinatorial (unweighted) graph Laplacian of the
mesh and V be the (n_vertices x 3) matrix of vertex positions. The
Laplacian row for vertex i is:

```
(LV)_i = v_i - (1 / deg_i) * sum_{j in N(i)} v_j
```

The fairness energy is the Frobenius norm squared of L applied to V:

```
E_fairness = || L V ||_F^2 = sum_i || (LV)_i ||^2
```

**Implementation.** The Laplacian L is a sparse matrix built once at mesh
construction and cached on `mesh.laplacian`. The computation in Tier-3 is:

```python
L = mesh.laplacian
laplacian_coords = L @ mesh.vertices
E_fairness = float(np.sum(laplacian_coords**2))
```

**Reference.** Crane et al. (2013).

---

### E_closeness — Shape Fidelity

**Physical meaning.** Without a regularising term, the optimiser could
reduce planarity energy to zero by collapsing the mesh to a flat plane,
which destroys the designer's intent. The closeness term penalises
displacement from the original vertex positions, preserving the overall
shape.

**Formula.** Let V be the current vertex positions and V_0 be the original
positions stored in `mesh.vertices_original` (set by the preprocessor):

```
E_closeness = sum_i || v_i - v_i^0 ||^2
```

E_closeness = 0 at the very start of optimisation and grows as vertices
move. A high closeness weight limits how freely vertices can move, trading
geometric quality for shape fidelity.

---

### E_angle_balance — Conical Vertex Constraint

**Physical meaning.** A conical mesh is one in which the four quad faces
meeting at each interior vertex satisfy the condition:

```
alpha_1 + alpha_3 = alpha_2 + alpha_4
```

where alpha_1, alpha_2, alpha_3, alpha_4 are the face corner angles at that
vertex in cyclic order. This is a necessary condition for the mesh to be
developable, meaning it can be unrolled flat without cutting or tearing.

**Formula.** For each 4-valent interior vertex v:

```
E_angle_balance = sum_v  ( (alpha_1 + alpha_3) - (alpha_2 + alpha_4) )^2
```

Vertices with valence other than 4 are skipped, as the conical condition
is only defined for 4-valent vertices.

**When to use.** This term is optional. Set `angle_balance = 0.0` (the
default) for general PQ mesh optimisation where fabrication by flat panels
is the only requirement. Enable it (typical values: 1 to 100) only when the
target application specifically requires a conical or near-developable mesh.

**Reference.** Liu et al. (2006).

---

### Combined Energy Function

`compute_total_energy(mesh, weights, return_components=False)` is the main
objective function called at every iteration. When `return_components=True`,
it also returns a dictionary with keys:

| Key | Description |
|---|---|
| `E_planarity` | Raw unweighted planarity energy |
| `E_fairness` | Raw unweighted fairness energy |
| `E_closeness` | Raw unweighted closeness energy |
| `E_angle_balance` | Raw unweighted angle balance energy |
| `weighted_planarity` | `w_p * E_planarity` |
| `weighted_fairness` | `w_f * E_fairness` |
| `weighted_closeness` | `w_c * E_closeness` |
| `weighted_angle_balance` | `w_a * E_angle_balance` |

The `angle_balance` term is only computed when `w_a > 0`, which avoids
unnecessary computation in the common case.

---

## Gradient Computation

Analytical gradients are implemented in `gradients.py`. Providing analytical
gradients to L-BFGS-B avoids finite-difference approximation, which would
require 6N energy evaluations per iteration for an N-vertex mesh compared to
a single analytical gradient evaluation.

The total gradient is the weighted sum of four individual gradients:

```
nabla E_total = w_p * nabla E_planarity
              + w_f * nabla E_fairness
              + w_c * nabla E_closeness
              + w_a * nabla E_angle_balance
```

This is valid because differentiation distributes over sums.

---

### Planarity Gradient

`compute_planarity_gradient(mesh)` returns a (n_vertices, 3) array.

**Derivation.** For each face f and each vertex v_k in that face, the
gradient contribution of v_k with respect to E_planarity(f) is:

```
d E_planarity(f) / d v_k = 2 * d_k * n_f
```

where `d_k = (v_k - centroid) . n_f` is the signed distance of vertex k
from the face's best-fit plane, and `n_f` is the face unit normal (the
singular vector corresponding to sigma_min). The total gradient for a vertex
is the sum of contributions from all faces containing it.

**Scatter matrix.** Rather than looping over faces in Python to accumulate
contributions, the code uses a precomputed sparse scatter matrix
(`mesh.scatter_matrix`, shape `n_verts x F*4`) which maps the flat per-face
contribution array directly to vertex gradients via a single BLAS-backed
sparse matrix multiply:

```python
# Tier-3 path:
contributions = 2.0 * signed_dists[:, :, None] * normals[:, None, :]  # (F, 4, 3)
return mesh.scatter_matrix @ contributions.reshape(-1, 3)              # (n_verts, 3)
```

**Tier-2 Numba kernel.** The Numba path uses Cardano's closed-form
eigenvalue formula to compute the face normal inside a `@njit(parallel=True)`
kernel without calling SVD, which is expensive inside Numba:

1. Compute the 3x3 covariance matrix A = M^T M from the centred face vertices.
2. Compute the smallest eigenvalue of A using Cardano's formula.
3. Recover the corresponding eigenvector using the cross-product method
   (taking the most numerically stable pair of row cross-products).
4. Compute `d_k` and the gradient contribution for each vertex.

This is numerically equivalent to the SVD-based Tier-3 path and is validated
by the test suite.

---

### Fairness Gradient

`compute_fairness_gradient(mesh)` returns a (n_vertices, 3) array.

**Formula.** Differentiating `|| L V ||^2` with respect to V gives:

```
nabla_V || LV ||^2 = 2 * L^T * (L * V)
```

Since the combinatorial Laplacian L is symmetric (L = L^T), this simplifies to:

```
nabla_V E_fairness = 2 * L^2 * V
```

**Implementation.** The Tier-3 path computes this as two sequential sparse
matrix multiplies:

```python
L = mesh.laplacian
return 2.0 * (L.T @ (L @ mesh.vertices))
```

There is no Tier-2 (Numba) path for this gradient because the sparse matrix
multiply is already highly optimised through SciPy's BLAS backend.

---

### Closeness Gradient

`compute_closeness_gradient(mesh)` returns a (n_vertices, 3) array.

**Formula.** Differentiating `sum_i || v_i - v_i^0 ||^2` with respect to v_i
gives simply:

```
d E_closeness / d v_i = 2 * (v_i - v_i^0)
```

This is the simplest of the four gradients. It acts as a spring pulling each
vertex back towards its original position:

```python
return 2.0 * (mesh.vertices - mesh.vertices_original)
```

---

### Angle Balance Gradient

`compute_angle_balance_gradient(mesh)` returns a (n_vertices, 3) array.

**Derivation.** For a 4-valent vertex v with imbalance
`delta = (alpha_0 + alpha_2) - (alpha_1 + alpha_3)`, the energy contribution
is `delta^2`. Differentiating with respect to the positions of v, its
previous neighbour v_prev, and its next neighbour v_next in each face gives
three gradient contributions per face angle.

For a face angle at vertex v between edge vectors e1 = v_prev - v and
e2 = v_next - v, the standard angle-gradient formula gives:

```
d alpha / d v      = (1/sin(alpha)) * ( (e2_hat - cos(alpha)*e1_hat)/||e1||
                                       + (e1_hat - cos(alpha)*e2_hat)/||e2|| )

d alpha / d v_prev = -(1/sin(alpha)) * (e2_hat - cos(alpha)*e1_hat) / ||e1||

d alpha / d v_next = -(1/sin(alpha)) * (e1_hat - cos(alpha)*e2_hat) / ||e2||
```

The minimum sine clamp `_SIN_MIN_ANGLE_GRAD = 1e-2` prevents the gradient
from becoming arbitrarily large near degenerate face angles.

The total gradient for the angle balance energy is accumulated by applying
these contributions with alternating signs (+1, -1, +1, -1) according to
whether the angle enters the imbalance with a positive or negative sign.

**Numba two-pass kernel.** Because the angle balance gradient involves
irregular vertex-to-face scatter, a two-pass strategy is used to enable
parallel computation without write conflicts:

- **Pass 1 (parallel):** Each 4-valent vertex computes its own central
  gradient contributions and stores neighbour contributions in pre-allocated
  scratch buffers (`scratch_gvp`, `scratch_gvn`, `scratch_prev`, `scratch_next`).
  Each vertex writes only to its own rows, so no conflicts arise.
- **Pass 2 (serial):** A single thread reads from the scratch buffers and
  accumulates neighbour contributions into the gradient array.

The scratch buffers are allocated once at mesh construction and reused across
all iterations to avoid repeated memory allocation.

---

### Numerical Gradient Verification

`verify_gradient(mesh, weights, tolerance=1e-4)` compares the analytical
gradient against a central finite-difference estimate:

```
numerical_gradient[i, j] = (E(v_ij + eps) - E(v_ij - eps)) / (2 * eps)
```

The relative error is the Euclidean norm of the difference divided by the
norm of the numerical gradient. Values below `1e-4` confirm the analytical
gradients are correctly derived. This check is run in the test suite.

---

## Two-Stage Optimisation Strategy

When `OptimisationConfig.two_stage = True` (the default), the optimiser runs
two sequential L-BFGS-B passes over the same mesh with different weight
configurations.

### Stage 1 — Rapid Planarity Pass

The planarity weight is multiplied by `stage1_planarity_multiplier` (default: 5).
Fairness and closeness weights are each reduced to 10% of their standard values.
Loose stopping tolerances and a capped iteration budget keep this stage fast.

| Parameter | Stage 1 value |
|---|---|
| Planarity weight | `w_p * stage1_planarity_multiplier` |
| Fairness weight | `w_f * 0.1` |
| Closeness weight | `w_c * 0.1` |
| Angle balance weight | `w_a` (unchanged) |
| `ftol` | `1e-7` |
| `gtol` | `1e-4` |
| `maxcor` | `10` |
| `maxls` | `20` |
| Max iterations | `min(200, max_iterations // 3)` |

The purpose of Stage 1 is to drive all face normals towards a planar
configuration quickly. Starting Stage 2 from a near-planar solution
significantly accelerates convergence compared to a cold start.

Note: the `angle_balance` weight is carried over unchanged rather than
boosted in Stage 1, because boosting it before faces are planar can
destabilise the topology.

### Stage 2 — Balanced Refinement

Restores the original weights specified in `OptimisationConfig.weights` and
applies tighter stopping tolerances. Stage 2 is warm-started from the Stage 1
solution.

| Parameter | Stage 2 value |
|---|---|
| All weights | Original values from `OptimisationConfig.weights` |
| `ftol` | `1e-9` |
| `gtol` | `1e-5` |
| `maxcor` | `20` |
| `maxls` | `40` |
| Max iterations | `max_iterations - stage1_budget` |

### Combined Budget

The combined iteration count across both stages is strictly bounded by
`config.max_iterations`. Stage 1 receives at most
`min(200, max_iterations // 3)` iterations; Stage 2 receives the remainder.
The `OptimisationResult.n_iterations` field reports the combined total.

---

## Backend Dispatch Hierarchy

Every energy and gradient function implements a three-tier dispatch hierarchy
evaluated at runtime based on available hardware and libraries:

```
Tier 1 -- CuPy (NVIDIA GPU, cuBLAS / cuSOLVER / cuSPARSE)
Tier 2 -- Numba (CPU-parallel, @njit(parallel=True, cache=True))
Tier 3 -- NumPy (CPU serial, always available)
```

The `HAS_CUDA` and `HAS_NUMBA` flags are determined once at import time by
`src/backends.py`. Every function checks these flags at the top and dispatches
accordingly, falling back gracefully if a higher tier raises an error.

### Per-Term Backend Notes

| Term | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| E_planarity | CuPy batched SVD | Numba: per-face SVD in `prange` | NumPy batched SVD |
| E_fairness | CuPy sparse matmul | Not applicable (SciPy sparse is sufficient) | SciPy sparse matmul |
| E_closeness | CuPy elementwise subtract | Not applicable | NumPy subtract |
| E_angle_balance | CuPy vectorised arccos | Numba: `prange` over vertices | Python loop over vertices |
| grad_planarity | CuPy batched SVD + GPU scatter | Numba: Cardano eigenvector + scatter | NumPy batched SVD + scatter |
| grad_fairness | CuPy sparse 2*L^T*(LV) | Not applicable | SciPy sparse 2*L^T*(LV) |
| grad_closeness | CuPy elementwise | Not applicable | NumPy elementwise |
| grad_angle_balance | Not applicable (irregular adjacency) | Numba two-pass kernel | Python loop |

### Numba Kernel Compiler Settings

All Numba kernels are compiled with:

- `parallel=True`: Numba spawns a thread pool; each face or vertex is processed
  by one thread.
- `cache=True`: The compiled native binary is saved to `__pycache__/` on first
  run. Subsequent runs load from cache, incurring no JIT compile overhead.
- `fastmath=False`: Preserves floating-point associativity, ensuring results
  numerically match the NumPy baseline.

The first call to any Numba kernel on a fresh environment incurs a one-time
compilation cost of approximately 2 to 5 seconds. Call
`warmup_numba_kernels()` before launching interactive sessions to avoid this
stall appearing during use.

### GPU OOM Fallback

All GPU functions are wrapped in a `gpu_memory_guard()` context manager from
`src/backends.py`. If a CUDA out-of-memory error occurs, the context manager
catches it, sets `_gpu_succeeded = False`, and the function falls back through
the remaining tiers automatically. The mesh state is always consistent after
a fallback.

---

## Configuration Reference

`OptimisationConfig` holds all settings for a single run. All fields have
default values suitable for most architectural quad meshes.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `weights` | `dict` | `{planarity:100, fairness:1, closeness:10}` | Per-term energy weights |
| `max_iterations` | `int` | `1000` | Combined iteration budget across both stages |
| `tolerance` | `float` | `1e-6` | Energy convergence threshold (Stage 2 `ftol`) |
| `gradient_tolerance` | `float` | `1e-4` | Gradient norm convergence threshold |
| `verbose` | `bool` | `True` | Print per-10-step progress to stdout |
| `history_tracking` | `bool` | `True` | Record energy and gradient norm at each iteration |
| `bounds_scale` | `float` or `None` | `None` | Constrain each vertex to ± bounds_scale × initial value |
| `two_stage` | `bool` | `True` | Enable two-stage strategy (recommended) |
| `stage1_planarity_multiplier` | `float` | `5.0` | Planarity weight multiplier in Stage 1 (recommended: 3 to 10) |

`OptimisationConfig.validate()` checks all parameters before optimisation
begins and raises `ValueError` with a descriptive message for the first
invalid setting found.

### Weight Selection Guidelines

Use `suggest_weight_scaling(mesh)` from `energy_terms.py` to obtain
data-driven starting weights. As a general guide:

- **Planarity** (`w_p`): highest weight; this is the primary fabrication
  constraint. Typical range: 10 to 1000. The default is 100.
- **Fairness** (`w_f`): prevents oscillation and maintains smooth curvature.
  Typical range: 0.1 to 10. The default is 1.
- **Closeness** (`w_c`): prevents collapse; increase if the mesh deforms too
  far from the original design. Typical range: 1 to 100. The default is 10.
- **Angle balance** (`w_a`): disabled by default (`0.0`). Enable only when
  the application requires conical vertices. Typical range when used: 1 to 100.

---

## Data Structures

### `OptimisationResult`

Returned by `MeshOptimiser.optimise()` and `optimise_mesh_simple()`. All
fields are listed below.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | `True` if SciPy reported successful convergence |
| `message` | `str` | Status message from `scipy.optimize.minimize` |
| `optimised_mesh` | `QuadMesh` | Mesh with vertices updated to the optimised positions |
| `initial_energy` | `float` | Total weighted energy before optimisation |
| `final_energy` | `float` | Total weighted energy after optimisation |
| `n_iterations` | `int` | Combined iterations across both stages |
| `n_function_evaluations` | `int` | Combined function evaluations across both stages |
| `n_gradient_evaluations` | `int` | Combined gradient evaluations across both stages |
| `execution_time` | `float` | Total wall-clock time in seconds |
| `energy_history` | `list[float]` or `None` | Energy at each iteration (if `history_tracking=True`) |
| `gradient_norm_history` | `list[float]` or `None` | Gradient norm at each iteration |
| `component_energies_initial` | `dict` or `None` | Per-term raw energies before optimisation |
| `component_energies_final` | `dict` or `None` | Per-term raw energies after optimisation |

Key methods:

- `energy_reduction()`: Returns `(initial - final) / initial` as a fraction in [0, 1].
- `energy_reduction_percentage()`: Returns the above multiplied by 100.
- `summary()`: Returns a formatted multi-line string covering convergence
  status, iteration counts, timing, and a per-term energy breakdown with
  plain-English trade-off explanations.

**Note on `success=False`.** A result with `success=False` does not
necessarily mean the mesh is unusable. A mesh that improved by 95% but
hit the iteration limit is still a valid, substantially optimised result.
The `summary()` method distinguishes between iteration-limit cases and true
failures.

### `OptimisationConfig` (see above)

---

## SciPy Interface Layer

L-BFGS-B in SciPy operates on flat one-dimensional arrays rather than the
two-dimensional vertex matrices used internally. Two wrapper functions in
`gradients.py` handle this translation.

### `energy_for_scipy(x_flat, mesh, weights)`

Reshapes the flat array to `(n_vertices, 3)`, updates `mesh.vertices`,
computes the total energy, and returns it as a scalar. If the energy is
non-finite (due to a degenerate mesh configuration), returns `1e300` as a
fallback. This causes SciPy to reject the trial point and backtrack rather
than diverging.

### `gradient_for_scipy(x_flat, mesh, weights)`

Reshapes the flat array, updates `mesh.vertices`, computes the total
analytical gradient, and returns it as a flat array of shape `(n_vertices * 3,)`.
If any gradient values are non-finite, they are replaced with zero and a
warning is printed. This prevents the optimiser from crashing on degenerate
faces while signalling that investigation may be needed.

Both functions are called by the `lambda` wrappers passed to
`scipy.optimize.minimize` inside `MeshOptimiser.optimise()`.

---

## Diagnostic and Verification Utilities

### `analyse_energy_components(mesh, weights)`

Prints a formatted table of raw and weighted energy values to stdout.
Intended for interactive exploration before or after optimisation.

### `suggest_weight_scaling(mesh, verbose=True)`

Computes all four energies at unit weight, then recommends weights scaled to
bring each weighted contribution to a comparable magnitude. The suggested
target ratio is `w_p*E_p : w_f*E_f : w_c*E_c ~ 10:1:5`. If `E_p < 1e-10`
(the mesh is already near-planar), a warning is printed and weights are
calibrated to the fairness energy alone.

### `verify_gradient(mesh, weights, tolerance=1e-4)`

Compares the analytical gradient against central finite differences and
returns `(is_correct, relative_error)`. A relative error below `1e-4`
confirms the analytical derivation is correct.

### `compute_gradient_statistics(gradient)`

Returns a dictionary with `norm`, `max_magnitude`, `mean_magnitude`,
`std_magnitude`, `max_component`, and `min_magnitude` computed from a
`(n_vertices, 3)` gradient array. Useful for monitoring convergence and
detecting numerical instability.

### `print_gradient_analysis(mesh, weights)`

Prints a breakdown of the norms of each individual gradient term alongside
the total gradient norm. Useful for diagnosing which term is dominating the
optimisation direction.

### `MeshOptimiser.validate_mesh(mesh)`

Pre-flight checks before optimisation begins. Validates:

1. At least 4 vertices and 1 face.
2. All vertex coordinates are finite (no NaN or Inf).
3. All faces are quads (exactly 4 vertex indices).
4. No face contains duplicate vertex indices.
5. No face has a near-zero area (below `1e-10`).

Returns `(True, "Valid")` on success, or `(False, message)` with a
descriptive error on the first problem found.

---

## References

- Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.
- Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  Architectural Geometry. Bentley Institute Press.
- Nocedal, J. and Wright, S. J. (2006).
  Numerical Optimization. 2nd ed. Springer.
- Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J. (1997).
  "Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale
  bound-constrained optimisation."
  ACM Transactions on Mathematical Software, 23(4), pp. 550-560.
- Crane, K., de Goes, F., Desbrun, M., and Schroder, P. (2013).
  "Digital geometry processing with discrete exterior calculus."
  ACM SIGGRAPH 2013 Courses.
