# Chapter 3: Implementation

## 3.1 Overview

This chapter documents the implementation of the PQ mesh optimisation tool, translating the
mathematical formulation of Chapter 2 into a functioning software system. The implementation
is structured as a single importable Python library under the `src/` package, comprising five
subpackages: `core`, `io`, `optimisation`, `preprocessing`, and `visualisation`. All
computation is expressed in terms of NumPy arrays; Numba JIT-compiled kernels provide
optional acceleration for the planarity gradient and angle-balance gradient without altering
the public interface or numerical contract of any function (Liu et al., 2006; Lam et al., 2015).

The central data structure, described in Section 3.2, is the `QuadMesh` dataclass defined in
`src/core/mesh.py`. All modules consume and return `QuadMesh` objects or NumPy arrays;
no file paths are threaded through the computation layer. This separation of concerns ensures
that every module is independently testable and that the I/O layer can be replaced without
affecting the optimisation logic.

---

## 3.2 Core Data Structure: `QuadMesh`

The `QuadMesh` dataclass in `src/core/mesh.py` is the single canonical representation of
mesh geometry throughout the system. It holds three attributes:

- `vertices`: a NumPy array of shape `(V, 3)`, dtype `float64`, storing the Cartesian
  coordinates of all V vertices.
- `faces`: a NumPy array of shape `(F, 4)`, dtype `int64`, storing zero-indexed vertex
  indices for each of the F quadrilateral faces.
- `vertices_original`: a shape `(V, 3)` copy of the vertex positions at load time, never
  modified after initialisation. It provides the reference geometry for the closeness energy
  term $E_c = w_c \|V - V_0\|^2_F$.

Two convenience properties, `n_vertices` and `n_faces`, return the vertex and face counts
respectively. The optimiser modifies `mesh.vertices` in place; callers requiring a
non-destructive run must pass `copy.deepcopy(mesh)` to `MeshOptimiser.optimise()`.

Derived attributes — the combinatorial Laplacian matrix `laplacian`, the sparse scatter
matrix `scatter_matrix`, and the padded vertex-to-face adjacency array
`vertex_face_ids_padded` — are computed once on first access and cached as instance
attributes. This lazy initialisation ensures that meshes loaded solely for I/O (without
running optimisation) do not incur the cost of these computations.

---

## 3.3 Input/Output Layer

### 3.3.1 OBJ Handling

`src/io/obj_handler.py` exposes two functions:

```python
def load_obj(path: str) -> QuadMesh: ...
def save_obj(mesh: QuadMesh, path: str) -> None: ...
```

`load_obj` parses all `v` (vertex) and `f` (face) records from a Wavefront OBJ file,
converts face indices to zero-based integer arrays, and raises `ValueError` if any face
contains fewer or more than four vertex references. Degenerate-face validation is delegated
to `MeshOptimiser.validate_mesh()` at optimisation time rather than at load time, so that
preprocessing may correct degeneracies before they cause a hard failure. `save_obj` writes
vertices and faces back to OBJ format in original vertex order, ensuring round-trip
compatibility with external tools such as Blender and Rhino.

### 3.3.2 Panel Exporter

`src/io/panel_exporter.py` provides fabrication-ready flat-pattern export:

```python
def unfold_mesh(mesh: QuadMesh) -> tuple[list[np.ndarray], list[int]]: ...
def export_dxf(panels: list[np.ndarray], path: str) -> None: ...
def export_svg(panels: list[np.ndarray], path: str) -> None: ...
```

`unfold_mesh` projects each 3D quad face onto its least-squares best-fit plane, preserving
true edge lengths, and returns a list of 2D panel polygons alongside their corresponding
face indices. `export_dxf` and `export_svg` arrange these panels in a grid layout and write
them to CAD-compatible DXF and vector SVG files respectively, suitable for direct use in
CNC cutting or laser fabrication workflows (Pottmann et al., 2015).

---

## 3.4 Preprocessing

`src/preprocessing/preprocessor.py` exposes a single public function:

```python
def preprocess_mesh(
    mesh: QuadMesh,
    normalise: bool = True,
    verbose: bool = True,
) -> QuadMesh: ...
```

The function executes six stages in a fixed order:

1. **Bounding-box recording**: the original bounding box is stored before any modification,
   providing a reference for the normalisation step.
2. **Duplicate vertex removal**: vertices within a Euclidean distance of $10^{-8}$ are
   merged, with face indices remapped to the surviving vertex. This prevents the closeness
   energy from accumulating spurious contributions from coincident vertices.
3. **Degenerate face removal**: faces containing zero-area or repeated vertex indices are
   deleted and the face array is recompacted.
4. **Optional unit-bounding-box normalisation**: vertex coordinates are shifted and scaled
   so that the longest axis of the mesh fits within a target scale centred at the origin.
   This stabilises the weight heuristic (Section 2.5.3) by ensuring that absolute coordinate
   magnitudes do not inflate or deflate energy ratios.
5. **Cleaned mesh construction**: a new `QuadMesh` is assembled from the cleaned vertex and
   face arrays, and `vertices_original` is set to the normalised positions as the closeness
   energy baseline.
6. **Automatic weight suggestion**: `suggest_weights_for_mesh` evaluates initial energy-term
   magnitudes at the starting configuration and computes proportional weights targeting the
   ratio $E_p : E_f : E_c \approx 10 : 1 : 5$. A guard for near-planar meshes activates
   when $E_p < 10^{-10}$, returning default weights without division. The default weights,
   used when the heuristic is not applied or the near-planar guard activates, correspond to
   the `OptimisationConfig` defaults: $w_p = 100.0$, $w_f = 1.0$, $w_c = 10.0$
   (Botsch et al., 2010).

---

## 3.5 Optimisation Pipeline

### 3.5.1 Energy Functionals

A fifth module, `src/optimisation/mesh_geometry.py`, provides geometric primitive functions
(face normal computation, best-fit plane projection, and angle calculations) consumed
internally by `energy_terms.py` and `gradients.py`; it has no public API surface and is
not documented separately.

`src/optimisation/energy_terms.py` implements the four energy terms defined in Chapter 2.
All functions accept a `QuadMesh` and, where applicable, a scalar weight, and return a
scalar `float64`:

| Function | Energy Term | Formula |
|---|---|---|
| `compute_planarity_energy(mesh)` | $E_p$ | $\sum_f \sum_{v \in f} d_{f,v}^2$ |
| `compute_fairness_energy(mesh)` | $E_f$ | $\|LV\|^2_F$ |
| `compute_closeness_energy(mesh)` | $E_c$ | $\|V - V_0\|^2_F$ |
| `compute_angle_balance_energy(mesh)` | $E_a$ | $\sum_v \left(\sum_{f \ni v} \theta_{f,v} - 2\pi\right)^2$ |
| `compute_total_energy(mesh, weights)` | $E$ | Weighted sum of all four terms |

where $d_{f,v}$ is the signed distance of vertex $v$ from the least-squares best-fit plane
of face $f$ (computed via SVD of the mean-centred vertex matrix $M_f$), and $L$ is the
combinatorial umbrella Laplacian matrix (Botsch et al., 2010). $\theta_{f,v}$ denotes the
interior angle of face $f$ at vertex $v$; the angle-balance term penalises deviation of the
angle sum at each vertex from the conical condition $\sum_{f \ni v} \theta_{f,v} = 2\pi$.

### 3.5.2 Analytic Gradients and Backend Dispatch

`src/optimisation/gradients.py` computes analytic gradients for all four energy terms via a
three-tier dispatch hierarchy:

- **Tier 1 (CuPy GPU)**: batched SVD and sparse matrix multiply executed on an NVIDIA GPU
  via CuPy. Active when `HAS_CUDA` is `True` in `src/backends.py`.
- **Tier 2 (Numba CPU-parallel)**: `@njit(parallel=True, cache=True, fastmath=False)` kernels
  compiled to native machine code. The planarity gradient kernel uses Cardano closed-form
  eigenvalue extraction to avoid calling SVD inside a Numba function. The angle-balance
  gradient uses a two-pass scatter-accumulate pattern to eliminate write conflicts across
  parallel face threads. `fastmath=False` preserves floating-point associativity, ensuring
  numerical equivalence with the NumPy baseline.
- **Tier 3 (NumPy baseline)**: fully vectorised batched-SVD operations with no Python loop
  over faces. Always available as a fallback.

Each tier produces numerically equivalent results. The Numba-versus-NumPy parity is
validated in `tests/test_numerical_equivalence.py` to $10^{-10}$ for $10 \times 10$ meshes
and $10^{-8}$ for $20 \times 20$ meshes (Section 2.4.3; Higham, 2002).

Two numerical safeguards are implemented in the SciPy interface functions
`energy_for_scipy` and `gradient_for_scipy`: the former returns a sentinel energy of
$10^{300}$ on non-finite energy, and the latter replaces non-finite gradient components
individually via `numpy.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)`. Both safeguards
are never activated on well-formed inputs but ensure graceful degradation on degenerate
mesh configurations.

All four analytic gradients are verified against central finite differences with step
$\varepsilon = 10^{-6}$ in `tests/test_gradients.py` and `tests/test_gradients_extended.py`.
The maximum relative error for all four terms is below $10^{-4}$ (the configured
verification tolerance) across flat, sinusoidally perturbed, and cylindrically curved
mesh geometries (Nocedal and Wright, 2006).

### 3.5.3 Optimisation Driver

`src/optimisation/optimiser.py` provides three public exports: `OptimisationConfig`,
`OptimisationResult`, and `MeshOptimiser`. The `optimise_mesh_simple` convenience function
creates a default `OptimisationConfig` and delegates to `MeshOptimiser.optimise()`.

**Two-stage strategy.** When `two_stage=True` (the default), the optimiser executes two
sequential L-BFGS-B passes. Stage 1 multiplies the planarity weight by
`stage1_planarity_multiplier` (default 5.0) and reduces fairness and closeness weights to
10% of their standard values, using loose tolerances (`ftol=1e-7`, `gtol=1e-4`,
`maxcor=10`, `maxls=20`) to terminate quickly once faces are approximately flat. Stage 2
restores the original balanced weights and overrides the `OptimisationConfig` defaults with
tighter tolerances (`ftol=1e-9`, `gtol=1e-5`, `maxcor=20`, `maxls=40`) applied locally
within `optimise()`, warm-starting from the Stage 1 solution. The combined iteration
budget across both stages is strictly bounded by `max_iterations` (default 1,000): Stage 1
receives at most `min(200, max_iterations // 3)` iterations and Stage 2 receives the
remainder.

**Mesh validation.** Before optimisation begins, `validate_mesh()` checks that: the mesh has
at least four vertices and one face; all vertex coordinates are finite; all faces are
quadrilaterals; no face contains duplicate vertex indices; and no face has a near-zero area
(threshold $10^{-10}$). A failed validation returns an `OptimisationResult` with
`success=False` without modifying the mesh.

**Result reporting.** `OptimisationResult` records the full iteration history
(`energy_history`, `gradient_norm_history`), per-component energy breakdowns before and
after optimisation, combined iteration and function-evaluation counts across both stages,
and wall-clock execution time. The `summary()` method generates a human-readable result
report including per-goal percentage improvements and plain-English convergence diagnosis.

---

## 3.6 Visualisation

`src/visualisation/interactive_optimisation.py` provides an optional interactive viewer
built on Polyscope and Dear ImGui. It is the only module in the `src/` tree intended for
direct execution. The viewer renders the quad mesh with per-face planarity deviation
colour-mapped to a diverging scale, exposes weight sliders and a run/reset control panel
via ImGui, and rerenders the mesh after each optimisation call. The visualisation layer has
no dependencies on the optimisation layer beyond the `QuadMesh` dataclass and the
`optimise_mesh_simple` function; it can be replaced or omitted without affecting the core
library.

---

## 3.7 Testing Architecture

The test suite in `tests/` comprises 20 test modules (229 tests total, 1 skipped; overall
coverage ≥79% of `src/`, excluding `interactive_optimisation.py`). The principal modules
relevant to the implementation claims of this chapter are:

| Test Module | Scope |
|---|---|
| `test_gradients.py` | Gradient verification: $E_p$, $E_f$, $E_c$ analytic vs. finite difference |
| `test_gradients_extended.py` | Gradient verification: $E_a$ (angle-balance); SciPy interface safeguards; extended geometries |
| `test_numerical_equivalence.py` | Numba-versus-NumPy parity for all four gradients (229 tests) |
| `test_energy_terms.py` | Unit tests for all five energy functions |
| `test_optimiser.py` | Integration tests for `MeshOptimiser` and `OptimisationConfig` |
| `test_preprocessor.py` | Unit tests for all six preprocessing stages and weight suggestion |
| `test_robustness.py` | Degenerate-input and boundary-condition handling |
| `test_geometry.py` | Geometric primitive correctness (face normals, best-fit planes, angle calculations) |

All tests are executable via `pytest tests/` from the repository root. The TESTING_GUIDE.md
in `tests/` documents the full suite structure, expected pass criteria, and instructions for
running individual modules in isolation.

---

## References

- Botsch, M., Kobbelt, L., Pauly, M., Alliez, P., and Lévy, B. (2010). *Polygon Mesh Processing*. AK Peters.
- Crane, K., de Goes, F., Desbrun, M., and Schröder, P. (2013). Digital geometry processing with discrete exterior calculus. *ACM SIGGRAPH 2013 Courses*.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*. 2nd ed. SIAM.
- Lam, S. K., Pitrou, A., and Seibert, S. (2015). Numba: a LLVM-based Python JIT compiler. *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*, pp. 1–6.
- Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006). Geometric modelling with conical meshes and developable surfaces. *ACM Transactions on Graphics*, 25(3), pp. 681–689.
- Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*. 2nd ed. Springer.
- Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015). Architectural geometry. *Computers and Graphics*, 47, pp. 145–164.
