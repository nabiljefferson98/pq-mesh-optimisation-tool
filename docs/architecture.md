# System Architecture

**PQ Mesh Optimisation — Software Architecture Reference**

---

## Overview

The system is structured as a layered library with a clean separation between
the core data model, the mathematical optimisation engine, I/O adapters, and
user-facing scripts. No layer depends on a layer above it.

```
┌────────────────────────────────────────────────────────────────────────┐
│  User Interface Layer                                                  │
│  scripts/                                                              │
│  (runnable scripts; import src/ but are never imported by it)          │
├────────────────────────────────────────────────────────────────────────┤
│  I/O & Visualisation Layer                                             │
│  src/io/obj_handler.py      — OBJ reader/writer                        │
│  src/io/panel_exporter.py   — DXF / SVG flat-panel export              │
│  src/visualisation/interactive_optimisation.py — Polyscope 3D viewer  │
├────────────────────────────────────────────────────────────────────────┤
│  Hardware Backend Layer                                                │
│  src/backends.py            — CUDA (cupy) · parallel (numba) · serial  │
├────────────────────────────────────────────────────────────────────────┤
│  Preprocessing Layer                                                   │
│  src/preprocessing/preprocessor.py                                     │
│  (normalisation · degenerate-face removal · weight suggestion)         │
├────────────────────────────────────────────────────────────────────────┤
│  Optimisation Engine                                                   │
│  src/optimisation/energy_terms.py  — E_p, E_f, E_c, E_a               │
│  src/optimisation/gradients.py     — ∂E/∂V analytical                 │
│  src/optimisation/mesh_geometry.py — geometric primitives              │
│  src/optimisation/optimiser.py     — L-BFGS-B driver                  │
├────────────────────────────────────────────────────────────────────────┤
│  Backend Acceleration Layer                                            │
│  src/backends.py  — HAS_CUDA / HAS_NUMBA flags; dispatch helpers      │
├────────────────────────────────────────────────────────────────────────┤
│  Core Data Model                                                       │
│  src/core/mesh.py  — QuadMesh · scatter_matrix · topology caches      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `src/core/mesh.py` — `QuadMesh`

The single shared data structure threaded through the entire pipeline.

| Attribute | Type | Description |
|-----------|------|-------------|
| `vertices` | `ndarray (n, 3)` | Current vertex positions (mutated in-place during optimisation) |
| `faces` | `ndarray (m, 4)` | Vertex index quads (immutable after construction) |
| `vertices_original` | `ndarray (n, 3)` | Baseline positions for closeness energy |
| `scatter_matrix` | `csr_matrix (n, m·4)` | Lazy sparse accumulation matrix for gradient scatter-add |
| `vertex_face_ids_padded` | `ndarray (n, k)` | Padded lookup table of face IDs per vertex for Numba acceleration |

**Lazy topology caches** (`_vertex_faces`, `_vertex_neighbors`, `_scatter_matrix`) are
built on first access and reused for the lifetime of the object. They are invalidated
only if `faces` changes (which never happens post-construction).

**Validation on construction:**
- vertices must be `(n, 3)` with `n ≥ 1`
- face indices must be non-negative and within `[0, n)`
- `update_vertices()` rejects NaN / Inf inputs

---

### `src/backends.py`

Provides runtime backend detection and dispatch helpers consumed by all
energy and gradient modules.

| Symbol | Type | Description |
|--------|------|-------------|
| `HAS_CUDA` | `bool` | `True` when CuPy is importable and a CUDA device is available |
| `HAS_NUMBA` | `bool` | `True` when Numba is importable |
| `to_device(arr)` | function | Copies a NumPy array to GPU (CuPy); no-op on CPU |
| `to_numpy(arr)` | function | Copies a CuPy array back to NumPy; no-op if already NumPy |
| `gpu_memory_guard()` | context manager | Catches `OutOfMemoryError` / `CUDARuntimeError`; sets `HAS_CUDA=False` and falls back to NumPy |

**Three-tier dispatch pattern** (used in every hot-path function):

```python
try:
    if HAS_CUDA:
        return _function_gpu(...)       # Tier 1: CuPy GPU
except ImportError:
    pass

try:
    if HAS_NUMBA:
        return _function_numba(...)     # Tier 2: Numba CPU-parallel
except Exception as _exc:
    warnings.warn(...)                  # Broad catch — Numba can raise non-ImportError

return _function_numpy(...)             # Tier 3: NumPy baseline (always available)
```

Note: Numba `try` blocks must use `except Exception` (not merely `except ImportError`) because
Numba `@njit` decoration can raise `numba.core.errors.TypingError`, `LoweringError`, and LLVM
compilation errors on platforms where Numba is installed but compilation fails. Narrowing to
`ImportError` causes a `NameError` when the decorated function is referenced in the fallback
path if compilation was attempted but failed.

---

### `src/optimisation/energy_terms.py`

Four additive energy components:

| Symbol | Function | Formula |
|--------|----------|---------|
| $E_p$ | `compute_planarity_energy` | $\sum_f \sum_{v \in f} d_{f,v}^2$ where $d_{f,v}$ = signed distance from face best-fit plane (SVD) |
| $E_f$ | `compute_fairness_energy` | $\sum_v \|\Delta v\|^2$ discrete Laplacian (umbrella operator) |
| $E_c$ | `compute_closeness_energy` | $\sum_v \|v - v^0\|^2$ L₂ fidelity to original positions |
| $E_a$ | `compute_angle_balance_energy` | $\sum_v (\sum_{f \ni v} \theta_{f,v} - 2\pi)^2$ conical vertex constraint |

Total energy:

$$E = w_p E_p + w_f E_f + w_c E_c + w_a E_a$$

All functions are **fully vectorised** (no Python loops over faces) using batched
`np.linalg.svd` on the face-vertex tensor of shape `(F, 4, 3)`. When `cupy` is
available, computations are dispatched to the GPU; otherwise, `numba` provides
parallel CPU acceleration where applicable.

---

### `src/backends.py` — Hardware Abstraction

Modular backend detection and routing:

| Backend | Mode | Requirements |
|---------|------|--------------|
| `cupy`  | NVIDIA GPU | `pip install cupy-cuda13x` |
| `numba` | Parallel CPU | `pip install numba` |
| `numpy` | Serial CPU | Baseline (no extra deps) |

Automatic detection follows the hierarchy `cupy` → `numba` → `numpy`. Users can
override via `PQ_BACKEND=numpy`.

**Planarity energy backend dispatch (added 15 Mar 2026):**

| Tier | Backend | Function | Notes |
|------|---------|----------|-------|
| 1 | CuPy GPU | `_planarity_energy_gpu` | Batched CuPy SVD; sparse GPU matmul |
| 2 | Numba CPU | `_planarity_energy_numba` | `@njit(parallel=True, cache=True, fastmath=False)`; `prange` over faces |
| 3 | NumPy | inline vectorised SVD | Validated reference; always available |

The Numba kernel `_planarity_energy_numba` calls the inner kernel
`_planarity_per_face_numba` which computes a per-face 4×3 thin SVD using
hand-unrolled Householder reduction, avoiding any heap allocation inside the JIT region.
`fastmath=False` preserves float associativity for numerical equivalence with the NumPy
baseline (required by `test_numerical_equivalence.py` — planarity energy + gradient + angle-balance Numba equivalence, 229 tests total). `cache=True` eliminates JIT
compile overhead on repeated runs by persisting compiled artefacts to `__pycache__`.

---

### `src/optimisation/gradients.py`

Provides analytical $\nabla_V E$ for all four energy terms. Gradients are
derived via the chain rule through the SVD and assembled into an
`(n_vertices, 3)` array via **sparse scatter-add**:

```
grad = scatter_matrix @ per_face_contributions   # BLAS-backed matmul
```

This replaces `np.add.at` and delivers the vectorised accumulation without
a Python loop.

**scipy interface:**

| Function | Role |
|----------|------|
| `energy_for_scipy(x, mesh, weights)` | Flattens `x` → `mesh.vertices`, returns scalar |
| `gradient_for_scipy(x, mesh, weights)` | Same reshape, returns flattened `(n·3,)` gradient |

Both functions guard against NaN/Inf propagation: non-finite results are
replaced by safe fallbacks (`1e300` / zeros) with a warning rather than
propagating into the L-BFGS-B line search.

**Planarity gradient backend dispatch (added 15 Mar 2026):**

`compute_planarity_gradient` uses the same three-tier dispatch as the energy path:

| Tier | Backend | Function | Notes |
|------|---------|----------|-------|
| 1 | CuPy GPU | `_planarity_gradient_gpu` | GPU SVD + CuPy sparse scatter-add |
| 2 | Numba CPU | `_planarity_gradient_contributions_numba` | `@njit(parallel=True, cache=True, fastmath=False)`; returns `(F, 4, 3)` contribution tensor; caller applies `scatter_matrix @` |
| 3 | NumPy | inline batched SVD + einsum | Validated reference; always available |

Numerical equivalence validated by `TestPlanarityGradientNumbaEquivalence`
(10 tests). Tolerance: 1e-10 for meshes ≤10×10; 1e-8 for 20×20
(LAPACK/LLVM SVD rounding divergence, deterministic).

---

**Numba angle-balance gradient (updated 15 Mar 2026):**

The Numba try/except block for `_angle_balance_gradient_numba` was broadened
from `except ImportError` to `except Exception as _numba_grad_exc`. This is
necessary because Numba can raise `TypingError`, `LoweringError`, and other
non-`ImportError` exceptions during `@njit` decoration on platforms where
LLVM compilation fails. The narrower clause caused a `NameError` when the
decorated name was referenced after a failed (non-`ImportError`) compilation,
breaking 7 tests. The `_ANGLE_SIGNS` module-level constant was also changed
to a typed `np.float64` array to satisfy the Numba closure type-inference
requirement.

---

### `src/optimisation/optimiser.py`

**`OptimisationConfig`** — all tunable hyperparameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `weights` | `{planarity: 100, fairness: 1, closeness: 10}` | Relative importance of energy terms |
| `max_iterations` | 1000 | Hard iteration cap |
| `tolerance` (ftol) | 1e-9 | Relative energy-change convergence |
| `gradient_tolerance` (gtol) | 1e-5 | Gradient-norm convergence (primary criterion) |

**`MeshOptimiser.optimise(mesh)`** pipeline:

```
validate_mesh()
  └─ size check · duplicate vertex check · zero-area face check · NaN/Inf check

store vertices_original (if not already set)

_initial_raw = compute_total_energy(mesh, weights)   # snapshot — must be float
assert isinstance(_initial_raw, float), ...          # active contract guard (added 15 Mar)

scipy.optimize.minimize(
    fun  = energy_for_scipy,
    jac  = gradient_for_scipy,
    method = 'L-BFGS-B',
    options = {maxcor: 20, maxls: 40, ftol: 1e-9, gtol: 1e-5}
)

update mesh.vertices = result.x.reshape(-1, 3)

_final_raw = compute_total_energy(mesh, weights)     # snapshot — must be float
assert isinstance(_final_raw, float), ...            # active contract guard (added 15 Mar)

return OptimisationResult
```

**Return-type contract guards (added 15 Mar 2026):**

The `isinstance(energy_raw, tuple)` checks that previously guarded the initial and final
energy snapshots were dead code: `compute_total_energy` only returns a `tuple` when called
with `return_components=True`, which is never the case in these two call sites. Dead code
of this form masks future API breakage silently. Both checks were replaced with active
`assert isinstance(_initial_raw / _final_raw, float)` guards that raise `AssertionError`
with a descriptive message if the return-type contract is ever violated.

The **callback** (used when `history_tracking=True`) temporarily copies vertex
positions, evaluates energy + gradient for logging, then **always restores**
the original positions via `try/finally` — mesh state is never corrupted by a
failing callback.

**CLI output** — when `verbose=True` the optimiser prints three sections:

1. **Start banner** — mesh size, weight settings, and per-goal starting scores
   in plain English (e.g. "Panel flatness score: 12.3456").
2. **Per-step progress** (every 10 iterations) — two lines:
   ```
   Step   10: score = 141.03,  rate of change = 0.6722,  time elapsed = 1.75s
          (technical: iteration 10,  energy E = 1.410300e+02,  || gradient E || = 6.7220e-04)
   ```
   The first line uses plain terms; the second gives the academic equivalents
   (`energy E` and gradient norm `|| gradient E ||`).
3. **Results summary** (`OptimisationResult.summary()`) — status in plain
   English (`FINISHED SUCCESSFULLY`, `STEP LIMIT REACHED`, `FAILED`,
   `NEARLY THERE`, `PARTIAL IMPROVEMENT`), overall score before/after, step
   counts, and a per-goal breakdown with readable labels
   (`Panel flatness`, `Surface smoothness`, `Shape fidelity`, `Corner balance`).

---

### `src/preprocessing/preprocessor.py`

Six-stage pipeline run before any mesh reaches the optimiser:

| Stage | Action |
|-------|--------|
| 1 | Record original bounding box |
| 2 | Merge duplicate vertices (O(n²), warns if n > 2000) |
| 3 | Remove zero-area faces |
| 4 | Normalise: centre at origin, scale longest axis → `target_scale` |
| 5 | Build cleaned `QuadMesh`; set `vertices_original` baseline |
| 6 | Auto-suggest energy weights via `suggest_weights_for_mesh` |

**Weight suggestion** targets ratio $E_p : E_f : E_c \approx 10 : 1 : 5$ at the
start of optimisation, with guard logic for already-planar meshes (near-PQ by
construction).

---

### `src/io/obj_handler.py`

- Handles mixed-topology OBJ files (triangles + quads in the same file)
- Skips malformed vertex / face tokens with per-line warnings (capped at 5)
- Pairs adjacent triangle faces into quads via shared-edge analysis
- `save_obj` rejects path-traversal inputs (`..` in resolved path parts)

---

### `src/io/panel_exporter.py`

Converts an optimised PQ mesh to fabrication-ready 2D layouts:

```
unfold_mesh(mesh)
  └─ per face: SVD best-fit plane → local {û, v̂} frame → project to 2D → FlatPanel

export_svg(panels, path)     ─┐ atomic write via tempfile + os.replace
export_dxf(panels, path)     ─┘ layer_name sanitised: [A-Za-z0-9_\-], ≤255 chars
```

---

### `src/visualisation/interactive_optimisation.py`

Polyscope-based interactive 3D viewer script. Displays:
- Original and optimised meshes side-by-side
- Planarity heatmap per face
- Conical angle imbalance heatmap per vertex
- Interactive weight sliders (via `imgui`) that re-run optimisation live

This script is excluded from coverage measurement (`pyproject.toml`) because it
requires a display and Polyscope context unavailable in CI.

---

## Data Flow Diagram

```
  ┌──────────┐    load_obj     ┌──────────┐   preprocess_mesh   ┌──────────────────┐
  │  .obj    │ ─────────────► │ QuadMesh │ ──────────────────► │  QuadMesh        │
  │  file    │                 │ (raw)    │                      │  (normalised)    │
  └──────────┘                 └──────────┘                      └────────┬─────────┘
                                                                          │
                                                                          ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  MeshOptimiser.optimise(mesh)                                                │
  │                                                                              │
  │  scipy.minimize(L-BFGS-B)                                                   │
  │    ├── energy_for_scipy  ←  E_p + E_f + E_c + E_a  (three-tier dispatch)    │
  │    └── gradient_for_scipy ← ∂E/∂V  (analytical, sparse scatter-add)        │
  └──────────────────────────────────────────────────────────────────────────────┘
                                    │
                       OptimisationResult
                                    │
               ┌────────────────────┼─────────────────────┐
               ▼                    ▼                       ▼
         save_obj            unfold_mesh             interactive_optimisation
         (.obj)          export_dxf / export_svg    (Polyscope)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Vertices mutated in-place during scipy calls | Avoids re-allocation on every function evaluation; safe because L-BFGS-B is single-threaded |
| Sparse scatter matrix built once | Accumulating per-face gradient contributions into per-vertex array is `O(F·4)` — doing it as a sparse matmul is 2–4× faster than `np.add.at` on large meshes |
| Batched SVD via `np.linalg.svd` on `(F, 4, 3)` tensor | Eliminates Python loop over faces; vectorised LAPACK batching |
| Numba planarity kernel with `fastmath=False` | Preserves float associativity for numerical equivalence with the NumPy baseline; validated by `test_numerical_equivalence.py` |
| Numba `try` blocks use `except Exception` | `ImportError` alone is insufficient: Numba `@njit` can raise `TypingError` / `LoweringError` / LLVM errors on platforms where Numba is installed but compilation fails; broad catch ensures graceful degradation |
| Active `assert isinstance(..., float)` guards on energy snapshots | Replaces dead `isinstance(x, tuple)` checks; enforces return-type contract and surfaces future API breakage immediately rather than silently computing a wrong result |
| Atomic file writes (tempfile + `os.replace`) | Prevents corrupt output files if the process is interrupted mid-write |
| Pre-commit hooks (black · isort · flake8 · bandit) | Enforces consistent style and catches common security issues on every commit |

---

## Test Coverage

```
tests/
├── test_mesh.py                    — QuadMesh construction, topology queries
├── test_energy_terms.py            — Energy function correctness + weights
├── test_gradients.py               — Analytical vs numerical gradient agreement
├── test_gradients_extended.py      — Angle-balance gradient, scipy interface
├── test_optimiser.py               — Optimisation pipeline end-to-end
├── test_obj_handler.py             — OBJ round-trip, edge cases
├── test_obj_handler_extended.py    — Mixed topology, pairing, path safety
├── test_panel_exporter.py          — Unfold + SVG/DXF export
├── test_preprocessor.py            — Preprocessing stages, weight suggestion
├── test_quad_topology_preservation.py — Quad-vs-triangle loading consistency
├── test_quad_loading.py            — igl quad-vs-triangle loading comparison (uses tmp_path)
├── test_geometry.py                — Geometric primitive correctness
├── test_error_handling.py          — Error paths and mesh validation
├── test_coverage_extended.py       — Branch coverage for config/result helpers
├── test_scalability.py             — 20×20 grid performance regression
├── test_robustness.py              — Regression tests for all audit fixes
│                                     (B1–B11 · S1–S2 · E1–E3)
├── test_numerical_equivalence.py   — Numba vs NumPy numerical equivalence validation
│                                     (fastmath=False contract; angle-balance Numba parity;
│                                      planarity gradient Numba parity — 10 tests, 15 Mar 2026)
└── conftest.py                     — Shared fixtures
```

**229 tests, 1 skipped** (GUI-dependent) | **Coverage**: ≥79% (src/, excl. `interactive_optimisation.py`)
