# Appendix E: Software Architecture Reference

## E.1 Package Structure

The `src/` package is organised into five subpackages. All modules are importable
Python libraries; none are intended for direct execution with the sole exception of
`src/visualisation/interactive_optimisation.py`.

```
src/
├── __init__.py
├── backends.py                        # Numba/CuPy/NumPy backend selector
├── OVERVIEW.md
├── core/
│   ├── __init__.py
│   └── mesh.py                        # QuadMesh dataclass
├── io/
│   ├── __init__.py
│   ├── obj_handler.py                 # OBJ reader and writer
│   └── panel_exporter.py              # Mesh unfolding; DXF and SVG export
├── optimisation/
│   ├── __init__.py
│   ├── energy_terms.py                # Four energy functionals
│   ├── gradients.py                   # Analytic gradient computation
│   ├── mesh_geometry.py               # Low-level geometric primitives
│   └── optimiser.py                   # OptimisationConfig, MeshOptimiser
├── preprocessing/
│   ├── __init__.py
│   └── preprocessor.py                # Normalisation, deduplication, validation
└── visualisation/
    ├── __init__.py
    ├── interactive_optimisation.py    # Polyscope viewer and ImGui control panel
    └── VISUALISATION_GUIDE.md
```

---

## E.2 Module API Reference

### E.2.1 `src/core/mesh.py` — QuadMesh

```python
@dataclass
class QuadMesh:
    vertices: np.ndarray            # (V, 3), float64 — current vertex positions
    faces: np.ndarray               # (F, 4), int64   — zero-indexed face indices
    vertices_original: np.ndarray   # (V, 3), float64 — reference positions (never modified)

    # Derived attributes (lazy, cached on first access)
    laplacian: scipy.sparse.csr_matrix   # (V, V) combinatorial Laplacian
    scatter_matrix: scipy.sparse.csr_matrix  # (V, F*4) face-to-vertex scatter
    vertex_face_ids_padded: np.ndarray   # (V, max_valence) padded adjacency

    @property
    def n_vertices(self) -> int: ...
    @property
    def n_faces(self) -> int: ...
```

The optimiser modifies `mesh.vertices` in place. Callers requiring a non-destructive run
must pass `copy.deepcopy(mesh)` to `MeshOptimiser.optimise()`.

---

### E.2.2 `src/io/obj_handler.py`

```python
def load_obj(path: str) -> QuadMesh: ...
def save_obj(mesh: QuadMesh, path: str) -> None: ...
```

`load_obj` raises `ValueError` if any face contains fewer or more than four vertex
references. Face indices are converted to zero-based integer arrays. `save_obj` preserves
original vertex order for round-trip compatibility with Blender and Rhino.

---

### E.2.3 `src/io/panel_exporter.py`

```python
def unfold_mesh(mesh: QuadMesh) -> tuple[list[np.ndarray], list[int]]: ...
def export_dxf(panels: list[np.ndarray], path: str) -> None: ...
def export_svg(panels: list[np.ndarray], path: str) -> None: ...
```

`unfold_mesh` projects each 3D quad face onto its least-squares best-fit plane, preserving
true edge lengths. `export_dxf` and `export_svg` arrange 2D panels in a grid layout for
CNC or laser fabrication workflows.

---

### E.2.4 `src/optimisation/energy_terms.py`

| Function | Returns | Notes |
|---|---|---|
| `compute_planarity_energy(mesh)` | `float64` | $\sum_f \sum_{v \in f} d_{f,v}^2$ where $d_{f,v}$ is the signed distance of vertex $v$ from the SVD best-fit plane of face $f$ |
| `compute_fairness_energy(mesh)` | `float64` | $\|LV\|^2_F$ (combinatorial umbrella Laplacian) |
| `compute_closeness_energy(mesh)` | `float64` | $\|V - V_0\|^2_F$ |
| `compute_angle_balance_energy(mesh)` | `float64` | $\sum_v \left(\sum_{f \ni v} \theta_{f,v} - 2\pi\right)^2$ (conical vertex constraint) |
| `compute_planarity_per_face(mesh)` | `np.ndarray (F,)` | Per-face signed-distance RMS |
| `compute_total_energy(mesh, weights)` | `float64` | Weighted sum |

---

### E.2.5 `src/optimisation/gradients.py`

```python
def compute_planarity_gradient(mesh) -> np.ndarray: ...   # (V, 3)
def compute_fairness_gradient(mesh) -> np.ndarray: ...    # (V, 3)
def compute_closeness_gradient(mesh) -> np.ndarray: ...   # (V, 3)
def compute_angle_balance_gradient(mesh) -> np.ndarray: ...  # (V, 3)
def compute_total_gradient(mesh, weights) -> np.ndarray: ... # (V, 3)
def energy_for_scipy(x, mesh, weights) -> float: ...      # SciPy interface
def gradient_for_scipy(x, mesh, weights) -> np.ndarray: ... # SciPy interface
def verify_gradient(mesh, weights, tolerance=1e-4) -> dict: ...
```

**Backend dispatch order:** Tier 1 CuPy GPU → Tier 2 Numba CPU-parallel → Tier 3 NumPy
baseline. The fairness gradient has no Tier 2 path (SciPy sparse BLAS is sufficient).
The angle-balance gradient has no Tier 1 path (irregular adjacency structure precludes
efficient GPU batching).

**Numerical safeguards in SciPy interface:**

| Function | Safeguard | Trigger |
|---|---|---|
| `energy_for_scipy` | Returns sentinel $10^{300}$ | Non-finite energy value |
| `gradient_for_scipy` | `nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)` | Non-finite gradient components |

Both safeguards are never activated on well-formed inputs.

---

### E.2.6 `src/optimisation/optimiser.py`

```python
@dataclass
class OptimisationConfig:
    weights: dict[str, float]
    # Required keys: "planarity", "fairness", "closeness"
    # Optional key:  "angle_balance" (default 0.0)
    max_iterations: int = 1000
    tolerance: float = 1e-6           # Stage 1 ftol; Stage 2 uses 1e-9
    gradient_tolerance: float = 1e-4  # Stage 1 gtol; Stage 2 uses 1e-5
    two_stage: bool = True
    stage1_planarity_multiplier: float = 5.0
    bounds_scale: Optional[float] = None
    verbose: bool = True
    history_tracking: bool = True
```

**L-BFGS-B hyperparameters by stage:**

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| `ftol` | $10^{-7}$ | $10^{-9}$ |
| `gtol` | $10^{-4}$ | $10^{-5}$ |
| `maxcor` | 10 | 20 |
| `maxls` | 20 | 40 |
| `maxiter` | `min(200, max_iter // 3)` | remainder |

Stage 1 weights: $w_p \times 5.0$, $w_f \times 0.1$, $w_c \times 0.1$.
Stage 2 weights: original values restored.

```python
@dataclass
class OptimisationResult:
    success: bool
    message: str
    optimised_mesh: QuadMesh
    initial_energy: float
    final_energy: float
    n_iterations: int
    n_function_evaluations: int
    n_gradient_evaluations: int
    execution_time: float
    energy_history: Optional[list[float]]
    gradient_norm_history: Optional[list[float]]
    component_energies_initial: Optional[dict[str, float]]
    component_energies_final: Optional[dict[str, float]]

    def energy_reduction(self) -> float: ...
    def energy_reduction_percentage(self) -> float: ...
    def summary(self) -> str: ...
```

---

### E.2.7 `src/preprocessing/preprocessor.py`

```python
def preprocess_mesh(
    mesh: QuadMesh,
    normalise: bool = True,
    verbose: bool = True,
) -> QuadMesh: ...

def suggest_weights(mesh: QuadMesh) -> dict[str, float]: ...
```

**Pre-processing stages (executed in fixed order):**

1. Bounding-box recording (stores original extents before any modification)
2. Duplicate vertex removal (merge threshold: $10^{-8}$; O(n²), warns if $n > 2000$)
3. Degenerate face removal (faces with zero area or repeated vertex indices)
4. Unit-bounding-box normalisation (optional, `normalise=True`; scales longest axis to target scale, centres at origin)
5. Cleaned `QuadMesh` construction; sets `vertices_original` to normalised positions
6. Automatic weight suggestion via `suggest_weights_for_mesh` (Stage 6; see below)

**Weight suggestion heuristic:**

- Target ratio: $E_p : E_f : E_c \approx 10 : 1 : 5$
- Near-planar guard: activates when $E_p < 10^{-10}$; returns default weights without division
- Default weights: $w_p = 100.0$, $w_f = 1.0$, $w_c = 10.0$

---

## E.3 Backend Availability Flags (`src/backends.py`)

| Flag | Type | Meaning |
|---|---|---|
| `HAS_NUMBA` | `bool` | Numba successfully imported and kernel compilation succeeded |
| `HAS_CUDA` | `bool` | CuPy available and CUDA device detected |
| `warmup_numba_kernels()` | function | Pre-compiles all Numba kernels to eliminate first-use JIT stall |

The dispatch logic in every gradient function checks `HAS_CUDA` before `HAS_NUMBA`,
falling back to NumPy if neither is available. Backend selection is transparent: the
public interface of every function is identical across all three tiers.

---

## E.4 Test Suite Reference

All 20 test modules (229 tests total, 1 skipped; coverage ≥79% of `src/`) are executable via `pytest tests/` from the repository root.

| Module | Primary Scope |
|---|---|
| `test_gradients.py` | $E_p$, $E_f$, $E_c$ analytic vs. finite-difference verification |
| `test_gradients_extended.py` | $E_a$ verification; extended mesh geometries |
| `test_numerical_equivalence.py` | Numba-versus-NumPy parity ($10^{-10}$ at $10\times10$; $10^{-8}$ at $20\times20$) |
| `test_energy_terms.py` | Unit tests for all five energy functions |
| `test_optimiser.py` | Integration tests for `MeshOptimiser`, `OptimisationConfig`, two-stage logic |
| `test_preprocessor.py` | Unit tests for all six preprocessing stages and weight suggestion heuristic |
| `test_geometry.py` | Low-level geometric primitive correctness (face normals, best-fit planes, angle calculations) |
| `test_robustness.py` | Degenerate-input and boundary-condition handling |
| `test_scalability.py` | Timing benchmarks across mesh sizes |
| `test_coverage_extended.py` | Branch coverage for backend fallback paths |
| `test_coverage_gaps.py` | Additional branch and edge-case coverage |
| `test_backends.py` | Backend availability detection and warmup |
| `test_mesh.py` | `QuadMesh` construction and property validation |
| `test_obj_handler.py` | OBJ round-trip fidelity |
| `test_obj_handler_extended.py` | Malformed OBJ error handling |
| `test_panel_exporter.py` | Unfolding geometry and DXF/SVG output |
| `test_error_handling.py` | Exception path coverage |
| `test_vertex_face_ids.py` | Adjacency array construction |
| `test_quad_loading.py` | Quad-only enforcement on load |
| `test_quad_topology_preservation.py` | Face index integrity through optimisation |

Full instructions for running individual modules in isolation are documented in
`tests/TESTING_GUIDE.md`.

---

## E.5 Typical Usage Pattern

```python
from src.io.obj_handler import load_obj, save_obj
from src.preprocessing.preprocessor import preprocess_mesh
from src.optimisation.optimiser import optimise_mesh_simple

mesh = load_obj("data/input/generated/plane_5x5_noisy.obj")
mesh = preprocess_mesh(mesh, normalise=True, verbose=True)

result = optimise_mesh_simple(
    mesh,
    weights={"planarity": 100.0, "fairness": 1.0, "closeness": 10.0},
    max_iter=1000,
    verbose=True,
)

print(result.summary())
save_obj(mesh, "data/output/optimised_meshes/result.obj")
```
