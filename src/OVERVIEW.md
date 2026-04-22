# src/ — Source Package Overview

This directory is the root of the project's core library. All subpackages are importable Python modules; none of the files in this tree are intended to be executed directly (with the sole exception of `src/visualisation/interactive_optimisation.py`, which provides an optional interactive viewer).

---

## Package tree

```
src/
├── __init__.py
├── backends.py                   # Numba / NumPy backend selector and JIT kernels
├── OVERVIEW.md                   # This file
├── core/
│   ├── __init__.py
│   └── mesh.py                   # QuadMesh dataclass
├── io/
│   ├── __init__.py
│   ├── obj_handler.py            # OBJ reader and writer
│   └── panel_exporter.py         # Mesh unfolding; DXF and SVG export
├── optimisation/
│   ├── __init__.py
│   ├── energy_terms.py           # Four energy functionals
│   ├── gradients.py              # Analytic gradient computation
│   ├── mesh_geometry.py          # Low-level geometric primitives
│   └── optimiser.py              # OptimisationConfig, OptimisationResult, MeshOptimiser
├── preprocessing/
│   ├── __init__.py
│   └── preprocessor.py           # Mesh normalisation, deduplication, validation
└── visualisation/
    ├── __init__.py
    ├── interactive_optimisation.py  # Polyscope viewer and ImGui control panel
    └── VISUALISATION_GUIDE.md
```

---

## Module reference

### `src/backends.py`

Provides a transparent backend abstraction that selects between a Numba JIT-compiled tier and a pure-NumPy fallback tier at import time, based on whether `numba` is installed and importable.

Key exports:

| Symbol | Type | Description |
| ------ | ---- | ----------- |
| `HAS_NUMBA` | `bool` | `True` if Numba is available and has been successfully imported |
| `warmup_numba_kernels()` | function | Pre-compiles all Numba kernels to avoid first-use JIT stalls |
| `compute_planarity_numba(vertices, faces, face_mask)` | function | Numba-accelerated planarity deviation per face |
| `compute_planarity_gradient_numba(vertices, faces, face_vertex_map)` | function | Numba-accelerated planarity gradient |

The module uses `@numba.njit(cache=True, parallel=True)` for the two-pass angle-balance gradient kernel and `@numba.njit(cache=True)` with Cardano closed-form eigenvalue extraction for the planarity gradient kernel.

---

### `src/core/mesh.py`

Defines the single central data structure used throughout the project.

```python
@dataclass
class QuadMesh:
    vertices: np.ndarray          # shape (V, 3), float64
    faces: np.ndarray             # shape (F, 4), int32/int64, zero-indexed
    vertices_original: np.ndarray # shape (V, 3), copy of vertices at load time

    @property
    def n_vertices(self) -> int: ...
    @property
    def n_faces(self) -> int: ...
```

`vertices_original` is set once at load time and is never modified. It provides the reference geometry for the closeness energy term `E_closeness = w_c * ||V - V_0||^2_F`.

---

### `src/io/obj_handler.py`

```python
def load_obj(path: str) -> QuadMesh: ...
def save_obj(mesh: QuadMesh, path: str) -> None: ...
```

`load_obj` reads a Wavefront OBJ file, extracts all `v` (vertex) and `f` (face) records, converts faces to zero-based indexing, and validates that all faces are quadrilaterals. It raises `ValueError` if the file contains non-quad faces. `save_obj` writes the mesh back to OBJ format, preserving vertex order.

---

### `src/io/panel_exporter.py`

Provides mesh unfolding and panel export to fabrication-ready flat formats.

```python
def unfold_mesh(mesh: QuadMesh) -> tuple[list[np.ndarray], list[int]]: ...
def export_dxf(panels: list[np.ndarray], path: str) -> None: ...
def export_svg(panels: list[np.ndarray], path: str) -> None: ...
```

`unfold_mesh` returns a list of 2D panel polygons (one per face) and a list of face indices. Each panel is obtained by projecting the 3D face vertices onto the plane of least-squares fit, preserving true edge lengths. `export_dxf` and `export_svg` write the 2D panels to CAD-ready DXF and vector SVG files respectively, with panels arranged in a grid layout.

---

### `src/optimisation/energy_terms.py`

Implements the four energy functionals that constitute the total objective:

| Function | Energy term | Formula |
| -------- | ----------- | ------- |
| `compute_planarity_energy(mesh, weight)` | `E_planarity` | `w_p * sum_f sigma_min(M_f)^2` |
| `compute_fairness_energy(mesh, weight)` | `E_fairness` | `w_f * ||L V||^2_F` |
| `compute_closeness_energy(mesh, weight)` | `E_closeness` | `w_c * ||V - V_0||^2_F` |
| `compute_angle_balance_energy(mesh, weight)` | `E_angle` | `w_a * sum_v (sum_f alpha_{v,f})^2` |
| `compute_planarity_per_face(mesh)` | diagnostic | `sigma_min(M_f)` per face, unweighted |
| `compute_total_energy(mesh, weights)` | combined | Weighted sum of all four terms |

where `M_f` is the (4 x 3) matrix of mean-centred face vertex positions and `sigma_min` is its smallest singular value. `L` is the combinatorial Laplacian matrix (degree matrix minus adjacency matrix).

---

### `src/optimisation/gradients.py`

Provides analytic gradient computation for the total energy functional and supporting diagnostic utilities.

```python
def compute_total_gradient(mesh, weights, backend="auto") -> np.ndarray: ...
def verify_gradient(mesh, weights, eps=1e-7, sample_size=10) -> dict: ...
def compute_gradient_statistics(mesh, weights) -> dict: ...
def print_gradient_analysis(mesh, weights) -> None: ...
```

`compute_total_gradient` returns a flat `(3V,)` array — the concatenated gradient of the total energy with respect to all vertex coordinates. It dispatches to either the Numba or NumPy backend based on the `backend` argument (`"auto"`, `"numba"`, or `"numpy"`).

`verify_gradient` performs a finite-difference gradient check on a random sample of `sample_size` coordinate components and returns a dict containing the maximum relative error and a boolean `passed` flag. The threshold for passing is a relative error below `1e-4`.

`suggest_weight_scaling(mesh)` analyses the per-component gradient magnitudes at the initial point and returns a dict of recommended weight values that balance gradient contributions, assisting with weight selection during experimental evaluation.

---

### `src/optimisation/mesh_geometry.py`

Low-level geometric primitives used by `energy_terms.py` and `gradients.py`:

```python
def compute_face_planarity_deviation(vertices: np.ndarray) -> float: ...
def compute_all_planarity_deviations(mesh: QuadMesh) -> np.ndarray: ...
def compute_conical_angle_imbalance(mesh: QuadMesh) -> np.ndarray: ...
```

`compute_face_planarity_deviation` returns the smallest singular value of the mean-centred vertex matrix for a single face (4 x 3 input). `compute_all_planarity_deviations` vectorises this over all faces. `compute_conical_angle_imbalance` returns a per-vertex array of the angle-sum imbalance, which is the quantity minimised by `E_angle`.

---

### `src/optimisation/optimiser.py`

The optimisation driver. See `src/optimisation/OPTIMISATION_PIPELINE.md` for full documentation.

Key exports:

```python
@dataclass
class OptimisationConfig:
    weights: dict[str, float]         # e.g. {"planarity": 10.0, "fairness": 1.0, "closeness": 5.0}
    max_iterations: int               # default 200
    tolerance: float                  # default 1e-6
    verbose: bool                     # default False
    backend: str                      # "auto", "numba", or "numpy"

@dataclass
class OptimisationResult:
    success: bool
    message: str
    n_iterations: int
    n_function_evaluations: int
    n_gradient_evaluations: int
    initial_energy: float
    final_energy: float
    execution_time: float
    component_energies_initial: dict
    component_energies_final: dict

    def energy_reduction_percentage(self) -> float: ...
    def summary(self) -> str: ...

class MeshOptimiser:
    def __init__(self, config: OptimisationConfig): ...
    def optimise(self, mesh: QuadMesh) -> OptimisationResult: ...
    def validate_mesh(self, mesh: QuadMesh) -> bool: ...

def optimise_mesh_simple(
    mesh: QuadMesh,
    weights: dict,
    max_iterations: int = 200,
    verbose: bool = False,
) -> OptimisationResult: ...
```

---

### `src/preprocessing/preprocessor.py`

```python
def preprocess_mesh(
    mesh: QuadMesh,
    normalise: bool = True,
    verbose: bool = True,
) -> QuadMesh: ...
```

Performs the following operations in order:
1. Removes duplicate vertices by merging vertices within a tolerance of `1e-8`.
2. Removes degenerate faces (faces with repeated vertex indices).
3. Optionally normalises the mesh to fit within a unit bounding box centred at the origin.
4. Prints a report of changes made if `verbose=True`.

---

## Typical usage pattern

```python
from src.io.obj_handler import load_obj, save_obj
from src.preprocessing.preprocessor import preprocess_mesh
from src.optimisation.optimiser import optimise_mesh_simple

mesh = load_obj("data/input/generated/plane_5x5_noisy.obj")
mesh = preprocess_mesh(mesh, normalise=True, verbose=True)

result = optimise_mesh_simple(
    mesh,
    weights={"planarity": 10.0, "fairness": 1.0, "closeness": 5.0},
    max_iterations=200,
    verbose=True,
)

print(result.summary())
save_obj(mesh, "data/output/optimised_meshes/result.obj")
```

---

## Design constraints

- All public functions accept and return `QuadMesh` objects or NumPy arrays; no file paths are threaded through the computation layer.
- The optimisation pipeline modifies `mesh.vertices` **in place**. `mesh.vertices_original` is never modified. Callers that need to preserve the original must pass `copy.deepcopy(mesh)` to the optimiser.
- The `backends.py` selection is transparent: all energy and gradient functions accept a `backend` argument that defaults to `"auto"`, falling back gracefully to NumPy if Numba is unavailable.
