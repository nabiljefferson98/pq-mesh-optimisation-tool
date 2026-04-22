# tests/ — Testing Guide

The test suite for this project comprises 20 test modules covering unit tests, integration tests, numerical equivalence tests, robustness tests, and scalability tests. All tests are written using `pytest` and are located in the `tests/` directory.

---

## Running the tests

```bash
# Run the full test suite from the project root
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a single test module
pytest tests/test_energy_terms.py -v

# Run a single test by name
pytest tests/test_gradients.py::TestPlanarityGradient::test_flat_face_zero_gradient -v

# Run all tests except those marked slow
pytest tests/ -m "not slow"
```

---

## Test module catalogue

### Unit tests — core data structures

#### `test_mesh.py`
Tests the `QuadMesh` dataclass.
- Verifies that `n_vertices` and `n_faces` properties return the correct values.
- Verifies that `vertices_original` is set at construction and is independent of `vertices` (i.e., modifying `vertices` does not affect `vertices_original`).

#### `test_obj_handler.py`
Tests `load_obj` and `save_obj` for a minimal quad mesh.
- Round-trip test: load, save, reload, and verify vertex and face arrays are numerically identical.
- Verifies that `load_obj` raises `ValueError` for a file containing non-quad faces.

#### `test_obj_handler_extended.py`
Extended OBJ I/O tests.
- Tests loading of meshes with comment lines, blank lines, texture coordinates (`vt`), and normals (`vn`) that must be ignored during parsing.
- Tests loading of meshes with one-based face indices (standard OBJ) and verifies correct conversion to zero-based indexing.
- Tests `save_obj` output format: verifies that vertex lines begin with `v ` and face lines begin with `f `, and that face indices are written as one-based.
- Tests error handling for missing files and malformed geometry records.

#### `test_geometry.py`
Tests `compute_face_planarity_deviation` and `compute_all_planarity_deviations` from `src.optimisation.mesh_geometry`.
- Verifies that a perfectly flat quad returns deviation `0.0`.
- Verifies that a non-planar quad (one vertex lifted) returns a positive deviation.
- Verifies that `compute_all_planarity_deviations` returns an array of shape `(F,)` with all non-negative values.

---

### Unit tests — energy terms

#### `test_energy_terms.py`
Tests each of the four energy functionals.

| Test class | What is tested |
| ---------- | -------------- |
| `TestPlanarityEnergy` | Zero energy for a flat quad mesh; positive energy after lifting one vertex; scaling linearly with weight |
| `TestFairnessEnergy` | Zero energy for a regular grid mesh (Laplacian of a regular grid is zero at interior vertices); positive energy after perturbation |
| `TestClosenessEnergy` | Zero energy when `vertices == vertices_original`; correct value after a known displacement |
| `TestAngleBalanceEnergy` | Zero energy for a mesh where all vertex angle sums are equal; positive energy for an irregular mesh |
| `TestTotalEnergy` | Weighted sum matches the sum of individual weighted terms; zero weights produce zero contribution |
| `TestPlanarityPerFace` | Output shape is `(F,)`; values are non-negative; flat faces return zero |

#### `test_vertex_face_ids.py`
Verifies that vertex and face index arrays produced by `load_obj` are zero-based, contiguous, and within bounds for several test meshes.

#### `test_quad_loading.py`
Verifies that all meshes in `data/input/generated/` and `data/input/reference_datasets/` load without error, return a `QuadMesh` with non-empty `vertices` and `faces`, and contain only quadrilateral faces (face array shape `(F, 4)`).

---

### Unit tests — gradients

#### `test_gradients.py`
Core gradient tests.

| Test class | What is tested |
| ---------- | -------------- |
| `TestPlanarityGradient` | Zero gradient for a flat mesh; non-zero gradient after perturbation; finite-difference check with relative error below `1e-4` |
| `TestFairnessGradient` | Zero gradient for a regular grid; correct direction (pointing towards reduced Laplacian norm) |
| `TestClosenessGradient` | Gradient equals `2 * w_c * (V - V_0)` exactly for a known displacement |
| `TestTotalGradient` | Weighted sum matches sum of individual gradients; output shape is `(3V,)` |

#### `test_gradients_extended.py`
Extended gradient tests covering edge cases.
- Gradient at a mesh where all four energy weights are non-zero.
- Gradient with `w_angle_balance > 0`: verifies the alternating-sign accumulation pattern `(+1, -1, +1, -1)` for adjacent vertices.
- Gradient for a mesh with boundary vertices (vertices with fewer than four adjacent faces): verifies no `NaN` or `Inf` values are produced.
- Finite-difference check for the angle-balance gradient component in isolation.

---

### Numerical equivalence tests

#### `test_numerical_equivalence.py`
Verifies that the Numba and NumPy backends produce numerically identical results (within `atol=1e-10`) for all energy terms and gradient components, using a fixed random seed for reproducibility.

Test conditions:
- Flat mesh (all energies should be near-zero on both backends)
- Noisy mesh (general case)
- High-curvature mesh (Scherk surface, where planarity deviation is large)
- Mesh with all four weights non-zero

These tests are skipped automatically if Numba is not installed (`pytest.importorskip("numba")`).

---

### Integration tests — optimiser

#### `test_optimiser.py`
End-to-end optimisation tests.

| Test | What is tested |
| ---- | -------------- |
| `test_optimise_flat_mesh` | Optimising an already-flat mesh produces `energy_reduction_percentage >= 0` and does not increase energy |
| `test_optimise_noisy_plane` | Optimising a perturbed plane reduces mean planarity deviation by at least 50% |
| `test_optimise_result_fields` | `OptimisationResult` contains correct types and non-negative values for all fields |
| `test_optimise_mesh_simple` | `optimise_mesh_simple` convenience wrapper returns an `OptimisationResult` and modifies `mesh.vertices` in place |
| `test_config_default_weights` | `OptimisationConfig` with default weights runs without error |
| `test_result_summary` | `result.summary()` returns a non-empty string containing the iteration count |
| `test_energy_reduction_percentage` | `energy_reduction_percentage()` returns a value in `[0, 100]` for a successful run |
| `test_verbose_output` | Setting `verbose=True` produces output to stdout (captured by `capsys`) |

---

### Robustness tests

#### `test_robustness.py`
Tests that the optimiser behaves correctly under pathological inputs.

- **Degenerate weights** — `w_planarity = 0`, `w_fairness = 0`, all weights zero: verifies no crash and that the result `success` field reflects whether convergence was achieved.
- **Single-face mesh** — a mesh with exactly one quadrilateral face: verifies that the fairness energy (which requires a Laplacian) handles the boundary case correctly.
- **Near-singular geometry** — a face where three vertices are nearly collinear: verifies that the SVD-based planarity computation does not produce `NaN`.
- **Large displacement** — vertices displaced far from `vertices_original`: verifies the closeness gradient does not overflow.
- **Non-finite energy fallback** — manually injects `NaN` into the vertex array and verifies that `energy_for_scipy` returns `1e300` rather than `NaN`.
- **Gradient `nan_to_num` guard** — verifies that `gradient_for_scipy` returns a finite array even when called on a mesh with `NaN` vertices.

#### `test_error_handling.py`
Tests explicit error conditions.
- `load_obj` raises `FileNotFoundError` for a non-existent path.
- `load_obj` raises `ValueError` for a file containing triangular faces.
- `MeshOptimiser.validate_mesh` returns `False` for a mesh with `NaN` vertices.

---

### Coverage-extension tests

#### `test_coverage_extended.py`
Targeted tests written to increase branch coverage of `src/optimisation/optimiser.py` and `src/io/panel_exporter.py`.

- Tests the `success=False` path of `OptimisationResult.summary()`.
- Tests `energy_reduction_percentage()` when `initial_energy = 0` (verifies no division-by-zero).
- Tests `export_dxf` and `export_svg` with a mesh that has already-flat faces (verifies that zero-area panels are handled without error).
- Tests `unfold_mesh` with a single-row mesh (edge case for the BFS unfolding traversal).

#### `test_coverage_gaps.py`
Additional coverage for previously untested branches in `backends.py`, `preprocessor.py`, and `energy_terms.py`.

- Tests `warmup_numba_kernels()` does not raise when called multiple times.
- Tests `preprocess_mesh` with `normalise=False`.
- Tests `preprocess_mesh` on a mesh with no duplicate vertices (verifies no vertices are removed).
- Tests `compute_angle_balance_energy` with `weight = 0.0` returns `0.0` without computation.

---

### Topology and structure tests

#### `test_quad_topology_preservation.py`
Verifies that the optimiser preserves mesh topology throughout optimisation.
- Face array is identical before and after optimisation (same vertex indices, same order).
- `n_vertices` and `n_faces` are unchanged.
- No new vertices are introduced.

#### `test_preprocessor.py`
Tests `preprocess_mesh` comprehensively.
- Duplicate vertex merging: a mesh with two identical vertices is reduced to one; face indices are updated accordingly.
- Degenerate face removal: a face with a repeated vertex index is removed.
- Normalisation: after `normalise=True`, the bounding box fits within `[-0.5, 0.5]^3`.
- `verbose=True` produces output to stdout.

---

### Scalability tests

#### `test_scalability.py`
Verifies that the optimiser completes within reasonable time limits for meshes of increasing size.

| Mesh size | Max allowed time |
| --------- | ---------------- |
| 5x5 (25 faces) | 10 seconds |
| 10x10 (100 faces) | 30 seconds |
| 16x16 (256 faces) | 60 seconds |

These tests are marked `@pytest.mark.slow` and are excluded from the default test run. Run with `pytest tests/test_scalability.py -v` to execute them explicitly.

---

### Backend tests

#### `test_backends.py`
Tests the Numba/NumPy backend selector.
- Verifies that `HAS_NUMBA` is a `bool`.
- If `HAS_NUMBA` is `True`, verifies that `warmup_numba_kernels()` runs without error.
- Verifies that forcing `backend="numpy"` on `compute_total_gradient` produces the same result as `backend="auto"` when Numba is unavailable.

---

## Test fixtures

Shared fixtures are defined inline within each test module rather than in a centralised `conftest.py`. The most frequently used fixture is a small noisy quad mesh created with NumPy:

```python
@pytest.fixture
def noisy_quad_mesh():
    rng = np.random.default_rng(42)
    V = 5 * 5
    vertices = np.zeros((V, 3))
    # ... grid construction with Gaussian noise
    return QuadMesh(vertices=vertices, faces=faces, vertices_original=vertices.copy())
```

The fixed seed `42` is used throughout to ensure reproducibility of all random perturbations.

---

## Continuous integration

The full test suite (excluding `@pytest.mark.slow` tests) runs automatically on every push and pull request via the GitHub Actions workflow defined in `.github/workflows/`. The CI environment does not have Numba installed; accordingly, all `test_numerical_equivalence.py` tests are skipped in CI and the NumPy backend is used exclusively.

To replicate the CI environment locally:

```bash
pip install pytest numpy scipy
pytest tests/ -m "not slow" -v
```
