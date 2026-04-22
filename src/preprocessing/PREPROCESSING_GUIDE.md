# Preprocessing Guide

`src/preprocessing/preprocessor.py`

This document explains the preprocessing pipeline that prepares a raw input
mesh for optimisation. Real-world meshes imported from CAD tools frequently
have issues that would cause the optimiser to fail or produce wrong results.
This pipeline fixes those issues automatically before any optimisation begins.

---

## Why Preprocessing Is Necessary

The synthetic test meshes used during development are clean, unit-scale, and
perfectly connected. Real-world meshes exported from Blender, Rhino, or
ArchiCAD typically are not. Three specific problems arise:

### Problem 1: Arbitrary Scale

CAD models are often authored in millimetres, meaning vertex coordinates
have values of the order of thousands. The planarity energy is based on
squared distances, so its value scales as the fourth power of the coordinate
size. For a millimetre-scale model:

    E_planarity ~ (1e3)^4 = 1e12

The L-BFGS-B solver used for optimisation has an internal step-size bound
that assumes energies of order 1. An energy of `1e12` causes the line search
to fail on the very first iteration. Scale normalisation brings the energy
back to order `1e-2`, well within the solver's operating range.

### Problem 2: Duplicate Vertices

Most mesh exporters write a separate copy of each vertex for every face that
shares it, producing a "polygon soup": faces are listed with their own private
vertex copies rather than pointing to shared vertices. This breaks the
vertex-to-face adjacency graph that the Laplacian fairness term and the angle
balance gradient both require. For a polygon soup, every vertex appears to
have exactly four neighbours (all from the same face), so the Laplacian and
all topology-dependent gradients are computed incorrectly.

### Problem 3: Degenerate Faces

A degenerate face is a quad where two or more vertices are at the same
position, giving the face zero area. The planarity energy computation uses
a batched SVD whose inputs include the face edge vectors. A zero-area face
has linearly dependent edge vectors, causing the SVD to return a zero
singular value and producing `NaN` or `Inf` in the gradient. A single such
face causes the entire optimisation to fail immediately.

---

## Pipeline Stages

The `preprocess_mesh()` function runs six stages in order:

```
Stage 1:  Record bounding box of the original mesh for diagnostics.
Stage 2:  Merge duplicate vertices (cKDTree union-find, O(n log n)).
Stage 3:  Remove degenerate (zero-area) faces.
Stage 4:  Normalise: centre at origin, scale longest axis to target_scale.
Stage 5:  Rebuild a QuadMesh with vertices_original set to post-normalised positions.
Stage 6:  Auto-suggest energy weights calibrated to this mesh.
```

---

## Usage

```python
from src.io.obj_handler import load_obj
from src.preprocessing.preprocessor import preprocess_mesh

raw = load_obj("data/input/reference_datasets/spot_quadrangulated.obj")
mesh, info = preprocess_mesh(raw, normalise=True, verbose=True)
```

After this call, `mesh` is ready to pass directly to `MeshOptimiser`. The
returned `info` object (a `PreprocessingInfo`) records everything that was
done, which is useful for dissertation documentation and for reproducing
results.

---

## Function Reference

### `preprocess_mesh(mesh, normalise=True, target_scale=1.0, ...)`

| Parameter | Default | Description |
|---|---|---|
| `mesh` | required | Raw `QuadMesh` as returned by `load_obj()` |
| `normalise` | `True` | Apply scale normalisation (Stage 4). Strongly recommended for real-world meshes |
| `target_scale` | `1.0` | Desired length of the longest bounding-box axis after normalisation |
| `remove_degenerates` | `True` | Remove zero-area faces (Stage 3) |
| `merge_duplicates` | `True` | Merge coincident vertices (Stage 2) |
| `duplicate_threshold` | `1e-8` | Distance threshold for considering two vertices identical (world units, before normalisation) |
| `verbose` | `True` | Print a preprocessing report to standard output |

Returns a tuple `(processed_mesh, info)` where `processed_mesh` is a new
`QuadMesh` ready for optimisation, and `info` is a `PreprocessingInfo`
recording all transformations.

Raises `ValueError` if no valid faces remain after cleaning, which indicates
a fundamentally broken input mesh.

---

## Stage Detail: Normalisation

Normalisation applies two transformations to all vertex positions:

**Step 1.** Translate so the centroid is at the origin:

    v'_i = v_i - mean(v)

**Step 2.** Scale uniformly so the longest bounding-box axis equals `target_scale`:

    v''_i = v'_i * (target_scale / max_axis_length)

For the default `target_scale = 1.0`, the resulting mesh fits inside a unit
cube centred at the origin. The scale factor and centroid offset are stored
in `PreprocessingInfo` so results can be mapped back to the original
coordinate system if needed.

---

## Stage Detail: Duplicate Vertex Merging

The naive approach to finding duplicate vertices compares every vertex pair,
which costs O(n^2) and becomes prohibitively slow for meshes with tens of
thousands of vertices. The implementation uses a `scipy.spatial.cKDTree`
spatial index to find all pairs within the threshold distance in O(n log n)
time, followed by a union-find data structure to cluster them:

1. Build a cKDTree over all vertex positions.
2. Query all pairs of vertices within `duplicate_threshold` using
   `query_pairs()`, which is an efficient range search.
3. Apply union-find with path compression: for each pair `(a, b)`, merge
   their clusters. The representative of each cluster is always the vertex
   with the lowest index, ensuring deterministic output regardless of the
   order pairs are processed.
4. Remap all face vertex indices to the representative of their cluster.
5. Build a compacted vertex array containing only one row per cluster.

The number of merged pairs is stored in `PreprocessingInfo.removed_duplicate`.

---

## Stage Detail: Degenerate Face Removal

For quad faces, area is approximated using the cross product of the two
diagonal vectors:

    area = 0.5 * || (v_2 - v_0) x (v_3 - v_1) ||

For triangle faces the standard formula is used:

    area = 0.5 * || (v_1 - v_0) x (v_2 - v_0) ||

Any face with area below `1e-10` is removed. Faces with out-of-bounds vertex
indices are also removed, as these indicate corrupt mesh connectivity that
would cause an index error in the optimiser.

---

## `PreprocessingInfo` — Transformation Record

The `PreprocessingInfo` dataclass returned by `preprocess_mesh()` records the
following:

| Field | Type | Description |
|---|---|---|
| `original_vertices` | int | Vertex count before preprocessing |
| `original_faces` | int | Face count before preprocessing |
| `final_vertices` | int | Vertex count after preprocessing |
| `final_faces` | int | Face count after preprocessing |
| `was_normalised` | bool | Whether scale normalisation was applied |
| `scale_factor` | float | Multiplier applied during normalisation (1.0 if not normalised) |
| `centroid_offset` | (3,) float64 | Translation subtracted from all vertices during normalisation |
| `removed_degenerate` | int | Number of zero-area faces removed |
| `removed_duplicate` | int | Number of duplicate vertex pairs merged |
| `bounding_box_size` | (float, float, float) | Original bounding box dimensions (dx, dy, dz) |
| `suggested_weights` | dict or None | Auto-suggested energy weights (see below) |
| `warnings` | list of str | Non-fatal warnings (for example, duplicate counts) |

---

## Auto-Suggested Energy Weights

`suggest_weights_for_mesh(mesh)` computes initial energy values for the
processed mesh and returns a weight dictionary scaled so that the weighted
energies are of comparable magnitude at the start of optimisation. The
target ratio is:

    w_p * E_p : w_f * E_f : w_c * E_c  ~  10 : 1 : 5

This prevents any single term from dominating the gradient, which would
cause the optimiser to improve one objective while ignoring the others.

The weights are computed as:

    w_p = clip(10 / E_p, 1, 1e6)
    w_f = clip(1  / E_f, 1e-4, 1e3)
    w_c = clip(5  / E_c_proxy, 1e-2, 1e5)

where `E_c_proxy` is a proxy for the closeness energy at the first meaningful
optimisation step, estimated from the mesh scale.

A special case applies when the initial planarity energy `E_p` is below
`1e-8` (the mesh is already near-planar, as is typical for meshes generated
along principal curvature directions). Setting `w_p = 10 / E_p` would then
produce weights of order `1e9`, amplifying floating-point noise rather than
driving useful convergence. In this case the weights are calibrated to the
fairness energy alone, and a diagnostic message is printed recommending the
use of a noisy mesh variant for planarity testing.

The `angle_balance` weight is always `0.0` in the auto-suggested
configuration. It should be set manually if the conical mesh condition is
required for a specific experiment.

---

## Preprocessing Report

When `verbose=True`, a formatted report is printed after processing. Example
output for a Blender-exported mesh:

```
============================================================
PREPROCESSING REPORT
============================================================
  Vertices :    162 -> 81
  Faces    :     64 -> 64
  Bounding box (original): 1000.0000 x 1000.0000 x 200.0000
  Normalised  : scale factor = 1.000000e-03
  Duplicates removed : 81
  Suggested weights -> planarity=12.50, fairness=0.0034, closeness=8.4500
============================================================
```

In this example, 81 duplicate vertices were merged (the exporter had written
two copies of each shared vertex), and the mesh was scaled from millimetres
to metres.

---

## Important Note on `vertices_original`

The `closeness` energy term measures how far each vertex has moved from its
baseline position during optimisation. It is critical that this baseline is
set to the **post-normalised** positions, not the original millimetre-scale
coordinates. The preprocessor sets `mesh.vertices_original = vertices.copy()`
after Stage 4 to ensure this. Do not modify `vertices_original` after calling
`preprocess_mesh()`.

---

## References

- Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.
- Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.
- Bentley, J.L. (1975). "Multidimensional binary search trees used for
  associative searching." Communications of the ACM, 18(9), pp. 509-517.
  (Foundational reference for kd-tree spatial indexing.)
