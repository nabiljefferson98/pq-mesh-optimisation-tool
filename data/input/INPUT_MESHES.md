# Input Meshes

`data/input/`

This document describes the contents, purpose, naming conventions, and
expected file format for all mesh data stored under `data/input/`.

---

## Directory Structure

```
data/input/
├── INPUT_MESHES.md               <- this file
├── generated/                    <- procedurally generated test meshes
└── reference_datasets/           <- external benchmark meshes from public repositories
    ├── spot/
    │   ├── spot_quadrangulated.obj
    │   ├── spot_control_mesh.obj
    │   └── README.txt
    ├── blub/
    │   ├── blub_quadrangulated.obj
    │   ├── blub_control_mesh.obj
    │   └── README.txt
    ├── oloid/
    │   ├── oloid64_quad.obj
    │   ├── oloid256_quad.obj
    │   ├── oloid1024_quad.obj
    │   └── oloid4096_quad.obj
    └── bob/
        ├── bob_quad.obj
        └── bob_controlmesh.obj
```

---

## generated/

These meshes are generated programmatically by the mesh generation scripts in
`scripts/mesh_generation/`. They cover a range of surface types, noise levels,
and resolutions designed to exercise all aspects of the optimisation pipeline.

All meshes in this folder are quad meshes in Wavefront OBJ format with exactly
four vertex indices per face (`f v1 v2 v3 v4`).

### Mesh Catalogue

| Filename | Vertices | Faces | Surface Type | Noise Level | Preprocessor Notes | Purpose |
|---|---|---|---|---|---|---|
| `cylinder_10x8.obj` | ~90 | 80 | Cylindrical | None | None required | Default mesh for the interactive viewer; smooth curved surface |
| `canopy_8x8_demo.obj` | ~81 | 64 | Canopy (NURBS) | None | None required | Architectural canopy shape for demo and visual validation |
| `plane_3x3_noisy.obj` | ~16 | 9 | Planar | Low | None required | Minimal smoke-test mesh; fast convergence expected |
| `plane_5x5_clean.obj` | ~36 | 25 | Planar | None | None required | Already-planar baseline; tests that the optimiser produces no change |
| `plane_5x5_perfect.obj` | ~36 | 25 | Planar | None | None required | Ideal planar reference; all planarity scores should be approximately 0 |
| `plane_5x5_noisy.obj` | ~36 | 25 | Planar | Moderate | None required | Standard regression mesh for unit and integration tests |
| `plane_5x5_subtle.obj` | ~36 | 25 | Planar | Subtle | None required | Tests detection of very small planarity violations |
| `plane_5x5_heavy.obj` | ~36 | 25 | Planar | High | None required | Tests robustness under severe initial non-planarity |
| `plane_5x5_very_noisy.obj` | ~36 | 25 | Planar | Very high | None required | Stress test for convergence under extreme perturbation |
| `plane_10x10_noisy.obj` | ~121 | 100 | Planar | Moderate | None required | Medium-scale regression mesh |
| `plane_20x20_noisy.obj` | ~441 | 400 | Planar | Moderate | None required | Large-scale regression mesh; tests vectorised performance |
| `saddle_8x8.obj` | ~81 | 64 | Saddle (hyperbolic paraboloid) | None | None required | Tests behaviour on a doubly-curved surface |
| `saddle_12x12.obj` | ~169 | 144 | Saddle | None | None required | Higher-resolution saddle for benchmarking and scaling tests |
| `scherk_8x8.obj` | ~81 | 64 | Scherk minimal surface | None | None required | Tests optimiser on a complex free-form geometry |
| `sphere_cap_10x8.obj` | ~90 | 80 | Spherical cap | None | None required | Tests behaviour on a synclastic (positive Gaussian curvature) surface |
| `torus_patch_8x8.obj` | ~81 | 64 | Torus patch | None | None required | Tests behaviour on a surface with mixed curvature sign |

### Noise Level Definitions

| Label | Vertex displacement (fraction of mean edge length) |
|---|---|
| None | 0.0 (mathematically exact surface) |
| Subtle | ~0.01 (barely perceptible) |
| Low | ~0.05 |
| Moderate | ~0.10 |
| High | ~0.20 |
| Very high | ~0.35 |

---

## reference_datasets/

These meshes are sourced from Keenan Crane's 3D Model Repository
(https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/) and are used to
evaluate the optimisation pipeline on community-standard, publicly recognised
models. They represent a curated range of organic surface types with varying
curvature complexity, and include a mathematically developable surface (oloid)
that directly connects the evaluation to the dissertation's theoretical thread
on developable geometry.

All meshes in this folder are native quad meshes requiring no conversion.

### Mesh Catalogue

| Folder | Filename | Approx. Vertices | Approx. Faces | Surface Character | Preprocessor Notes | EXP Usage | Attribution |
|---|---|---|---|---|---|---|---|
| `spot/` | `spot_quadrangulated.obj` | ~3,030 | ~3,700 | Smooth organic, moderate curvature | None required; unit-scale, clean | EXP-05 primary benchmark | Keenan Crane (2013), CC0 |
| `spot/` | `spot_control_mesh.obj` | ~410 | ~470 | Low-resolution control cage | None required | Reference / visual inspection only | Keenan Crane (2013), CC0 |
| `blub/` | `blub_quadrangulated.obj` | ~1,200 | ~1,400 | Smooth organic, higher local curvature variation (fins) | None required; unit-scale, clean | EXP-05 secondary benchmark | Keenan Crane (2013), CC0 |
| `blub/` | `blub_control_mesh.obj` | ~100 | ~120 | Low-resolution control cage | None required | Reference / visual inspection only | Keenan Crane (2013), CC0 |
| `oloid/` | `oloid64_quad.obj` | ~66 | ~64 | Developable surface, zero Gaussian curvature | None required; already quad | EXP-01 scalability series (lowest res) | Keenan Crane, CC0 |
| `oloid/` | `oloid256_quad.obj` | ~258 | ~256 | Developable surface, zero Gaussian curvature | None required | EXP-01 scalability series; EXP-05 developable test | Keenan Crane, CC0 |
| `oloid/` | `oloid1024_quad.obj` | ~1,026 | ~1,024 | Developable surface, zero Gaussian curvature | Scale normalisation recommended | EXP-01 scalability series | Keenan Crane, CC0 |
| `oloid/` | `oloid4096_quad.obj` | ~4,098 | ~4,096 | Developable surface, zero Gaussian curvature | Scale normalisation recommended | EXP-01 scalability series (stress test) | Keenan Crane, CC0 |
| `bob/` | `bob_quad.obj` | ~large | ~large | Complex organic, highly irregular connectivity | Scale normalisation required; check for duplicate vertices at seams | EXP-05 large-scale stress test | Keenan Crane, CC0 |
| `bob/` | `bob_controlmesh.obj` | ~medium | ~medium | Low-resolution control cage | None required | Reference / visual inspection only | Keenan Crane, CC0 |

### Attribution

All meshes in `reference_datasets/` were obtained from:

> Keenan Crane, *3D Model Repository*, Carnegie Mellon University.
> https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/
> Licence: Creative Commons CC0 (public domain dedication).

The oloid mesh is a mathematically exact developable surface. An oloid is a
geometric solid formed by placing two circles of equal radius in perpendicular
planes such that each circle passes through the centre of the other. Its
entire surface is developable (Gaussian curvature K = 0 everywhere except
the two generating circular edges), making it a theoretically well-motivated
benchmark for a PQ planarity optimiser: on the flat ruling regions, the
planarity energy functional should converge towards zero, while the
circular-edge regions impose a fundamental geometric limit on achievable
planarity with fixed connectivity.

---

## File Format Requirements

All input meshes must be in Wavefront OBJ format (`.obj`) with the following
constraints for correct parsing by `src/io/obj_handler.py`:

- Every face must be a quad (exactly four vertex indices):
  ```
  f v1 v2 v3 v4
  ```
- Vertex positions must be defined before the face lines:
  ```
  v x y z
  ```
- Vertex indices in the OBJ file are **one-based**; the parser converts them
  to zero-based indices automatically.
- Faces with three vertex indices (triangles) are accepted by `QuadMesh`
  internally but will raise a `ValueError` in the standard load path if the
  file contains a mix of triangles and quads. Ensure all faces are quads
  before importing.
- OBJ groups (`g`), object names (`o`), texture coordinates (`vt`), and
  vertex normals (`vn`) are parsed and silently ignored.
- Files must be UTF-8 or ASCII encoded.

### Example Minimal OBJ

```
# Minimal single-quad mesh
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
```

---

## Preprocessor Notes

The preprocessing pipeline in `src/preprocessing/preprocessor.py` handles
three common issues that affect real-world mesh imports. The table below
describes each issue and how to detect it.

| Issue | Description | Typical cause | Effect on optimiser if untreated |
|---|---|---|---|
| Scale mismatch | Vertices are not in unit scale | Scanner or CAD export in millimetres | Planarity energy values become extremely large (~10^8), making convergence very slow or impossible |
| Duplicate vertices | Two or more vertex rows share the same 3D coordinate | Seams from export or mesh joining | Disconnected topology; the optimiser treats coincident vertices as independent, producing visible cracks |
| Degenerate faces | A face has two or more coincident vertices (zero area) | Import artefacts or repeated indices in OBJ file | Division by zero in the planarity cross-product computation |

The `preprocess_mesh()` function addresses all three issues automatically.
See `src/preprocessing/PREPROCESSING_GUIDE.md` for full details.

### Weight Auto-Suggestion

The `suggest_weights_for_mesh()` function in `src/preprocessing/preprocessor.py`
chooses energy weights automatically based on the initial planarity of the
input mesh. It targets a ratio of approximately 10:1:5 for planarity, fairness,
and closeness weights respectively.

---

## Generating New Test Meshes

When adding a new mesh to either input subfolder:

1. Save it to `data/input/generated/` if produced by a generation script, or
   `data/input/reference_datasets/` if sourced from an external repository.
2. Use the naming convention `<surface_type>_<resolution>_<variant>.obj`
   for generated meshes, e.g. `cone_10x10_clean.obj`.
3. Ensure the mesh is in Wavefront OBJ format with exactly four vertex
   indices per face.
4. Update the mesh catalogue table above.
5. If the mesh is intended for regression testing, add a corresponding entry
   in the relevant test file under `tests/`.
