# I/O Guide

`src/io/`

This module handles all file input and output for the project. It contains
two files: `obj_handler.py` for loading and saving meshes in Wavefront OBJ
format, and `panel_exporter.py` for exporting fabrication-ready flat panels
in SVG and DXF formats. Both are described in full below.

---

## Module Overview

| File | Responsibility |
|---|---|
| `obj_handler.py` | Load a quad mesh from an OBJ file; save an optimised mesh back to OBJ |
| `panel_exporter.py` | Unfold optimised quad faces into flat 2D panels; export as SVG and DXF |

---

## `obj_handler.py` — Mesh Import and Export

### What Is the Wavefront OBJ Format?

A Wavefront OBJ file is a plain-text file that stores mesh geometry as a
sequence of lines. Vertex lines begin with `v` and list three coordinates:

```
v 0.0 0.0 0.0
```

Face lines begin with `f` and list vertex indices (one-based, meaning the
first vertex is index 1 not 0):

```
f 1 2 3 4
```

Face tokens may also carry texture and normal annotations in the form
`v/vt/vn`, for example `f 1/1/1 2/2/2 3/3/3 4/4/4`. The parser reads only
the vertex index before the first slash and silently discards the rest.

---

### `load_obj(filepath, require_quads=False)`

Loads a mesh from an OBJ file and returns a `QuadMesh` object.

```python
from src.io.obj_handler import load_obj

mesh = load_obj("data/input/generated/saddle_8x8.obj")
print(mesh.n_vertices, mesh.n_faces)
```

#### What the loader does

1. Parses all `v` lines into a vertex array.
2. Parses all `f` lines into a face list, resolving any one-based or
   relative (negative) indices to zero-based integers.
3. Separates faces by type: quads (four vertices) and triangles (three
   vertices). Faces with five or more vertices are silently skipped.
4. If quad faces are present, uses them directly.
5. If only triangle faces are present, attempts to pair adjacent triangles
   back into quads using the shared-edge algorithm (see below).
6. Returns a `QuadMesh` containing the resulting faces.

#### Triangle-to-Quad Pairing

Many mesh exporters, including Blender and most CAD tools, triangulate all
faces before writing OBJ. This splits every quad into two triangles along
its shorter diagonal. The loader can reverse this by finding pairs of
triangles that share an edge and merging them back into a single quad.

The algorithm works as follows:

1. Build an edge-to-face lookup table from all triangle edges.
2. For every edge shared by exactly two triangles, merge them into a quad
   with vertices ordered as: `v0, va, v1, vb`, where `v0` and `v1` are
   the shared edge vertices and `va`, `vb` are the opposite vertices.
3. Each triangle may be used in at most one pairing (greedy, first match wins).

If pairing fails and `require_quads=False` (the default), the loader falls
back to returning a triangular mesh with a warning. If `require_quads=True`,
a `ValueError` is raised with a message recommending remeshing tools.

#### Failure Modes Handled

| Situation | Behaviour |
|---|---|
| File not found | Raises `FileNotFoundError` |
| No vertices in file | Raises `ValueError` |
| No faces of any type | Raises `ValueError` |
| Negative face indices (relative OBJ indices) | Resolved to absolute zero-based indices |
| Malformed vertex or face lines | Skipped with a printed warning; first five are shown |
| Mixed quad and triangle faces | Triangle faces ignored when quads are present |
| N-gon faces (5+ vertices) | Silently skipped |

---

### `save_obj(mesh, filepath)`

Exports a `QuadMesh` to an OBJ file.

```python
from src.io.obj_handler import save_obj

save_obj(mesh, "data/output/optimised_meshes/saddle_8x8_optimised.obj")
```

- Vertex coordinates are written with six decimal places of precision.
- Face indices are written as one-based integers as required by the OBJ
  specification.
- Parent directories are created automatically if they do not exist.
- A path traversal guard rejects any filepath containing `..` to prevent
  writing files outside the intended directory.

---

## `panel_exporter.py` — Fabrication Panel Export

After optimisation, every quad face is approximately planar. This module
unfolds each face into a flat 2D panel and exports the results in two
formats: SVG for visual inspection and DXF for CNC or laser cutting.

### Mathematical Unfolding Procedure

For each quad face with four 3D vertex positions `v_0, v_1, v_2, v_3`:

**Step 1.** Compute the centroid:

    c = (v_0 + v_1 + v_2 + v_3) / 4

**Step 2.** Centre the vertices:

    w_i = v_i - c

**Step 3.** Fit a plane using Singular Value Decomposition (SVD). Arrange the
four centred vertices as rows of a (4, 3) matrix W. The SVD decomposes W as:

    W = U * S * V^T

The plane normal `n` is the row of `V^T` corresponding to the **smallest**
singular value, because this is the direction of least variance, i.e., the
direction most perpendicular to all four vertices.

**Step 4.** Define a local 2D coordinate frame on the plane:

    u_hat = normalise(w_1 - w_0)         (first edge direction)
    v_hat = n x u_hat                     (right-hand perpendicular)

**Step 5.** Project all four vertices into the frame:

    p_i = ( w_i . u_hat,  w_i . v_hat )

where `.` denotes the dot product. This gives the 2D coordinates of each
vertex in the local panel frame.

**Step 6.** The planarity residual is:

    residual = max_i | w_i . n |

This is the largest out-of-plane distance before projection. For a well
optimised mesh this should be below `1e-3` model units, which corresponds
to less than 1 mm of warp for a panel of 1 m side length.

---

### Key Data Classes

#### `FlatPanel`

Represents one unfolded quad face. Key attributes:

| Attribute | Shape | Description |
|---|---|---|
| `face_id` | int | Index of the source face in the mesh |
| `vertices_2d` | (4, 2) float64 | 2D corner coordinates in the local panel frame |
| `planarity_residual` | float | Maximum out-of-plane distance before projection (model units) |
| `area_3d` | float | Approximate face area in 3D (two-triangle split) |
| `area_2d` | float | Area of the unfolded panel (shoelace formula) |
| `centroid_3d` | (3,) float64 | 3D centroid of the face for traceability |
| `normal` | (3,) float64 | Unit normal of the best-fit plane |

The area distortion `|area_2d - area_3d| / area_3d` should be below 0.01
(1%) for faces that are genuinely near-planar.

#### `UnfoldReport`

Aggregate statistics printed after a batch unfolding:

| Field | Description |
|---|---|
| `n_panels` | Total number of panels exported |
| `max_planarity_residual` | Worst-case residual across all panels |
| `mean_planarity_residual` | Mean residual across all panels |
| `max_area_distortion` | Maximum relative area change (dimensionless) |
| `warnings` | Per-face warnings for faces that exceed the tolerance |

---

### Functions

#### `unfold_mesh(mesh, planarity_tolerance=1e-3, verbose=True)`

Unfolds all faces of a mesh and returns a list of `FlatPanel` objects and
an `UnfoldReport`. This is the main entry point for the unfolding step.

```python
from src.io.panel_exporter import unfold_mesh

panels, report = unfold_mesh(mesh)
report.print()
```

#### `export_svg(panels, filepath, ...)`

Exports the flat panels to an SVG file arranged in a grid. When
`colour_by_residual=True` (the default), each panel is coloured on a
green-to-red gradient proportional to its planarity residual, giving an
immediate visual map of fabrication readiness. The SVG canvas defaults to
A4 landscape (297 mm x 210 mm). Files are written atomically to prevent
corrupt output if an error occurs mid-write.

#### `export_dxf(panels, filepath, scale_factor=1000.0, layer_name="PANELS")`

Exports the flat panels to a DXF R2010 (AC1024) file. Each panel is written
as a closed `LWPOLYLINE` entity. A companion `MTEXT` label at the panel
centroid shows the face index and planarity residual. The `scale_factor`
parameter converts model units to millimetres: the default of `1000.0`
assumes the mesh is in metres after normalisation by the preprocessor.

The DXF is written entirely from scratch without any external library
dependency, guaranteeing compatibility with AutoCAD, Rhino, LibreCAD, and
most CNC controllers.

#### `export_panels(mesh, output_dir, stem="mesh", ...)` — Recommended Entry Point

Convenience wrapper that calls `unfold_mesh`, `export_svg`, and `export_dxf`
in a single call. Output files are named `<stem>.svg` and `<stem>.dxf`
inside `output_dir`.

```python
from src.io.obj_handler import load_obj
from src.io.panel_exporter import export_panels
from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig

mesh = load_obj("data/input/generated/saddle_8x8.obj")
config = OptimisationConfig(weights={"planarity": 10.0, "fairness": 1.0, "closeness": 5.0})
result = MeshOptimiser(config).optimise(mesh)

report = export_panels(
    result.optimised_mesh,
    output_dir="data/output/panels",
    stem="saddle_8x8"
)
```

---

## Output Locations

| Format | Default location | Opened by |
|---|---|---|
| `.obj` (optimised mesh) | `data/output/optimised_meshes/` | Any 3D viewer (Blender, MeshLab) |
| `.svg` (flat panels) | `data/output/panels/` | Web browser, Inkscape, Illustrator |
| `.dxf` (flat panels) | `data/output/panels/` | AutoCAD, Rhino, LibreCAD, CNC controller |

---

## References

- Wavefront Technologies (1992). "Object Files (.obj)." OBJ format specification.
  Available at: http://paulbourke.net/dataformats/obj/
- Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.
- Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.
- Autodesk (2024). DXF R2010 Reference (AC1024).
  https://help.autodesk.com/view/OARX/2024/ENU/
