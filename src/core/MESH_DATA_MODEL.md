# Mesh Data Model

`src/core/mesh.py`

This document describes the `QuadMesh` class, which is the central data
structure of the entire project. Every other module — the energy functions,
gradient functions, optimiser, preprocessor, and exporters — operates on a
`QuadMesh` instance. Understanding this class is essential before reading any
other part of the codebase.

---

## What Is a QuadMesh?

A quad mesh is a surface made up entirely of four-sided polygons called
quadrilateral faces, or "quads". Each quad is defined by four vertices
connected in a loop. In architectural geometry, quad meshes are preferred
because they can approximate developable and near-planar surfaces in a way
that is manufacturable: if each face is close to planar, the panel can be
cut from a flat sheet of material.

The `QuadMesh` class stores two things:

- **Vertices**: the 3D coordinates of every point in the mesh.
- **Faces**: the connectivity, meaning which four vertices form each quad.

Everything else — the Laplacian matrix, adjacency tables, scatter matrices —
is derived from these two arrays and computed only when first needed.

---

## Core Arrays

### `vertices`

```
shape:  (n_vertices, 3)
dtype:  float64
```

Each row is one vertex, storing its x, y, and z coordinate. This array is
updated in-place by the optimiser at every iteration as vertices are moved
to improve planarity. It is the only part of the mesh that changes during
optimisation.

### `faces`

```
shape:  (n_faces, 4)
dtype:  int32
```

Each row is one quad face, storing four integer indices into the `vertices`
array. The indices are zero-based. The order of the four vertices in each row
follows the quad boundary in a consistent winding direction (either clockwise
or counter-clockwise), which ensures the face normal is well-defined. This
array is fixed at construction and never modified during optimisation.

### `vertices_original`

```
shape:  (n_vertices, 3)
dtype:  float64
```

A copy of `vertices` taken at construction time. This is used by the
**closeness energy** term, which measures how far each vertex has moved from
its original position. Without this reference, the optimiser would be free
to collapse the mesh to a degenerate flat plane. The `vertices_original`
array is never updated — it remains the fixed anchor for the entire run.

---

## Constructing a QuadMesh

```python
import numpy as np
from src.core.mesh import QuadMesh

# Four vertices forming a single quad face in the XY plane
vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=float)

faces = np.array([[0, 1, 2, 3]])

mesh = QuadMesh(vertices, faces)
print(mesh.n_vertices)  # 4
print(mesh.n_faces)     # 1
```

The constructor validates the inputs immediately. It will raise a `ValueError`
if any of the following conditions are violated:

- `vertices` is not a two-dimensional array with exactly three columns.
- `faces` is not a two-dimensional array.
- Any face index is negative.
- Any face index is greater than or equal to the number of vertices (out of
  bounds).

---

## Lazy-Cached Topology Properties

Computing topology quantities from scratch at every optimisation iteration
would be very slow. Instead, these quantities are computed once on first
access and stored on the mesh object. Because face connectivity never changes
during optimisation, these caches remain valid for the entire run.

### `scatter_matrix`

```
shape:  (n_vertices, n_faces * 4)
type:   scipy.sparse.csr_matrix, float64
```

The planarity gradient produces a contribution for each vertex of each face.
To assemble the full gradient, contributions must be summed back to the
correct vertex row. The scatter matrix encodes this as a single sparse matrix
multiplication, which is faster than a Python loop over all faces.

Entry `(v, k)` of the scatter matrix equals 1 if the k-th entry in the
flattened face array belongs to vertex `v`. Multiplying the scatter matrix by
a `(n_faces * 4, 3)` array of per-face contributions gives the correct
`(n_vertices, 3)` gradient array.

### `laplacian`

```
shape:  (n_vertices, n_vertices)
type:   scipy.sparse.csr_matrix, float64
```

The combinatorial Laplacian matrix encodes the relationship between each
vertex and its direct neighbours. For a vertex `i` with neighbours
`j_1, j_2, ..., j_k`, the Laplacian satisfies:

```
L[i, i]   =  k       (degree of vertex i)
L[i, j_r] = -1       (for each neighbour j_r)
L[i, *]   =  0       (all other entries)
```

Multiplying the Laplacian by the vertex matrix `V` of shape
`(n_vertices, 3)` gives the Laplacian coordinates, where row `i` is the
vector from the average neighbour position to vertex `i` itself:

    delta_i = V_i - (1/k) * sum_r(V_{j_r})

This quantity directly measures local smoothness and is used in the
**fairness energy** term. The uniform weighting (all neighbours contribute
equally) is used throughout, consistent with the formulation in Liu et al.
(2006).

### `vertex_face_ids_padded`

```
shape:  (n_vertices, max_valence)
dtype:  int32
```

For each vertex, this table stores the indices of all faces that contain it.
It is stored as a fixed-width two-dimensional array, padded with `-1` where
a vertex has fewer incident faces than the maximum in the mesh. This format
is required by the Numba-compiled gradient kernels, which cannot use Python
lists.

The maximum stored valence is capped at 16 to avoid excessive memory use.
For typical architectural quad meshes, interior vertices have exactly four
incident faces (valence 4), and boundary vertices have two or three. The
angle balance gradient only acts on interior vertices with exactly four
incident faces, so the cap does not affect correctness for standard meshes.

### `angle_balance_scratch`

A tuple of five pre-allocated NumPy arrays used by the Numba angle balance
gradient kernel. These arrays hold temporary intermediate values during the
computation. Pre-allocating them here avoids repeated memory allocation
inside the compiled kernel, which would add measurable overhead per
iteration.

| Buffer | Shape | Dtype | Purpose |
|---|---|---|---|
| `scratch_gvp` | `(n_vertices, 4, 3)` | float64 | Previous-vertex gradient contributions |
| `scratch_gvn` | `(n_vertices, 4, 3)` | float64 | Next-vertex gradient contributions |
| `scratch_prev` | `(n_vertices, 4)` | int32 | Previous-vertex indices per incident face |
| `scratch_next` | `(n_vertices, 4)` | int32 | Next-vertex indices per incident face |
| `scratch_active` | `(n_vertices,)` | int8 | Flags marking active 4-valent interior vertices |

---

## Key Methods

### `get_vertex_faces(vertex_id)`

Returns the list of face indices that contain a given vertex. This is the
one-ring face neighbourhood of that vertex. The result is cached on first
call.

### `get_face_vertices(face_id)`

Returns the 3D coordinates of the four corners of a given face as a
`(4, 3)` NumPy array.

### `update_vertices(new_positions)`

Replaces `self.vertices` with a validated copy of `new_positions`. Raises a
`ValueError` if the shape does not match or if any value is `NaN` or
infinite. The optimiser updates vertices directly by assignment for speed,
but external code should call this method to benefit from the validation.

### `reset_to_original()`

Restores `self.vertices` from `self.vertices_original`. Useful for re-running
optimisation from scratch without reloading the mesh from disk.

### `reset_topology_cache()`

Invalidates all cached topology quantities so they are recomputed on next
access. This is only needed if face connectivity is modified after
construction, which does not occur in the normal optimisation workflow.

---

## Why Faces Do Not Change During Optimisation

The optimisation problem solved in this project is purely a vertex relocation
problem. Given a fixed mesh topology (the connectivity encoded in `faces`),
the optimiser moves the vertices to make each face as close to planar as
possible while preserving the overall shape. Because topology never changes,
all topology-derived caches computed from `faces` remain valid from
construction until the mesh is replaced. This design choice is fundamental to
the efficiency of the pipeline.

---

## Supported Face Types

The class is designed for quad meshes (four vertices per face). Triangular
faces (three vertices per face) are accepted without error for compatibility
with mixed-topology OBJ files, but the angle balance energy and its gradient
are only defined for interior vertices of quad meshes and will be silently
skipped for triangular faces.

---

## References

- Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.
- Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.
- Crane, K., de Goes, F., Desbrun, M., and Schroder, P. (2013).
  "Digital geometry processing with discrete exterior calculus."
  ACM SIGGRAPH 2013 Courses.
