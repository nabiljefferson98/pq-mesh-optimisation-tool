"""
src/io/obj_handler.py

OBJ file import and export for planar quad mesh optimisation.

This module provides the two primary I/O functions for the project: 'load_obj'
for importing a mesh from a Wavefront OBJ file, and 'save_obj' for exporting
an optimised mesh back to OBJ format. Both functions are designed to handle
the inconsistencies found in real mesh files exported from architectural CAD
tools such as Blender, Rhino, and ArchiCAD.

The OBJ format
--------------
A Wavefront OBJ file stores mesh geometry as a plain-text sequence of vertex
lines ('v x y z') and face lines ('f i j k l'). Face indices in OBJ files
are one-based (the first vertex is index 1, not 0) and may be negative
(relative to the end of the current vertex list). Face lines may also carry
texture and normal annotations in the form 'v/vt/vn'. This module
handles all of these conventions transparently.

Failure modes handled
---------------------
The following real-dataset edge cases are handled explicitly, as they
frequently appear in OBJ files exported from architectural modelling tools
but are absent from synthetically generated test meshes:

  Mixed topology:
    A single OBJ file may contain both triangular and quad faces. The parser
    separates them; quad faces are used directly, and triangle faces are either
    ignored (if quads are present) or paired into quads (if no quads exist).

  Relative (negative) face indices:
    OBJ allows indices such as '-1' meaning "the last vertex added so far".
    These are resolved to absolute zero-based indices during parsing.

  Texture and normal annotations:
    Face tokens of the form 'v/vt' or 'v/vt/vn' are parsed correctly; only
    the vertex index before the first slash is used.

  Malformed lines:
    Lines with unparseable vertex coordinates or face tokens are skipped
    with a warning rather than crashing the loader.

  Fully triangulated meshes:
    When a mesh has been triangulated by an exporter (the default in
    Blender and most CAD tools), the loader attempts to pair adjacent
    triangles that share an edge back into quads. This recovers the original
    quad topology for meshes that were quadrilateral before export.

References
----------
Wavefront Technologies (1992). "Object Files (.obj)." Wavefront OBJ format
  specification. Available at: http://paulbourke.net/dataformats/obj/

Author: Muhammad Nabil
Date: March 2026
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.core.mesh import QuadMesh


# ============================================================================
# PUBLIC API
# ============================================================================
def load_obj(filepath: str, require_quads: bool = False) -> QuadMesh:
    """
    Load a quad mesh from a Wavefront OBJ file.

    Parses the OBJ file, separates faces by type (triangles and quads),
    and returns a 'QuadMesh' containing only quad faces. If the file
    contains no quad faces but does contain triangles, the function
    attempts to pair adjacent triangles into quads using the shared-edge
    method. If pairing fails and 'require_quads' is False, a triangular
    mesh is returned as a fallback.

    Parameters
    ----------
    filepath : str
        Path to the '.obj' file to load.
    require_quads : bool, optional
        If True, raises a 'ValueError' when no quad faces can be produced
        (either directly or via triangle pairing). If False, fall back to
        a triangular mesh. Default is False.

    Returns
    -------
    QuadMesh
        A mesh containing only quad faces (or triangle faces if pairing
        failed and 'require_quads' is False).

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If no vertices are found in the file, if no faces of any supported
        type are found, or if 'require_quads=True' and no quads could be
        produced.

    Notes
    -----
    N-gons faces (faces with five or more vertices) are silently skipped.
    A summary of face counts by type is printed to standard output after
    parsing so that the caller can see what the file contained.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    vertices, all_faces = _parse_obj(path)

    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    if not all_faces:
        raise ValueError(f"No faces found in {path}")

    vert_array = np.array(vertices, dtype=np.float64)

    # Partition faces by vertex count
    tri_faces = [f for f in all_faces if len(f) == 3]
    quad_faces = [f for f in all_faces if len(f) == 4]
    other_faces = [f for f in all_faces if len(f) not in (3, 4)]

    print(f"✓ Parsed {len(vert_array)} vertices, {len(all_faces)} faces total")
    print(f"  Quads:     {len(quad_faces)}")
    print(f"  Triangles: {len(tri_faces)}")
    if other_faces:
        sizes = sorted({len(f) for f in other_faces})
        print(f"  Other (n-gons, sizes={sizes}): {len(other_faces)} — skipped")

    if quad_faces:
        if tri_faces:
            print(
                f"  ⚠️  Ignoring {len(tri_faces)} triangle face(s); "
                "optimiser requires quads."
            )
        face_array = np.array(quad_faces, dtype=np.int32)

    elif tri_faces:
        print("  ℹ️  No quads found. Attempting triangle→quad pairing...")
        paired = _pair_triangles_to_quads(tri_faces)
        if paired:
            print(f"  ✓ Paired {len(paired)} quads from {len(tri_faces)} triangles.")
            face_array = np.array(paired, dtype=np.int32)
        else:
            if require_quads:
                raise ValueError(
                    "No quad faces found and triangle pairing failed.\n"
                    "Please remesh to a quad-dominant topology before optimisation.\n"
                    "Recommended tools: Blender (QuadRemesh), Instant Meshes (free)."
                )
            print(
                "  ⚠️  Triangle pairing failed. Falling back to triangle mesh "
                "(planarity checks will be trivial)."
            )
            face_array = np.array(tri_faces, dtype=np.int32)

    else:
        raise ValueError(
            f"No quad or triangle faces found in {path}.\n"
            "Supported face types: triangles (f v1 v2 v3) and quads (f v1 v2 v3 v4)."
        )

    print(f"✓ Loaded mesh: {len(vert_array)} vertices, {len(face_array)} faces\n")
    return QuadMesh(vert_array, face_array)


def save_obj(mesh: QuadMesh, filepath: str) -> None:
    """
    Export a mesh to a Wavefront OBJ file.

    Writes vertex positions and face connectivity in standard OBJ format.
    Face indices are written as one-based integers as required by the OBJ
    specification. The output file includes a short header comment with
    the vertex and face counts. Parent directories are created automatically
    if they do not already exist.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to export.
    filepath : str
        Output file path. Maybe relative or absolute, but must not contain
        directory traversal sequences ('..').

    Raises
    ------
    ValueError
        If the filepath contains '..' path components, which could cause
        the file to be written outside the intended directory.

    Notes
    -----
    Vertex coordinates are written with six decimal places of precision,
    which is enough for architectural-scale meshes and keeps file sizes
    manageable. Files are written in UTF-8 encoding.
    """
    out_path = Path(filepath).resolve()
    # Guard against path traversal: reject paths that escape the current
    # working directory tree via '../' sequences in the *original* input.
    raw = Path(filepath)
    if ".." in raw.parts:
        raise ValueError(
            f"Unsafe output path — directory traversal sequences ('..') "
            f"are not permitted: {filepath!r}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# OBJ file generated by PQ Mesh Optimisation Tool\n")
        f.write(f"# Vertices: {len(mesh.vertices)}\n")
        f.write(f"# Faces: {len(mesh.faces)}\n\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in mesh.faces:
            indices = " ".join(str(idx + 1) for idx in face)
            f.write(f"f {indices}\n")

    print(f"✓ Saved {len(mesh.faces)} faces to {out_path}")


# ============================================================================
# PRIVATE HELPERS
# ============================================================================
def _parse_obj(path: Path) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Parse a Wavefront OBJ file into raw vertex and face lists.

    Reads the file line by line, extracting vertex coordinates from 'v'
    lines and face connectivity from 'f' lines. Comment lines (starting
    with '#') and blank lines are ignored. Material library references
    ('mtllib') and object group markers ('g', 'o', 's') are also ignored,
    as they are not relevant to the mesh geometry.

    Face tokens of the form 'v/vt', 'v/vt/vn', or 'v//vn' are handled
    by splitting on '/' and using only the first component. Negative
    face indices are resolved to absolute zero-based indices relative to
    the number of vertices read so far at the point the face line appears.
    Faces with fewer than three valid vertex indices are discarded.

    Malformed vertex or face lines (for example, lines with non-numeric
    values) are counted, and the first five are reported to standard output.
    If more than five malformed lines are present, a summary count is
    printed instead to avoid flooding the console.

    Parameters
    ----------
    path : pathlib.Path
        Path to the OBJ file to parse.

    Returns
    -------
    tuple
        A two-element tuple (vertices, faces) where:
        'vertices' is a list of [x, y, z] float lists, and
        'faces' is a list of integer index lists (zero-based).
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    n_malformed = 0

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                try:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    n_malformed += 1
                    if n_malformed <= 5:
                        print(
                            f"  ⚠️  Skipping malformed vertex at line "
                            f"{lineno}: {line!r}"
                        )
            elif parts[0] == "f" and len(parts) >= 4:
                face_indices: List[int] = []
                bad_token = False
                for token in parts[1:]:
                    try:
                        raw = int(token.split("/")[0])
                    except ValueError:
                        bad_token = True
                        break
                    # OBJ negative index: -1 = last vertex added so far
                    idx = (len(vertices) + raw) if raw < 0 else (raw - 1)
                    face_indices.append(idx)
                if bad_token:
                    n_malformed += 1
                    if n_malformed <= 5:
                        print(
                            f"  ⚠️  Skipping malformed face at line {lineno}: {line!r}"
                        )
                elif len(face_indices) >= 3:
                    faces.append(face_indices)

    if n_malformed > 5:
        print(f"  ⚠️  ({n_malformed - 5} additional malformed lines skipped)")

    return vertices, faces


def _pair_triangles_to_quads(tri_faces: List[List[int]]) -> List[List[int]]:
    """
    Merge pairs of adjacent triangles sharing an edge into quad faces.

    Many mesh exporters (including Blender's default OBJ exporter and most
    CAD tools) triangulate quad meshes before export, splitting each quad
    into two triangles along a diagonal. This function attempts to reverse
    that triangulation by finding pairs of triangles that share an edge and
    merging them back into a single quad.

    The algorithm builds an edge-to-face lookup table from all triangle
    edges. For each edge shared by exactly two triangles, the two triangles
    are merged into a quad by combining their four unique vertices. Vertices
    are ordered as v0, va, v1, vb (counter-clockwise), where v0 and v1 are
    the shared-edge vertices and va and vb are the opposite vertices of each
    triangle. Each triangle is used in at most one pairing (greedy strategy:
    first valid pair wins).

    This approach correctly recovers the original quad topology when the
    triangulation was performed by splitting quads along their shorter
    diagonal, which is the most common case in architectural mesh exports.

    Parameters
    ----------
    tri_faces : list of list of int
        List of triangle faces, each represented as a list of three
        zero-based vertex indices.

    Returns
    -------
    list of int
        List of quad faces, each represented as a list of four zero-based
        vertex indices. May be empty if no adjacent triangle pairs are found.
    """
    edge_to_faces: dict = {}
    for fi, face in enumerate(tri_faces):
        for k in range(3):
            edge = tuple(sorted([face[k], face[(k + 1) % 3]]))
            edge_to_faces.setdefault(edge, []).append(fi)

    used: set = set()
    quads: List[List[int]] = []

    for edge, adj in edge_to_faces.items():
        if len(adj) != 2:
            continue
        fi, fj = adj
        if fi in used or fj in used:
            continue

        fa, fb = tri_faces[fi], tri_faces[fj]
        shared = set(edge)
        a_only = [v for v in fa if v not in shared]
        b_only = [v for v in fb if v not in shared]

        if len(a_only) == 1 and len(b_only) == 1:
            v0, v1 = sorted(edge)  # shared edge vertices
            va = a_only[0]
            vb = b_only[0]
            # Reconstruct CCW quad: v0 → va → v1 → vb
            quads.append([v0, va, v1, vb])
            used.add(fi)
            used.add(fj)

    return quads
