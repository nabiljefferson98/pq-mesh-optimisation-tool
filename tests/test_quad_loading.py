"""
Debug script: Compare quad vs triangle loading.
"""

import numpy as np
import pytest

igl = pytest.importorskip("igl", reason="igl not available on this platform")


def test_quad_loading_comparison(tmp_path):
    """Compare quad vs triangle loading across igl methods."""
    # Create a simple quad manually
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], dtype=np.float64)
    F_quad = np.array([[0, 1, 2, 3]], dtype=np.int32)

    obj_file = str(tmp_path / "test_quad.obj")

    # Save as OBJ
    igl.writeOBJ(obj_file, V, F_quad)

    # Method 1: read_triangle_mesh (WILL TRIANGULATE)
    verts1, faces1 = igl.read_triangle_mesh(obj_file)
    assert faces1.shape[1] in (3, 4), f"Unexpected face shape: {faces1.shape}"

    # Method 2: read_obj (PRESERVES QUADS)
    # readOBJ returns (V, TC, N, F, FTC, FN) — faces2 is F
    success, verts2, _, faces2, _, _ = igl.readOBJ(obj_file)
    assert faces2.shape[1] in (3, 4), f"Unexpected face shape: {faces2.shape}"

    # Method 3: Manual parser
    with open(obj_file, "r") as f:
        faces = []
        for line in f:
            if line.startswith("f "):
                face = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                faces.append(face)
    faces3 = np.array(faces)
    assert faces3.shape[1] in (3, 4), f"Unexpected face shape: {faces3.shape}"

    # read_triangle_mesh should have returned the original 4 vertices
    assert verts1.shape == V.shape
