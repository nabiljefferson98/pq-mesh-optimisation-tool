"""
Regression test: Ensure quad meshes are not triangulated during import.

This test was added after discovering that igl.read_triangle_mesh()
automatically triangulates quad faces, causing all planarity deviations
to be zero (since triangles are inherently planar).

Date: 2 February 2026
Critical bug: Week 2, Day 1
"""

from pathlib import Path

import numpy as np
import pytest

from src.io.obj_handler import load_obj, save_obj

igl = pytest.importorskip("igl", reason="igl not available on this Python version")


def test_manual_parser_preserves_quads(tmp_path):
    """
    Verify that our manual OBJ parser preserves quad topology.

    This is a regression test for the triangulation bug discovered
    in Week 2.
    """
    # Create a simple quad
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], dtype=np.float64)
    F_quad = np.array([[0, 1, 2, 3]], dtype=np.int32)

    # Save as temporary OBJ file using igl's camelCase function
    test_file = tmp_path / "test_quad.obj"
    igl.writeOBJ(str(test_file), V, F_quad)  # ✅ CORRECTED: writeOBJ not write_obj

    # Load using our parser
    mesh = load_obj(str(test_file))

    # CRITICAL ASSERTIONS
    assert (
        mesh.n_faces == 1
    ), f"Expected 1 quad, got {mesh.n_faces} faces (triangulation detected!)"

    assert (
        mesh.faces.shape[1] == 4
    ), f"Expected quads (4 vertices/face), got {mesh.faces.shape[1]} vertices/face"

    # Verify vertex preservation
    assert mesh.n_vertices == 4, f"Expected 4 vertices, got {mesh.n_vertices}"


def test_compare_triangle_vs_quad_loading(tmp_path):
    """
    Demonstrate the triangulation issue with igl.read_triangle_mesh().

    This test documents the bug for future reference.
    """
    # Create a single quad
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    F_quad = np.array([[0, 1, 2, 3]], dtype=np.int32)

    test_file = tmp_path / "quad_comparison.obj"
    igl.writeOBJ(str(test_file), V, F_quad)  # ✅ CORRECTED

    # Method 1: igl.read_triangle_mesh (WILL TRIANGULATE)
    verts_tri, faces_tri = igl.read_triangle_mesh(str(test_file))

    # Method 2: Our manual parser (PRESERVES QUADS)
    mesh_quad = load_obj(str(test_file))

    # Document the behaviour difference
    assert (
        len(faces_tri) == 2
    ), "igl.read_triangle_mesh should triangulate 1 quad → 2 triangles"

    assert mesh_quad.n_faces == 1, "Manual parser should preserve 1 quad"

    print("\n" + "=" * 60)
    print("TRIANGULATION BEHAVIOUR COMPARISON")
    print("=" * 60)
    print(
        f"igl.read_triangle_mesh():  {len(faces_tri)} faces"
        f" (shape: {faces_tri.shape})"
    )
    print(
        f"Manual OBJ parser:         {mesh_quad.n_faces} faces (shape: {mesh_quad.faces.shape})"  # noqa: E501
    )
    print("=" * 60)


def test_5x5_grid_has_25_quads_not_50_triangles():
    """
    Regression test for the specific bug found in plane_5x5_noisy.obj.

    Original issue: Mesh loaded with 50 triangular faces instead of 25 quads.
    Result: All planarity deviations were zero.
    """
    # This test uses the actual generated mesh
    mesh_path = Path("data/input/generated/plane_5x5_noisy.obj")

    if not mesh_path.exists():
        pytest.skip(f"Test mesh not found: {mesh_path}")

    mesh = load_obj(str(mesh_path))

    # CRITICAL: Must be 25 quads, not 50 triangles
    assert (
        mesh.n_faces == 25
    ), f"5×5 grid should have 25 quads, got {mesh.n_faces} faces"

    assert (
        mesh.faces.shape[1] == 4
    ), f"Faces should be quads (4 verts), got {mesh.faces.shape[1]}-gons"

    assert (
        mesh.n_vertices == 36
    ), f"5×5 grid should have 36 vertices, got {mesh.n_vertices}"


def test_round_trip_quad_preservation(tmp_path):
    """
    Test that saving and reloading a quad mesh preserves topology.

    This tests our custom save_obj() function.
    """
    # Create a 2-quad mesh
    V = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # First quad
            [1, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [1, 1, 0],  # Second quad (shares edge)
        ],
        dtype=np.float64,
    )

    F = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

    from src.core.mesh import QuadMesh

    mesh_original = QuadMesh(V, F)

    # Save
    output_file = tmp_path / "round_trip.obj"
    save_obj(mesh_original, str(output_file))

    # Reload
    mesh_reloaded = load_obj(str(output_file))

    # Verify topology preserved
    assert (
        mesh_reloaded.n_faces == 2
    ), f"Expected 2 quads after round-trip, got {mesh_reloaded.n_faces}"

    assert (
        mesh_reloaded.faces.shape[1] == 4
    ), f"Expected quads after round-trip, got {mesh_reloaded.faces.shape[1]}-gons"

    assert (
        mesh_reloaded.n_vertices == 8
    ), f"Expected 8 vertices after round-trip, got {mesh_reloaded.n_vertices}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
