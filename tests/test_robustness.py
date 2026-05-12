"""
tests/test_robustness.py

Regression tests for bugs and security issues fixed in the audit:
  B1  — Negative face indices in QuadMesh.__init__
  B2  — Out-of-bounds id in get_vertex_faces / get_face_vertices
  B3  — NaN/Inf in update_vertices
  B4  — Malformed OBJ lines (bad floats, bad face tokens)
  B5  — compute_numerical_gradient restores mesh state even on exception
  B6  — _create_callback restores mesh state even on exception
  B7  — validate_mesh handles triangle faces without crashing
  B8  — unfold_mesh on 0-face mesh returns empty list
  B9  — suggest_weights_for_mesh on empty mesh returns defaults
  B10 — compute_total_gradient with missing weight key raises ValueError
  B11 — gradient_for_scipy / energy_for_scipy replace NaN with safe fallback
  S1  — save_obj raises ValueError on directory-traversal path
  S2  — export_dxf sanitises layer_name
  E3  — _merge_duplicate_vertices emits UserWarning for large meshes
"""

import os

import numpy as np
import pytest

from src.core.mesh import QuadMesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_quad() -> QuadMesh:
    """A single flat unit quad mesh."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return QuadMesh(V, F)


def _small_grid(rows: int = 3, cols: int = 3) -> QuadMesh:
    """
    Create a small flat planar grid of (rows-1)*(cols-1) quads.
    """
    verts = []
    for r in range(rows):
        for c in range(cols):
            verts.append([c, r, 0.0])
    V = np.array(verts, dtype=np.float64)

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v0 = r * cols + c
            v1 = v0 + 1
            v2 = v0 + cols + 1
            v3 = v0 + cols
            faces.append([v0, v1, v2, v3])
    F = np.array(faces, dtype=np.int32)
    return QuadMesh(V, F)


# ===========================================================================
# T1 — B1: Negative face index raises ValueError
# ===========================================================================


def test_negative_face_index_raises():
    """QuadMesh.__init__ must reject faces with negative vertex indices (B1)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    F = np.array([[-1, 1, 2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="negative"):
        QuadMesh(V, F)


# ===========================================================================
# T2 — B2: Out-of-bounds IDs raise IndexError
# ===========================================================================


def test_get_vertex_faces_oob_raises():
    """get_vertex_faces(-1) and get_vertex_faces(n) must raise IndexError (B2)."""
    mesh = _unit_quad()
    with pytest.raises(IndexError):
        mesh.get_vertex_faces(-1)
    with pytest.raises(IndexError):
        mesh.get_vertex_faces(mesh.n_vertices)


def test_get_face_vertices_oob_raises():
    """get_face_vertices(-1) and get_face_vertices(n) must raise IndexError (B2)."""
    mesh = _unit_quad()
    with pytest.raises(IndexError):
        mesh.get_face_vertices(-1)
    with pytest.raises(IndexError):
        mesh.get_face_vertices(mesh.n_faces)


# ===========================================================================
# T3 — B3: update_vertices with NaN or Inf raises ValueError
# ===========================================================================


def test_update_vertices_nan_raises():
    """update_vertices with NaN must raise ValueError (B3)."""
    mesh = _unit_quad()
    bad = mesh.vertices.copy()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN|Inf|finite"):
        mesh.update_vertices(bad)


def test_update_vertices_inf_raises():
    """update_vertices with Inf must raise ValueError (B3)."""
    mesh = _unit_quad()
    bad = mesh.vertices.copy()
    bad[1, 2] = float("inf")
    with pytest.raises(ValueError, match="NaN|Inf|finite"):
        mesh.update_vertices(bad)


# ===========================================================================
# T4 — B4: Malformed OBJ lines are skipped without crashing
# ===========================================================================


def test_load_malformed_obj_partial(tmp_path):
    """
    An OBJ file with bad vertex/face lines must not crash; valid entries
    are loaded normally (B4).
    """
    from src.io.obj_handler import load_obj

    # 4 valid vertex lines + 1 good face + 1 malformed face token line.
    # After parsing: 4 vertices (0-indexed 0-3), 1 valid quad face.
    obj_content = "\n".join(
        [
            "# test obj with bad lines",
            "v 0.0 0.0 0.0",
            "v 1.0 0.0 0.0",
            "v 1.0 1.0 0.0",
            "v 0.0 1.0 0.0",
            "f 1 BAD/2 3 4",  # malformed face token -> skipped
            "f 1 2 3 4",  # valid face (1-indexed, all within 4 verts)
        ]
    )
    p = tmp_path / "bad.obj"
    p.write_text(obj_content)

    # Should not raise; the valid face is loaded
    mesh = load_obj(str(p))
    assert mesh is not None
    assert mesh.n_faces >= 1


# ===========================================================================
# T5 — B8: unfold_mesh on 0-face mesh returns empty list, no crash
# ===========================================================================


def test_unfold_mesh_empty():
    """unfold_mesh on a 0-face mesh must return ([], report) without crashing (B8)."""
    from src.io.panel_exporter import unfold_mesh

    V = np.zeros((4, 3), dtype=np.float64)
    F = np.empty((0, 4), dtype=np.int32)
    empty_mesh = QuadMesh(V, F)

    panels, report = unfold_mesh(empty_mesh)

    assert panels == []
    assert report.n_panels == 0


# ===========================================================================
# T6 — S2: export_dxf sanitises layer_name
# ===========================================================================


def test_export_dxf_layer_name_sanitised(tmp_path):
    """export_dxf must sanitise special characters in layer_name (S2)."""
    from src.io.panel_exporter import export_dxf, unfold_mesh

    mesh = _small_grid()
    panels, _ = unfold_mesh(mesh)

    out_path = str(tmp_path / "test.dxf")

    # Pass a layer name that contains injection-like characters
    dangerous_name = "../../../etc/passwd\x00<script>"
    export_dxf(panels, out_path, layer_name=dangerous_name)

    assert os.path.exists(out_path)
    content = open(out_path).read()
    # Original dangerous chars must NOT appear verbatim
    assert "../../../etc/passwd" not in content
    assert "<script>" not in content


# ===========================================================================
# T7 — B5: compute_numerical_gradient restores mesh even when energy_func raises
# ===========================================================================


def test_numerical_gradient_restores_mesh_on_exception():
    """
    If energy_func raises inside compute_numerical_gradient, the mesh vertices
    must be restored to their original values (B5).
    """
    from src.optimisation.gradients import compute_numerical_gradient

    mesh = _small_grid()
    original_verts = mesh.vertices.copy()

    call_count = [0]

    def bad_energy_func(m):
        call_count[0] += 1
        if call_count[0] > 3:
            raise RuntimeError("simulated energy failure")
        return 0.0

    try:
        compute_numerical_gradient(mesh, bad_energy_func)
    except RuntimeError:
        pass  # expected

    np.testing.assert_array_equal(
        mesh.vertices,
        original_verts,
        err_msg="Mesh vertices were not restored after exception in energy_func",
    )


# ===========================================================================
# T8 — B10: compute_total_gradient with missing weight key raises ValueError
# ===========================================================================


def test_compute_total_gradient_missing_weight_raises():
    """compute_total_gradient raises ValueError for a missing weight key (B10)."""
    from src.optimisation.gradients import compute_total_gradient

    mesh = _small_grid()
    incomplete_weights = {"planarity": 1.0}  # missing 'fairness' and 'closeness'

    with pytest.raises((ValueError, KeyError)):
        compute_total_gradient(mesh, incomplete_weights)


# ===========================================================================
# T9 — B9: suggest_weights_for_mesh on empty mesh returns defaults, no crash
# ===========================================================================


def test_suggest_weights_empty_mesh():
    """suggest_weights_for_mesh on an empty mesh returns defaults, no crash (B9)."""
    from src.preprocessing.preprocessor import suggest_weights_for_mesh

    V = np.zeros((0, 3), dtype=np.float64)
    F = np.empty((0, 4), dtype=np.int32)
    empty_mesh = QuadMesh(V, F)

    weights = suggest_weights_for_mesh(empty_mesh)

    assert "planarity" in weights
    assert "fairness" in weights
    assert "closeness" in weights
    # All weights should be finite positive numbers
    for key, val in weights.items():
        if key != "angle_balance":
            assert np.isfinite(val) and val >= 0, f"Weight '{key}' is invalid: {val}"


# ===========================================================================
# T10 — B7: validate_mesh handles triangle faces without crashing
# ===========================================================================


def test_validate_mesh_triangle_face_no_crash():
    """
    validate_mesh must not crash when a mesh has 3-vertex faces (B7).
    It may return (False, reason) but must NOT raise an exception.
    """
    from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig

    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2]], dtype=np.int32)  # triangle, not quad
    mesh = QuadMesh(V, F)

    optimiser = MeshOptimiser(OptimisationConfig())

    # Must not raise; is_valid may be True or False
    is_valid, message = optimiser.validate_mesh(mesh)
    assert isinstance(is_valid, bool)
    assert isinstance(message, str)


# ===========================================================================
# T11 — S1: save_obj raises ValueError on directory-traversal path
# ===========================================================================


def test_save_obj_path_traversal_raises(tmp_path):
    """save_obj must raise ValueError when given a path traversal (S1)."""
    from src.io.obj_handler import save_obj

    mesh = _unit_quad()
    traversal_path = str(tmp_path / ".." / ".." / "evil.obj")

    with pytest.raises(ValueError, match="traversal|path|directory"):
        save_obj(mesh, traversal_path)


# ===========================================================================
# T12 — E3: _merge_duplicate_vertices emits UserWarning for large meshes
# ===========================================================================


def test_merge_duplicate_vertices_large_mesh_scalable_path():
    """
    Large meshes should use the scalable duplicate-merge path without emitting
    the legacy O(n²) warning.
    """
    import warnings

    import numpy as np

    from src.preprocessing.preprocessor import _merge_duplicate_vertices

    n = 2501
    vertices = np.zeros((n, 3), dtype=np.float64)
    vertices[:, 0] = np.arange(n, dtype=np.float64)

    # Introduce one duplicate so the remapping path is actually exercised.
    vertices[-1] = vertices[0]

    faces = [
        [0, 1, 2, 3],
        [n - 4, n - 3, n - 2, n - 1],
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_vertices, new_faces, n_merged = _merge_duplicate_vertices(
            vertices,
            faces,
            threshold=1e-12,
        )

    assert n_merged == 1
    assert new_vertices.shape == (n - 1, 3)
    assert len(new_faces) == len(faces)
    assert new_faces[1][-1] == 0
    assert not any("O(n²)" in str(w.message) for w in caught), (
        f"Did not expect the legacy O(n²) warning, got: "
        f"{[str(w.message) for w in caught]}"
    )
