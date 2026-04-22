"""
Tests for src/io/panel_exporter.py
"""

import os
import tempfile

import numpy as np
import pytest

from src.core.mesh import QuadMesh
from src.io.panel_exporter import (
    FlatPanel,
    UnfoldReport,
    _quad_area_2d,
    _quad_area_3d,
    export_dxf,
    export_panels,
    export_svg,
    unfold_face,
    unfold_mesh,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _planar_quad() -> np.ndarray:
    """A perfectly flat 1×1 quad in the XY plane."""
    return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)


def _slightly_nonplanar_quad() -> np.ndarray:
    """Quad with one vertex slightly lifted — small planarity residual."""
    return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.01]], dtype=float)


def _flat_quad_mesh(rows: int = 2, cols: int = 2) -> QuadMesh:
    """Small flat quad mesh."""
    vertices = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            vertices.append([c, r, 0.0])
    vertices = np.array(vertices, dtype=np.float64)

    faces = []
    for r in range(rows):
        for c in range(cols):
            v0 = r * (cols + 1) + c
            faces.append([v0, v0 + 1, v0 + (cols + 1) + 1, v0 + (cols + 1)])
    faces = np.array(faces, dtype=np.int32)
    return QuadMesh(vertices, faces)


# ---------------------------------------------------------------------------
# unfold_face
# ---------------------------------------------------------------------------


def test_unfold_face_returns_correct_shapes():
    verts = _planar_quad()
    coords_2d, residual, normal = unfold_face(verts)
    assert coords_2d.shape == (4, 2)
    assert isinstance(residual, float)
    assert normal.shape == (3,)


def test_unfold_face_planar_has_zero_residual():
    verts = _planar_quad()
    _, residual, _ = unfold_face(verts)
    assert residual < 1e-10


def test_unfold_face_nonplanar_has_positive_residual():
    verts = _slightly_nonplanar_quad()
    _, residual, _ = unfold_face(verts)
    assert residual > 0.0


def test_unfold_face_normal_is_unit_vector():
    verts = _planar_quad()
    _, _, normal = unfold_face(verts)
    assert pytest.approx(np.linalg.norm(normal), abs=1e-9) == 1.0


def test_unfold_face_preserves_area():
    """2D area should ≈ 3D area for a planar face."""
    verts = _planar_quad()
    coords_2d, _, _ = unfold_face(verts)
    area_2d = _quad_area_2d(coords_2d)
    area_3d = _quad_area_3d(verts)
    assert pytest.approx(area_2d, rel=1e-4) == area_3d


def test_unfold_face_wrong_shape_raises():
    bad_verts = np.zeros((3, 3))
    with pytest.raises(ValueError, match="Expected \\(4, 3\\)"):
        unfold_face(bad_verts)


def test_unfold_face_coincident_vertices():
    """All vertices at same point — degenerate u_hat fallback should not crash."""
    verts = np.zeros((4, 3))
    coords_2d, residual, normal = unfold_face(verts)
    assert coords_2d.shape == (4, 2)


# ---------------------------------------------------------------------------
# _quad_area_2d / _quad_area_3d
# ---------------------------------------------------------------------------


def test_quad_area_2d_unit_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert pytest.approx(_quad_area_2d(pts), abs=1e-10) == 1.0


def test_quad_area_3d_flat_unit_square():
    verts = _planar_quad()
    assert pytest.approx(_quad_area_3d(verts), abs=1e-10) == 1.0


def test_quad_area_3d_positive():
    verts = _slightly_nonplanar_quad()
    assert _quad_area_3d(verts) > 0.0


# ---------------------------------------------------------------------------
# unfold_mesh
# ---------------------------------------------------------------------------


def test_unfold_mesh_returns_panel_list_and_report():
    mesh = _flat_quad_mesh()
    panels, report = unfold_mesh(mesh, verbose=False)
    assert isinstance(panels, list)
    assert isinstance(report, UnfoldReport)
    assert len(panels) == mesh.n_faces


def test_unfold_mesh_report_counts():
    mesh = _flat_quad_mesh(3, 3)
    panels, report = unfold_mesh(mesh, verbose=False)
    assert report.n_panels == mesh.n_faces
    assert report.max_planarity_residual >= 0.0
    assert report.mean_planarity_residual >= 0.0


def test_unfold_mesh_panel_fields():
    mesh = _flat_quad_mesh()
    panels, _ = unfold_mesh(mesh, verbose=False)
    for panel in panels:
        assert isinstance(panel, FlatPanel)
        assert panel.vertices_2d.shape == (4, 2)
        assert panel.planarity_residual >= 0.0
        assert panel.area_3d > 0.0
        assert panel.centroid_3d.shape == (3,)


def test_unfold_mesh_planarity_warning(capsys):
    """Mesh with large residual (> tolerance) should add warnings."""
    # Large nonplanar mesh — set a tiny tolerance to force warnings
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # flat face
            [2, 0, 0],
            [3, 0, 0],
            [3, 1, 1.0],
            [2, 1, 0],  # nonplanar face
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    mesh = QuadMesh(verts, faces)
    panels, report = unfold_mesh(mesh, planarity_tolerance=1e-6, verbose=False)
    assert len(report.warnings) > 0


def test_unfold_mesh_verbose(capsys):
    mesh = _flat_quad_mesh()
    unfold_mesh(mesh, verbose=True)
    captured = capsys.readouterr()
    assert "UNFOLDING REPORT" in captured.out


# ---------------------------------------------------------------------------
# export_svg
# ---------------------------------------------------------------------------


def test_export_svg_creates_file():
    mesh = _flat_quad_mesh()
    panels, _ = unfold_mesh(mesh, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.svg")
        export_svg(panels, path)
        assert os.path.exists(path)
        content = open(path).read()
        assert "<svg" in content
        assert "<polygon" in content


def test_export_svg_no_colour():
    mesh = _flat_quad_mesh()
    panels, _ = unfold_mesh(mesh, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nc.svg")
        export_svg(panels, path, colour_by_residual=False)
        content = open(path).read()
        assert "b0c8e8" in content  # flat colour


def test_export_svg_empty_panels(capsys):
    """Empty panel list should print a warning and not crash."""
    with tempfile.TemporaryDirectory() as tmp:
        export_svg([], os.path.join(tmp, "empty.svg"))
    captured = capsys.readouterr()
    assert "No panels" in captured.out


def test_export_svg_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "subdir", "nested", "out.svg")
        mesh = _flat_quad_mesh()
        panels, _ = unfold_mesh(mesh, verbose=False)
        export_svg(panels, path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# export_dxf
# ---------------------------------------------------------------------------


def test_export_dxf_creates_file():
    mesh = _flat_quad_mesh()
    panels, _ = unfold_mesh(mesh, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.dxf")
        export_dxf(panels, path)
        assert os.path.exists(path)
        content = open(path).read()
        assert "AC1024" in content
        assert "LWPOLYLINE" in content


def test_export_dxf_panel_labels():
    mesh = _flat_quad_mesh()
    panels, _ = unfold_mesh(mesh, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "labels.dxf")
        export_dxf(panels, path)
        content = open(path).read()
        assert "MTEXT" in content


def test_export_dxf_empty_panels(capsys):
    """Empty panel list should print a warning and not crash."""
    with tempfile.TemporaryDirectory() as tmp:
        export_dxf([], os.path.join(tmp, "empty.dxf"))
    captured = capsys.readouterr()
    assert "No panels" in captured.out


def test_export_dxf_custom_layer():
    mesh = _flat_quad_mesh()
    panels, _ = unfold_mesh(mesh, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "layer.dxf")
        export_dxf(panels, path, layer_name="MYCUTS")
        content = open(path).read()
        assert "MYCUTS" in content


def test_export_dxf_scale_factor():
    """Scale factor should multiply coordinates — check a scaled value appears."""
    mesh = _flat_quad_mesh()
    panels, _ = unfold_mesh(mesh, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scaled.dxf")
        export_dxf(panels, path, scale_factor=500.0)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# export_panels (convenience wrapper)
# ---------------------------------------------------------------------------


def test_export_panels_creates_svg_and_dxf():
    mesh = _flat_quad_mesh()
    with tempfile.TemporaryDirectory() as tmp:
        report = export_panels(mesh, output_dir=tmp, stem="test_mesh", verbose=False)
        assert os.path.exists(os.path.join(tmp, "test_mesh.svg"))
        assert os.path.exists(os.path.join(tmp, "test_mesh.dxf"))
        assert isinstance(report, UnfoldReport)
        assert report.n_panels == mesh.n_faces


def test_export_panels_returns_report():
    mesh = _flat_quad_mesh(3, 3)
    with tempfile.TemporaryDirectory() as tmp:
        report = export_panels(mesh, output_dir=tmp, stem="grid", verbose=False)
        assert report.n_panels == 9


# ---------------------------------------------------------------------------
# UnfoldReport
# ---------------------------------------------------------------------------


def test_unfold_report_print(capsys):
    report = UnfoldReport(
        n_panels=4,
        max_planarity_residual=1e-5,
        mean_planarity_residual=5e-6,
        max_area_distortion=0.001,
        warnings=[],
    )
    report.print()
    captured = capsys.readouterr()
    assert "UNFOLDING REPORT" in captured.out
    assert "4" in captured.out


def test_unfold_report_print_with_warnings(capsys):
    report = UnfoldReport(
        n_panels=2,
        max_planarity_residual=0.1,
        mean_planarity_residual=0.05,
        max_area_distortion=0.05,
        warnings=["Face 0: too nonplanar"],
    )
    report.print()
    captured = capsys.readouterr()
    assert "too nonplanar" in captured.out
