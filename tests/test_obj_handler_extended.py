"""
Extended tests for src/io/obj_handler.py — targeting uncovered lines:
  46        FileNotFoundError
  51, 53    ValueError: no vertices / no faces
  66-67     other_faces (n-gons) present
  71        tri_faces present alongside quads → warning
  77-97     tri-only mesh → pairing or fallback
  186-216   _pair_triangles_to_quads
"""

import os
import tempfile

import numpy as np
import pytest

from src.core.mesh import QuadMesh
from src.io.obj_handler import _pair_triangles_to_quads, load_obj, save_obj

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tmp_obj(content: str) -> str:
    """Write OBJ text to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".obj")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# load_obj — error paths
# ---------------------------------------------------------------------------


def test_load_obj_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_obj("/no/such/file/ever.obj")


def test_load_obj_no_vertices():
    path = _write_tmp_obj("# empty\nf 1 2 3 4\n")
    try:
        with pytest.raises(ValueError, match="No vertices"):
            load_obj(path)
    finally:
        os.unlink(path)


def test_load_obj_no_faces():
    path = _write_tmp_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\n")
    try:
        with pytest.raises(ValueError, match="No faces"):
            load_obj(path)
    finally:
        os.unlink(path)


def test_load_obj_no_quad_or_triangle_faces():
    """File with only n-gons (5 verts) → ValueError."""
    content = (
        "v 0 0 0\nv 1 0 0\nv 2 0 0\nv 2 1 0\nv 0 1 0\n"
        "f 1 2 3 4 5\n"  # pentagon — skipped as 'other'
    )
    path = _write_tmp_obj(content)
    try:
        with pytest.raises(ValueError):
            load_obj(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# load_obj — n-gon / mixed branches
# ---------------------------------------------------------------------------


def test_load_obj_ngon_skipped_but_quads_kept(capsys):
    """OBJ with quads + one pentagon: pentagon is skipped, quads loaded."""
    content = (
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "v 2 0 0\nv 3 0 0\nv 3 1 0\nv 2 1 0\n"
        "v 4 0 0\n"
        "f 1 2 3 4\n"  # quad
        "f 5 6 7 8 9\n"  # pentagon — should be skipped
    )
    path = _write_tmp_obj(content)
    try:
        mesh = load_obj(path)
        assert mesh.n_faces == 1  # only the quad
    finally:
        os.unlink(path)


def test_load_obj_triangles_alongside_quads_warns(capsys):
    """OBJ with quads + triangles: triangles are ignored with a warning."""
    content = (
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "v 2 0 0\nv 2 1 0\n"
        "f 1 2 3 4\n"  # quad
        "f 1 5 6\n"  # triangle — should be ignored
    )
    path = _write_tmp_obj(content)
    try:
        mesh = load_obj(path)
        assert mesh.n_faces == 1
        out = capsys.readouterr().out
        assert "Ignoring" in out or "triangle" in out.lower()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# load_obj — triangle-only mesh → pairing
# ---------------------------------------------------------------------------


def test_load_obj_triangles_only_paired_to_quads(capsys):
    """OBJ with two triangles sharing an edge → paired into 1 quad."""
    content = (
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "f 1 2 3\n"  # tri 1
        "f 1 3 4\n"  # tri 2 (shares edge 1-3 → pairs with tri 1)
    )
    path = _write_tmp_obj(content)
    try:
        mesh = load_obj(path)
        assert mesh.n_faces == 1
        out = capsys.readouterr().out
        assert "quad" in out.lower() or "pair" in out.lower()
    finally:
        os.unlink(path)


def test_load_obj_triangles_unpaired_fallback_no_require_quads(capsys):
    """Single triangle that cannot be paired: falls back to triangle mesh."""
    content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    path = _write_tmp_obj(content)
    try:
        mesh = load_obj(path, require_quads=False)
        assert mesh.n_faces == 1  # single triangle
        out = capsys.readouterr().out
        assert (
            "Triangle" in out
            or "triangle" in out.lower()
            or "pairing failed" in out.lower()
        )
    finally:
        os.unlink(path)


def test_load_obj_triangles_unpaired_require_quads_raises():
    """Single triangle with require_quads=True (default) → ValueError."""
    content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    path = _write_tmp_obj(content)
    try:
        with pytest.raises(ValueError, match="No quad"):
            load_obj(path, require_quads=True)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# save_obj
# ---------------------------------------------------------------------------


def test_save_obj_creates_file():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.obj")
        save_obj(mesh, path)
        assert os.path.exists(path)
        content = open(path).read()
        assert "v " in content
        assert "f " in content


def test_save_obj_round_trip():
    """Save then load: vertex count and face count must be preserved."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rt.obj")
        save_obj(mesh, path)
        loaded = load_obj(path)
    assert loaded.n_vertices == mesh.n_vertices
    assert loaded.n_faces == mesh.n_faces
    assert np.allclose(loaded.vertices, mesh.vertices, atol=1e-5)


def test_save_obj_creates_parent_directory():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "subdir", "out.obj")
        save_obj(mesh, path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# _pair_triangles_to_quads
# ---------------------------------------------------------------------------


def test_pair_triangles_basic():
    tris = [[0, 1, 2], [0, 2, 3]]  # share edge 0-2
    quads = _pair_triangles_to_quads(tris)
    assert len(quads) == 1
    assert len(quads[0]) == 4


def test_pair_triangles_no_shared_edge():
    """Triangles with no shared edge → nothing paired."""
    tris = [[0, 1, 2], [3, 4, 5]]
    quads = _pair_triangles_to_quads(tris)
    assert quads == []


def test_pair_triangles_four_triangles_two_quads():
    """Two independent pairs → two quads."""
    tris = [
        [0, 1, 2],
        [0, 2, 3],  # pair 1
        [4, 5, 6],
        [4, 6, 7],  # pair 2
    ]
    quads = _pair_triangles_to_quads(tris)
    assert len(quads) == 2


def test_pair_triangles_used_flag():
    """Same triangle cannot be used in two pairs (greedy first-wins)."""
    # tri 0 shares edge with both tri 1 and tri 2 → only one pairing
    tris = [
        [0, 1, 2],  # shares 0-1 with tri1, shares 1-2 with tri2
        [0, 1, 3],
        [1, 2, 4],
    ]
    quads = _pair_triangles_to_quads(tris)
    # At most one quad from tri0
    assert len(quads) <= 1
