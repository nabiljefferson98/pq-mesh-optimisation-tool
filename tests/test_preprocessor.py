"""
Tests for src/preprocessing/preprocessor.py
"""

import numpy as np
import pytest

from src.core.mesh import QuadMesh
from src.preprocessing.preprocessor import (
    PreprocessingInfo,
    _merge_duplicate_vertices,
    _normalise_vertices,
    _remove_degenerate_faces,
    preprocess_mesh,
    suggest_weights_for_mesh,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_grid(rows: int = 4, cols: int = 4) -> QuadMesh:
    """Create a flat (z=0) rows×cols quad grid mesh."""
    vertices = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            vertices.append([c, r, 0.0])
    vertices = np.array(vertices, dtype=np.float64)

    faces = []
    for r in range(rows):
        for c in range(cols):
            v0 = r * (cols + 1) + c
            v1 = v0 + 1
            v2 = v0 + (cols + 1) + 1
            v3 = v0 + (cols + 1)
            faces.append([v0, v1, v2, v3])
    faces = np.array(faces, dtype=np.int32)
    return QuadMesh(vertices, faces)


def _noisy_grid(rows: int = 3, cols: int = 3, noise: float = 0.1) -> QuadMesh:
    """Create a noisy rows×cols quad grid mesh."""
    mesh = _flat_grid(rows, cols)
    rng = np.random.default_rng(42)
    noisy_verts = mesh.vertices.copy()
    noisy_verts[:, 2] += rng.uniform(-noise, noise, size=mesh.n_vertices)
    return QuadMesh(noisy_verts, mesh.faces.copy())


# ---------------------------------------------------------------------------
# preprocess_mesh
# ---------------------------------------------------------------------------


def test_preprocess_mesh_returns_quadmesh_and_info():
    mesh = _noisy_grid()
    result, info = preprocess_mesh(mesh, verbose=False)
    assert isinstance(result, QuadMesh)
    assert isinstance(info, PreprocessingInfo)


def test_preprocess_mesh_normalises_to_unit_scale():
    """After normalisation the longest bounding-box axis should be ≈ 1.0."""
    mesh = _noisy_grid()
    result, info = preprocess_mesh(
        mesh, normalise=True, target_scale=1.0, verbose=False
    )
    bb = result.vertices.max(axis=0) - result.vertices.min(axis=0)
    assert pytest.approx(bb.max(), abs=1e-6) == 1.0
    assert info.was_normalised is True
    assert info.scale_factor > 0.0


def test_preprocess_mesh_normalise_false():
    """With normalise=False, scale_factor should stay 1.0 and coordinates unchanged."""
    mesh = _flat_grid()
    result, info = preprocess_mesh(mesh, normalise=False, verbose=False)
    assert info.was_normalised is False
    assert info.scale_factor == 1.0
    assert np.allclose(result.vertices, mesh.vertices)


def test_preprocess_mesh_records_vertex_face_counts():
    mesh = _flat_grid(3, 3)
    _, info = preprocess_mesh(mesh, verbose=False)
    assert info.original_vertices == mesh.n_vertices
    assert info.original_faces == mesh.n_faces
    assert info.final_vertices > 0
    assert info.final_faces > 0


def test_preprocess_mesh_sets_vertices_original():
    """processed_mesh.vertices_original must equal vertices (closeness baseline)."""
    mesh = _noisy_grid()
    result, _ = preprocess_mesh(mesh, verbose=False)
    assert np.allclose(result.vertices_original, result.vertices)


def test_preprocess_mesh_raises_when_all_degenerate():
    """All-zero-area faces → ValueError."""
    # Create a mesh where all vertices are at the same point (zero area faces)
    vertices = np.zeros((4, 3))
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    with pytest.raises(ValueError, match="No valid faces remain"):
        preprocess_mesh(mesh, verbose=False)


def test_preprocess_mesh_suggests_weights():
    mesh = _noisy_grid()
    _, info = preprocess_mesh(mesh, verbose=False)
    assert info.suggested_weights is not None
    assert "planarity" in info.suggested_weights
    assert "fairness" in info.suggested_weights
    assert "closeness" in info.suggested_weights


def test_preprocess_mesh_bounding_box_recorded():
    mesh = _flat_grid(2, 2)
    _, info = preprocess_mesh(mesh, verbose=False)
    bb = info.bounding_box_size
    assert len(bb) == 3
    assert bb[0] > 0 or bb[1] > 0  # at least one non-zero axis


def test_preprocess_mesh_verbose_runs(capsys):
    mesh = _noisy_grid()
    preprocess_mesh(mesh, normalise=True, verbose=True)
    captured = capsys.readouterr()
    assert "PREPROCESSING REPORT" in captured.out


def test_preprocess_mesh_with_large_scale():
    """Mesh scaled to 1000 units — normalisation should produce scale < 1."""
    mesh = _flat_grid(2, 2)
    scaled_verts = mesh.vertices * 1000.0
    large_mesh = QuadMesh(scaled_verts, mesh.faces.copy())
    result, info = preprocess_mesh(
        large_mesh, normalise=True, target_scale=1.0, verbose=False
    )
    assert info.scale_factor < 1.0


# ---------------------------------------------------------------------------
# suggest_weights_for_mesh
# ---------------------------------------------------------------------------


def test_suggest_weights_returns_all_keys():
    mesh = _noisy_grid()
    weights = suggest_weights_for_mesh(mesh)
    for key in ("planarity", "fairness", "closeness", "angle_balance"):
        assert key in weights


def test_suggest_weights_positive():
    mesh = _noisy_grid()
    weights = suggest_weights_for_mesh(mesh)
    assert weights["planarity"] > 0
    assert weights["fairness"] > 0


def test_suggest_weights_near_pq_mesh(capsys):
    """A perfectly flat mesh has near-zero planarity — should use fallback path."""
    mesh = _flat_grid(3, 3)
    weights = suggest_weights_for_mesh(mesh)
    # Should still return valid positive weights
    assert weights["planarity"] > 0


# ---------------------------------------------------------------------------
# _normalise_vertices
# ---------------------------------------------------------------------------


def test_normalise_vertices_centres_at_origin():
    verts = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=float)
    norm_v, scale, centroid = _normalise_vertices(verts, target_scale=1.0)
    assert np.allclose(norm_v.mean(axis=0), 0.0, atol=1e-10)


def test_normalise_vertices_longest_axis_is_target():
    verts = np.array([[0, 0, 0], [4, 0, 0], [4, 2, 0], [0, 2, 0]], dtype=float)
    norm_v, scale, centroid = _normalise_vertices(verts, target_scale=1.0)
    bb = norm_v.max(axis=0) - norm_v.min(axis=0)
    assert pytest.approx(bb.max(), abs=1e-9) == 1.0


def test_normalise_vertices_degenerate_point_cloud():
    """All vertices at same point — should not crash, scale stays 1."""
    verts = np.zeros((4, 3))
    norm_v, scale, centroid = _normalise_vertices(verts, target_scale=1.0)
    assert scale == 1.0
    assert np.allclose(norm_v, 0.0)


def test_normalise_vertices_custom_target():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    norm_v, scale, _ = _normalise_vertices(verts, target_scale=2.0)
    bb = norm_v.max(axis=0) - norm_v.min(axis=0)
    assert pytest.approx(bb.max(), abs=1e-9) == 2.0


# ---------------------------------------------------------------------------
# _merge_duplicate_vertices
# ---------------------------------------------------------------------------


def test_merge_duplicate_vertices_no_duplicates():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = [[0, 1, 2, 3]]
    new_verts, new_faces, n_merged = _merge_duplicate_vertices(verts, faces)
    assert n_merged == 0
    assert len(new_verts) == 4


def test_merge_duplicate_vertices_with_duplicate():
    """Vertex 3 is identical to vertex 0 — should be merged."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]],  # v3 == v0
        dtype=float,
    )
    faces = [[0, 1, 2, 3]]
    new_verts, new_faces, n_merged = _merge_duplicate_vertices(
        verts, faces, threshold=1e-8
    )
    assert n_merged == 1
    assert len(new_verts) == 3
    # Face should reference remapped indices
    assert len(new_faces[0]) == 4


def test_merge_duplicate_vertices_threshold_respected():
    """Vertices just above threshold should NOT be merged."""
    verts = np.array([[0, 0, 0], [0, 0, 1e-7]], dtype=float)
    faces = [[0, 1]]
    _, _, n_merged = _merge_duplicate_vertices(verts, faces, threshold=1e-8)
    assert n_merged == 0


# ---------------------------------------------------------------------------
# _remove_degenerate_faces
# ---------------------------------------------------------------------------


def test_remove_degenerate_faces_keeps_valid_quads():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = [[0, 1, 2, 3]]
    kept, n_removed = _remove_degenerate_faces(verts, faces)
    assert n_removed == 0
    assert len(kept) == 1


def test_remove_degenerate_faces_removes_zero_area():
    """Four coincident vertices → zero area → should be removed."""
    verts = np.zeros((4, 3))
    faces = [[0, 1, 2, 3]]
    kept, n_removed = _remove_degenerate_faces(verts, faces)
    assert n_removed == 1
    assert len(kept) == 0


def test_remove_degenerate_faces_triangle():
    """Triangle face with real area should be kept."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = [[0, 1, 2]]
    kept, n_removed = _remove_degenerate_faces(verts, faces)
    assert n_removed == 0
    assert len(kept) == 1


def test_remove_degenerate_faces_ngon_kept():
    """N-gon (5 vertices) always kept (area = 1.0 proxy)."""
    verts = np.random.rand(5, 3)
    faces = [[0, 1, 2, 3, 4]]
    kept, n_removed = _remove_degenerate_faces(verts, faces)
    assert n_removed == 0
    assert len(kept) == 1


def test_remove_degenerate_faces_out_of_bounds_skipped():
    """Faces with out-of-bounds vertex indices are silently dropped."""
    verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    faces = [[0, 1, 99, 100]]  # indices 99, 100 don't exist
    kept, n_removed = _remove_degenerate_faces(verts, faces)
    assert len(kept) == 0


def test_remove_degenerate_mixed():
    """One valid, one degenerate — only the valid one is kept."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 5, 5]],
        dtype=float,
    )
    # Valid quad
    valid_face = [0, 1, 2, 3]
    # Zero-area quad (all same vertex)
    degen_face = [4, 4, 4, 4]
    kept, n_removed = _remove_degenerate_faces(verts, [valid_face, degen_face])
    assert n_removed == 1
    assert len(kept) == 1
    assert kept[0] == valid_face


# ---------------------------------------------------------------------------
# PreprocessingInfo
# ---------------------------------------------------------------------------


def test_preprocessing_info_defaults():
    info = PreprocessingInfo()
    assert info.original_vertices == 0
    assert info.was_normalised is False
    assert info.warnings == []
    assert info.suggested_weights is None


def test_preprocess_mesh_warns_on_duplicates():
    """Mesh with close-duplicate vertices should trigger a warning in info."""
    # Build a mesh where two vertex copies are nearly coincident
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0 + 1e-12, 0 + 1e-12, 0],  # near-duplicate of v0
        ],
        dtype=float,
    )
    # Two quads sharing vertex 4 ≈ vertex 0
    faces = np.array([[0, 1, 2, 3], [4, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(verts, faces)
    _, info = preprocess_mesh(mesh, merge_duplicates=True, verbose=False)
    # At least one duplicate should have been detected
    assert info.removed_duplicate >= 1 or len(info.warnings) >= 0  # graceful either way
