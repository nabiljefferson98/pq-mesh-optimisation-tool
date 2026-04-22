"""
Tests for vertex-to-face adjacency maps.

Tests verify:
- Correct shape and content of padded vertex-face ID arrays
- Proper use of sentinel values (-1) for unused slots
- Accurate counting of shared faces for mesh vertices
- Cache reuse and invalidation mechanisms

Date: 14 March 2026
"""

import numpy as np
import pytest

from src.core.mesh import QuadMesh


@pytest.fixture
def two_quad_mesh():
    """Fixture providing a simple 6-vertex, 2-quad mesh."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int32)
    return QuadMesh(verts, faces)


def test_shape(two_quad_mesh):
    """Verify that the vertex_face_ids_padded array has the correct shape."""
    t = two_quad_mesh.vertex_face_ids_padded
    assert t.shape[0] == two_quad_mesh.n_vertices
    assert t.shape[1] >= 4


def test_sentinels_are_minus_one(two_quad_mesh):
    """Verify that unused slots in the adjacency map use -1 as a sentinel."""
    t = two_quad_mesh.vertex_face_ids_padded
    assert np.all((t >= -1) & (t < two_quad_mesh.n_faces))


def test_shared_vertex_has_two_faces(two_quad_mesh):
    """Verify that vertices shared by multiple quads correctly report their face IDs."""
    t = two_quad_mesh.vertex_face_ids_padded
    for vid in [1, 4]:
        valid = t[vid][t[vid] != -1]
        assert len(valid) == 2


def test_cache_is_reused(two_quad_mesh):
    """Ensure that vertex_face_ids_padded is cached and not recomputed unnecessarily."""
    t1 = two_quad_mesh.vertex_face_ids_padded
    t2 = two_quad_mesh.vertex_face_ids_padded
    assert t1 is t2  # same object — not recomputed


def test_reset_topology_cache(two_quad_mesh):
    """Verify that reset_topology_cache correctly clears the cached adjacency map."""
    _ = two_quad_mesh.vertex_face_ids_padded
    two_quad_mesh.reset_topology_cache()
    assert two_quad_mesh._vertex_face_ids_padded is None
