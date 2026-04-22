"""
Tests for error handling and edge cases.
"""

import numpy as np

from src.core.mesh import QuadMesh
from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig


def test_empty_mesh():
    """Test handling of empty mesh."""
    V = np.array([], dtype=np.float64).reshape(0, 3)
    F = np.array([], dtype=np.int32).reshape(0, 4)

    # Empty mesh should be created but fail validation
    mesh = QuadMesh(V, F)
    assert mesh.n_vertices == 0
    assert mesh.n_faces == 0

    optimiser = MeshOptimiser(OptimisationConfig())
    result = optimiser.optimise(mesh)

    assert not result.success
    assert (
        "vertices" in result.message.lower() or "validation" in result.message.lower()
    )


def test_mesh_with_nan():
    """Test handling of NaN values."""
    V = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, np.nan], [0, 1, 0]], dtype=np.float64  # Invalid
    )
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(V, F)

    optimiser = MeshOptimiser(OptimisationConfig())
    result = optimiser.optimise(mesh)

    assert not result.success
    assert "nan" in result.message.lower() or "inf" in result.message.lower()


def test_degenerate_zero_area_face():
    """Test face with all vertices collinear (zero area)."""
    V = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],  # Collinear  # Collinear
        dtype=np.float64,
    )
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(V, F)

    optimiser = MeshOptimiser(OptimisationConfig())
    result = optimiser.optimise(mesh)

    assert not result.success
    assert "area" in result.message.lower()


def test_duplicate_vertices_in_face():
    """Test face with duplicate vertex indices."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 1, 2]], dtype=np.int32)  # Vertex 1 repeated
    mesh = QuadMesh(V, F)

    optimiser = MeshOptimiser(OptimisationConfig())
    result = optimiser.optimise(mesh)

    assert not result.success
    assert "duplicate" in result.message.lower()
