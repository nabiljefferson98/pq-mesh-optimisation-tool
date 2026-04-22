import numpy as np
import pytest

from src.core.mesh import QuadMesh
from src.optimisation.mesh_geometry import (
    compute_all_planarity_deviations,
    compute_angle_at_vertex_in_face,
    compute_face_planarity_deviation,
)


def test_planar_quad_has_zero_deviation():
    """Perfectly planar quad should have deviation ≈ 0."""
    face_verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    deviation = compute_face_planarity_deviation(face_verts)
    assert deviation < 1e-10


def test_non_planar_quad_has_positive_deviation():
    """Non-planar quad should have deviation > 0."""
    face_verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]]  # Lifted out of plane
    )

    deviation = compute_face_planarity_deviation(face_verts)
    assert deviation > 0.05


def test_right_angle_computation():
    """90-degree angle should be π/2."""
    face_verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    angle = compute_angle_at_vertex_in_face(face_verts, 0)
    assert np.abs(angle - np.pi / 2) < 1e-6


def test_planarity_on_mesh():
    """Test planarity computation on a mesh."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2, 3]])
    mesh = QuadMesh(V, F)

    deviations = compute_all_planarity_deviations(mesh)

    assert len(deviations) == 1
    assert deviations[0] < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
