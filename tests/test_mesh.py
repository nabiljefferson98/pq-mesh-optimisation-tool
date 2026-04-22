import numpy as np
import pytest

from src.core.mesh import QuadMesh


def test_quad_mesh_creation():
    """Test basic mesh creation."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)

    mesh = QuadMesh(V, F)

    assert mesh.n_vertices == 4
    assert mesh.n_faces == 1
    assert np.allclose(mesh.vertices, V)


def test_get_face_vertices():
    """Test face vertex retrieval."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2, 3]])

    mesh = QuadMesh(V, F)
    face_verts = mesh.get_face_vertices(0)

    assert face_verts.shape == (4, 3)
    assert np.allclose(face_verts[0], [0, 0, 0])


def test_invalid_mesh():
    """Test validation catches errors."""
    V = np.array([[0, 0, 0], [1, 0, 0]])
    F = np.array([[0, 1, 2, 3]])  # Index 2, 3 don't exist

    with pytest.raises(ValueError, match="Face indices out of bounds"):
        QuadMesh(V, F)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
