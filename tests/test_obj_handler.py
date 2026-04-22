import pytest

from src.io.obj_handler import load_obj, save_obj


def test_load_plane_grid():
    """Test loading generated plane grid."""
    mesh = load_obj("data/input/generated/plane_5x5_clean.obj")

    assert mesh.n_vertices == 36  # (5+1) * (5+1)
    assert mesh.n_faces == 25  # 5 * 5


def test_save_and_reload():
    """Test round-trip save/load."""
    mesh1 = load_obj("data/input/generated/plane_5x5_clean.obj")
    save_obj(mesh1, "data/output/test_export.obj")

    mesh2 = load_obj("data/output/test_export.obj")

    assert mesh1.n_vertices == mesh2.n_vertices
    assert mesh1.n_faces == mesh2.n_faces


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
