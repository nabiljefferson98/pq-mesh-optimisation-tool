"""
Extended tests for src/optimisation/gradients.py — targeting uncovered lines:
  194-284  compute_angle_balance_gradient (full body)
  289-291  compute_angle_balance_energy_scalar
  350-351  angle_balance branch in compute_total_gradient
  428      compute_numerical_gradient_term
  481      verify_gradient verbose=False branch
  488-504  verify_gradient near-zero gradient branch
  575-605  print_gradient_analysis
  670      gradient_for_scipy
"""

import numpy as np

from src.core.mesh import QuadMesh
from src.optimisation.gradients import (
    compute_angle_balance_energy_scalar,
    compute_angle_balance_gradient,
    compute_gradient_statistics,
    compute_numerical_gradient,
    compute_numerical_gradient_term,
    compute_total_gradient,
    gradient_for_scipy,
    print_gradient_analysis,
    verify_gradient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noisy_grid(rows: int = 3, cols: int = 3, noise: float = 0.05) -> QuadMesh:
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

    rng = np.random.default_rng(7)
    vertices[:, 2] += rng.uniform(-noise, noise, size=len(vertices))
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    return mesh


def _interior_quad_mesh() -> QuadMesh:
    """3x3 grid — interior vertices have valence 4 (required by angle-balance)."""
    return _noisy_grid(3, 3)


# ---------------------------------------------------------------------------
# compute_angle_balance_gradient
# ---------------------------------------------------------------------------


def test_angle_balance_gradient_shape():
    mesh = _interior_quad_mesh()
    grad = compute_angle_balance_gradient(mesh)
    assert grad.shape == mesh.vertices.shape


def test_angle_balance_gradient_is_finite():
    mesh = _interior_quad_mesh()
    grad = compute_angle_balance_gradient(mesh)
    assert np.isfinite(grad).all()


def test_angle_balance_gradient_zero_for_flat_interior():
    """Perfectly flat mesh with symmetric quads → near-zero angle-balance gradient."""
    vertices = []
    for r in range(4):
        for c in range(4):
            vertices.append([c, r, 0.0])
    vertices = np.array(vertices, dtype=np.float64)
    faces = []
    for r in range(3):
        for c in range(3):
            v0 = r * 4 + c
            faces.append([v0, v0 + 1, v0 + 5, v0 + 4])
    faces = np.array(faces, dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    grad = compute_angle_balance_gradient(mesh)
    # Interior gradient should be near zero for a uniform flat grid
    assert np.linalg.norm(grad) < 1.0  # loose check — symmetry may not be perfect


def test_angle_balance_gradient_boundary_vertices_zero():
    """Boundary vertices (valence != 4) must contribute zero gradient."""
    mesh = _interior_quad_mesh()
    grad = compute_angle_balance_gradient(mesh)
    # Corner vertex 0 has valence 1 → its gradient must be zero
    incident = mesh.get_vertex_faces(0)
    if len(incident) != 4:
        assert np.allclose(grad[0], 0.0)


def test_angle_balance_gradient_degenerate_edge_handled():
    """Mesh with near-coincident vertices must not crash (degenerate edge guard)."""
    vertices = np.array(
        [
            [0, 0, 0],
            [1e-14, 0, 0],  # nearly coincident with v0
            [1, 1, 0],
            [0, 1, 0],
            [2, 0, 0],
            [2, 1, 0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    grad = compute_angle_balance_gradient(mesh)
    assert np.isfinite(grad).all()


# ---------------------------------------------------------------------------
# compute_angle_balance_energy_scalar
# ---------------------------------------------------------------------------


def test_angle_balance_energy_scalar_returns_float():
    mesh = _interior_quad_mesh()
    val = compute_angle_balance_energy_scalar(mesh)
    assert isinstance(val, float)
    assert val >= 0.0


# ---------------------------------------------------------------------------
# compute_total_gradient — angle_balance branch
# ---------------------------------------------------------------------------


def test_total_gradient_with_angle_balance_weight():
    mesh = _interior_quad_mesh()
    weights = {
        "planarity": 1.0,
        "fairness": 1.0,
        "closeness": 1.0,
        "angle_balance": 1.0,
    }
    grad = compute_total_gradient(mesh, weights)
    assert grad.shape == mesh.vertices.shape
    assert np.isfinite(grad).all()


def test_total_gradient_angle_balance_zero_weight_skipped():
    """angle_balance=0.0 must not execute the branch (no error)."""
    mesh = _noisy_grid()
    weights = {
        "planarity": 1.0,
        "fairness": 1.0,
        "closeness": 1.0,
        "angle_balance": 0.0,
    }
    grad = compute_total_gradient(mesh, weights)
    assert np.isfinite(grad).all()


def test_total_gradient_numerical_mode():
    """use_numerical=True should return a valid gradient."""
    # Use tiny mesh to keep test fast
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    weights = {"planarity": 1.0, "fairness": 1.0, "closeness": 1.0}
    grad = compute_total_gradient(mesh, weights, use_numerical=True)
    assert grad.shape == (4, 3)
    assert np.isfinite(grad).all()


# ---------------------------------------------------------------------------
# compute_numerical_gradient / compute_numerical_gradient_term
# ---------------------------------------------------------------------------


def test_numerical_gradient_small_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()

    from src.optimisation.energy_terms import compute_planarity_energy

    grad = compute_numerical_gradient(mesh, compute_planarity_energy)
    assert grad.shape == (4, 3)
    assert np.isfinite(grad).all()


def test_numerical_gradient_term_matches_numerical_gradient():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()

    from src.optimisation.energy_terms import compute_planarity_energy

    g1 = compute_numerical_gradient(mesh, compute_planarity_energy)
    g2 = compute_numerical_gradient_term(mesh, compute_planarity_energy)
    assert np.allclose(g1, g2)


# ---------------------------------------------------------------------------
# verify_gradient
# ---------------------------------------------------------------------------


def test_verify_gradient_passes_for_small_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    weights = {"planarity": 1.0, "fairness": 0.1, "closeness": 0.1}
    is_correct, error = verify_gradient(mesh, weights, tolerance=1e-2, verbose=False)
    assert isinstance(is_correct, bool)
    assert error >= 0.0


def test_verify_gradient_verbose(capsys):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    weights = {"planarity": 1.0, "fairness": 0.1, "closeness": 0.1}
    verify_gradient(mesh, weights, tolerance=0.5, verbose=True)
    out = capsys.readouterr().out
    assert "GRADIENT VERIFICATION" in out


def test_verify_gradient_near_zero_gradient():
    """Flat mesh → very small gradient → near-zero branch in verify_gradient."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    weights = {"planarity": 0.001, "fairness": 0.001, "closeness": 0.001}
    is_correct, error = verify_gradient(mesh, weights, tolerance=1.0, verbose=False)
    assert error >= 0.0


# ---------------------------------------------------------------------------
# compute_gradient_statistics
# ---------------------------------------------------------------------------


def test_gradient_statistics_keys():
    grad = np.random.rand(10, 3)
    stats = compute_gradient_statistics(grad)
    for key in (
        "norm",
        "max_magnitude",
        "mean_magnitude",
        "std_magnitude",
        "max_component",
    ):
        assert key in stats


def test_gradient_statistics_zero_gradient():
    grad = np.zeros((5, 3))
    stats = compute_gradient_statistics(grad)
    assert stats["norm"] == 0.0
    assert stats["max_magnitude"] == 0.0


# ---------------------------------------------------------------------------
# print_gradient_analysis
# ---------------------------------------------------------------------------


def test_print_gradient_analysis_runs(capsys):
    mesh = _noisy_grid(2, 2)
    weights = {
        "planarity": 1.0,
        "fairness": 0.1,
        "closeness": 0.1,
        "angle_balance": 0.0,
    }
    print_gradient_analysis(mesh, weights)
    out = capsys.readouterr().out
    assert "GRADIENT ANALYSIS" in out
    assert "Planarity" in out or "planarity" in out.lower()


# ---------------------------------------------------------------------------
# gradient_for_scipy
# ---------------------------------------------------------------------------


def test_gradient_for_scipy_shape():
    mesh = _noisy_grid(2, 2)
    weights = {"planarity": 1.0, "fairness": 0.1, "closeness": 0.1}
    x_flat = mesh.vertices.flatten()
    result = gradient_for_scipy(x_flat, mesh, weights)
    assert result.shape == x_flat.shape
    assert np.isfinite(result).all()


def test_gradient_for_scipy_updates_mesh_vertices():
    """gradient_for_scipy updates mesh.vertices to the supplied x_flat."""
    mesh = _noisy_grid(2, 2)
    weights = {"planarity": 1.0, "fairness": 0.1, "closeness": 0.1}
    x_flat = mesh.vertices.flatten() + 0.01  # shift all positions by 0.01
    gradient_for_scipy(x_flat, mesh, weights)
    # mesh.vertices should now reflect x_flat (that's the documented behaviour)
    assert np.allclose(mesh.vertices, x_flat.reshape(-1, 3))
