"""
Extended tests targeting remaining coverage gaps:

  src/optimisation/energy_terms.py  lines 96-116, 426, 476-479, 495-513
  src/optimisation/optimiser.py     lines 47-48, 83, 85, 131, 154-167,
                                         207, 223, 266, 326, 343-360,
                                         373-376, 458, 495-496
  src/core/mesh.py                  lines 40, 43, 47, 141-146
"""

import numpy as np
import pytest

from src.core.mesh import QuadMesh
from src.optimisation.energy_terms import (
    analyse_energy_components,
    compute_planarity_per_face,
    compute_total_energy,
    suggest_weight_scaling,
)
from src.optimisation.optimiser import (
    MeshOptimiser,
    OptimisationConfig,
    OptimisationResult,
    optimise_mesh_simple,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noisy_quad(noise: float = 0.2) -> QuadMesh:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, noise], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    return mesh


def _noisy_grid(rows: int = 3, cols: int = 3) -> QuadMesh:
    vertices = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            vertices.append([float(c), float(r), 0.0])
    vertices = np.array(vertices, dtype=np.float64)
    rng = np.random.default_rng(1)
    vertices[:, 2] += rng.uniform(-0.1, 0.1, size=len(vertices))
    faces = []
    for r in range(rows):
        for c in range(cols):
            v0 = r * (cols + 1) + c
            faces.append([v0, v0 + 1, v0 + (cols + 1) + 1, v0 + (cols + 1)])
    faces = np.array(faces, dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    return mesh


# ===========================================================================
# energy_terms — compute_planarity_per_face  (lines 96-116)
# ===========================================================================


def test_planarity_per_face_shape():
    mesh = _noisy_grid()
    devs = compute_planarity_per_face(mesh)
    assert devs.shape == (mesh.n_faces,)


def test_planarity_per_face_nonnegative():
    mesh = _noisy_grid()
    devs = compute_planarity_per_face(mesh)
    assert (devs >= 0).all()


def test_planarity_per_face_flat_near_zero():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    devs = compute_planarity_per_face(mesh)
    assert devs[0] < 1e-10


# ===========================================================================
# energy_terms — compute_total_energy with return_components (line 426)
# ===========================================================================


def test_compute_total_energy_with_components():
    mesh = _noisy_quad()
    weights = {
        "planarity": 10.0,
        "fairness": 1.0,
        "closeness": 5.0,
        "angle_balance": 0.0,
    }
    result = compute_total_energy(mesh, weights, return_components=True)
    assert isinstance(result, tuple)
    total, components = result
    assert total >= 0.0
    assert "E_planarity" in components
    assert "E_fairness" in components
    assert "E_closeness" in components


def test_compute_total_energy_angle_balance_component():
    mesh = _noisy_grid()
    weights = {
        "planarity": 1.0,
        "fairness": 1.0,
        "closeness": 1.0,
        "angle_balance": 1.0,
    }
    result = compute_total_energy(mesh, weights, return_components=True)
    total, components = result
    assert "E_angle_balance" in components
    assert components["E_angle_balance"] >= 0.0


# ===========================================================================
# energy_terms — print_energy_breakdown (lines 476-479)
# ===========================================================================


def test_analyse_energy_components_runs(capsys):
    mesh = _noisy_quad()
    weights = {
        "planarity": 10.0,
        "fairness": 1.0,
        "closeness": 5.0,
        "angle_balance": 0.0,
    }
    analyse_energy_components(mesh, weights)
    out = capsys.readouterr().out
    assert "ENERGY" in out or "Energy" in out


def test_analyse_energy_components_with_angle_balance(capsys):
    mesh = _noisy_grid()
    weights = {
        "planarity": 1.0,
        "fairness": 1.0,
        "closeness": 1.0,
        "angle_balance": 1.0,
    }
    analyse_energy_components(mesh, weights)
    out = capsys.readouterr().out
    assert "angle" in out.lower() or "balance" in out.lower()


# ===========================================================================
# energy_terms — suggest_weight_scaling (lines 495-513)
# ===========================================================================


def test_suggest_weight_scaling_noisy_mesh(capsys):
    mesh = _noisy_grid()
    weights = suggest_weight_scaling(mesh, verbose=True)
    assert "planarity" in weights
    out = capsys.readouterr().out
    assert "WEIGHT" in out or "weight" in out.lower()


def test_suggest_weight_scaling_flat_mesh():
    """Flat mesh → planarity ≈ 0 → 'already planar' branch."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(vertices, faces)
    mesh.vertices_original = vertices.copy()
    weights = suggest_weight_scaling(mesh, verbose=False)
    assert weights["planarity"] > 0


# ===========================================================================
# core/mesh.py — uncovered branches (lines 40, 43, 47, 141-146)
# ===========================================================================


def test_mesh_vertices_wrong_shape():
    """Vertices must be (n, 3) — passing 2D array with 2 columns raises."""
    bad_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="Vertices must be"):
        QuadMesh(bad_verts, faces)


def test_mesh_faces_not_2d():
    """1-D faces array raises ValueError."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    bad_faces = np.array([0, 1, 2, 3], dtype=np.int32)  # 1D
    with pytest.raises(ValueError, match="Faces must be 2D array"):
        QuadMesh(verts, bad_faces)


def test_mesh_update_vertices():
    """update_vertices must replace positions."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(verts, faces)
    new_pos = verts + 1.0
    mesh.update_vertices(new_pos)
    assert np.allclose(mesh.vertices, new_pos)


def test_mesh_update_vertices_wrong_shape():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(verts, faces)
    with pytest.raises(ValueError, match="Shape mismatch"):
        mesh.update_vertices(np.zeros((3, 3)))


def test_mesh_reset_to_original():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(verts, faces)
    mesh.vertices_original = verts.copy()
    mesh.vertices = verts + 5.0
    mesh.reset_to_original()
    assert np.allclose(mesh.vertices, verts)


# ===========================================================================
# optimiser — OptimisationConfig.validate  (lines 83, 85)
# ===========================================================================


def test_config_validate_bad_max_iterations():
    cfg = OptimisationConfig(max_iterations=0)
    with pytest.raises(ValueError, match="max_iterations"):
        cfg.validate()


def test_config_validate_bad_tolerance():
    cfg = OptimisationConfig(tolerance=-1.0)
    with pytest.raises(ValueError, match="tolerance"):
        cfg.validate()


def test_config_validate_bad_gradient_tolerance():
    cfg = OptimisationConfig(gradient_tolerance=0.0)
    with pytest.raises(ValueError, match="gradient_tolerance"):
        cfg.validate()


def test_config_validate_missing_weight_key():
    cfg = OptimisationConfig(weights={"planarity": 1.0, "fairness": 1.0})
    with pytest.raises(ValueError, match="Missing weight"):
        cfg.validate()


def test_config_validate_negative_weight():
    cfg = OptimisationConfig(
        weights={"planarity": -1.0, "fairness": 1.0, "closeness": 1.0}
    )
    with pytest.raises(ValueError, match="non-negative"):
        cfg.validate()


# ===========================================================================
# optimiser — OptimisationResult.summary  (lines 154-167, 207, 223, 266)
# ===========================================================================


def _make_result(success=True, message="CONVERGED", initial=1.0, final=0.1):
    mesh = _noisy_quad()
    return OptimisationResult(
        success=success,
        message=message,
        optimised_mesh=mesh,
        initial_energy=initial,
        final_energy=final,
        n_iterations=5,
        n_function_evaluations=6,
        n_gradient_evaluations=6,
        execution_time=0.1,
        component_energies_initial={
            "planarity": 0.5,
            "fairness": 0.3,
            "closeness": 0.2,
            "angle_balance": 0.0,
        },
        component_energies_final={
            "planarity": 0.05,
            "fairness": 0.03,
            "closeness": 0.02,
            "angle_balance": 0.0,
        },
    )


def test_summary_converged():
    r = _make_result(success=True, message="CONVERGED")
    s = r.summary()
    assert "CONVERGED" in s.upper()


def test_summary_iteration_limit():
    r = _make_result(success=False, message="TOTAL NO. OF ITERATIONS REACHED LIMIT")
    s = r.summary()
    assert "ITERATION LIMIT" in s or "increase" in s.lower()


def test_summary_abnormal():
    r = _make_result(success=False, message="ABNORMAL_TERMINATION_IN_LNSRCH")
    s = r.summary()
    assert "FAILED" in s or "ABNORMAL" in s


def test_summary_gradient_tolerance_not_met():
    """Energy reduced by >95% but not converged → special message."""
    r = _make_result(success=False, message="SOME OTHER", initial=100.0, final=0.1)
    s = r.summary()
    assert (
        "GRADIENT TOLERANCE NOT MET" in s
        or "nearly there" in s.lower()
        or "good enough" in s.lower()
    )


def test_summary_did_not_converge():
    r = _make_result(success=False, message="SOME OTHER", initial=1.0, final=0.9)
    s = r.summary()
    assert (
        "DID NOT CONVERGE" in s
        or "did not converge" in s.lower()
        or "partial" in s.lower()
    )


def test_summary_component_near_zero_initial():
    """initial ≈ 0, final ≈ 0 → 'both ≈ 0' branch."""
    mesh = _noisy_quad()
    r = OptimisationResult(
        success=True,
        message="CONVERGED",
        optimised_mesh=mesh,
        initial_energy=1.0,
        final_energy=0.1,
        n_iterations=1,
        n_function_evaluations=1,
        n_gradient_evaluations=1,
        execution_time=0.0,
        component_energies_initial={
            "planarity": 0.0,
            "fairness": 0.0,
            "closeness": 0.0,
            "angle_balance": 0.0,
        },
        component_energies_final={
            "planarity": 0.0,
            "fairness": 0.0,
            "closeness": 0.0,
            "angle_balance": 0.0,
        },
    )
    s = r.summary()
    assert "N/A" in s or "≈ 0" in s or "already at 0" in s


def test_summary_component_initial_zero_final_nonzero():
    """initial ≈ 0, final > 0 → 'initial ≈ 0, final abs' branch."""
    mesh = _noisy_quad()
    r = OptimisationResult(
        success=True,
        message="CONVERGED",
        optimised_mesh=mesh,
        initial_energy=1.0,
        final_energy=0.1,
        n_iterations=1,
        n_function_evaluations=1,
        n_gradient_evaluations=1,
        execution_time=0.0,
        component_energies_initial={
            "planarity": 0.0,
            "fairness": 1.0,
            "closeness": 1.0,
            "angle_balance": 0.0,
        },
        component_energies_final={
            "planarity": 0.5,
            "fairness": 0.5,
            "closeness": 0.5,
            "angle_balance": 0.0,
        },
    )
    s = r.summary()
    assert "initial ≈ 0" in s or "N/A" in s or "started at 0" in s


def test_summary_energy_increase_trade_off():
    """final > initial for some component → '⚠ expected trade-off' branch."""
    mesh = _noisy_quad()
    r = OptimisationResult(
        success=True,
        message="CONVERGED",
        optimised_mesh=mesh,
        initial_energy=1.0,
        final_energy=0.5,
        n_iterations=1,
        n_function_evaluations=1,
        n_gradient_evaluations=1,
        execution_time=0.0,
        component_energies_initial={
            "planarity": 1.0,
            "fairness": 0.1,
            "closeness": 0.1,
            "angle_balance": 0.0,
        },
        component_energies_final={
            "planarity": 0.05,
            "fairness": 5.0,
            "closeness": 0.1,
            "angle_balance": 0.0,
        },
    )
    s = r.summary()
    assert "trade-off" in s or "expected" in s.lower()


def test_summary_zero_initial_energy():
    """initial_energy == 0 → energy_reduction returns 0."""
    mesh = _noisy_quad()
    r = OptimisationResult(
        success=True,
        message="CONVERGED",
        optimised_mesh=mesh,
        initial_energy=0.0,
        final_energy=0.0,
        n_iterations=0,
        n_function_evaluations=0,
        n_gradient_evaluations=0,
        execution_time=0.0,
    )
    assert r.energy_reduction() == 0.0
    assert r.energy_reduction_percentage() == 0.0


# ===========================================================================
# optimiser — MeshOptimiser.validate_mesh  (lines 326, 343-360, 373-376)
# ===========================================================================


def test_validate_mesh_too_few_vertices():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = QuadMesh.__new__(QuadMesh)
    mesh.vertices = verts
    mesh.faces = faces  # shape (1, 3), n_vertices = 3 < 4
    mesh.vertices_original = verts.copy()

    cfg = OptimisationConfig(verbose=False)
    opt = MeshOptimiser(cfg)
    valid, msg = opt.validate_mesh(mesh)
    assert valid is False
    assert "vertices" in msg.lower()


def test_validate_mesh_no_faces():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.zeros((0, 4), dtype=np.int32)
    mesh = QuadMesh.__new__(QuadMesh)
    mesh.vertices = verts
    mesh.faces = faces
    mesh.vertices_original = verts.copy()
    cfg = OptimisationConfig(verbose=False)
    opt = MeshOptimiser(cfg)
    valid, msg = opt.validate_mesh(mesh)
    assert valid is False


def test_validate_mesh_duplicate_vertices_in_face():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    # Face with duplicate vertex 0 repeated
    faces = np.array([[0, 0, 2, 3]], dtype=np.int32)
    mesh = QuadMesh.__new__(QuadMesh)
    mesh.vertices = verts
    mesh.faces = faces
    mesh.vertices_original = verts.copy()
    cfg = OptimisationConfig(verbose=False)
    opt = MeshOptimiser(cfg)
    valid, msg = opt.validate_mesh(mesh)
    assert valid is False
    assert "duplicate" in msg.lower()


def test_validate_mesh_zero_area_face():
    verts = np.zeros((4, 3))
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh.__new__(QuadMesh)
    mesh.vertices = verts
    mesh.faces = faces
    mesh.vertices_original = verts.copy()
    cfg = OptimisationConfig(verbose=False)
    opt = MeshOptimiser(cfg)
    valid, msg = opt.validate_mesh(mesh)
    assert valid is False
    assert "area" in msg.lower()


def test_validate_mesh_nan_vertices():
    verts = np.array([[np.nan, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh.__new__(QuadMesh)
    mesh.vertices = verts
    mesh.faces = faces
    mesh.vertices_original = np.zeros((4, 3))
    cfg = OptimisationConfig(verbose=False)
    opt = MeshOptimiser(cfg)
    valid, msg = opt.validate_mesh(mesh)
    assert valid is False
    assert "NaN" in msg or "Inf" in msg


# ===========================================================================
# optimiser — MeshOptimiser.optimise with invalid mesh  (line 458)
# ===========================================================================


def test_optimiser_returns_failed_result_for_invalid_mesh():
    """optimise() on invalid mesh returns OptimisationResult with success=False."""
    verts = np.zeros((4, 3))
    mesh = QuadMesh.__new__(QuadMesh)
    mesh.vertices = verts
    mesh.faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh.vertices_original = verts.copy()

    opt = MeshOptimiser(OptimisationConfig(verbose=False))
    result = opt.optimise(mesh)
    assert result.success is False
    assert "VALIDATION_ERROR" in result.message


# ===========================================================================
# optimiser — optimise_mesh_simple  (lines 495-496)
# ===========================================================================


def test_optimise_mesh_simple_uses_default_weights():
    mesh = _noisy_grid(2, 2)
    result = optimise_mesh_simple(mesh, weights=None, max_iter=5, verbose=False)
    assert result is not None
    assert result.n_iterations >= 0


def test_optimise_mesh_simple_verbose(capsys):
    mesh = _noisy_grid(2, 2)
    optimise_mesh_simple(mesh, max_iter=3, verbose=True)
    # Should print something
    out = capsys.readouterr().out
    assert len(out) > 0
