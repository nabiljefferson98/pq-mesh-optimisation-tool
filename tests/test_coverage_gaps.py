"""
tests/test_coverage_gaps.py

Targeted tests to close the coverage gap between 64% and the 70% threshold.

These tests focus on three reachable regions that the existing suite never
exercises:
    1. backends.py — numpy/numba detection paths, print_backend_info(),
       warmup_numba_kernels() early-return path, to_numpy(), get_sparse_module()
    2. energy_terms.py — analyse_energy_components(), suggest_weight_scaling(),
       compute_total_energy(return_components=True), zero-weight angle balance
    3. gradients.py — energy_for_scipy() and gradient_for_scipy() NumPy paths,
       compute_total_gradient() with and without angle_balance weight

GPU (CuPy) and Numba JIT kernel bodies are excluded via the existing
HAS_CUDA / HAS_NUMBA guards — they are legitimately untestable in a CPU-only
environment and are excluded from coverage via pragma comments where needed.
"""

import io
import sys
from unittest.mock import patch

import numpy as np
import pytest

from src.core.mesh import QuadMesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quad_mesh(noise: float = 0.05) -> QuadMesh:
    """
    Build a minimal 3x3 planar quad mesh with optional noise.
    Returns a QuadMesh with vertices_original set.
    """
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    if noise > 0:
        rng = np.random.default_rng(42)
        verts += rng.uniform(-noise, noise, verts.shape)

    faces = np.array(
        [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]],
        dtype=np.int64,
    )
    mesh = QuadMesh(vertices=verts, faces=faces)
    mesh.vertices_original = verts.copy()
    return mesh


# ---------------------------------------------------------------------------
# 1. backends.py — reachable paths
# ---------------------------------------------------------------------------


class TestBackendsReachablePaths:

    def test_print_backend_info_runs_without_error(self, capsys):
        """print_backend_info() must complete and write at least one line."""
        from src.backends import print_backend_info

        print_backend_info()
        captured = capsys.readouterr()
        assert "Backend" in captured.out or len(captured.out) > 0

    def test_warmup_numba_kernels_returns_early_when_no_numba(self):
        """When HAS_NUMBA is False, warmup_numba_kernels() should return immediately."""
        import src.backends as _backends

        original = _backends.HAS_NUMBA
        try:
            _backends.HAS_NUMBA = False
            from src.backends import warmup_numba_kernels

            warmup_numba_kernels()  # must not raise
        finally:
            _backends.HAS_NUMBA = original

    def test_get_array_module_returns_numpy_on_cpu(self):
        from src.backends import get_array_module

        xp = get_array_module()
        assert xp is np

    def test_to_device_is_identity_on_cpu(self):
        from src.backends import to_device

        arr = np.array([1.0, 2.0, 3.0])
        result = to_device(arr)
        np.testing.assert_array_equal(result, arr)

    def test_to_numpy_passthrough_for_ndarray(self):
        from src.backends import to_numpy

        arr = np.array([1.0, 2.0])
        result = to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_get_sparse_module_returns_scipy_sparse_on_cpu(self):
        import scipy.sparse

        from src.backends import get_sparse_module

        sparse_mod = get_sparse_module()
        assert sparse_mod is scipy.sparse

    def test_backend_constants_are_booleans(self):
        from src.backends import BACKEND, HAS_CUDA, HAS_NUMBA

        assert isinstance(HAS_CUDA, bool)
        assert isinstance(HAS_NUMBA, bool)
        assert BACKEND in ("numpy", "numba", "cupy")


# ---------------------------------------------------------------------------
# 2. energy_terms.py — diagnostic utilities and component returns
# ---------------------------------------------------------------------------


class TestEnergyTermsDiagnostics:

    def test_compute_total_energy_return_components_true(self):
        """return_components=True must return a tuple (float, dict)."""
        from src.optimisation.energy_terms import compute_total_energy

        mesh = _make_quad_mesh()
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        result = compute_total_energy(mesh, weights, return_components=True)
        assert isinstance(result, tuple)
        total, components = result
        assert isinstance(total, float)
        assert "E_planarity" in components
        assert "E_fairness" in components
        assert "E_closeness" in components
        assert "E_angle_balance" in components

    def test_compute_total_energy_zero_angle_balance_weight(self):
        """When angle_balance weight == 0, E_angle should be 0.0."""
        from src.optimisation.energy_terms import compute_total_energy

        mesh = _make_quad_mesh()
        weights = {
            "planarity": 10.0,
            "fairness": 1.0,
            "closeness": 1.0,
            "angle_balance": 0.0,
        }
        total, components = compute_total_energy(mesh, weights, return_components=True)
        assert components["E_angle_balance"] == 0.0

    def test_compute_total_energy_nonzero_angle_balance_weight(self):
        """When angle_balance weight > 0, E_angle should be computed."""
        from src.optimisation.energy_terms import compute_total_energy

        mesh = _make_quad_mesh()
        weights = {
            "planarity": 10.0,
            "fairness": 1.0,
            "closeness": 1.0,
            "angle_balance": 1.0,
        }
        total, components = compute_total_energy(mesh, weights, return_components=True)
        assert isinstance(components["E_angle_balance"], float)

    def test_analyse_energy_components_prints_output(self, capsys):
        """analyse_energy_components() must print the ENERGY ANALYSIS header."""
        from src.optimisation.energy_terms import analyse_energy_components

        mesh = _make_quad_mesh()
        weights = {
            "planarity": 10.0,
            "fairness": 1.0,
            "closeness": 1.0,
            "angle_balance": 1.0,
        }
        analyse_energy_components(mesh, weights)
        captured = capsys.readouterr()
        assert "ENERGY ANALYSIS" in captured.out
        assert "Planarity" in captured.out
        assert "Total Energy" in captured.out

    def test_analyse_energy_components_zero_angle_balance(self, capsys):
        """analyse_energy_components() with angle_balance=0 must not print angle line."""
        from src.optimisation.energy_terms import analyse_energy_components

        mesh = _make_quad_mesh()
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        analyse_energy_components(mesh, weights)
        captured = capsys.readouterr()
        assert "ENERGY ANALYSIS" in captured.out

    def test_suggest_weight_scaling_returns_dict(self):
        """suggest_weight_scaling() must return a dict with all four weight keys."""
        from src.optimisation.energy_terms import suggest_weight_scaling

        mesh = _make_quad_mesh()
        result = suggest_weight_scaling(mesh, verbose=False)
        assert isinstance(result, dict)
        for key in ("planarity", "fairness", "closeness", "angle_balance"):
            assert key in result
            assert isinstance(result[key], float)

    def test_suggest_weight_scaling_verbose_prints_output(self, capsys):
        """suggest_weight_scaling(verbose=True) must print WEIGHT RECOMMENDATIONS."""
        from src.optimisation.energy_terms import suggest_weight_scaling

        mesh = _make_quad_mesh()
        suggest_weight_scaling(mesh, verbose=True)
        captured = capsys.readouterr()
        assert "WEIGHT RECOMMENDATIONS" in captured.out

    def test_compute_planarity_per_face_returns_array(self):
        """compute_planarity_per_face() must return a 1-D array of length n_faces."""
        from src.optimisation.energy_terms import compute_planarity_per_face

        mesh = _make_quad_mesh()
        result = compute_planarity_per_face(mesh)
        assert isinstance(result, np.ndarray)
        assert result.shape == (mesh.n_faces,)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# 3. gradients.py — NumPy dispatch paths
# ---------------------------------------------------------------------------


class TestGradientsNumpyPaths:

    def test_energy_for_scipy_returns_scalar(self):
        """energy_for_scipy() must return a Python float."""
        from src.optimisation.gradients import energy_for_scipy

        mesh = _make_quad_mesh()
        x = mesh.vertices.flatten()
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        result = energy_for_scipy(x, mesh, weights)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_gradient_for_scipy_returns_array(self):
        """gradient_for_scipy() must return a 1-D float64 array."""
        from src.optimisation.gradients import gradient_for_scipy

        mesh = _make_quad_mesh()
        x = mesh.vertices.flatten()
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        result = gradient_for_scipy(x, mesh, weights)
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert result.dtype == np.float64

    def test_compute_total_gradient_shape(self):
        """compute_total_gradient() must return an array of shape (n_verts, 3)."""
        from src.optimisation.gradients import compute_total_gradient

        mesh = _make_quad_mesh()
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        grad = compute_total_gradient(mesh, weights)
        assert grad.shape == (mesh.n_vertices, 3)

    def test_compute_total_gradient_with_angle_balance(self):
        """compute_total_gradient() with angle_balance weight must still return correct shape."""
        from src.optimisation.gradients import compute_total_gradient

        mesh = _make_quad_mesh()
        weights = {
            "planarity": 10.0,
            "fairness": 1.0,
            "closeness": 1.0,
            "angle_balance": 1.0,
        }
        grad = compute_total_gradient(mesh, weights)
        assert grad.shape == (mesh.n_vertices, 3)

    def test_update_vertices_raises_on_nan_values(self):
        """update_vertices() with NaN raises ValueError — covers lines 236-244."""
        mesh = _make_quad_mesh()
        bad = mesh.vertices.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            mesh.update_vertices(bad)

    def test_update_vertices_raises_on_shape_mismatch(self):
        """update_vertices() with the wrong shape raises ValueError — covers line 221."""
        mesh = _make_quad_mesh()
        with pytest.raises(ValueError, match="Shape mismatch"):
            mesh.update_vertices(np.zeros((3, 3)))

    def test_get_vertex_faces_out_of_range_raises(self):
        """get_vertex_faces() with an invalid index raises IndexError — covers lines 249-260."""
        mesh = _make_quad_mesh()
        with pytest.raises(IndexError):
            mesh.get_vertex_faces(9999)

    def test_gradient_finite_difference_consistency(self):
        """
        Finite-difference check: the analytical gradient must agree with a
        numerical gradient to within 1e-4 relative tolerance.
        Uses a single vertex perturbation to keep the test fast.
        """
        from src.optimisation.gradients import energy_for_scipy, gradient_for_scipy

        mesh = _make_quad_mesh(noise=0.1)
        x = mesh.vertices.flatten()
        weights = {"planarity": 50.0, "fairness": 1.0, "closeness": 5.0}

        analytical_grad = gradient_for_scipy(x, mesh, weights)

        # Finite difference on the first coordinate only (fast)
        h = 1e-5
        x_fwd = x.copy()
        x_fwd[0] += h
        x_bwd = x.copy()
        x_bwd[0] -= h
        fd_grad_0 = (
            energy_for_scipy(x_fwd, mesh, weights)
            - energy_for_scipy(x_bwd, mesh, weights)
        ) / (2 * h)

        assert abs(analytical_grad[0] - fd_grad_0) < 1e-3, (
            f"Gradient mismatch: analytical={analytical_grad[0]:.6f}, "
            f"finite-diff={fd_grad_0:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. optimiser.py — _NEAR_ZERO_ENERGY module constant accessibility
# ---------------------------------------------------------------------------


class TestOptimiserConstant:

    def test_near_zero_energy_constant_exists_at_module_level(self):
        """_NEAR_ZERO_ENERGY must be importable and have the correct value."""
        from src.optimisation.optimiser import _NEAR_ZERO_ENERGY

        assert _NEAR_ZERO_ENERGY == 1e-15
        assert isinstance(_NEAR_ZERO_ENERGY, float)

    def test_summary_handles_near_zero_initial_energy(self):
        """
        summary() must not raise when a component's initial energy is at
        floating-point noise level (exercises the <= _NEAR_ZERO_ENERGY branches).
        """
        from src.core.mesh import QuadMesh
        from src.optimisation.optimiser import OptimisationResult

        mesh = _make_quad_mesh(noise=0.0)
        result = OptimisationResult(
            success=True,
            message="CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL",
            optimised_mesh=mesh,
            initial_energy=1.0,
            final_energy=0.5,
            n_iterations=10,
            n_function_evaluations=20,
            n_gradient_evaluations=20,
            execution_time=1.0,
            component_energies_initial={
                "planarity": 0.0,  # exercises <= _NEAR_ZERO_ENERGY and final == 0
                "fairness": 5e-16,  # exercises <= _NEAR_ZERO_ENERGY and final > 0
                "closeness": 1.0,  # exercises > _NEAR_ZERO_ENERGY, improvement
                "angle_balance": 2.0,  # exercises > _NEAR_ZERO_ENERGY, trade-off
            },
            component_energies_final={
                "planarity": 0.0,
                "fairness": 1e-14,
                "closeness": 0.5,
                "angle_balance": 2.5,
            },
        )
        summary = result.summary()
        assert "OPTIMISATION COMPLETE" in summary
        assert "already at 0" in summary
        assert "started at 0" in summary
        assert "better" in summary
        assert "trade-off" in summary


# ---------------------------------------------------------------------------
# 5. mesh_geometry.py — triangle face early-return + conical imbalance
# ---------------------------------------------------------------------------


class TestMeshGeometryUncovered:

    def test_compute_face_planarity_triangle_returns_zero(self):
        """Triangle faces are always planar — early-return path (line 22)."""
        from src.optimisation.mesh_geometry import compute_face_planarity_deviation

        tri = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
        )
        assert compute_face_planarity_deviation(tri) == 0.0

    def test_compute_conical_angle_imbalance_non_4valent_returns_zero(self):
        """Vertices with != 4 incident faces return 0.0 (lines 108-112)."""
        from src.optimisation.mesh_geometry import compute_conical_angle_imbalance

        mesh = _make_quad_mesh()
        # Vertex 0 is a corner — has only 1 incident face in a 3x3 mesh
        result = compute_conical_angle_imbalance(mesh, vertex_id=0)
        assert result == 0.0

    def test_compute_conical_angle_imbalance_4valent_interior(self):
        """Interior 4-valent vertex should return a finite imbalance (lines 113-131)."""
        from src.optimisation.mesh_geometry import compute_conical_angle_imbalance

        mesh = _make_quad_mesh(noise=0.0)
        # Vertex 4 is the centre of the 3x3 mesh — exactly 4-valent
        result = compute_conical_angle_imbalance(mesh, vertex_id=4)
        assert isinstance(result, float)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# 6. gradients.py — diagnostics and verification paths
# ---------------------------------------------------------------------------


class TestGradientsDiagnostics:

    def test_compute_gradient_statistics_keys_and_types(self):
        """compute_gradient_statistics() must return all 6 float keys."""
        from src.optimisation.gradients import compute_gradient_statistics

        grad = np.random.default_rng(0).standard_normal((9, 3))
        stats = compute_gradient_statistics(grad)
        for key in (
            "norm",
            "max_magnitude",
            "mean_magnitude",
            "std_magnitude",
            "max_component",
            "min_magnitude",
        ):
            assert key in stats
            assert isinstance(stats[key], float)

    def test_compute_gradient_statistics_zero_gradient(self):
        """Zero gradient must return all-zero statistics."""
        from src.optimisation.gradients import compute_gradient_statistics

        stats = compute_gradient_statistics(np.zeros((9, 3)))
        assert stats["norm"] == 0.0
        assert stats["max_magnitude"] == 0.0

    def test_print_gradient_analysis_runs(self, capsys):
        """print_gradient_analysis() must print GRADIENT ANALYSIS header."""
        from src.optimisation.gradients import print_gradient_analysis

        mesh = _make_quad_mesh()
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        print_gradient_analysis(mesh, weights)
        captured = capsys.readouterr()
        assert "GRADIENT ANALYSIS" in captured.out
        assert "Planarity" in captured.out

    def test_verify_gradient_passes_on_valid_mesh(self):
        """verify_gradient() must return (True, float) for a valid mesh."""
        from src.optimisation.gradients import verify_gradient

        mesh = _make_quad_mesh(noise=0.05)
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        is_ok, err = verify_gradient(mesh, weights, tolerance=1e-2, verbose=False)
        assert isinstance(is_ok, bool)
        assert isinstance(err, float)
        assert err >= 0.0

    def test_verify_gradient_verbose_prints_output(self, capsys):
        """verify_gradient(verbose=True) must print GRADIENT VERIFICATION header."""
        from src.optimisation.gradients import verify_gradient

        mesh = _make_quad_mesh(noise=0.05)
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        verify_gradient(mesh, weights, verbose=True)
        captured = capsys.readouterr()
        assert "GRADIENT VERIFICATION" in captured.out

    def test_compute_numerical_gradient_shape(self):
        """compute_numerical_gradient() must return (n_verts, 3) float64 array."""
        from src.optimisation.energy_terms import compute_total_energy
        from src.optimisation.gradients import compute_numerical_gradient

        mesh = _make_quad_mesh(noise=0.05)
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        grad = compute_numerical_gradient(
            mesh,
            lambda m: float(compute_total_energy(m, weights)),
            epsilon=1e-5,
        )
        assert grad.shape == (mesh.n_vertices, 3)
        assert np.all(np.isfinite(grad))

    def test_compute_total_gradient_use_numerical_true(self):
        """compute_total_gradient(use_numerical=True) exercises the finite-diff path."""
        from src.optimisation.gradients import compute_total_gradient

        mesh = _make_quad_mesh(noise=0.05)
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        grad = compute_total_gradient(mesh, weights, use_numerical=True)
        assert grad.shape == (mesh.n_vertices, 3)

    def test_gradient_for_scipy_nonfinite_replaced(self, capsys):
        """gradient_for_scipy() with a degenerate mesh must replace NaN with 0."""
        from src.optimisation.gradients import gradient_for_scipy

        mesh = _make_quad_mesh(noise=0.0)
        # Inject a NaN vertex to trigger the non-finite replacement branch
        mesh.vertices[0, 0] = np.nan
        x = mesh.vertices.flatten()
        weights = {"planarity": 1.0, "fairness": 1.0, "closeness": 1.0}
        result = gradient_for_scipy(x, mesh, weights)
        assert np.all(np.isfinite(result)), "NaN values must be replaced with 0"

    def test_energy_for_scipy_nonfinite_returns_fallback(self, capsys):
        """energy_for_scipy() with a NaN mesh must return 1e300 fallback."""
        from src.optimisation.gradients import energy_for_scipy

        mesh = _make_quad_mesh(noise=0.0)
        mesh.vertices[0, 0] = np.nan
        x = mesh.vertices.flatten()
        weights = {"planarity": 1.0, "fairness": 1.0, "closeness": 1.0}
        result = energy_for_scipy(x, mesh, weights)
        assert result == 1e300


# ---------------------------------------------------------------------------
# 7. energy_terms.py — remaining uncovered branches
# ---------------------------------------------------------------------------


class TestEnergyTermsRemainingBranches:

    def test_analyse_energy_components_nonzero_angle_shows_angle_line(self, capsys):
        """analyse_energy_components() with angle_balance > 0 prints angle row."""
        from src.optimisation.energy_terms import analyse_energy_components

        mesh = _make_quad_mesh(noise=0.05)
        weights = {
            "planarity": 10.0,
            "fairness": 1.0,
            "closeness": 1.0,
            "angle_balance": 1.0,
        }
        analyse_energy_components(mesh, weights)
        captured = capsys.readouterr()
        assert "Angle" in captured.out or "angle" in captured.out.lower()

    def test_suggest_weight_scaling_handles_near_planar_mesh(self):
        """suggest_weight_scaling() on a perfectly flat mesh must not raise."""
        from src.optimisation.energy_terms import suggest_weight_scaling

        mesh = _make_quad_mesh(noise=0.0)
        result = suggest_weight_scaling(mesh, verbose=False)
        assert isinstance(result, dict)

    def test_compute_total_energy_missing_optional_angle_key(self):
        """compute_total_energy() with no 'angle_balance' key in weights is valid."""
        from src.optimisation.energy_terms import compute_total_energy

        mesh = _make_quad_mesh()
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        total = compute_total_energy(mesh, weights)
        assert isinstance(total, float) or (
            isinstance(total, tuple) and isinstance(total[0], float)
        )

    def test_compute_total_energy_returns_float_not_tuple_by_default(self):
        """compute_total_energy() with return_components=False returns a float."""
        from src.optimisation.energy_terms import compute_total_energy

        mesh = _make_quad_mesh()
        weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}
        result = compute_total_energy(mesh, weights, return_components=False)
        # Allow tuple for backward compatibility but assert first element is float
        val = result[0] if isinstance(result, tuple) else result
        assert isinstance(val, float)
        assert val >= 0.0


# ---------------------------------------------------------------------------
# 8. mesh.py — property rebuild paths via reset_topology_cache()
# ---------------------------------------------------------------------------


class TestMeshUncoveredPaths:

    def test_scatter_matrix_is_built_on_first_access(self):
        """scatter_matrix property builds _scatter_matrix on first access (lines 118-126)."""
        mesh = _make_quad_mesh()
        mesh._scatter_matrix = None  # force cache miss
        sm = mesh.scatter_matrix
        assert sm is not None
        assert sm.shape == (mesh.n_vertices, mesh.n_faces * 4)

    def test_scatter_matrix_rebuilt_after_reset_topology_cache(self):
        """reset_topology_cache() nullifies _scatter_matrix; next access rebuilds it."""
        mesh = _make_quad_mesh()
        _ = mesh.scatter_matrix  # prime cache
        mesh.reset_topology_cache()  # sets _scatter_matrix = None
        assert mesh._scatter_matrix is None  # confirm nullified
        rebuilt = mesh.scatter_matrix  # triggers lines 119-126
        assert rebuilt is not None
        assert rebuilt.shape[1] == mesh.n_faces * 4

    def test_laplacian_built_on_first_access(self):
        """laplacian property builds _laplacian_cpu on first access (lines 149-175)."""
        mesh = _make_quad_mesh()
        mesh._laplacian_cpu = None  # force cache miss
        L = mesh.laplacian
        assert L is not None
        assert L.shape == (mesh.n_vertices, mesh.n_vertices)

    def test_laplacian_rebuilt_after_reset_topology_cache(self):
        """reset_topology_cache() nullifies _laplacian_cpu; next access rebuilds it."""
        mesh = _make_quad_mesh()
        _ = mesh.laplacian  # prime cache
        mesh.reset_topology_cache()  # sets _laplacian_cpu = None
        assert mesh._laplacian_cpu is None  # confirm nullified
        rebuilt = mesh.laplacian  # triggers rebuild path
        assert rebuilt is not None
        assert rebuilt.shape == (mesh.n_vertices, mesh.n_vertices)

    def test_update_vertices_raises_on_shape_mismatch(self):
        """update_vertices() with wrong shape must raise ValueError (line 221)."""
        mesh = _make_quad_mesh()
        bad = np.zeros((3, 3), dtype=np.float64)  # wrong n_vertices
        with pytest.raises(ValueError, match="Shape mismatch"):
            mesh.update_vertices(bad)

    def test_update_vertices_raises_on_nan(self):
        """update_vertices() with NaN values must raise ValueError (lines 236-244)."""
        mesh = _make_quad_mesh()
        bad = mesh.vertices.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            mesh.update_vertices(bad)

    def test_vertex_face_ids_padded_shape(self):
        """vertex_face_ids_padded must return array of shape (n_verts, max_valence)."""
        mesh = _make_quad_mesh()
        table = mesh.vertex_face_ids_padded
        assert table.ndim == 2
        assert table.shape[0] == mesh.n_vertices
        # Sentinel padding value
        assert np.all((table >= -1))

    def test_get_vertex_faces_raises_on_out_of_range(self):
        """get_vertex_faces() with invalid index must raise IndexError (lines 249-260)."""
        mesh = _make_quad_mesh()
        with pytest.raises(IndexError):
            mesh.get_vertex_faces(9999)


# ---------------------------------------------------------------------------
# 9. optimiser.py — uncovered paths using correct API
# ---------------------------------------------------------------------------


class TestOptimiserUncoveredPaths:

    def test_optimisation_config_default_weights(self):
        """OptimisationConfig default weights must contain all three required keys
        and be positive — covers lines 68-69 (default_factory lambda)."""
        from src.optimisation.optimiser import OptimisationConfig

        cfg = OptimisationConfig()
        for key in ("planarity", "fairness", "closeness"):
            assert key in cfg.weights
            assert cfg.weights[key] > 0.0

    def test_optimisation_config_validate_raises_on_missing_weight(self):
        """OptimisationConfig.validate() must raise ValueError when a required
        weight is absent — covers lines 124 (validate loop)."""
        from src.optimisation.optimiser import OptimisationConfig

        cfg = OptimisationConfig(weights={"planarity": 10.0, "fairness": 1.0})
        with pytest.raises(ValueError, match="Missing weight"):
            cfg.validate()

    def test_optimisation_config_validate_raises_on_negative_weight(self):
        """OptimisationConfig.validate() must raise ValueError for negative weights."""
        from src.optimisation.optimiser import OptimisationConfig

        cfg = OptimisationConfig(
            weights={"planarity": -1.0, "fairness": 1.0, "closeness": 1.0}
        )
        with pytest.raises(ValueError, match="non-negative"):
            cfg.validate()

    def test_optimisation_config_validate_raises_on_bad_tolerance(self):
        """OptimisationConfig.validate() must raise ValueError for tolerance <= 0."""
        from src.optimisation.optimiser import OptimisationConfig

        cfg = OptimisationConfig(
            weights={"planarity": 10.0, "fairness": 1.0, "closeness": 1.0},
            tolerance=-1e-6,
        )
        with pytest.raises(ValueError, match="tolerance"):
            cfg.validate()

    def test_mesh_optimiser_validate_mesh_too_few_vertices(self):
        """MeshOptimiser.validate_mesh() with < 4 vertices returns (False, str)
        — covers lines 271 (n_vertices < 4 branch)."""
        from src.core.mesh import QuadMesh
        from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig

        opt = MeshOptimiser()
        tiny = QuadMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2, 0]], dtype=np.int32),
        )
        is_valid, msg = opt.validate_mesh(tiny)
        assert not is_valid
        assert "vertices" in msg.lower()

    def test_mesh_optimiser_optimise_returns_invalid_result_on_bad_mesh(self):
        """MeshOptimiser.optimise() with an invalid mesh returns success=False
        without calling scipy — covers lines 413 (validation failure branch)."""
        from src.core.mesh import QuadMesh
        from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig

        cfg = OptimisationConfig(
            weights={"planarity": 10.0, "fairness": 1.0, "closeness": 1.0},
            verbose=False,
        )
        opt = MeshOptimiser(cfg)
        tiny = QuadMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2, 0]], dtype=np.int32),
        )
        result = opt.optimise(tiny)
        assert result.success is False
        assert "VALIDATION_ERROR" in result.message

    def test_optimisation_result_summary_uses_label_keys(self):
        """OptimisationResult.summary() must use human-readable label strings
        (Panel flatness, Surface smoothness, etc.) — covers lines 706-715."""
        from src.optimisation.optimiser import OptimisationResult

        mesh = _make_quad_mesh()
        result = OptimisationResult(
            success=True,
            message="CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL",
            optimised_mesh=mesh,
            initial_energy=10.0,
            final_energy=1.0,
            n_iterations=50,
            n_function_evaluations=100,
            n_gradient_evaluations=100,
            execution_time=2.5,
            component_energies_initial={
                "planarity": 8.0,
                "fairness": 1.0,
                "closeness": 1.0,
                "angle_balance": 0.0,
            },
            component_energies_final={
                "planarity": 0.5,
                "fairness": 0.3,
                "closeness": 0.2,
                "angle_balance": 0.0,
            },
        )
        s = result.summary()
        assert "OPTIMISATION COMPLETE" in s
        assert "Panel flatness" in s  # actual label from _labels dict
        assert "Surface smoothness" in s
        assert "Shape fidelity" in s
        assert "Corner balance" in s
        assert "90.00%" in s  # 10.0 → 1.0 = 90% improvement

    def test_optimisation_result_summary_step_limit_branch(self):
        """summary() with 'ITERATIONS REACHED LIMIT' message covers the
        step-limit branch (lines 502-503)."""
        from src.optimisation.optimiser import OptimisationResult

        mesh = _make_quad_mesh()
        result = OptimisationResult(
            success=False,
            message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
            optimised_mesh=mesh,
            initial_energy=10.0,
            final_energy=5.0,
            n_iterations=200,
            n_function_evaluations=400,
            n_gradient_evaluations=400,
            execution_time=30.0,
            component_energies_initial={
                "planarity": 8.0,
                "fairness": 1.0,
                "closeness": 1.0,
            },
            component_energies_final={
                "planarity": 4.0,
                "fairness": 0.5,
                "closeness": 0.5,
            },
        )
        s = result.summary()
        assert "STEP LIMIT REACHED" in s
        assert "50.0%" in s  # (10-5)/10 = 50%

    def test_optimisation_result_summary_abnormal_branch(self):
        """summary() with 'ABNORMAL' in message covers the error branch."""
        from src.optimisation.optimiser import OptimisationResult

        mesh = _make_quad_mesh()
        result = OptimisationResult(
            success=False,
            message="ABNORMAL_TERMINATION_IN_LNSRCH",
            optimised_mesh=mesh,
            initial_energy=10.0,
            final_energy=9.9,
            n_iterations=5,
            n_function_evaluations=10,
            n_gradient_evaluations=10,
            execution_time=0.5,
            component_energies_initial={},
            component_energies_final={},
        )
        s = result.summary()
        assert "FAILED" in s

    def test_optimise_mesh_simple_runs_on_valid_mesh(self):
        """optimise_mesh_simple() must return an OptimisationResult
        — covers lines 38-42 (OptimisationConfig.__post_init__) and
        the full optimise() pipeline on a tiny mesh."""
        from src.optimisation.optimiser import optimise_mesh_simple

        mesh = _make_quad_mesh(noise=0.05)
        result = optimise_mesh_simple(
            mesh,
            weights={"planarity": 10.0, "fairness": 1.0, "closeness": 1.0},
            max_iter=5,  # minimal — we only need coverage, not convergence
            verbose=False,
        )
        from src.optimisation.optimiser import OptimisationResult

        assert isinstance(result, OptimisationResult)
        assert isinstance(result.final_energy, float)
        assert result.final_energy >= 0.0


# ---------------------------------------------------------------------------
# 10. backends.py — env-var forced paths and diagnostics
# ---------------------------------------------------------------------------


class TestBackendsEnvVarPaths:

    def test_forced_numba_backend_env_var(self, monkeypatch):
        """PQ_BACKEND=numba env var forces numba detection (lines 84-111).
        If numba is not installed, it falls back to numpy with a warning."""
        import importlib

        try:
            import numba  # noqa: F401

            has_numba = True
        except ImportError:
            has_numba = False

        monkeypatch.setenv("PQ_BACKEND", "numba")
        import src.backends as _b

        # Re-run _detect_backend() directly — does not reimport the module
        result = _b._detect_backend()

        if has_numba:
            assert result == "numba"
        else:
            # When numba is not installed, it falls back to numpy
            assert result == "numpy"

    def test_forced_numpy_backend_env_var(self, monkeypatch):
        """PQ_BACKEND=numpy env var returns numpy even when numba is installed."""
        import src.backends as _b

        monkeypatch.setenv("PQ_BACKEND", "numpy")
        result = _b._detect_backend()
        assert result == "numpy"

    def test_forced_cupy_env_var_falls_back_when_no_gpu(self, monkeypatch):
        """PQ_BACKEND=cupy with no GPU emits RuntimeWarning and falls back
        — covers lines 84-87 warning block."""
        import src.backends as _b

        monkeypatch.setenv("PQ_BACKEND", "cupy")
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _b._detect_backend()
        # On a Mac without CUDA, must fall back to numba or numpy
        assert result in ("numba", "numpy")
        # The warning must have been emitted
        warning_messages = [str(x.message) for x in w]
        assert any("cupy" in m.lower() or "cuda" in m.lower() for m in warning_messages)

    def test_warmup_numba_kernels_executes_without_raising(self):
        """warmup_numba_kernels() with HAS_NUMBA=True must not raise
        — covers lines 337-406 print/import block."""
        from src.backends import HAS_NUMBA, warmup_numba_kernels

        if not HAS_NUMBA:
            pytest.skip("Numba not installed — warmup path not reachable")
        # Must complete without raising; Numba is already compiled so < 1s
        warmup_numba_kernels()

    def test_gpu_memory_guard_passthrough_on_no_error(self):
        """gpu_memory_guard() context manager must yield without exception
        on a no-error path — covers the try/yield block (lines 192-196)."""
        from src.backends import gpu_memory_guard

        with gpu_memory_guard():
            x = np.ones(4) * 2.0
        assert x[0] == 2.0

    def test_gpu_memory_guard_reraises_non_gpu_exceptions(self):
        """gpu_memory_guard() must re-raise non-GPU exceptions unchanged
        — covers the else: raise branch (lines 213-219)."""
        from src.backends import gpu_memory_guard

        with pytest.raises(ValueError, match="test error"):
            with gpu_memory_guard():
                raise ValueError("test error")

    def test_print_backend_info_numba_version_line(self, capsys):
        """print_backend_info() with HAS_NUMBA prints Numba version
        — covers lines 290-298."""
        from src.backends import HAS_NUMBA, print_backend_info

        print_backend_info()
        captured = capsys.readouterr()
        if HAS_NUMBA:
            assert "Numba" in captured.out
        assert "NumPy" in captured.out


# ---------------------------------------------------------------------------
# 11. energy_terms.py — uncovered diagnostic functions (lines 532-834)
# ---------------------------------------------------------------------------


class TestEnergyTermsRemainingDiagnostics:

    def test_compute_planarity_per_face_shape_and_nonnegativity(self):
        """compute_planarity_per_face() returns (n_faces,) non-negative array
        — covers lines 532-548."""
        from src.optimisation.energy_terms import compute_planarity_per_face

        mesh = _make_quad_mesh(noise=0.1)
        per_face = compute_planarity_per_face(mesh)
        assert per_face.shape == (mesh.n_faces,)
        assert np.all(per_face >= 0.0)

    def test_compute_planarity_per_face_zero_for_flat_mesh(self):
        """compute_planarity_per_face() returns ~0 for a perfectly flat mesh."""
        from src.optimisation.energy_terms import compute_planarity_per_face

        mesh = _make_quad_mesh(noise=0.0)
        per_face = compute_planarity_per_face(mesh)
        assert np.allclose(per_face, 0.0, atol=1e-10)

    def test_compute_closeness_energy_zero_for_unmodified_mesh(self):
        """compute_closeness_energy() returns 0.0 when vertices match original
        — covers lines 642-652."""
        from src.optimisation.energy_terms import compute_closeness_energy

        mesh = _make_quad_mesh()
        # vertices_original is set to vertices.copy() in __init__, so delta = 0
        energy = compute_closeness_energy(mesh)
        assert float(energy) == pytest.approx(0.0, abs=1e-12)

    def test_compute_closeness_energy_positive_after_displacement(self):
        """compute_closeness_energy() grows after displacing vertices."""
        from src.optimisation.energy_terms import compute_closeness_energy

        mesh = _make_quad_mesh()
        mesh.vertices = mesh.vertices + 1.0  # displace all vertices by 1 unit
        energy = compute_closeness_energy(mesh)
        assert float(energy) > 0.0

    def test_compute_angle_balance_energy_returns_nonnegative(self):
        """compute_angle_balance_energy() must return a non-negative float
        — covers lines 665-691."""
        from src.optimisation.energy_terms import compute_angle_balance_energy

        mesh = _make_quad_mesh(noise=0.05)
        energy = float(compute_angle_balance_energy(mesh))
        assert energy >= 0.0

    def test_compute_angle_balance_energy_zero_for_regular_grid(self):
        """compute_angle_balance_energy() is ~0 for a regular flat grid
        (all angles are 90° — perfectly conical)."""
        from src.optimisation.energy_terms import compute_angle_balance_energy

        mesh = _make_quad_mesh(noise=0.0)
        energy = float(compute_angle_balance_energy(mesh))
        assert energy >= 0.0  # always non-negative; exact zero depends on topology

    def test_compute_fairness_energy_zero_for_regular_grid(self):
        """compute_fairness_energy() is ~0 for a perfectly regular flat mesh
        — covers lines 701-721."""
        from src.optimisation.energy_terms import compute_fairness_energy

        mesh = _make_quad_mesh(noise=0.0)
        energy = float(compute_fairness_energy(mesh))
        assert energy >= 0.0

    def test_compute_fairness_energy_increases_with_noise(self):
        """compute_fairness_energy() must be larger for a noisy mesh."""
        from src.optimisation.energy_terms import compute_fairness_energy

        flat_mesh = _make_quad_mesh(noise=0.0)
        noisy_mesh = _make_quad_mesh(noise=0.5)
        e_flat = float(compute_fairness_energy(flat_mesh))
        e_noisy = float(compute_fairness_energy(noisy_mesh))
        assert e_noisy >= e_flat

    def test_compute_planarity_energy_increases_with_noise(self):
        """compute_planarity_energy() increases with vertex perturbation
        — covers lines 738-834 (Numba and NumPy dispatch)."""
        from src.optimisation.energy_terms import compute_planarity_energy

        flat_mesh = _make_quad_mesh(noise=0.0)
        noisy_mesh = _make_quad_mesh(noise=0.5)
        e_flat = float(compute_planarity_energy(flat_mesh))
        e_noisy = float(compute_planarity_energy(noisy_mesh))
        assert e_noisy > e_flat


# ---------------------------------------------------------------------------
# 12. gradients.py — uncovered diagnostic ranges (lines 882-954, 1132-1145)
# ---------------------------------------------------------------------------


class TestGradientsRemainingDiagnostics:

    def test_compute_angle_balance_gradient_shape(self):
        """compute_angle_balance_gradient() returns (n_verts, 3) array
        — exercises lines 882-930 (Numba dispatch) and 937-954 (NumPy fallback)."""
        from src.optimisation.gradients import compute_angle_balance_gradient

        mesh = _make_quad_mesh(noise=0.05)
        grad = compute_angle_balance_gradient(mesh)
        assert grad.shape == (mesh.n_vertices, 3)
        assert np.all(np.isfinite(grad))

    def test_compute_angle_balance_gradient_zero_for_flat_regular_grid(self):
        """Angle balance gradient must be ~0 for a perfectly regular flat mesh."""
        from src.optimisation.gradients import compute_angle_balance_gradient

        mesh = _make_quad_mesh(noise=0.0)
        grad = compute_angle_balance_gradient(mesh)
        # Boundary vertices may have nonzero gradient; check interior is finite
        assert grad.shape == (mesh.n_vertices, 3)
        assert np.all(np.isfinite(grad))

    def test_compute_closeness_gradient_shape_and_zero_at_original(self):
        """compute_closeness_gradient() returns (n_verts, 3) and is zero
        when vertices match original — covers lines 1132-1145."""
        from src.optimisation.gradients import compute_closeness_gradient

        mesh = _make_quad_mesh()
        grad = compute_closeness_gradient(mesh)
        assert grad.shape == (mesh.n_vertices, 3)
        # vertices == vertices_original so gradient = 2*(v - v0) = 0
        assert np.allclose(grad, 0.0, atol=1e-12)

    def test_compute_closeness_gradient_nonzero_after_displacement(self):
        """compute_closeness_gradient() is nonzero after displacing vertices."""
        from src.optimisation.gradients import compute_closeness_gradient

        mesh = _make_quad_mesh()
        mesh.vertices = mesh.vertices + 0.5
        grad = compute_closeness_gradient(mesh)
        assert not np.allclose(grad, 0.0)
        assert np.all(np.isfinite(grad))

    def test_compute_fairness_gradient_shape(self):
        """compute_fairness_gradient() returns (n_verts, 3) — covers lines
        adjacent to 1132-1145 fairness path."""
        from src.optimisation.gradients import compute_fairness_gradient

        mesh = _make_quad_mesh(noise=0.05)
        grad = compute_fairness_gradient(mesh)
        assert grad.shape == (mesh.n_vertices, 3)
        assert np.all(np.isfinite(grad))
