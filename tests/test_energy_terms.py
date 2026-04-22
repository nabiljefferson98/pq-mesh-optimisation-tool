"""
Unit tests for energy term implementations.

Tests verify:
- Mathematical correctness of energy functionals
- Boundary conditions (e.g., zero energy for perfect meshes)
- Sensitivity to mesh perturbations
- Numerical stability

Date: 2 February 2026
"""

import numpy as np
import pytest

from src.core.mesh import QuadMesh
from src.optimisation.energy_terms import (
    analyse_energy_components,
    compute_closeness_energy,
    compute_fairness_energy,
    compute_planarity_energy,
    compute_total_energy,
    suggest_weight_scaling,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def planar_quad():
    """Single perfectly planar quad."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return QuadMesh(V, F)


@pytest.fixture
def nonplanar_quad():
    """Single non-planar quad (one vertex lifted)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return QuadMesh(V, F)


@pytest.fixture
def quad_grid_2x2():
    """2×2 grid of quads."""
    V = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [0, 2, 0],
            [1, 2, 0],
            [2, 2, 0],
        ],
        dtype=np.float64,
    )

    F = np.array(
        [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.int32
    )

    return QuadMesh(V, F)


# ============================================================================
# PLANARITY ENERGY TESTS
# ============================================================================


def test_planar_quad_has_zero_planarity_energy(planar_quad):
    """Perfectly planar quad should have E_planarity ≈ 0."""
    energy = compute_planarity_energy(planar_quad)
    assert energy < 1e-10, f"Expected ~0, got {energy}"


def test_nonplanar_quad_has_positive_planarity_energy(nonplanar_quad):
    """Non-planar quad should have E_planarity > 0."""
    energy = compute_planarity_energy(nonplanar_quad)
    assert energy > 0.001, f"Expected positive energy, got {energy}"


def test_planarity_energy_increases_with_deviation(planar_quad):
    """Lifting vertex should increase planarity energy."""
    E_initial = compute_planarity_energy(planar_quad)

    # Lift one vertex
    planar_quad.vertices[2, 2] = 0.1
    E_perturbed = compute_planarity_energy(planar_quad)

    assert E_perturbed > E_initial, "Energy should increase with perturbation"


# ============================================================================
# FAIRNESS ENERGY TESTS
# ============================================================================


def test_fairness_energy_positive(quad_grid_2x2):
    """Fairness energy should be non-negative."""
    energy = compute_fairness_energy(quad_grid_2x2)
    assert energy >= 0, f"Energy must be non-negative, got {energy}"


def test_fairness_energy_increases_with_irregularity(quad_grid_2x2):
    """Displacing interior vertex should increase fairness energy."""
    E_initial = compute_fairness_energy(quad_grid_2x2)

    # Move interior vertex (index 4: centre of grid)
    quad_grid_2x2.vertices[4] += [0.2, 0.2, 0.3]
    E_perturbed = compute_fairness_energy(quad_grid_2x2)

    assert E_perturbed > E_initial, "Fairness energy should increase with irregularity"


# ============================================================================
# CLOSENESS ENERGY TESTS
# ============================================================================


def test_closeness_energy_zero_for_unchanged_mesh(planar_quad):
    """Closeness energy should be zero if vertices unchanged."""
    energy = compute_closeness_energy(planar_quad)
    assert energy < 1e-10, f"Expected ~0 for unchanged mesh, got {energy}"


def test_closeness_energy_increases_with_displacement(planar_quad):
    """Moving vertices away from original should increase closeness energy."""
    # Move one vertex
    planar_quad.vertices[0] += [0.1, 0.1, 0.1]

    energy = compute_closeness_energy(planar_quad)

    # Expected: 0.1² + 0.1² + 0.1² = 0.03
    expected = 3 * 0.1**2
    assert np.abs(energy - expected) < 1e-6, f"Expected {expected}, got {energy}"


def test_closeness_energy_is_quadratic(planar_quad):
    """Closeness energy should scale quadratically with displacement."""
    # Displacement of magnitude d should give energy ∝ d²

    planar_quad.vertices[0] += [0.1, 0, 0]
    E1 = compute_closeness_energy(planar_quad)

    planar_quad.reset_to_original()
    planar_quad.vertices[0] += [0.2, 0, 0]  # Double displacement
    E2 = compute_closeness_energy(planar_quad)

    # E2 should be ≈ 4 * E1 (quadratic scaling)
    assert np.abs(E2 / E1 - 4.0) < 0.1, f"Expected quadratic scaling, got ratio {E2/E1}"


# ============================================================================
# TOTAL ENERGY TESTS
# ============================================================================


def test_total_energy_combines_components(planar_quad):
    """Total energy should be weighted sum of components."""
    weights = {"planarity": 100.0, "fairness": 1.0, "closeness": 10.0}

    E_total, components = compute_total_energy(
        planar_quad, weights, return_components=True
    )

    # Manually compute weighted sum
    expected = (
        weights["planarity"] * components["E_planarity"]
        + weights["fairness"] * components["E_fairness"]
        + weights["closeness"] * components["E_closeness"]
    )

    assert np.abs(E_total - expected) < 1e-6, "Total should equal weighted sum"


def test_total_energy_respects_weights(nonplanar_quad):
    """Changing weights should affect total energy proportionally."""
    weights1 = {"planarity": 1.0, "fairness": 1.0, "closeness": 1.0}
    weights2 = {"planarity": 10.0, "fairness": 1.0, "closeness": 1.0}

    E1 = compute_total_energy(nonplanar_quad, weights1)
    E2 = compute_total_energy(nonplanar_quad, weights2)

    # E2 should be larger due to increased planarity weight
    assert E2 > E1, "Higher planarity weight should increase total energy"


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


def test_analyse_energy_components_runs(quad_grid_2x2, capsys):
    """Ensure analysis function runs without error and prints output."""
    weights = {"planarity": 100.0, "fairness": 1.0, "closeness": 10.0}

    analyse_energy_components(quad_grid_2x2, weights)

    captured = capsys.readouterr()
    assert "ENERGY ANALYSIS" in captured.out
    assert "Total Energy:" in captured.out


def test_suggest_weight_scaling_returns_dict(quad_grid_2x2):
    """Weight suggestion should return valid dictionary."""
    suggested = suggest_weight_scaling(quad_grid_2x2, verbose=False)

    assert isinstance(suggested, dict)
    assert "planarity" in suggested
    assert "fairness" in suggested
    assert "closeness" in suggested
    assert all(w >= 0 for w in suggested.values()), "Weights should be non-negative"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_energy_with_zero_weights(planar_quad):
    """Total energy with all zero weights should be zero."""
    weights = {"planarity": 0.0, "fairness": 0.0, "closeness": 0.0}

    E_total = compute_total_energy(planar_quad, weights)

    assert E_total == 0.0, "Zero weights should give zero total energy"


def test_energy_handles_degenerate_mesh():
    """Energy computation should not crash on degenerate (flat) mesh."""
    # All vertices in a line
    V = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(V, F)

    weights = {"planarity": 1.0, "fairness": 1.0, "closeness": 1.0}

    # Should not crash (may produce warnings)
    E_total = compute_total_energy(mesh, weights)

    assert np.isfinite(E_total), "Energy should be finite even for degenerate mesh"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
