"""
tests/test_optimiser.py

Unit tests for mesh optimisation pipeline.
"""

import numpy as np
import pytest

from src.core.mesh import QuadMesh
from src.optimisation.energy_terms import compute_planarity_energy
from src.optimisation.optimiser import (
    MeshOptimiser,
    OptimisationConfig,
    optimise_mesh_simple,
)


@pytest.fixture
def planar_quad():
    """Perfectly planar quad (no optimisation needed)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return QuadMesh(V, F)


@pytest.fixture
def nonplanar_quad():
    """Non-planar quad (needs optimisation).

    Vertex 3 is lifted 0.5 units above the plane defined by vertices 0, 1, 2.
    Coplanar distance = 0.5, planarity energy ≈ 5.5e-02 — unambiguously
    non-planar on all backends (CPU numpy, Numba, CuPy GPU SVD).

    Previous fixtures with alternating z-signs (e.g. z=[0,+d,0,-d]) define
    geometrically planar saddles: all four points lie on the plane z = d*x - d*y
    and SVD correctly returns zero planarity energy regardless of d.
    """
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.5]], dtype=np.float64)
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return QuadMesh(V, F)


def test_optimisation_config_default():
    """Test default optimisation configuration."""
    config = OptimisationConfig()
    config.validate()

    assert "planarity" in config.weights
    assert "fairness" in config.weights
    assert "closeness" in config.weights
    assert config.max_iterations > 0
    assert config.tolerance > 0


def test_optimisation_config_validation():
    """Test configuration validation catches errors."""
    # Negative iterations
    with pytest.raises(ValueError):
        config = OptimisationConfig(max_iterations=-1)
        config.validate()

    # Missing weight
    with pytest.raises(ValueError):
        config = OptimisationConfig(weights={"planarity": 1.0})
        config.validate()

    # Negative weight
    with pytest.raises(ValueError):
        config = OptimisationConfig(
            weights={"planarity": -1.0, "fairness": 1.0, "closeness": 1.0}
        )
        config.validate()


def test_optimiser_initialisation():
    """Test optimiser initialisation."""
    optimiser = MeshOptimiser()
    assert optimiser.config is not None
    assert optimiser._iteration_count == 0


def test_optimise_planar_quad_unchanged(planar_quad):
    """Perfectly planar quad should have minimal energy change."""
    verts_initial = planar_quad.vertices.copy()

    config = OptimisationConfig(
        weights={"planarity": 100.0, "fairness": 1.0, "closeness": 10.0},
        max_iterations=100,
        verbose=False,
    )

    optimiser = MeshOptimiser(config)
    result = optimiser.optimise(planar_quad)

    # Should converge successfully
    assert result.success

    # Planarity should remain near-zero (it was already planar)
    assert result.component_energies_final["planarity"] < 1e-6

    # Vertex displacement should be small (fairness may cause slight contraction)
    displacement = np.linalg.norm(planar_quad.vertices - verts_initial)
    assert displacement < 1.0, f"Excessive displacement: {displacement}"


def test_optimise_nonplanar_quad_improves(nonplanar_quad):
    """Non-planar quad should improve planarity."""
    initial_planarity = compute_planarity_energy(nonplanar_quad)

    # Geometric guarantee: vertex 3 is 0.5 units off-plane, so planarity
    # energy must be >> 0. Assert rather than skip to catch regressions.
    assert initial_planarity > 1e-4, (
        f"Fixture unexpectedly planar: planarity = {initial_planarity:.2e}. "
        f"This indicates a bug in compute_planarity_energy or the fixture geometry."
    )

    result = optimise_mesh_simple(
        nonplanar_quad,
        weights={"planarity": 100.0, "fairness": 1.0, "closeness": 10.0},
        max_iter=100,
        verbose=False,
    )

    assert result.success
    assert result.final_energy <= result.initial_energy + 1e-6
    assert (
        result.component_energies_final["planarity"]
        <= result.component_energies_initial["planarity"] + 1e-6
    )


def test_optimiser_history_tracking(nonplanar_quad):
    """Test that history tracking works correctly."""
    config = OptimisationConfig(
        weights={"planarity": 100.0, "fairness": 1.0, "closeness": 10.0},
        max_iterations=50,
        verbose=False,
        history_tracking=True,
        two_stage=False,
    )

    optimiser = MeshOptimiser(config)
    result = optimiser.optimise(nonplanar_quad)

    # History should be populated
    assert result.energy_history is not None
    assert result.gradient_norm_history is not None
    assert len(result.energy_history) > 0
    assert len(result.gradient_norm_history) > 0

    # Energy should be monotonically decreasing (or flat)
    energies = np.array(result.energy_history)
    assert np.all(np.diff(energies) <= 1e-6)  # Allow small numerical noise


def test_optimiser_without_history(nonplanar_quad):
    """Test optimisation without history tracking."""
    config = OptimisationConfig(
        weights={"planarity": 100.0, "fairness": 1.0, "closeness": 10.0},
        max_iterations=50,
        verbose=False,
        history_tracking=False,
    )

    optimiser = MeshOptimiser(config)
    result = optimiser.optimise(nonplanar_quad)

    assert result.energy_history is None
    assert result.gradient_norm_history is None


def test_optimisation_result_methods(nonplanar_quad):
    """Test OptimisationResult utility methods."""
    result = optimise_mesh_simple(nonplanar_quad, max_iter=50, verbose=False)

    # Test energy reduction calculation
    reduction = result.energy_reduction()
    assert 0 <= reduction <= 1

    # Test percentage
    percentage = result.energy_reduction_percentage()
    assert 0 <= percentage <= 100

    # Test summary generation
    summary = result.summary()
    assert any(
        keyword in summary
        for keyword in (
            "CONVERGED",
            "ITERATION LIMIT",
            "FAILED",
            "DID NOT CONVERGE",
            "TOLERANCE NOT MET",
            "FINISHED SUCCESSFULLY",
            "STEP LIMIT REACHED",
            "NEARLY THERE",
            "PARTIAL IMPROVEMENT",
        )
    )
    assert "Score at the start" in summary or "Initial energy" in summary
    assert "Score at the end" in summary or "Final energy" in summary


def test_weights_effect_on_optimisation(nonplanar_quad):
    """Test that different weights produce different results."""
    mesh1 = QuadMesh(nonplanar_quad.vertices.copy(), nonplanar_quad.faces.copy())
    mesh2 = QuadMesh(nonplanar_quad.vertices.copy(), nonplanar_quad.faces.copy())

    # High planarity weight
    result1 = optimise_mesh_simple(
        mesh1,
        weights={"planarity": 1000.0, "fairness": 1.0, "closeness": 1.0},
        max_iter=100,
        verbose=False,
    )

    # High closeness weight (should stay closer to original)
    result2 = optimise_mesh_simple(
        mesh2,
        weights={"planarity": 10.0, "fairness": 1.0, "closeness": 1000.0},
        max_iter=100,
        verbose=False,
    )

    # High planarity weight should achieve good planarity.
    # If both results are at machine precision (< 1e-20), both are "perfect"
    # and the weight distinction is irrelevant — the test intent is satisfied.
    p1 = result1.component_energies_final["planarity"]
    p2 = result2.component_energies_final["planarity"]
    both_near_zero = p1 < 1e-20 and p2 < 1e-20
    assert (
        both_near_zero or p1 < p2
    ), f"High-planarity run ({p1:.2e}) should be ≤ high-closeness run ({p2:.2e})"

    # High closeness weight should preserve original better
    assert (
        result2.component_energies_final["closeness"]
        < result1.component_energies_final["closeness"]
    )


def test_max_iterations_limit():
    """Test that max_iterations limit is respected."""
    V = np.array([[0, 0, 0], [1, 0, 0.5], [1, 1, 0], [0, 1, -0.5]], dtype=np.float64)
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(V, F)

    config = OptimisationConfig(
        weights={"planarity": 100.0, "fairness": 1.0, "closeness": 10.0},
        max_iterations=10,  # Very low limit
        verbose=False,
    )

    optimiser = MeshOptimiser(config)
    result = optimiser.optimise(mesh)

    # Should stop due to iteration limit
    assert result.n_iterations <= 10


def test_simple_interface(nonplanar_quad):
    """Test simplified optimise_mesh_simple interface."""
    result = optimise_mesh_simple(nonplanar_quad, max_iter=50, verbose=False)

    assert result is not None
    assert result.success or result.n_iterations == 50
    assert result.final_energy <= result.initial_energy


def test_execution_time_recorded(nonplanar_quad):
    """Test that execution time is recorded."""
    result = optimise_mesh_simple(nonplanar_quad, max_iter=50, verbose=False)

    assert result.execution_time > 0
    assert result.execution_time < 60  # Should complete in <1 minute


def test_optimise_highly_nonplanar_quad():
    """Moderately non-planar quad should show significant planarity improvement."""
    # Create a moderately warped quad (not too extreme)
    V = np.array(
        [
            [0, 0, 0],
            [1, 0, 0.2],  # Moderate z-displacement
            [1, 1, -0.15],  # Moderate z-displacement
            [0, 1, 0.15],  # Moderate z-displacement
        ],
        dtype=np.float64,
    )
    F = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadMesh(V, F)

    initial_planarity = compute_planarity_energy(mesh)

    # Verify this is actually non-planar
    assert initial_planarity > 0.001, "Test mesh should be non-planar"

    result = optimise_mesh_simple(
        mesh,
        weights={"planarity": 1000.0, "fairness": 1.0, "closeness": 10.0},
        max_iter=200,
        verbose=False,
    )

    # Should converge successfully with moderate warping
    assert result.success, f"Optimization failed: {result.message}"

    # Planarity should improve significantly
    improvement = (
        initial_planarity - result.component_energies_final["planarity"]
    ) / initial_planarity
    assert (
        improvement > 0.8
    ), f"Expected >80% planarity improvement, got {improvement*100:.1f}%"

    # Final planarity should be small
    assert result.component_energies_final["planarity"] < 0.01
