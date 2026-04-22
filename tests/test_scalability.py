"""
Tests for complex and noisy mesh scenarios.
"""

from pathlib import Path

import numpy as np
import pytest

from src.core.mesh import QuadMesh
from src.io.obj_handler import load_obj
from src.optimisation.energy_terms import compute_planarity_energy
from src.optimisation.optimiser import optimise_mesh_simple


def test_large_mesh_10x10():
    """Test optimisation on 10x10 grid (121 vertices)."""
    mesh_path = Path("data/input/generated/plane_10x10_noisy.obj")

    if not mesh_path.exists():
        pytest.skip(f"Test mesh not found: {mesh_path}")

    mesh = load_obj(str(mesh_path))

    assert mesh.n_vertices == 121
    assert mesh.n_faces == 100

    initial_planarity = compute_planarity_energy(mesh)

    result = optimise_mesh_simple(
        mesh,
        weights={"planarity": 10.0, "fairness": 1.0, "closeness": 5.0},
        max_iter=500,
        verbose=False,
    )

    assert result is not None
    assert result.component_energies_final["planarity"] < initial_planarity
    # Should achieve >90% planarity improvement
    improvement = (
        initial_planarity - result.component_energies_final["planarity"]
    ) / initial_planarity
    assert improvement > 0.9


def test_very_noisy_mesh():
    """Test on mesh with 10% noise (extreme case)."""
    mesh_path = Path("data/input/generated/plane_5x5_very_noisy.obj")

    if not mesh_path.exists():
        pytest.skip(f"Test mesh not found: {mesh_path}")

    mesh = load_obj(str(mesh_path))

    initial_planarity = compute_planarity_energy(mesh)

    # Should handle high noise
    result = optimise_mesh_simple(
        mesh,
        weights={"planarity": 20.0, "fairness": 1.0, "closeness": 5.0},
        max_iter=500,
        verbose=False,
    )

    # Should still achieve significant improvement
    assert result.component_energies_final["planarity"] < initial_planarity * 0.1


def test_non_uniform_quad_sizes():
    """Test mesh with varying quad sizes."""
    # Create mesh with non-uniform spacing
    V = np.array(
        [
            [0, 0, 0],
            [1, 0, 0.1],
            [1, 0.5, 0],
            [0, 0.5, -0.1],
            [2, 0, 0.05],
            [2, 0.5, 0],
        ],
        dtype=np.float64,
    )

    F = np.array(
        [
            [0, 1, 2, 3],  # First quad
            [1, 4, 5, 2],  # Second quad (different size)
        ],
        dtype=np.int32,
    )

    mesh = QuadMesh(V, F)

    result = optimise_mesh_simple(
        mesh,
        weights={"planarity": 10.0, "fairness": 1.0, "closeness": 5.0},
        max_iter=200,
        verbose=False,
    )

    # Should handle different sizes gracefully
    assert result is not None
    assert result.success or result.energy_reduction() > 0.5


def test_boundary_preservation():
    """Test that optimisation respects mesh boundaries."""
    mesh_path = Path("data/input/generated/plane_5x5_noisy.obj")

    if not mesh_path.exists():
        pytest.skip(f"Test mesh not found: {mesh_path}")

    mesh = load_obj(str(mesh_path))

    # Get boundary vertices (corners in this case)
    boundary_indices = [0, 5, 30, 35]  # Corners of 5x5 grid
    boundary_positions_initial = mesh.vertices[boundary_indices].copy()

    optimise_mesh_simple(
        mesh,
        weights={
            "planarity": 10.0,
            "fairness": 1.0,
            "closeness": 50.0,
        },  # High closeness
        max_iter=100,
        verbose=False,
    )

    boundary_positions_final = mesh.vertices[boundary_indices]

    # Boundary should not move much with high closeness weight
    displacement = np.linalg.norm(
        boundary_positions_final - boundary_positions_initial, axis=1
    ).mean()
    assert displacement < 0.5  # Less than 50% of characteristic length


def test_convergence_on_perfect_mesh():
    """Perfect mesh should converge immediately."""
    mesh_path = Path("data/input/generated/plane_5x5_perfect.obj")

    if not mesh_path.exists():
        pytest.skip(f"Test mesh not found: {mesh_path}")

    mesh = load_obj(str(mesh_path))

    result = optimise_mesh_simple(
        mesh,
        weights={"planarity": 10.0, "fairness": 1.0, "closeness": 5.0},
        max_iter=100,
        verbose=False,
    )

    # Should recognize it's already optimal
    assert result.n_iterations < 5  # Very few iterations needed
    assert result.component_energies_final["planarity"] < 1e-6


def test_scalability_20x20():
    """Test that 20x20 mesh (441 vertices) completes in reasonable time."""
    mesh_path = Path("data/input/generated/plane_20x20_noisy.obj")

    if not mesh_path.exists():
        pytest.skip(f"Test mesh not found: {mesh_path}")

    mesh = load_obj(str(mesh_path))

    result = optimise_mesh_simple(
        mesh,
        weights={"planarity": 10.0, "fairness": 1.0, "closeness": 5.0},
        max_iter=100,  # Limited iterations for speed
        verbose=False,
    )

    # Should complete in reasonable time (CI runners are slower than local machines)
    assert result.execution_time < 30.0  # Less than 30 seconds

    # Should still make progress
    assert result.energy_reduction() > 0.5  # At least 50% reduction
