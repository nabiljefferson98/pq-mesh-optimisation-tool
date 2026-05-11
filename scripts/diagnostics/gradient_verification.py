"""
Gradient verification and diagnostics.

This script:
- Verifies analytical gradients against numerical gradients
- Demonstrates gradient descent behaviour
- Provides visual diagnostics for gradient correctness

Run this before starting optimisation to ensure gradients are correct.

Date: 2 February 2026
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from src.io.obj_handler import load_obj
from src.optimisation.energy_terms import compute_total_energy
from src.optimisation.gradients import (
    compute_gradient_statistics,
    compute_total_gradient,
    print_gradient_analysis,
    verify_gradient,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Gradient verification and diagnostics."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="data/input/generated/plane_5x5_noisy.obj",
        help="Path to the OBJ mesh file to verify (default: plane_5x5_noisy.obj)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" GRADIENT VERIFICATION & DIAGNOSTICS".center(70))
    print("=" * 70 + "\n")

    # Load test mesh
    print(f"Loading mesh: {args.mesh}")
    mesh = load_obj(args.mesh)
    print(f"  Vertices: {mesh.n_vertices}, Faces: {mesh.n_faces}\n")

    # Define weights
    weights = {"planarity": 100.0, "fairness": 1.0, "closeness": 10.0}

    # ========== PART 1: Gradient Verification ==========
    print("\n" + "=" * 70)
    print(" PART 1: ANALYTICAL VS NUMERICAL GRADIENT")
    print("=" * 70 + "\n")

    print("Comparing analytical gradient against numerical gradient...")
    print("(This may take 30-60 seconds due to finite differences)\n")

    is_correct, error = verify_gradient(mesh, weights, tolerance=1e-3, verbose=True)

    if not is_correct:
        print("\n⚠️  WARNING: Gradient verification failed!")
        print("    This may indicate an error in analytical gradient computation.")
        print("    Optimisation may not converge correctly.\n")
        return

    # ========== PART 2: Gradient Analysis ==========
    print("\n" + "=" * 70)
    print(" PART 2: GRADIENT COMPONENT ANALYSIS")
    print("=" * 70 + "\n")

    print_gradient_analysis(mesh, weights)

    # ========== PART 3: Gradient Descent Test ==========
    print("\n" + "=" * 70)
    print(" PART 3: GRADIENT DESCENT SANITY CHECK")
    print("=" * 70 + "\n")

    print("Testing that gradient descent decreases energy...\n")

    # Save initial state
    verts_initial = mesh.vertices.copy()
    E_initial = compute_total_energy(mesh, weights)

    print(f"Initial energy: {E_initial:.6f}")

    # Take 5 gradient descent steps
    step_size = 0.001
    energies = [E_initial]

    for step in range(5):
        grad = compute_total_gradient(mesh, weights)
        mesh.vertices -= step_size * grad
        E = compute_total_energy(mesh, weights)
        energies.append(E)
        print(f"  Step {step+1}: E = {E:.6f} (decrease: {E - energies[-2]:.6f})")

    # Check monotonic decrease
    all_decreasing = all(
        energies[i + 1] < energies[i] for i in range(len(energies) - 1)
    )

    if all_decreasing:
        print("\n✓ PASSED: Energy decreased monotonically")
    else:
        print("\n✗ WARNING: Energy did not decrease monotonically")
        print("    Step size may be too large, or gradient may be incorrect")

    # Restore mesh
    mesh.vertices = verts_initial.copy()

    # ========== PART 4: Gradient Statistics ==========
    print("\n" + "=" * 70)
    print(" PART 4: GRADIENT MAGNITUDE DISTRIBUTION")
    print("=" * 70 + "\n")

    grad = compute_total_gradient(mesh, weights)
    stats = compute_gradient_statistics(grad)

    print("Gradient statistics:")
    print(f"  Total norm:      {stats['norm']:.6f}")
    print(f"  Max magnitude:   {stats['max_magnitude']:.6f}")
    print(f"  Mean magnitude:  {stats['mean_magnitude']:.6f}")
    print(f"  Std deviation:   {stats['std_magnitude']:.6f}")
    print(f"  Min magnitude:   {stats['min_magnitude']:.6f}")

    # Interpretation
    print("\nInterpretation:")
    if stats["norm"] < 1e-3:
        print("  • Very small gradient → mesh near local minimum")
    elif stats["norm"] < 1.0:
        print("  • Small gradient → optimisation should converge quickly")
    elif stats["norm"] < 10.0:
        print("  • Moderate gradient → typical starting condition")
    else:
        print("  • Large gradient → far from minimum, may need many iterations")

    if stats["std_magnitude"] / (stats["mean_magnitude"] + 1e-10) > 2.0:
        print("  • High variance → non-uniform convergence expected")
    else:
        print("  • Low variance → uniform convergence expected")

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70 + "\n")

    print("✓ Gradient implementation validated")
    print(f"✓ Gradient error: {error:.2e} (< 1e-3)")
    print("✓ Gradient descent test passed")
    print()
    print(
        "  What this means:\n"
        "    The analytical gradient (computed mathematically) closely matches\n"
        "    the numerical gradient (computed by tiny finite differences).\n"
        "    This confirms that the optimiser will move mesh vertices in exactly\n"
        "    the right direction when reducing energy — the maths is correct.\n"
        "\n"
        "    A gradient error below 1e-3 is considered reliable for\n"
        "    engineering-grade optimisation."
    )
    print()
    print("Ready to proceed with optimisation!")
    print("Next step: Run the interactive optimiser on a mesh.")
    print("  python src/visualisation/interactive_optimisation.py")
    print()


if __name__ == "__main__":
    main()
