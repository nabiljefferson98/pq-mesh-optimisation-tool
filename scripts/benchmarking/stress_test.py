"""
Stress test: find the upper limit of optimisable mesh size.

Generates planar quad grids of increasing resolution, runs the optimiser
on each, and reports:
  - Vertices / faces
  - Optimisation time (seconds)
  - Peak RAM usage (MB)
  - Convergence status
  - Energy reduction (%)
  - Final mean/max planarity deviation

Usage:
    python scripts/stress_test.py
    python scripts/stress_test.py --max-n 200   # override max grid size
    python scripts/stress_test.py --max-iter 200 # override max iterations per run
"""

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from src.core.mesh import QuadMesh
from src.optimisation.energy_terms import compute_planarity_per_face
from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def generate_noisy_grid(n: int, noise: float = 0.05, seed: int = 42) -> QuadMesh:
    """Generate an n×n planar quad grid with Gaussian z-noise."""
    np.random.seed(seed)
    x = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(x, x)
    Z = np.random.normal(0, noise, X.shape)
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    faces = []
    for j in range(n):
        for i in range(n):
            v0 = j * (n + 1) + i
            faces.append([v0, v0 + 1, v0 + (n + 1) + 1, v0 + (n + 1)])
    F = np.array(faces, dtype=np.int32)
    return QuadMesh(V, F)


def format_row(n, n_verts, n_faces, elapsed, peak_mb, status, reduction, max_p, mean_p):
    status_icon = "✓" if status else "✗"
    return (
        f"  {n:>5d}x{n:<5d}  {n_verts:>7,}  {n_faces:>7,}  "
        f"{elapsed:>8.2f}s  {peak_mb:>8.1f}MB  "
        f"{status_icon}  {reduction:>7.1f}%  "
        f"{max_p:>10.3e}  {mean_p:>10.3e}"
    )


# ─────────────────────────────────────────────
# Main stress test
# ─────────────────────────────────────────────


def run_stress_test(grid_sizes: list[int], max_iter: int, timeout_s: float):
    weights = {"planarity": 10.0, "fairness": 1.0, "closeness": 5.0}

    header = (
        f"  {'Grid':>11}  {'Verts':>7}  {'Faces':>7}  "
        f"{'Time':>9}  {'PeakRAM':>9}  "
        f"{'OK'}  {'Reduce':>7}  "
        f"{'MaxPlan':>10}  {'MeanPlan':>10}"
    )
    divider = "─" * len(header)

    print("\n" + divider)
    print("  PQ MESH OPTIMISER  ─  STRESS TEST")
    print(
        "  What this test does:\n"
        "    Generates quad grids of increasing size, runs the optimiser on each,\n"
        "    and records how long it takes and how much memory it uses.\n"
        "    This tells you the practical upper limit for real-time or batch use.\n"
        "\n"
        "    OK (✓) = converged within the iteration limit.\n"
        "    Reduce = how much the total energy fell (higher is better).\n"
        "    MaxPlan = largest planarity deviation across all panels (< 0.001 = fabrication-ready)."
    )
    print(
        f"  Weights: planarity={weights['planarity']}, "
        f"fairness={weights['fairness']}, closeness={weights['closeness']}"
    )
    print(f"  Max iterations per run: {max_iter}  |  Timeout: {timeout_s}s")
    print(divider)
    print(header)
    print(divider)

    results = []

    for n in grid_sizes:
        mesh = generate_noisy_grid(n)
        n_verts = mesh.n_vertices
        n_faces = mesh.n_faces

        config = OptimisationConfig(
            weights=weights,
            max_iterations=max_iter,
            verbose=False,
            history_tracking=False,
        )
        optimiser = MeshOptimiser(config)

        tracemalloc.start()
        t0 = time.perf_counter()

        try:
            result = optimiser.optimise(mesh)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / 1024 / 1024
            planarity = compute_planarity_per_face(mesh)
            reduction = result.energy_reduction_percentage()
            timed_out = elapsed > timeout_s

            row = format_row(
                n,
                n_verts,
                n_faces,
                elapsed,
                peak_mb,
                result.success and not timed_out,
                reduction,
                planarity.max(),
                planarity.mean(),
            )
            print(row)

            results.append(
                {
                    "n": n,
                    "n_verts": n_verts,
                    "n_faces": n_faces,
                    "elapsed": elapsed,
                    "peak_mb": peak_mb,
                    "success": result.success,
                    "reduction": reduction,
                    "max_planarity": planarity.max(),
                    "mean_planarity": planarity.mean(),
                    "timed_out": timed_out,
                }
            )

            # Stop if this run timed out — larger meshes will only be slower
            if timed_out:
                print(
                    f"\n  ⚠  Timed out at {n}×{n} ({elapsed:.1f}s > {timeout_s}s limit). Stopping."
                )
                break

        except Exception as e:
            elapsed = time.perf_counter() - t0
            tracemalloc.stop()
            print(f"  {n:>5d}x{n:<5d}  {n_verts:>7,}  {n_faces:>7,}  " f"  ERROR: {e}")
            break

    print(divider)

    # Summary
    if results:
        successful = [r for r in results if r["success"] and not r["timed_out"]]
        if successful:
            largest = successful[-1]
            print(
                f"\n  Largest successfully optimised: "
                f"{largest['n']}×{largest['n']}  "
                f"({largest['n_verts']:,} verts, {largest['n_faces']:,} faces)  "
                f"in {largest['elapsed']:.2f}s using {largest['peak_mb']:.1f}MB RAM"
            )
            if largest["mean_planarity"] < 1e-3:
                print(
                    "  Final planarity:  ✅ Fabrication-ready (panels are flat enough)"
                )
            elif largest["mean_planarity"] < 0.01:
                print(
                    "  Final planarity:  ⚠️  Acceptable — minor gaps expected at joints"
                )
            else:
                print(
                    "  Final planarity:  ❌ Too curved for fabrication — increase iterations or planarity weight"
                )
        print()


def main():
    parser = argparse.ArgumentParser(description="PQ mesh optimiser stress test")
    parser.add_argument(
        "--max-n",
        type=int,
        default=150,
        help="Maximum grid dimension to test (default: 150)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=150,
        help="Max optimiser iterations per mesh (default: 150)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-mesh time limit in seconds (default: 120)",
    )
    args = parser.parse_args()

    # Exponentially spaced grid sizes up to max_n
    sizes = [5, 10, 20, 30, 40, 50, 75, 100]
    sizes = [s for s in sizes if s <= args.max_n]
    if args.max_n not in sizes:
        sizes.append(args.max_n)
    sizes = sorted(set(sizes))

    run_stress_test(sizes, args.max_iter, args.timeout)


if __name__ == "__main__":
    main()
