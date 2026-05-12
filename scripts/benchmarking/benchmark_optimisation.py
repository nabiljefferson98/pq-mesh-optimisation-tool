"""
benchmark_optimisation.py
-------------------------
Runs the PQ mesh optimiser on one or more mesh files and saves the results
to a JSON file for later analysis and plotting.

What this script does (in plain English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. You give it mesh file paths either via the --mesh flag (one or more .obj
   files) or, if you give no flags at all, it falls back to scanning
   data/input/generated/ for any file matching plane_*_noisy.obj.
2. For each mesh it runs the optimiser THREE times (to get a stable average)
   and skips the first run's time if the backend is Numba (because Numba
   needs one "warm-up" run to compile its code before it runs at full speed).
3. It measures wall-clock time and peak RAM for each run using tracemalloc.
4. It collects per-face planarity deviation statistics (mean, median,
   standard deviation, 95th percentile) before and after optimisation.
5. It saves everything to data/output/benchmarks/performance_data.json
   so that analyse_results.py and the plotting scripts can read it later.

Usage examples
~~~~~~~~~~~~~~
# Run on all generated test meshes (default fallback)
python scripts/benchmarking/benchmark_optimisation.py

# Run on specific reference dataset meshes (EXP-01 Series B, EXP-05)
python scripts/benchmarking/benchmark_optimisation.py \\
    --mesh data/input/reference_datasets/oloid/oloid64_quad.obj \\
    --mesh data/input/reference_datasets/oloid/oloid256_quad.obj \\
    --mesh data/input/reference_datasets/oloid/oloid1024_quad.obj \\
    --mesh data/input/reference_datasets/oloid/oloid4096_quad.obj

python scripts/benchmarking/benchmark_optimisation.py \\
    --mesh data/input/reference_datasets/spot/spot_quadrangulated.obj \\
    --mesh data/input/reference_datasets/blub/blub_quadrangulated.obj \\
    --mesh data/input/reference_datasets/oloid/oloid256_quad.obj \\
    --mesh data/input/reference_datasets/bob/bob_quad.obj

# Override number of repetitions (default is 3)
python scripts/benchmarking/benchmark_optimisation.py --reps 5
"""

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.io.obj_handler import load_obj
from src.optimisation.energy_terms import compute_planarity_per_face
from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {"planarity": 10.0, "fairness": 1.0, "closeness": 5.0}
DEFAULT_MAX_ITER = 500
DEFAULT_REPS = 3


# ---------------------------------------------------------------------------
# Core benchmarking function
# ---------------------------------------------------------------------------


def benchmark_mesh(mesh_path: Path, weights: dict, max_iter: int, reps: int) -> dict:
    """
    Run the optimiser on one mesh file and collect all performance metrics.

    Parameters
    ----------
    mesh_path : Path
        Full path to the .obj file to benchmark.
    weights : dict
        Energy term weights passed directly to the optimiser.
    max_iter : int
        Maximum number of L-BFGS-B iterations per run.
    reps : int
        Number of repetitions.

    Returns
    -------
    dict
        All metrics for this mesh, including planarity_raw_before and
        planarity_raw_after (full per-face deviation arrays as lists)
        needed for Figure 4.9.
    """
    times = []
    memory_peaks = []
    last_result = None
    planarity_before_stats = None
    planarity_raw_before = None

    for rep_idx in range(reps):
        mesh = load_obj(str(mesh_path))

        if rep_idx == 0:
            pf_before = compute_planarity_per_face(mesh)
            planarity_before_stats = _deviation_stats(pf_before)
            planarity_raw_before = pf_before.tolist()

        config = OptimisationConfig(
            weights=weights,
            max_iterations=max_iter,
            verbose=False,
            history_tracking=False,
        )
        optimiser = MeshOptimiser(config)

        tracemalloc.start()
        t0 = time.perf_counter()
        result = optimiser.optimise(mesh)
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        memory_peaks.append(peak_bytes / 1024 / 1024)
        last_result = result

    # Exclude first run if it looks like a Numba warm-up
    timing_runs = times
    if reps >= 2:
        other_times = times[1:]
        median_rest = float(np.median(other_times))
        if median_rest > 0 and times[0] > 3 * median_rest:
            timing_runs = other_times

    time_mean = float(np.mean(timing_runs))
    time_std = float(np.std(timing_runs, ddof=1)) if len(timing_runs) > 1 else 0.0

    # Per-face planarity AFTER optimisation (from the last rep)
    pf_after = compute_planarity_per_face(last_result.optimised_mesh)
    planarity_after_stats = _deviation_stats(pf_after)
    planarity_raw_after = pf_after.tolist()

    plan_initial = last_result.component_energies_initial.get("planarity", 0.0)
    plan_final = last_result.component_energies_final.get("planarity", 0.0)
    planarity_improvement = (
        (plan_initial - plan_final) / plan_initial * 100.0 if plan_initial > 0 else 0.0
    )

    return {
        "mesh_name": mesh_path.stem,
        "mesh_path": str(mesh_path),
        "n_vertices": last_result.optimised_mesh.n_vertices,
        "n_faces": last_result.optimised_mesh.n_faces,
        "time_mean_s": time_mean,
        "time_std_s": time_std,
        "time_all_s": times,
        "memory_peak_mb": float(max(memory_peaks)),
        "iterations": last_result.n_iterations,
        "function_evals": last_result.n_function_evaluations,
        "execution_time": time_mean,
        "initial_energy": float(last_result.initial_energy),
        "final_energy": float(last_result.final_energy),
        "energy_reduction": float(last_result.energy_reduction_percentage()),
        "planarity_improvement": planarity_improvement,
        "success": bool(last_result.success),
        "convergence_message": last_result.message,
        "planarity_before": planarity_before_stats,
        "planarity_after": planarity_after_stats,
        # Full per-face arrays — required for Figure 4.9 (oloid spatial heatmap)
        "planarity_raw_before": planarity_raw_before,
        "planarity_raw_after": planarity_raw_after,
    }


def _deviation_stats(pf_array: np.ndarray) -> dict:
    """
    Compute summary statistics for a per-face planarity deviation array.
    """
    return {
        "mean": float(np.mean(pf_array)),
        "median": float(np.median(pf_array)),
        "std": float(np.std(pf_array)),
        "p95": float(np.percentile(pf_array, 95)),
        "max": float(np.max(pf_array)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """
    Parse command-line arguments, run benchmarks, and save results to JSON.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the PQ mesh optimiser on one or more mesh files. "
            "Results are saved to data/output/benchmarks/performance_data.json."
        )
    )
    parser.add_argument(
        "--mesh",
        dest="meshes",
        action="append",
        metavar="PATH",
        help=(
            "Path to an .obj mesh file to benchmark. "
            "Repeat this flag to benchmark multiple files. "
            "If omitted, scans data/input/generated/ for plane_*_noisy.obj files."
        ),
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        metavar="N",
        help=f"Number of repetitions per mesh (default: {DEFAULT_REPS}).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        metavar="N",
        help=f"Maximum L-BFGS-B iterations per run (default: {DEFAULT_MAX_ITER}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to save the results JSON "
            "(default: data/output/benchmarks/performance_data.json)."
        ),
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parents[2]

    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data/output/benchmarks/performance_data.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.meshes:
        test_meshes = [Path(p) for p in args.meshes]
        for p in test_meshes:
            if not p.exists():
                print(f"\u274c  File not found: {p}")
                sys.exit(1)
    else:
        generated_dir = PROJECT_ROOT / "data/input/generated"
        test_meshes = sorted(generated_dir.glob("plane_*_noisy.obj"))
        if not test_meshes:
            print(
                "No test meshes found. Either pass --mesh paths explicitly, "
                "or run: python scripts/mesh_generation/generate_test_meshes.py"
            )
            sys.exit(1)

    print(
        f"\nBenchmarking {len(test_meshes)} mesh(es) "
        f"({args.reps} repetition(s) each, max_iter={args.max_iter})...\n"
    )

    results = []
    for mesh_path in test_meshes:
        print(f"  Processing {mesh_path.name} ...", end=" ", flush=True)
        try:
            metrics = benchmark_mesh(
                mesh_path,
                weights=DEFAULT_WEIGHTS,
                max_iter=args.max_iter,
                reps=args.reps,
            )
            results.append(metrics)
            print(
                f"\u2713  {metrics['iterations']} iter, "
                f"{metrics['time_mean_s']:.2f}s \u00b1 {metrics['time_std_s']:.2f}s, "
                f"{metrics['memory_peak_mb']:.1f} MB"
            )
        except Exception as e:
            print(f"\u2717  Error: {e}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\u2705 Results saved to {output_path}")

    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    down_arrow = "\u2193"
    ok_check = "\u2705"
    x_cross = "\u274c"
    header_cols = (
        f"  {'Mesh':<28} {'Vertices':>8} {'Faces':>7} "
        f"{'Time (s)':>12} {'Mem (MB)':>10} {'Energy ':>10}{down_arrow} {'OK':>4}"
    )
    print(header_cols)
    print("-" * 90)
    for r in results:
        ok = ok_check if r["success"] else x_cross
        pm_symbol = "\u00b1"
        row = (
            f"  {r['mesh_name']:<28} {r['n_vertices']:>8,} {r['n_faces']:>7,} "
            f"  {r['time_mean_s']:>6.2f}{pm_symbol}{r['time_std_s']:>4.2f}s "
            f"  {r['memory_peak_mb']:>8.1f} "
            f"  {r['energy_reduction']:>8.1f}%  {ok}"
        )
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    main()
