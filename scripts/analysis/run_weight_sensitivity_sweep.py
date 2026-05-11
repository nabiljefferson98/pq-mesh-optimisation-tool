"""
run_weight_sensitivity_sweep.py
--------------------------------
Runs the full weight sensitivity sweep for EXP-04.

What this script does (in plain English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Systematically explores the effect of weight configurations on
optimisation outcomes by testing all combinations of planarity weight,
fairness weight, and closeness weight. For each combination it runs
the optimiser on a noisy planar mesh and records the result.

The goal is to find the 'Pareto-optimal' configurations: the weight
combinations where you cannot improve one objective (e.g. flatness)
without making another worse (e.g. smoothness or shape change).

Results are saved to:
  data/output/weight_sensitivity/weight_sensitivity_data.json  (raw data)
  data/output/weight_sensitivity/sensitivity_analysis_report.txt (human-readable report)

Covers: EXP-04 (weight sensitivity analysis).

Usage
~~~~~
    python scripts/analysis/run_weight_sensitivity_sweep.py

    # Use a different input mesh:
    python scripts/analysis/run_weight_sensitivity_sweep.py \\
        --mesh data/input/generated/plane_10x10_noisy.obj
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.io.obj_handler import load_obj
from src.optimisation.optimiser import optimise_mesh_simple


def run_weight_sweep(
    mesh_path: Path,
    planarity_weights: list[float],
    fairness_weights: list[float],
    closeness_weights: list[float],
) -> list[dict]:
    """
    Run optimisation with all combinations of weights.

    Args:
        mesh_path: Path to input mesh
        planarity_weights: List of planarity weight values to test
        fairness_weights: List of fairness weight values to test
        closeness_weights: List of closeness weight values to test

    Returns:
        List of result dictionaries
    """
    mesh_original = load_obj(str(mesh_path))

    results = []
    total_combinations = (
        len(planarity_weights) * len(fairness_weights) * len(closeness_weights)
    )

    print(f"\nTesting {total_combinations} weight combinations on {mesh_path.name}")
    print(f"{'='*70}")

    for i, (wp, wf, wc) in enumerate(
        product(planarity_weights, fairness_weights, closeness_weights)
    ):
        mesh = load_obj(str(mesh_path))
        weights = {"planarity": wp, "fairness": wf, "closeness": wc}

        print(
            f"[{i+1}/{total_combinations}] wp={wp:.1f}, wf={wf:.1f}, wc={wc:.1f}...",
            end=" ",
            flush=True,
        )

        try:
            result = optimise_mesh_simple(
                mesh, weights=weights, max_iter=500, verbose=False
            )

            planarity_improvement = (
                (
                    result.component_energies_initial["planarity"]
                    - result.component_energies_final["planarity"]
                )
                / result.component_energies_initial["planarity"]
                * 100
                if result.component_energies_initial["planarity"] > 0
                else 0
            )

            fairness_improvement = (
                (
                    result.component_energies_initial["fairness"]
                    - result.component_energies_final["fairness"]
                )
                / result.component_energies_initial["fairness"]
                * 100
                if result.component_energies_initial["fairness"] > 0
                else 0
            )

            vertex_displacement = np.linalg.norm(
                mesh.vertices - mesh_original.vertices, axis=1
            ).mean()

            results.append(
                {
                    "weights": weights,
                    "success": result.success,
                    "iterations": result.n_iterations,
                    "execution_time": result.execution_time,
                    "initial_energy": float(result.initial_energy),
                    "final_energy": float(result.final_energy),
                    "energy_reduction": float(result.energy_reduction_percentage()),
                    "planarity_initial": float(
                        result.component_energies_initial["planarity"]
                    ),
                    "planarity_final": float(
                        result.component_energies_final["planarity"]
                    ),
                    "planarity_improvement": float(planarity_improvement),
                    "fairness_initial": float(
                        result.component_energies_initial["fairness"]
                    ),
                    "fairness_final": float(
                        result.component_energies_final["fairness"]
                    ),
                    "fairness_improvement": float(fairness_improvement),
                    "closeness_final": float(
                        result.component_energies_final["closeness"]
                    ),
                    "vertex_displacement": float(vertex_displacement),
                }
            )

            print(
                f"\u2713 E\u2193={result.energy_reduction_percentage():.1f}%, "
                f"iter={result.n_iterations}"
            )

        except Exception as e:
            print(f"\u2717 Error: {e}")
            results.append({"weights": weights, "success": False, "error": str(e)})

    return results


def analyze_pareto_frontier(results: list[dict]) -> dict:
    """
    Identify Pareto-optimal weight configurations.

    A configuration is Pareto-optimal if no other configuration
    is better in all objectives simultaneously.
    """
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        return {"pareto_optimal": []}

    pareto_optimal = []

    for r in successful_results:
        is_dominated = False

        for other in successful_results:
            if other == r:
                continue

            planarity_better = (
                other["planarity_improvement"] >= r["planarity_improvement"]
            )
            fairness_better = other["fairness_improvement"] >= r["fairness_improvement"]
            displacement_better = (
                other["vertex_displacement"] <= r["vertex_displacement"]
            )

            if (
                planarity_better
                and fairness_better
                and displacement_better
                and (
                    other["planarity_improvement"] > r["planarity_improvement"]
                    or other["fairness_improvement"] > r["fairness_improvement"]
                    or other["vertex_displacement"] < r["vertex_displacement"]
                )
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_optimal.append(r)

    return {
        "pareto_optimal": pareto_optimal,
        "n_pareto": len(pareto_optimal),
        "n_total": len(successful_results),
    }


def generate_analysis_report(results: list[dict], output_path: Path):
    """Generate human-readable analysis report."""
    successful = [r for r in results if r.get("success", False)]

    if not successful:
        print("\n\u26a0\ufe0f  No successful optimizations to analyze")
        return

    report_lines = [
        "=" * 80,
        "WEIGHT SENSITIVITY ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Total configurations tested: {len(results)}",
        f"Successful optimizations: {len(successful)}",
        f"Failed optimizations: {len(results) - len(successful)}",
        "",
        "=" * 80,
        "BEST CONFIGURATIONS BY OBJECTIVE",
        "=" * 80,
        "",
    ]

    best_planarity = max(successful, key=lambda r: r["planarity_improvement"])
    report_lines.extend(
        [
            "Best Planarity Improvement:",
            f"  Weights: planarity={best_planarity['weights']['planarity']:.1f}, "
            f"fairness={best_planarity['weights']['fairness']:.1f}, "
            f"closeness={best_planarity['weights']['closeness']:.1f}",
            f"  Planarity improvement: {best_planarity['planarity_improvement']:.2f}%",
            f"  Fairness improvement: {best_planarity['fairness_improvement']:.2f}%",
            f"  Vertex displacement: {best_planarity['vertex_displacement']:.6f}",
            "",
        ]
    )

    best_fairness = max(successful, key=lambda r: r["fairness_improvement"])
    report_lines.extend(
        [
            "Best Fairness Improvement:",
            f"  Weights: planarity={best_fairness['weights']['planarity']:.1f}, "
            f"fairness={best_fairness['weights']['fairness']:.1f}, "
            f"closeness={best_fairness['weights']['closeness']:.1f}",
            f"  Planarity improvement: {best_fairness['planarity_improvement']:.2f}%",
            f"  Fairness improvement: {best_fairness['fairness_improvement']:.2f}%",
            f"  Vertex displacement: {best_fairness['vertex_displacement']:.6f}",
            "",
        ]
    )

    min_displacement = min(successful, key=lambda r: r["vertex_displacement"])
    report_lines.extend(
        [
            "Minimal Vertex Displacement:",
            f"  Weights: planarity={min_displacement['weights']['planarity']:.1f}, "
            f"fairness={min_displacement['weights']['fairness']:.1f}, "
            f"closeness={min_displacement['weights']['closeness']:.1f}",
            f"  Planarity improvement: {min_displacement['planarity_improvement']:.2f}%",
            f"  Fairness improvement: {min_displacement['fairness_improvement']:.2f}%",
            f"  Vertex displacement: {min_displacement['vertex_displacement']:.6f}",
            "",
        ]
    )

    pareto_analysis = analyze_pareto_frontier(successful)
    report_lines.extend(
        [
            "=" * 80,
            "PARETO FRONTIER ANALYSIS",
            "=" * 80,
            "",
            f"Pareto-optimal configurations: {pareto_analysis['n_pareto']}",
            f"Percentage: {pareto_analysis['n_pareto']/pareto_analysis['n_total']*100:.1f}%",
            "",
            "Pareto-optimal weight configurations:",
        ]
    )

    for i, config in enumerate(pareto_analysis["pareto_optimal"][:10], 1):
        w = config["weights"]
        report_lines.append(
            f"  {i}. wp={w['planarity']:.1f}, wf={w['fairness']:.1f}, wc={w['closeness']:.1f} | "
            f"Plan={config['planarity_improvement']:.1f}%, "
            f"Fair={config['fairness_improvement']:.1f}%, "
            f"Disp={config['vertex_displacement']:.4f}"
        )

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    output_path.write_text(report_text)
    print(f"\n\u2705 Generated analysis report: {output_path}")
    print("\n" + report_text)


def main():
    PROJECT_ROOT = Path(__file__).parents[2]

    parser = argparse.ArgumentParser(
        description="Run the EXP-04 weight sensitivity sweep."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to input mesh OBJ (default: data/input/generated/plane_10x10_noisy.obj).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to save results JSON (default: data/output/weight_sensitivity/weight_sensitivity_data.json).",
    )
    args = parser.parse_args()

    mesh_path = (
        Path(args.mesh)
        if args.mesh
        else PROJECT_ROOT / "data/input/generated/plane_10x10_noisy.obj"
    )

    output_dir = PROJECT_ROOT / "data/output/weight_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        print("Run: python scripts/mesh_generation/generate_test_meshes.py")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("WEIGHT SENSITIVITY ANALYSIS (EXP-04)")
    print("=" * 70)
    print(
        "  What this script does:\n"
        "    Tests many combinations of the three optimisation weights\n"
        "    (planarity, fairness, closeness) and records how each one\n"
        "    affects the final mesh quality.\n"
        "\n"
        "    The goal is to find the 'Pareto-optimal' configurations:   \n"
        "    weight combinations where you cannot improve one objective  \n"
        "    (e.g. flatness) without making another worse               \n"
        "    (e.g. smoothness or shape change).\n"
        "\n"
        "    Results are saved to data/output/weight_sensitivity/.\n"
    )
    print("=" * 70 + "\n")

    planarity_weights = [1.0, 5.0, 10.0, 50.0, 100.0]
    fairness_weights = [0.1, 1.0, 5.0, 10.0]
    closeness_weights = [1.0, 5.0, 10.0, 20.0]

    results = run_weight_sweep(
        mesh_path, planarity_weights, fairness_weights, closeness_weights
    )

    output_json = (
        Path(args.output)
        if args.output
        else output_dir / "weight_sensitivity_data.json"
    )
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\u2705 Saved raw data: {output_json}")

    output_report = output_dir / "sensitivity_analysis_report.txt"
    generate_analysis_report(results, output_report)

    successful = [r for r in results if r.get("success", False)]
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Configurations tested : {len(results)}")
    print(
        f"  Successful runs       : {len(successful)} ({100*len(successful)/max(len(results), 1):.0f}%)"
    )
    if successful:
        best = max(successful, key=lambda r: r.get("planarity_improvement", 0))
        w = best["weights"]
        print(
            f"\n  Best planarity result : {best['planarity_improvement']:.1f}% improvement\n"
            f"    weights: planarity={w['planarity']:.1f}, "
            f"fairness={w['fairness']:.1f}, closeness={w['closeness']:.1f}"
        )
        if best["planarity_final"] < 1e-3:
            print("    Fabrication status: \u2705 Panels are flat enough to fabricate")
        elif best["planarity_final"] < 0.01:
            print(
                "    Fabrication status: \u26a0\ufe0f  Nearly flat \u2014 minor gaps expected"
            )
        else:
            print(
                "    Fabrication status: \u274c Still curved \u2014 try higher planarity weight"
            )
    print(f"\n  Full report saved to: {output_report}")
    print(f"  Raw data saved to   : {output_json}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
