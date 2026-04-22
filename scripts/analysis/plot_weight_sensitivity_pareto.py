"""
plot_weight_sensitivity_pareto.py
----------------------------------
Visualises weight sensitivity analysis results for EXP-04.

What this script does (in plain English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads the JSON written by run_weight_sensitivity_sweep.py and produces:
  1. A Pareto frontier scatter plot: planarity improvement vs. fairness
     improvement, coloured by vertex displacement. Shows the trade-off
     between making panels flat vs. keeping the shape smooth.
  2. Weight heatmaps: for each closeness weight value, a 2D heatmap
     showing how planarity weight vs. fairness weight affects total
     energy reduction and planarity improvement.

Outputs are saved to data/output/weight_sensitivity/plots/.

Covers: EXP-04 (weight sensitivity and Pareto optimality).

Usage
~~~~~
    python scripts/analysis/plot_weight_sensitivity_pareto.py

    # Run the sweep first if the data file does not exist:
    python scripts/analysis/run_weight_sensitivity_sweep.py \\
        --mesh data/input/generated/plane_10x10_noisy.obj
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def plot_pareto_frontier_2d(results: list[dict], output_path: Path):
    """Plot 2D Pareto frontier: Planarity vs Fairness improvement."""
    successful = [r for r in results if r.get("success", False)]

    planarity_improvements = [r["planarity_improvement"] for r in successful]
    fairness_improvements = [r["fairness_improvement"] for r in successful]

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        planarity_improvements,
        fairness_improvements,
        c=[r["vertex_displacement"] for r in successful],
        cmap="viridis",
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Planarity Improvement (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Fairness Improvement (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Figure 4.6: Multi-Objective Trade-off: Planarity vs Fairness (EXP-04)",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Average Vertex Displacement", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\u2713 Saved Pareto frontier plot: {output_path}")
    plt.close()


def plot_weight_heatmaps(results: list[dict], output_dir: Path):
    """Generate heatmaps showing effect of weight pairs on planarity and energy."""
    successful = [r for r in results if r.get("success", False)]

    planarity_weights = sorted(set(r["weights"]["planarity"] for r in successful))
    fairness_weights = sorted(set(r["weights"]["fairness"] for r in successful))
    closeness_weights = sorted(set(r["weights"]["closeness"] for r in successful))

    for wc in closeness_weights:
        filtered = [r for r in successful if r["weights"]["closeness"] == wc]

        if not filtered:
            continue

        grid_energy = np.zeros((len(fairness_weights), len(planarity_weights)))
        grid_planarity = np.zeros_like(grid_energy)

        for r in filtered:
            wp = r["weights"]["planarity"]
            wf = r["weights"]["fairness"]
            i = fairness_weights.index(wf)
            j = planarity_weights.index(wp)
            grid_energy[i, j] = r["energy_reduction"]
            grid_planarity[i, j] = r["planarity_improvement"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        im1 = ax1.imshow(
            grid_energy,
            cmap="RdYlGn",
            aspect="auto",
            extent=[
                min(planarity_weights),
                max(planarity_weights),
                min(fairness_weights),
                max(fairness_weights),
            ],
            origin="lower",
        )
        ax1.set_xlabel("Planarity Weight", fontsize=12)
        ax1.set_ylabel("Fairness Weight", fontsize=12)
        ax1.set_title(
            f"Total Energy Reduction (%) | Closeness Weight = {wc}",
            fontsize=14,
            fontweight="bold",
        )
        plt.colorbar(im1, ax=ax1, label="Energy Reduction (%)")

        im2 = ax2.imshow(
            grid_planarity,
            cmap="viridis",
            aspect="auto",
            extent=[
                min(planarity_weights),
                max(planarity_weights),
                min(fairness_weights),
                max(fairness_weights),
            ],
            origin="lower",
        )
        ax2.set_xlabel("Planarity Weight", fontsize=12)
        ax2.set_ylabel("Fairness Weight", fontsize=12)
        ax2.set_title(
            f"Planarity Improvement (%) | Closeness Weight = {wc}",
            fontsize=14,
            fontweight="bold",
        )
        plt.colorbar(im2, ax=ax2, label="Planarity Improvement (%)")

        plt.tight_layout()
        output_path = output_dir / f"heatmap_closeness_{wc:.1f}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\u2713 Saved heatmap: {output_path}")
        plt.close()


def main():
    PROJECT_ROOT = Path(__file__).parents[2]
    data_path = (
        PROJECT_ROOT / "data/output/weight_sensitivity/weight_sensitivity_data.json"
    )
    output_dir = PROJECT_ROOT / "data/output/weight_sensitivity/plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print("Run: python scripts/analysis/run_weight_sensitivity_sweep.py")
        sys.exit(1)

    print("Loading sensitivity analysis data...")
    with open(data_path) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} experiments\n")

    print("Generating visualisations...")
    plot_pareto_frontier_2d(results, output_dir / "pareto_frontier_2d.png")
    plot_weight_heatmaps(results, output_dir)

    print(f"\n\u2705 All visualisations saved to {output_dir}/")


if __name__ == "__main__":
    main()
