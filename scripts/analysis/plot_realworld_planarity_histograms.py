"""
plot_realworld_planarity_histograms.py
---------------------------------------
Generates Figure 4.8: per-face planarity deviation histograms for all four
EXP-05 real-world reference dataset meshes (Spot, Blub, Oloid-256, Bob).

What this script does (in plain English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For each mesh in the EXP-05 benchmark run, this script draws a histogram
showing how many faces have a given planarity deviation value (how far each
quad face is from being perfectly flat). Two overlapping histograms are drawn
per mesh:
  - Red/coral bars:  planarity deviations BEFORE optimisation
  - Green bars:      planarity deviations AFTER optimisation

A good result looks like the green bars clustering tightly near zero while
the red bars are spread across a wider range.

The output is a 2x2 grid of subplots (one per mesh) saved to:
    data/output/figures/EXP-05_planarity_histogram.png

For the oloid mesh specifically, an extra second figure (Figure 4.9) is
produced showing the per-face deviation as a bar chart indexed by face
number so you can see which faces are planar and which are not.

Covers: EXP-05 (real-world mesh evaluation).

Usage
~~~~~
    python scripts/analysis/plot_realworld_planarity_histograms.py

    # Point at a non-default JSON file:
    python scripts/analysis/plot_realworld_planarity_histograms.py \\
        --data data/output/benchmarks/exp05_results.json \\
        --output data/output/figures/EXP-05_planarity_histograms.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from plot_style_config import COLOURS, apply_dissertation_style  # noqa: E402

PROJECT_ROOT = Path(__file__).parents[2]

EXP05_MESHES = [
    "spot_quadrangulated",
    "blub_quadrangulated",
    "oloid256_quad",
    "bob_quad",
]

MESH_LABELS = {
    "spot_quadrangulated": "Spot (quad)",
    "blub_quadrangulated": "Blub (quad)",
    "oloid256_quad": "Oloid-256 (developable)",
    "bob_quad": "Bob (large organic)",
}

_RUN_CMD = (
    "  python scripts/benchmarking/benchmark_optimisation.py\n"
    "      --mesh data/input/reference_datasets/spot/spot_quadrangulated.obj\n"
    "      --mesh data/input/reference_datasets/blub/blub_quadrangulated.obj\n"
    "      --mesh data/input/reference_datasets/oloid/oloid256_quad.obj\n"
    "      --mesh data/input/reference_datasets/bob/bob_quad.obj"
)


def plot_exp05_histograms(data_path: Path, output_path: Path):
    """
    Draw a 2x2 grid of before/after planarity deviation histograms (Figure 4.8).

    Parameters
    ----------
    data_path : Path
        Path to the EXP-05 benchmark JSON.
    output_path : Path
        Where to save the output PNG.
    """
    if not data_path.exists():
        print(f"\u26a0\ufe0f  EXP-05 benchmark file not found: {data_path}")
        print("    Run benchmark_optimisation.py with the EXP-05 mesh list first:")
        print(_RUN_CMD)
        return

    with open(data_path) as f:
        all_results = json.load(f)

    results_by_name = {r["mesh_name"]: r for r in all_results}

    apply_dissertation_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Figure 4.8: Per-Face Planarity Deviation \u2014 Before vs. After Optimisation",
        fontsize=14,
        fontweight="bold",
    )

    for ax, mesh_name in zip(axes.flat, EXP05_MESHES):
        label = MESH_LABELS.get(mesh_name, mesh_name)

        if mesh_name not in results_by_name:
            ax.set_title(f"{label}\n(not found in data)")
            ax.text(
                0.5,
                0.5,
                "Data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        r = results_by_name[mesh_name]
        pb = r.get("planarity_before", {})
        pa = r.get("planarity_after", {})

        raw_before = r.get("planarity_raw_before", None)
        raw_after = r.get("planarity_raw_after", None)

        if raw_before is not None and raw_after is not None:
            before_arr = np.array(raw_before)
            after_arr = np.array(raw_after)
            data_note = ""
        else:
            rng = np.random.default_rng(seed=42)
            n_faces = r["n_faces"]
            before_arr = np.abs(
                rng.normal(pb.get("mean", 0), pb.get("std", 0), n_faces)
            )
            after_arr = np.abs(rng.normal(pa.get("mean", 0), pa.get("std", 0), n_faces))
            data_note = " (approx.)"

        combined_max = max(before_arr.max(), after_arr.max())
        bins = np.linspace(0, combined_max, 50)

        ax.hist(
            before_arr,
            bins=bins,
            color=COLOURS["before"],
            alpha=0.6,
            label=f"Before{data_note}",
            edgecolor="none",
        )
        ax.hist(
            after_arr,
            bins=bins,
            color=COLOURS["after"],
            alpha=0.7,
            label=f"After{data_note}",
            edgecolor="none",
        )
        ax.axvline(
            pa.get("mean", 0),
            color=COLOURS["after"],
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
            label=f"After mean = {pa.get('mean', 0):.4f}",
        )
        ax.axvline(
            pa.get("p95", 0),
            color=COLOURS["neutral"],
            linestyle=":",
            linewidth=1.2,
            label=f"After 95th pctile = {pa.get('p95', 0):.4f}",
        )
        ax.set_title(label)
        ax.set_xlabel("|d_f| \u2014 per-face planarity deviation")
        ax.set_ylabel("Number of faces")
        ax.legend(fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\u2713 Figure 4.8 saved: {output_path}")
    plt.close()


def plot_oloid_spatial_heatmap(data_path: Path, output_path: Path):
    """
    Draw Figure 4.9: per-face planarity deviation indexed by face number
    for the oloid mesh, illustrating ruling vs. circular-edge regions.

    Parameters
    ----------
    data_path : Path
        Path to the EXP-05 benchmark JSON containing oloid256 results.
    output_path : Path
        Where to save the output PNG.
    """
    if not data_path.exists():
        print(f"\u26a0\ufe0f  Data file not found: {data_path}. Skipping Figure 4.9.")
        return

    with open(data_path) as f:
        all_results = json.load(f)

    oloid_result = next(
        (r for r in all_results if r["mesh_name"] == "oloid256_quad"), None
    )
    if oloid_result is None:
        print("\u26a0\ufe0f  oloid256_quad not found in data. Skipping Figure 4.9.")
        return

    raw_after = oloid_result.get("planarity_raw_after", None)
    if raw_after is None:
        print(
            "\u26a0\ufe0f  planarity_raw_after not available for oloid256_quad. "
            "Re-run benchmark_optimisation.py on the EXP-05 meshes to populate "
            "this field (now stored automatically)."
        )
        return

    pf = np.array(raw_after)
    n_faces = len(pf)

    apply_dissertation_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(
        np.arange(n_faces),
        pf,
        color=COLOURS["planarity"],
        width=1.0,
        edgecolor="none",
        alpha=0.8,
    )
    ax.set_xlabel("Face index (0 to n_faces-1)")
    ax.set_ylabel("|d_f| \u2014 planarity deviation after optimisation")
    ax.set_title(
        "Figure 4.9: Oloid-256 \u2014 Spatial Distribution of Planarity Deviation\n"
        "(Ruling-region faces expected near zero; circular-edge faces show residual error)"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\u2713 Figure 4.9 saved: {output_path}")
    plt.close()


def main():
    """
    Parse arguments and generate EXP-05 histogram plots (Figures 4.8 and 4.9).
    """
    parser = argparse.ArgumentParser(
        description="Generate Figures 4.8 and 4.9: EXP-05 real-world planarity histograms."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to EXP-05 benchmark JSON (default: data/output/benchmarks/performance_data.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output path for Figure 4.8 PNG. Figure 4.9 is saved alongside it "
            "with the suffix _oloid_spatial_heatmap.png. "
            "(default: data/output/benchmarks/EXP-05_planarity_histogram.png)"
        ),
    )
    args = parser.parse_args()

    data_path = (
        Path(args.data)
        if args.data
        else PROJECT_ROOT / "data/output/benchmarks/performance_data.json"
    )

    if args.output:
        fig48_path = Path(args.output)
    else:
        fig48_path = (
            PROJECT_ROOT / "data/output/benchmarks/EXP-05_planarity_histogram.png"
        )

    fig49_path = fig48_path.parent / "EXP-05_oloid_spatial_heatmap.png"

    plot_exp05_histograms(data_path, fig48_path)
    plot_oloid_spatial_heatmap(data_path, fig49_path)


if __name__ == "__main__":
    main()
