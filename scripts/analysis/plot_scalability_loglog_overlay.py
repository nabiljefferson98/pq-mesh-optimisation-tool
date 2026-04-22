"""
plot_scalability_loglog_overlay.py
-----------------------------------
Generates Figure 4.3: a single log-log scalability plot overlaying
EXP-01 Series A (synthetic planar grids) and EXP-01 Series B (oloid
resolution series) to verify that the empirical complexity exponent is
consistent across both dataset types.

What this script does (in plain English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You run benchmark_optimisation.py twice:
  - Once on the synthetic grids   -> save to performance_data_seriesA.json
  - Once on the oloid meshes      -> save to performance_data_seriesB.json

Then run this script, pointing it at both JSON files. It draws them both
on the same log-log chart with different marker styles so you can visually
check that the oloid points lie close to the same fitted power-law line
as the synthetic grid points. If they do, the complexity exponent is robust
and not an artefact of the structured grid topology.

Covers: EXP-01 (Series A vs Series B scalability confirmation).

Usage
~~~~~
    python scripts/analysis/plot_scalability_loglog_overlay.py \\
        --series-a data/output/benchmarks/performance_data_seriesA.json \\
        --series-b data/output/benchmarks/performance_data_seriesB.json \\
        --output   data/output/benchmarks/EXP-01_scalability_loglog.png

    # If you only have one file containing all meshes combined:
    python scripts/analysis/plot_scalability_loglog_overlay.py \\
        --series-a data/output/benchmarks/performance_data.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running from the project root without installing scripts/ as a package.
sys.path.insert(0, str(Path(__file__).parent))
from plot_style_config import COLOURS, MARKERS, apply_dissertation_style  # noqa: E402

PROJECT_ROOT = Path(__file__).parents[2]


def load_series(json_path: Path) -> list:
    """
    Load benchmark results from a JSON file.

    Parameters
    ----------
    json_path : Path
        Path to a performance_data JSON file.

    Returns
    -------
    list
        List of result dicts. Returns an empty list if the file does
        not exist or is empty.
    """
    if not json_path.exists():
        print(f"\u26a0\ufe0f  File not found: {json_path}")
        return []
    with open(json_path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def plot_overlay(series_a: list, series_b: list, output_path: Path):
    """
    Produce the log-log overlay plot (Figure 4.3).

    Draws Series A (synthetic grids) and Series B (oloid resolution
    series) on the same log-log axes. Fits a single power-law regression
    line to Series A only (the larger dataset) and overlays it on both
    series so the reader can see how closely Series B follows.

    Parameters
    ----------
    series_a : list
        Benchmark results for the synthetic grid meshes (EXP-01 Series A).
    series_b : list
        Benchmark results for the oloid resolution series (EXP-01 Series B).
        May be empty if only one file was provided.
    output_path : Path
        Where to save the output PNG.
    """
    if not series_a and not series_b:
        print("\u26a0\ufe0f  No data to plot. Provide at least --series-a.")
        return

    apply_dissertation_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    def _extract(series):
        """Extract (vertices, times) arrays from a results list."""
        v = np.array([d["n_vertices"] for d in series], dtype=float)
        t = np.array(
            [d.get("time_mean_s", d.get("execution_time", 0.0)) for d in series],
            dtype=float,
        )
        return v, t

    # -- Plot Series A
    if series_a:
        va, ta = _extract(series_a)
        order = np.argsort(va)
        ax.loglog(
            va[order], ta[order],
            marker=MARKERS.get("spot", "o"),
            color=COLOURS["planarity"],
            linestyle="-",
            label="Series A \u2014 synthetic grids",
        )

        # Fit power law to Series A
        valid = (va > 0) & (ta > 0)
        if valid.sum() >= 2:
            lv = np.log10(va[valid])
            lt = np.log10(ta[valid])
            coeffs = np.polyfit(lv, lt, 1)
            exp_a, c_a = coeffs

            fit_v = np.logspace(
                np.log10(min(va[valid].min(),
                             series_b[0]["n_vertices"] if series_b else va.min())),
                np.log10(max(va[valid].max(),
                             series_b[-1]["n_vertices"] if series_b else va.max())),
                200,
            )
            fit_t = 10 ** c_a * fit_v ** exp_a
            ax.loglog(
                fit_v, fit_t,
                "--",
                color=COLOURS["fit_line"],
                alpha=0.8,
                label=f"Power-law fit (Series A): O(n^{exp_a:.2f})",
            )
            print(f"  Series A complexity: O(n^{exp_a:.2f})")

    # -- Plot Series B (oloid)
    if series_b:
        vb, tb = _extract(series_b)
        order = np.argsort(vb)
        ax.loglog(
            vb[order], tb[order],
            marker=MARKERS.get("oloid", "^"),
            color=COLOURS["oloid"],
            linestyle="--",
            label="Series B \u2014 oloid (developable)",
        )

    ax.set_xlabel("Number of Vertices")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title(
        "Figure 4.3: Scalability \u2014 Synthetic Grids vs. Oloid (log-log)"
    )
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\u2713 Figure 4.3 saved: {output_path}")
    plt.close()


def main():
    """
    Parse arguments and produce Figure 4.3.
    """
    parser = argparse.ArgumentParser(
        description="Generate Figure 4.3: EXP-01 log-log scalability overlay (Series A + B)."
    )
    parser.add_argument(
        "--series-a",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to Series A benchmark JSON (synthetic grids).",
    )
    parser.add_argument(
        "--series-b",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to Series B benchmark JSON (oloid meshes). Optional.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output PNG path "
            "(default: data/output/benchmarks/EXP-01_scalability_loglog.png)."
        ),
    )
    args = parser.parse_args()

    series_a = load_series(Path(args.series_a))
    series_b = load_series(Path(args.series_b)) if args.series_b else []

    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data/output/benchmarks/EXP-01_scalability_loglog.png"
    )

    plot_overlay(series_a, series_b, output_path)


if __name__ == "__main__":
    main()
