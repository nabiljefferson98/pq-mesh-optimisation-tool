"""
plot_convergence_and_scaling.py
--------------------------------
Generates convergence and scalability plots for EXP-01 and EXP-03 of the
dissertation (Figures 4.1, 4.2).

What this script does (in plain English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads data/output/benchmarks/performance_data.json (written by
benchmark_optimisation.py) and produces two side-by-side plots:

  Left panel (Figure 4.1):
    Iterations to convergence vs. number of vertices (semi-log x-axis).
    Shows that the optimiser always converges in a small number of steps
    regardless of mesh size.

  Right panel (Figure 4.2):
    Wall-clock time vs. number of vertices on a log-log scale, with a
    fitted power-law line O(n^exponent). The exponent is printed on the
    plot and on the terminal.

Covers: EXP-01 (scalability timing) and EXP-03 (convergence behaviour).

Output is saved to data/output/benchmarks/convergence_analysis.png.

Usage
~~~~~
    python scripts/analysis/plot_convergence_and_scaling.py
    python scripts/analysis/plot_convergence_and_scaling.py \\
        --data data/output/benchmarks/performance_data_exp01.json \\
        --output data/output/benchmarks/EXP-01_scalability_time.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running from the project root without installing scripts/ as a package.
sys.path.insert(0, str(Path(__file__).parent))
from plot_style_config import COLOURS, apply_dissertation_style  # noqa: E402

PROJECT_ROOT = Path(__file__).parents[2]


def plot_convergence_comparison(data_path: Path, output_path: Path):
    """
    Load benchmark data and produce Figures 4.1 and 4.2.

    Creates a side-by-side figure:
      - Left: iterations to convergence vs. vertex count (log x-axis)
      - Right: execution time vs. vertex count (log-log), with power-law
               regression line and its exponent printed on the plot.

    Parameters
    ----------
    data_path : Path
        Path to performance_data.json produced by benchmark_optimisation.py.
    output_path : Path
        Where to save the output PNG.
    """
    if not data_path.exists():
        print(f"\u26a0\ufe0f  Data file not found: {data_path}")
        print("    Run benchmark_optimisation.py first.")
        return

    with open(data_path) as f:
        data = json.load(f)

    if not data:
        print("\u26a0\ufe0f  No data found in benchmark file.")
        return

    # Sort by mesh size (ascending) so lines are drawn left-to-right
    data_sorted = sorted(data, key=lambda x: x["n_vertices"])

    vertices = np.array([d["n_vertices"] for d in data_sorted], dtype=float)
    iterations = np.array([d["iterations"] for d in data_sorted], dtype=float)
    # Use time_mean_s if available (new format), fall back to execution_time
    times = np.array(
        [d.get("time_mean_s", d.get("execution_time", 0.0)) for d in data_sorted],
        dtype=float,
    )
    time_stds = np.array(
        [d.get("time_std_s", 0.0) for d in data_sorted],
        dtype=float,
    )

    apply_dissertation_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- Left plot: iterations vs vertices
    ax1.plot(
        vertices,
        iterations,
        "o-",
        color=COLOURS["planarity"],
        label="L-BFGS-B iterations",
    )
    ax1.set_xlabel("Number of Vertices")
    ax1.set_ylabel("Iterations to Convergence")
    ax1.set_title("Figure 4.1: Convergence Scaling")
    ax1.set_xscale("log")
    ax1.legend()

    # -- Right plot: time vs vertices (log-log), with error bars for SD
    # Only use points where both time and vertex count are positive
    valid = (vertices > 0) & (times > 0)
    v_valid = vertices[valid]
    t_valid = times[valid]
    sd_valid = time_stds[valid]

    ax2.errorbar(
        v_valid,
        t_valid,
        yerr=sd_valid,
        fmt="o-",
        color=COLOURS["closeness"],
        capsize=4,
        label="Mean time \u00b1 1 SD",
    )
    ax2.set_xlabel("Number of Vertices")
    ax2.set_ylabel("Execution Time (seconds)")
    ax2.set_title("Figure 4.2: Computational Complexity (log-log)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Power-law fit: log(t) = exponent * log(v) + intercept
    log_v = np.log10(v_valid)
    log_t = np.log10(t_valid)
    coeffs = np.polyfit(log_v, log_t, 1)
    exponent, intercept_log = coeffs

    fit_v = np.logspace(np.log10(v_valid.min()), np.log10(v_valid.max()), 200)
    fit_t = 10**intercept_log * fit_v**exponent
    ax2.plot(
        fit_v,
        fit_t,
        "--",
        color=COLOURS["fit_line"],
        alpha=0.8,
        label=f"Fitted O(n^{exponent:.2f})",
    )
    ax2.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\u2713 Saved: {output_path}")
    plt.close()

    checkmark = "\u2713 Yes" if exponent < 2 else "\u2717 No"
    print(f"\n  Empirical complexity: T(n) \u2248 O(n^{exponent:.2f})")
    print(f"  Sub-quadratic: {checkmark}")


def main():
    """
    Parse command-line arguments and generate the convergence plot.
    """
    parser = argparse.ArgumentParser(
        description="Generate Figures 4.1 and 4.2: convergence and scalability (EXP-01, EXP-03)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to performance_data.json (default: data/output/benchmarks/performance_data.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output PNG path (default: data/output/benchmarks/convergence_analysis.png).",
    )
    args = parser.parse_args()

    data_path = (
        Path(args.data)
        if args.data
        else PROJECT_ROOT / "data/output/benchmarks/performance_data.json"
    )
    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data/output/benchmarks/convergence_analysis.png"
    )

    plot_convergence_comparison(data_path, output_path)


if __name__ == "__main__":
    main()
