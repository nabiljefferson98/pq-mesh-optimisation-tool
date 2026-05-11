"""
summarise_and_export_results.py
--------------------------------
Reads the JSON output produced by benchmark_optimisation.py and
run_weight_sensitivity_sweep.py, then prints human-readable summary tables to the
terminal and writes dissertation-ready output files.

What this script does (in plain English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Reads data/output/benchmarks/performance_data.json and prints:
   - A performance summary table (vertices, faces, time, memory,
     energy reduction, planarity improvement)
   - Per-face planarity deviation statistics (mean/median/std/95th
     percentile before and after optimisation) matching the EXP-05
     results tables in docs/results/
   - An empirical complexity estimate T(n) ~ O(n^exponent) from a
     log-log linear regression of time vs. vertex count

2. Reads data/output/weight_sensitivity/weight_sensitivity_data.json
   and prints the best weight configuration identified by the sweep.

3. Writes LaTeX and CSV tables to data/output/tables/ ready for
   pasting into Chapter 4 of the dissertation.

Covers: EXP-01 (complexity fit), EXP-05 (real-world planarity stats),
        EXP-04 (best weight configuration from sensitivity sweep).

Usage
~~~~~
# Analyse default benchmark output
python scripts/analysis/summarise_and_export_results.py

# Point at a non-default benchmark file (e.g. EXP-05 reference datasets)
python scripts/analysis/summarise_and_export_results.py \\
    --benchmark data/output/benchmarks/performance_data_exp05.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]


# ---------------------------------------------------------------------------
# Benchmark results analysis
# ---------------------------------------------------------------------------


def analyse_benchmark_results(benchmark_file: Path):
    """
    Load the benchmark JSON and print a full performance summary.

    Reads the JSON file written by benchmark_optimisation.py and prints:
      - A performance table (one row per mesh)
      - Per-face planarity deviation statistics before and after
      - An empirical complexity estimate using log-log linear regression

    Parameters
    ----------
    benchmark_file : Path
        Path to the performance_data.json file.

    Returns
    -------
    pd.DataFrame or None
        A tidy DataFrame of benchmark results, or None if the file
        is missing or empty.

    Notes
    -----
    Important: energy_reduction in the JSON is already stored as a
    percentage (0-100). This function does NOT multiply it by 100 again.
    """
    if not benchmark_file.exists():
        print(f"\u26a0\ufe0f  Benchmark file not found: {benchmark_file}")
        print("    Run benchmark_optimisation.py first.")
        return None

    with open(benchmark_file, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print("\u26a0\ufe0f  Benchmark file is empty or has unexpected format.")
        return None

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 80)

    rows = []
    for r in data:
        # Use time_mean_s / time_std_s if present (new format),
        # fall back to execution_time for backwards compatibility.
        t_mean = r.get("time_mean_s", r.get("execution_time", 0.0))
        t_std = r.get("time_std_s", 0.0)
        mem = r.get("memory_peak_mb", 0.0)

        rows.append(
            {
                "Mesh": r["mesh_name"],
                "Vertices": r["n_vertices"],
                "Faces": r["n_faces"],
                "Time (s)": t_mean,
                "Time SD (s)": t_std,
                "Memory (MB)": mem,
                "Iterations": r["iterations"],
                # energy_reduction is already a percentage — do NOT *100
                "Energy Reduction (%)": r["energy_reduction"],
                "Planarity Improv. (%)": r["planarity_improvement"],
                "Success": "\u2713" if r["success"] else "\u2717",
            }
        )

    df = pd.DataFrame(rows)

    print("\n### Table: Performance Benchmarks")
    print(df.to_string(index=False))

    # -- Statistics
    print("\n### Summary Statistics")
    n_ok = df["Success"].value_counts().get("\u2713", 0)
    print(f"  Meshes tested:           {len(df)}")
    print(f"  Successful convergence:  {n_ok}/{len(df)} ({n_ok/len(df)*100:.0f}%)")
    print(f"  Mean time:               {df['Time (s)'].mean():.2f} s")
    print(f"  Mean energy reduction:   {df['Energy Reduction (%)'].mean():.1f}%")
    print(f"  Mean planarity improv.:  {df['Planarity Improv. (%)'].mean():.1f}%")
    print(f"  Mean iterations:         {df['Iterations'].mean():.0f}")

    # -- Per-face planarity deviation statistics
    has_planarity = all(
        "planarity_before" in r and "planarity_after" in r for r in data
    )
    if has_planarity:
        print("\n### Per-Face Planarity Deviation Statistics (|d_f|, unit-normalised)")
        print(
            f"  {'Mesh':<30} "
            f"{'Mean Before':>12} {'Mean After':>11} "
            f"{'Median After':>13} {'Std After':>10} {'95th Pctile After':>18}"
        )
        print("  " + "-" * 96)
        for r in data:
            pb = r["planarity_before"]
            pa = r["planarity_after"]
            print(
                f"  {r['mesh_name']:<30} "
                f"{pb['mean']:>12.4f} {pa['mean']:>11.4f} "
                f"{pa['median']:>13.4f} {pa['std']:>10.4f} {pa['p95']:>18.4f}"
            )

    # -- Empirical complexity estimate from log-log regression
    if len(df) >= 3:
        print("\n### Empirical Complexity Estimate")
        df_sorted = df.sort_values("Vertices")
        valid = df_sorted[(df_sorted["Vertices"] > 0) & (df_sorted["Time (s)"] > 0)]
        if len(valid) >= 2:
            try:
                from scipy.stats import linregress

                log_v = np.log(valid["Vertices"].values.astype(float))
                log_t = np.log(valid["Time (s)"].values.astype(float))
                slope, intercept, r_val, p_val, _ = linregress(log_v, log_t)
                print(f"  T(n) \u2248 {np.exp(intercept):.4f} * n^{slope:.2f}")
                print(f"  R\u00b2 = {r_val**2:.3f}")
                print("  Projected times for larger meshes:")
                for n in [1000, 2500, 5000, 10000]:
                    pred = np.exp(intercept) * (n**slope)
                    print(f"    {n:>6} vertices -> ~{pred:.1f}s")
            except Exception as e:
                print(f"  Could not compute complexity estimate: {e}")

    print("=" * 80)
    return df


# ---------------------------------------------------------------------------
# Sensitivity sweep analysis
# ---------------------------------------------------------------------------


def analyse_sensitivity_results(sensitivity_file: Path):
    """
    Load the sensitivity sweep JSON and print the best weight configuration.

    Reads the JSON written by run_weight_sensitivity_sweep.py. Groups results
    by which weight was varied (planarity, fairness, or closeness) and
    identifies the best configuration for each sweep axis.

    Parameters
    ----------
    sensitivity_file : Path
        Path to weight_sensitivity_data.json.

    Returns
    -------
    dict or None
        Dictionary with keys 'planarity_sweep', 'fairness_sweep',
        'closeness_sweep', each a list of result dicts. None if the
        file is missing.

    Notes
    -----
    Weight group thresholds are derived from the actual data rather than
    hard-coded constants so the function works regardless of which base
    weights were used in the sweep (e.g. default w_p=10, w_f=1, w_c=5).
    """
    if not sensitivity_file.exists():
        print(f"\u26a0\ufe0f  Sensitivity file not found: {sensitivity_file}")
        print("    Run run_weight_sensitivity_sweep.py first.")
        return None

    with open(sensitivity_file, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print("\u26a0\ufe0f  Sensitivity file is empty.")
        return None

    print("\n" + "=" * 70)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"  Total configurations tested: {len(data)}")

    all_wp = np.array([r["weights"]["planarity"] for r in data])
    all_wf = np.array([r["weights"]["fairness"] for r in data])
    all_wc = np.array([r["weights"]["closeness"] for r in data])

    def _spread(arr):
        """Return coefficient of variation (std/mean) to measure how much
        a weight varies across all experiment configurations."""
        m = np.mean(arr)
        return (np.std(arr) / m) if m > 0 else 0.0

    planarity_sweep, fairness_sweep, closeness_sweep = [], [], []
    for item in data:
        wp = item["weights"]["planarity"]
        wf = item["weights"]["fairness"]
        wc = item["weights"]["closeness"]
        dev_p = abs(wp - np.mean(all_wp)) / (np.mean(all_wp) + 1e-12)
        dev_f = abs(wf - np.mean(all_wf)) / (np.mean(all_wf) + 1e-12)
        dev_c = abs(wc - np.mean(all_wc)) / (np.mean(all_wc) + 1e-12)
        max_dev = max(dev_p, dev_f, dev_c)
        if max_dev == dev_p:
            planarity_sweep.append({"weight": wp, **item})
        elif max_dev == dev_f:
            fairness_sweep.append({"weight": wf, **item})
        else:
            closeness_sweep.append({"weight": wc, **item})

    def _best(sweep, key_fn):
        """Return the successful configuration that minimises key_fn."""
        ok = [x for x in sweep if x.get("success", False)]
        return min(ok, key=key_fn) if ok else None

    for label, sweep in [
        ("Planarity", planarity_sweep),
        ("Fairness", fairness_sweep),
        ("Closeness", closeness_sweep),
    ]:
        if not sweep:
            continue
        print(f"\n### {label} Weight Sweep ({len(sweep)} configurations)")
        best = _best(sweep, lambda x: x.get("planarity_final", float("inf")))
        if best:
            print(f"  Best weight: {best['weight']:.3f}")
            print(f"  -> Final planarity:      {best.get('planarity_final', 'N/A')}")
            print(
                f"  -> Planarity improvement: {best.get('planarity_improvement', 0)*100:.2f}%"
            )
            print(
                f"  -> Energy reduction:      {best.get('energy_reduction', 0)*100:.1f}%"
            )
        else:
            print("  No successful configurations found.")

    print("=" * 70)
    return {
        "planarity_sweep": planarity_sweep,
        "fairness_sweep": fairness_sweep,
        "closeness_sweep": closeness_sweep,
    }


# ---------------------------------------------------------------------------
# Dissertation table generation
# ---------------------------------------------------------------------------


def generate_dissertation_tables(benchmark_df, output_dir: Path):
    """
    Write LaTeX and CSV tables for Chapter 4 of the dissertation.

    Generates two files:
      - benchmark_table.tex  — LaTeX tabular suitable for copying directly
                               into the dissertation source.
      - benchmark_table.csv  — CSV version for cross-checking in Excel/Word.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        The DataFrame returned by analyse_benchmark_results().
    output_dir : Path
        Directory to write the output files into. Created if it does
        not exist.
    """
    if benchmark_df is None:
        print("\u26a0\ufe0f  No benchmark data to generate tables from.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    cols_for_latex = [
        "Mesh",
        "Vertices",
        "Faces",
        "Time (s)",
        "Memory (MB)",
        "Iterations",
        "Planarity Improv. (%)",
        "Success",
    ]
    cols_present = [c for c in cols_for_latex if c in benchmark_df.columns]

    latex_table = benchmark_df[cols_present].to_latex(
        index=False,
        float_format="%.2f",
        caption="Performance benchmarks across different mesh sizes.",
        label="tab:benchmarks",
    )
    tex_path = output_dir / "benchmark_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex_table)
    print(f"\u2705 LaTeX table saved to {tex_path}")

    csv_path = output_dir / "benchmark_table.csv"
    benchmark_df.to_csv(csv_path, index=False)
    print(f"\u2705 CSV table saved to {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """
    Parse arguments, run all analyses, and write output tables.
    """
    parser = argparse.ArgumentParser(
        description="Summarise PQ mesh optimiser results and export dissertation tables."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the benchmark JSON file "
            "(default: data/output/benchmarks/performance_data.json)."
        ),
    )
    parser.add_argument(
        "--sensitivity",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the sensitivity JSON file "
            "(default: data/output/weight_sensitivity/weight_sensitivity_data.json)."
        ),
    )
    args = parser.parse_args()

    benchmark_file = (
        Path(args.benchmark)
        if args.benchmark
        else PROJECT_ROOT / "data/output/benchmarks/performance_data.json"
    )
    sensitivity_file = (
        Path(args.sensitivity)
        if args.sensitivity
        else PROJECT_ROOT
        / "data/output/weight_sensitivity/weight_sensitivity_data.json"
    )
    tables_dir = PROJECT_ROOT / "data/output/tables"

    print("\n" + "=" * 80)
    print("DISSERTATION RESULTS ANALYSIS")
    print("=" * 80)

    benchmark_df = analyse_benchmark_results(benchmark_file)
    analyse_sensitivity_results(sensitivity_file)
    generate_dissertation_tables(benchmark_df, tables_dir)

    print("\n\u2705 Analysis complete.")
    print("\nOutput files:")
    print(f"  {tables_dir}/benchmark_table.tex")
    print(f"  {tables_dir}/benchmark_table.csv")


if __name__ == "__main__":
    main()
