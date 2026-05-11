"""
Dissertation-quality benchmark figures for PQ-Mesh Optimisation.
Reads all EXP-01 to EXP-05 JSON files and produces 7 publication-ready PNG figures.

Usage:
    python scripts/plotting/plot_benchmarks.py

Output:
    data/output/figures/FIG-01_energy_reduction_seriesA.png
    data/output/figures/FIG-02_scalability_seriesB.png
    data/output/figures/FIG-03_backend_comparison.png
    data/output/figures/FIG-04_convergence_profile.png
    data/output/figures/FIG-05_realworld_timing.png
    data/output/figures/FIG-06_planarity_improvement.png
    data/output/figures/FIG-07_platform_timing_heatmap.png
"""

import glob
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Global style ────────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

BENCH_DIR = Path("data/output/benchmarks")
FIG_DIR = Path("data/output/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAC_COLOUR = "#2563EB"  # blue
WIN_COLOUR = "#DC2626"  # red
NUMPY_COLOUR = "#16A34A"  # green
NUMBA_COLOUR = "#9333EA"  # purple

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_json(path):
    with open(path) as f:
        return json.load(f)


def mean_runs(file_pattern):
    """
    Given a glob pattern matching multiple run JSON files,
    return a dict keyed by mesh_name with averaged metrics.
    """
    files = sorted(glob.glob(str(file_pattern)))
    if not files:
        return {}
    accum = {}
    for fpath in files:
        records = load_json(fpath)
        for r in records:
            name = r["mesh_name"]
            if name not in accum:
                accum[name] = {
                    "n_faces": r["n_faces"],
                    "n_vertices": r["n_vertices"],
                    "time_mean_s": [],
                    "time_std_s": [],
                    "memory_peak_mb": [],
                    "iterations": [],
                    "energy_reduction": [],
                    "planarity_improvement": [],
                    "initial_energy": [],
                    "final_energy": [],
                }
            accum[name]["time_mean_s"].append(r["time_mean_s"])
            accum[name]["time_std_s"].append(r["time_std_s"])
            accum[name]["memory_peak_mb"].append(r["memory_peak_mb"])
            accum[name]["iterations"].append(r["iterations"])
            accum[name]["energy_reduction"].append(r["energy_reduction"])
            accum[name]["planarity_improvement"].append(r["planarity_improvement"])
            accum[name]["initial_energy"].append(r["initial_energy"])
            accum[name]["final_energy"].append(r["final_energy"])

    result = {}
    for name, vals in accum.items():
        result[name] = {
            "n_faces": vals["n_faces"],
            "n_vertices": vals["n_vertices"],
            "time_mean": np.mean(vals["time_mean_s"]),
            "time_std": np.mean(vals["time_std_s"]),
            "memory_mb": np.mean(vals["memory_peak_mb"]),
            "iterations": round(np.mean(vals["iterations"])),
            "energy_reduction": np.mean(vals["energy_reduction"]),
            "planarity_improvement": np.mean(vals["planarity_improvement"]),
            "initial_energy": np.mean(vals["initial_energy"]),
            "final_energy": np.mean(vals["final_energy"]),
        }
    return result


# ════════════════════════════════════════════════════════════════════════════
# FIG-01  Energy Reduction — Series A Synthetic Grids (Mac + Win)
# ════════════════════════════════════════════════════════════════════════════
def fig01_energy_reduction():
    mac = mean_runs(BENCH_DIR / "EXP-01_seriesA_mac_run*.json")
    win = mean_runs(BENCH_DIR / "EXP-01_seriesA_win_run*.json")

    # sort by face count
    meshes = sorted(mac.keys(), key=lambda m: mac[m]["n_faces"])
    labels = [
        m.replace("plane_", "").replace("_noisy", "").replace("_", "×") for m in meshes
    ]
    labels = [l.replace("very×", "5×5\nvery ") for l in labels]

    x = np.arange(len(meshes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_mac = ax.bar(
        x - width / 2,
        [mac[m]["energy_reduction"] for m in meshes],
        width,
        label="Mac M3 (Numba)",
        color=MAC_COLOUR,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    bars_win = ax.bar(
        x + width / 2,
        [win[m]["energy_reduction"] for m in meshes],
        width,
        label="Windows RTX (Numba)",
        color=WIN_COLOUR,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar in list(bars_mac) + list(bars_win):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xlabel("Mesh (grid size)")
    ax.set_ylabel("Energy Reduction (%)")
    ax.set_title(
        "Figure 1 — Energy Reduction on Synthetic Planar Grids\n"
        "EXP-01 Series A: Mac M3 vs Windows RTX, Numba Backend"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG-01_energy_reduction_seriesA.png")
    plt.close(fig)
    print("✓ FIG-01 saved")


# ════════════════════════════════════════════════════════════════════════════
# FIG-02  Scalability — Series B Oloid Resolution (log-log, Mac + Win)
# ════════════════════════════════════════════════════════════════════════════
def fig02_scalability():
    mac = mean_runs(BENCH_DIR / "EXP-01_seriesB_mac_run*.json")
    win = mean_runs(BENCH_DIR / "EXP-01_seriesB_win_run*.json")

    meshes = sorted(mac.keys(), key=lambda m: mac[m]["n_faces"])
    faces = [mac[m]["n_faces"] for m in meshes]

    mac_times = [mac[m]["time_mean"] for m in meshes]
    win_times = [win[m]["time_mean"] for m in meshes]
    mac_stds = [mac[m]["time_std"] for m in meshes]
    win_stds = [win[m]["time_std"] for m in meshes]

    # fit O(n^alpha) power law
    def fit_power(x, y):
        lx, ly = np.log(x), np.log(y)
        alpha, logc = np.polyfit(lx, ly, 1)
        return alpha, np.exp(logc)

    alpha_mac, c_mac = fit_power(faces, mac_times)
    alpha_win, c_win = fit_power(faces, win_times)

    x_smooth = np.logspace(np.log10(min(faces) * 0.8), np.log10(max(faces) * 1.2), 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        faces,
        mac_times,
        yerr=mac_stds,
        fmt="o-",
        color=MAC_COLOUR,
        label=f"Mac M3  (α={alpha_mac:.2f})",
        capsize=4,
        linewidth=1.8,
        markersize=7,
    )
    ax.errorbar(
        faces,
        win_times,
        yerr=win_stds,
        fmt="s--",
        color=WIN_COLOUR,
        label=f"Windows RTX  (α={alpha_win:.2f})",
        capsize=4,
        linewidth=1.8,
        markersize=7,
    )
    ax.plot(
        x_smooth,
        c_mac * x_smooth**alpha_mac,
        color=MAC_COLOUR,
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
    )
    ax.plot(
        x_smooth,
        c_win * x_smooth**alpha_win,
        color=WIN_COLOUR,
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Quad Faces (log scale)")
    ax.set_ylabel("Wall-Clock Time / s (log scale)")
    ax.set_title(
        "Figure 2 — Scalability: Wall-Clock Time vs Mesh Resolution\n"
        "EXP-01 Series B: Oloid Family (64, 256, 1024 faces)"
    )
    ax.set_xticks(faces)
    ax.set_xticklabels([str(f) for f in faces])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG-02_scalability_seriesB.png")
    plt.close(fig)
    print("✓ FIG-02 saved")


# ════════════════════════════════════════════════════════════════════════════
# FIG-03  Backend Comparison — NumPy vs Numba (Mac + Win, grouped bar)
# ════════════════════════════════════════════════════════════════════════════
def fig03_backend_comparison():
    data = {}
    for platform in ("mac", "win"):
        for backend in ("numpy", "numba"):
            key = f"{platform}_{backend}"
            pat = BENCH_DIR / f"EXP-02_{platform}_{backend}_run*.json"
            data[key] = mean_runs(pat)

    meshes = sorted(
        data["mac_numpy"].keys(), key=lambda m: data["mac_numpy"][m]["n_faces"]
    )
    labels = [f"{data['mac_numpy'][m]['n_faces']}f" for m in meshes]
    x = np.arange(len(meshes))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, platform, pname, pcolour in [
        (axes[0], "mac", "Mac M3", MAC_COLOUR),
        (axes[1], "win", "Windows RTX", WIN_COLOUR),
    ]:
        np_times = [data[f"{platform}_numpy"][m]["time_mean"] for m in meshes]
        nb_times = [data[f"{platform}_numba"][m]["time_mean"] for m in meshes]
        np_stds = [data[f"{platform}_numpy"][m]["time_std"] for m in meshes]
        nb_stds = [data[f"{platform}_numba"][m]["time_std"] for m in meshes]

        b1 = ax.bar(
            x - w / 2,
            np_times,
            w,
            yerr=np_stds,
            label="NumPy",
            color=NUMPY_COLOUR,
            alpha=0.85,
            capsize=3,
            error_kw={"linewidth": 1.2},
            edgecolor="white",
        )
        b2 = ax.bar(
            x + w / 2,
            nb_times,
            w,
            yerr=nb_stds,
            label="Numba",
            color=NUMBA_COLOUR,
            alpha=0.85,
            capsize=3,
            error_kw={"linewidth": 1.2},
            edgecolor="white",
        )

        # annotate slowdown ratios
        for i, (np_t, nb_t) in enumerate(zip(np_times, nb_times)):
            ratio = nb_t / np_t
            ax.text(
                x[i],
                max(np_t, nb_t) * 1.05,
                f"{ratio:.1f}×",
                ha="center",
                fontsize=8,
                color="#6B7280",
            )

        ax.set_xlabel("Mesh (face count)")
        ax.set_ylabel("Wall-Clock Time / s")
        ax.set_title(f"{pname}: NumPy vs Numba")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()

    fig.suptitle(
        "Figure 3 — Backend Comparison: NumPy vs Numba\n"
        "EXP-02: Numba slowdown factor annotated above each pair",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG-03_backend_comparison.png")
    plt.close(fig)
    print("✓ FIG-03 saved")


# ════════════════════════════════════════════════════════════════════════════
# FIG-04  Convergence Profile — Time & Energy per Mesh (Mac + Win)
# ════════════════════════════════════════════════════════════════════════════
def fig04_convergence():
    meshes_ordered = [
        "plane_20x20_noisy",
        "oloid256_quad",
        "spot_quadrangulated",
        "blub_quadrangulated",
    ]
    labels = [
        "Plane 20×20\n(400 faces)",
        "Oloid 256\n(256 faces)",
        "Spot\n(2,928 faces)",
        "Blub\n(7,104 faces)",
    ]

    mac_data, win_data = {}, {}
    for mesh in meshes_ordered:
        mac_file = BENCH_DIR / f"EXP-03_{mesh}_mac.json"
        win_file = BENCH_DIR / f"EXP-03_{mesh}_win.json"
        if mac_file.exists():
            mac_data[mesh] = load_json(mac_file)[0]
        if win_file.exists():
            win_data[mesh] = load_json(win_file)[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left — wall-clock time
    ax = axes[0]
    x = np.arange(len(meshes_ordered))
    w = 0.35
    mac_times = [mac_data[m]["time_mean_s"] for m in meshes_ordered]
    win_times = [win_data[m]["time_mean_s"] for m in meshes_ordered]
    mac_stds = [mac_data[m]["time_std_s"] for m in meshes_ordered]
    win_stds = [win_data[m]["time_std_s"] for m in meshes_ordered]

    ax.bar(
        x - w / 2,
        mac_times,
        w,
        yerr=mac_stds,
        label="Mac M3",
        color=MAC_COLOUR,
        alpha=0.85,
        capsize=4,
        edgecolor="white",
    )
    ax.bar(
        x + w / 2,
        win_times,
        w,
        yerr=win_stds,
        label="Windows RTX",
        color=WIN_COLOUR,
        alpha=0.85,
        capsize=4,
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-Clock Time / s")
    ax.set_title("(a) Optimisation Time per Mesh")
    ax.legend()
    ax.set_yscale("log")

    # Right — energy reduction
    ax = axes[1]
    mac_er = [mac_data[m]["energy_reduction"] for m in meshes_ordered]
    win_er = [win_data[m]["energy_reduction"] for m in meshes_ordered]

    ax.bar(
        x - w / 2,
        mac_er,
        w,
        label="Mac M3",
        color=MAC_COLOUR,
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar(
        x + w / 2,
        win_er,
        w,
        label="Windows RTX",
        color=WIN_COLOUR,
        alpha=0.85,
        edgecolor="white",
    )
    for i, (me, we) in enumerate(zip(mac_er, win_er)):
        ax.text(x[i] - w / 2, me + 0.5, f"{me:.1f}%", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, we + 0.5, f"{we:.1f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy Reduction (%)")
    ax.set_title("(b) Energy Reduction per Mesh")
    ax.set_ylim(0, 105)
    ax.legend()

    fig.suptitle(
        "Figure 4 — Convergence Analysis: Time and Energy Reduction\n"
        "EXP-03: Synthetic and Reference Meshes, Numba Backend",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG-04_convergence_profile.png")
    plt.close(fig)
    print("✓ FIG-04 saved")


# ════════════════════════════════════════════════════════════════════════════
# FIG-05  Real-World Mesh Timing — Mac vs Win (EXP-05, 3-run means)
# ════════════════════════════════════════════════════════════════════════════
def fig05_realworld_timing():
    mac = mean_runs(BENCH_DIR / "EXP-05_mac_run*.json")
    win = mean_runs(BENCH_DIR / "EXP-05_win_run*.json")

    # only meshes present in both
    shared = sorted(set(mac) & set(win), key=lambda m: mac[m]["n_faces"])
    labels = [
        f"{m.replace('_quadrangulated','').replace('_quad','')}\n"
        f"({mac[m]['n_faces']:,} faces)"
        for m in shared
    ]

    x, w = np.arange(len(shared)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(
        x - w / 2,
        [mac[m]["time_mean"] for m in shared],
        w,
        yerr=[mac[m]["time_std"] for m in shared],
        label="Mac M3",
        color=MAC_COLOUR,
        alpha=0.85,
        capsize=4,
        edgecolor="white",
    )
    b2 = ax.bar(
        x + w / 2,
        [win[m]["time_mean"] for m in shared],
        w,
        yerr=[win[m]["time_std"] for m in shared],
        label="Windows RTX",
        color=WIN_COLOUR,
        alpha=0.85,
        capsize=4,
        edgecolor="white",
    )

    # annotate Mac/Win ratio
    for i, m in enumerate(shared):
        ratio = win[m]["time_mean"] / mac[m]["time_mean"]
        top = max(mac[m]["time_mean"], win[m]["time_mean"])
        sign = "Mac faster" if ratio > 1 else "Win faster"
        ax.text(
            x[i],
            top * 1.04,
            f"{'↑' if ratio>1 else '↓'}{abs(ratio):.2f}×\n({sign})",
            ha="center",
            fontsize=8,
            color="#374151",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-Clock Time / s")
    ax.set_title(
        "Figure 5 — Real-World Mesh Optimisation Time\n"
        "EXP-05: Mac M3 vs Windows RTX, Numba Backend (mean of 3 runs ± std)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG-05_realworld_timing.png")
    plt.close(fig)
    print("✓ FIG-05 saved")


# ════════════════════════════════════════════════════════════════════════════
# FIG-06  Planarity Improvement — Before vs After (EXP-03, Mac)
# ════════════════════════════════════════════════════════════════════════════
def fig06_planarity():
    meshes_ordered = [
        "plane_20x20_noisy",
        "oloid256_quad",
        "spot_quadrangulated",
        "blub_quadrangulated",
    ]
    labels = ["Plane 20×20", "Oloid 256", "Spot", "Blub"]

    before_means, after_means = [], []
    before_stds, after_stds = [], []

    for mesh in meshes_ordered:
        rec = load_json(BENCH_DIR / f"EXP-03_{mesh}_mac.json")[0]
        before_means.append(rec["planarity_before"]["mean"] * 1000)
        after_means.append(rec["planarity_after"]["mean"] * 1000)
        before_stds.append(rec["planarity_before"]["std"] * 1000)
        after_stds.append(rec["planarity_after"]["std"] * 1000)

    x, w = np.arange(len(meshes_ordered)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - w / 2,
        before_means,
        w,
        yerr=before_stds,
        label="Before optimisation",
        color="#F59E0B",
        alpha=0.85,
        capsize=4,
        edgecolor="white",
    )
    ax.bar(
        x + w / 2,
        after_means,
        w,
        yerr=after_stds,
        label="After optimisation",
        color="#10B981",
        alpha=0.85,
        capsize=4,
        edgecolor="white",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Planarity Deviation (×10⁻³)")
    ax.set_title(
        "Figure 6 — Planarity Improvement: Before vs After Optimisation\n"
        "EXP-03: Mean face planarity deviation (lower is better), Mac M3"
    )
    ax.legend()

    # Note about oloid — planarity worsens because the mesh starts near-perfect
    ax.annotate(
        "*Oloid starts near-planar;\noptimiser trades planarity\nfor angle balance",
        xy=(1, after_means[1]),
        xytext=(1.6, after_means[1] * 1.8),
        fontsize=8,
        color="#6B7280",
        arrowprops=dict(arrowstyle="->", color="#6B7280", lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG-06_planarity_improvement.png")
    plt.close(fig)
    print("✓ FIG-06 saved")


# ════════════════════════════════════════════════════════════════════════════
# FIG-07  Platform × Mesh Timing Heatmap (EXP-05 + EXP-03 combined)
# ════════════════════════════════════════════════════════════════════════════
def fig07_heatmap():
    # Collect all per-mesh mean times on both platforms
    exp03_meshes = [
        "plane_20x20_noisy",
        "oloid256_quad",
        "spot_quadrangulated",
        "blub_quadrangulated",
    ]
    mesh_labels = ["Plane\n20×20", "Oloid\n256", "Spot", "Blub"]

    mac_times, win_times = [], []
    for mesh in exp03_meshes:
        m_rec = load_json(BENCH_DIR / f"EXP-03_{mesh}_mac.json")[0]
        w_rec = load_json(BENCH_DIR / f"EXP-03_{mesh}_win.json")[0]
        mac_times.append(m_rec["time_mean_s"])
        win_times.append(w_rec["time_mean_s"])

    data = np.array([mac_times, win_times])
    # normalise each column to [0,1] for colour mapping
    norm_d = data / data.max(axis=0)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(norm_d, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Mac M3", "Windows RTX"])
    ax.set_xticks(range(len(exp03_meshes)))
    ax.set_xticklabels(mesh_labels)

    # annotate each cell with actual time
    for row, platform_times in enumerate([mac_times, win_times]):
        for col, t in enumerate(platform_times):
            txt = f"{t:.2f}s" if t < 10 else f"{t:.1f}s"
            ax.text(
                col,
                row,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if norm_d[row, col] > 0.6 else "black",
            )

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, fraction=0.03)
    cbar.set_label("Relative time\n(column-normalised)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Faster", "Mid", "Slower"])

    ax.set_title(
        "Figure 7 — Platform Timing Heatmap: Mac M3 vs Windows RTX\n"
        "EXP-03: Wall-clock time per mesh (green = faster, red = slower)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG-07_platform_heatmap.png")
    plt.close(fig)
    print("✓ FIG-07 saved")


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating dissertation figures...")
    fig01_energy_reduction()
    fig02_scalability()
    fig03_backend_comparison()
    fig04_convergence()
    fig05_realworld_timing()
    fig06_planarity()
    fig07_heatmap()
    print(f"\n✅ All figures saved to {FIG_DIR}/")
