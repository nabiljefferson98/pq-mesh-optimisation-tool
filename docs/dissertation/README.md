# docs/dissertation/ — LaTeX Dissertation Source

This directory contains the LaTeX source files for the undergraduate dissertation:

> **Real-Time Planar Quad Mesh Optimisation for Designing Developable Surfaces**
> Muhammad Nabil Bin Muhammad Saiful Wong
> University of Leeds, School of Computing, 2025/26

---

## File Map

```
docs/dissertation/
├── README.md                   ← This file
├── main.tex                    ← Top-level document (\input all chapters)
├── chapter4_results.tex        ← Chapter 4: Results (§4.1–§4.5)
├── chapter5_discussion.tex     ← Chapter 5: Discussion (§5.1–§5.3)
└── figures.tex                 ← Centralised \includegraphics{} figure definitions
```

## Figure Assets

All figure PNG files are committed at `data/output/figures/`:

| File | Figure number | Caption |
|---|---|---|
| `EXP-01_scalability_loglog.png` | 4.2 | Log-log scalability plot with fitted complexity line |
| `EXP-01_convergence_scaling.png` | 4.3 | Convergence and scaling overlay (Series A + B) |
| `EXP-05_planarity_histograms.png` | 4.8 | Per-mesh planarity deviation histograms (EXP-05) |
| `EXP-05_oloid_spatial_heatmap.png` | 4.9 | Oloid per-face planarity deviation spatial heatmap |

## Benchmark Table

The corrected LaTeX benchmark table (EXP-05) is at:
`data/output/tables/benchmark_table.tex`

In `chapter4_results.tex`, include it with:
```latex
\input{../../data/output/tables/benchmark_table}
```

## Notes

- Figures are referenced relative to the repository root via the `graphicspath` set in `main.tex`.
- Do **not** commit compiled `.pdf`, `.aux`, `.log`, `.synctex.gz` artefacts — these are excluded by `.gitignore`.
- The dissertation word count target is 10,000–15,000 words (Leeds undergraduate requirement).
