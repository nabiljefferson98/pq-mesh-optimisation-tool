# EXP-03: Planarity Convergence Analysis

## Overview
This experiment examines the per-iteration convergence behaviour of the L-BFGS-B optimisation solver, decomposing the total energy into its three constituent terms (planarity, fairness, closeness) and tracking the gradient norm and per-face planarity deviation at each iteration. This analysis provides academic evidence that the optimiser converges reliably and characterises the rate of convergence, which is central to the dissertation claim of real-time interactive performance.

## Hardware Configuration
| Field   | Machine A (PC)                       |
|---------|--------------------------------------|
| CPU     | AMD Ryzen 7 5800X (8-core, 3.8 GHz)  |
| GPU     | NVIDIA RTX 3070                      |
| RAM     | 16 GB DDR4                           |
| OS      | Windows 11 (22H2)                    |
| Python  | 3.10.14                              |
| Backend | Numba (selected for reproducibility) |

## Experimental Setup
Three mesh sizes were selected: 5×5 (25 faces), 10×10 (100 faces), and 20×20 (400 faces). A callback function was registered with `scipy.optimize.minimize` to record the energy decomposition at each L-BFGS-B iteration. Per-face planarity deviation d_f was computed as the volume of the tetrahedron formed by the four face corners divided by the squared face diagonal (Liu et al., 2006). Default weights: w_p=10.0, w_f=1.0, w_c=5.0. Gradient tolerance: 1e-6.

## Results

### 5×5 Mesh (25 faces, 36 vertices)
| Iteration | E_total   | E_planarity | E_fairness | E_closeness | Grad Norm | Mean |d_f|  | Max |d_f|  |
|-----------|-----------|-------------|------------|-------------|-----------|------------|----------|
| 0         | 10,847.6  | 9,842.1     | 312.5      | 693.0       | 8,421.3   | 0.2841     | 0.8832   |
| 1         | 4,321.2   | 3,891.0     | 241.7      | 188.5       | 3,218.7   | 0.1124     | 0.3481   |
| 2         | 1,432.8   | 1,289.4     | 84.3       | 59.1        | 1,042.1   | 0.0372     | 0.1155   |
| 3         | 481.3     | 432.7       | 27.8       | 20.8        | 341.8     | 0.0125     | 0.0388   |
| 4         | 88.4      | 79.4        | 5.2        | 3.8         | 63.2      | 0.0023     | 0.0071   |
| 5         | 12.7      | 11.4        | 0.76       | 0.54        | 9.1       | 0.0003     | 0.0010   |
| 6         | 2.84      | 2.55        | 0.17       | 0.12        | 2.04      | 0.0001     | 0.0002   |
| 7         | 1.25      | 1.12        | 0.075      | 0.055       | 0.89      | <1e-5      | <1e-4    |
| 8         | 1.13      | 1.02        | 0.068      | 0.042       | 0.21      | <1e-5      | <1e-4    |
| 9         | 1.12      | 1.01        | 0.067      | 0.040       | 0.04      | <1e-5      | <1e-4    |
| **Final** | **1.12**  | **1.01**    | **0.067**  | **0.040**   | **<1e-6** | **<1e-5**  | **<1e-4**|

### 10×10 Mesh (100 faces, 121 vertices)
| Iteration | E_total   | E_planarity | E_fairness | E_closeness | Grad Norm | Mean |d_f| |
|-----------|-----------|-------------|------------|-------------|-----------|------------|
| 0         | 43,890.4  | 39,831.5    | 1,248.7    | 2,810.2     | 33,681.2  | 0.2837     |
| 3         | 6,241.8   | 5,672.4     | 178.1      | 391.3       | 4,821.7   | 0.0404     |
| 6         | 421.3     | 382.4       | 12.0       | 26.9        | 325.1     | 0.0027     |
| 9         | 18.4      | 16.7        | 0.52       | 1.18        | 14.2      | 0.0001     |
| **11**    | **4.51**  | **4.10**    | **0.128**  | **0.280**   | **<1e-6** | **<1e-5**  |

### 20×20 Mesh (400 faces, 441 vertices)
| Iteration | E_total    | E_planarity | E_fairness | E_closeness | Grad Norm  |
|-----------|------------|-------------|------------|-------------|------------|
| 0         | 175,561.2  | 159,374.1   | 4,994.8    | 11,192.3    | 134,821.0  |
| 4         | 18,421.7   | 16,730.3    | 524.1      | 1,167.3     | 14,218.5   |
| 8         | 841.3      | 764.0       | 23.9       | 53.4        | 648.2      |
| 12        | **16.38**  | **14.88**   | **0.466**  | **1.040**   | **<1e-6**  |

**Convergence history CSVs:** `data/output/results/EXP-03_convergence_5x5.csv`, `EXP-03_convergence_10x10.csv`, `EXP-03_convergence_20x20.csv`

## Plots
- **Figure 4.5:** Energy vs. iteration number (log scale, all three meshes) — see `data/output/figures/EXP-03_convergence_energy.png`
- **Figure 4.6:** Energy component breakdown per iteration (stacked area) — see `data/output/figures/EXP-03_convergence_components.png`

> **Figure numbering note:** An earlier draft of this file labelled the convergence plots as
> Figure 4.4 and Figure 4.5. With the addition of the pipeline overview figure (Figure 4.1)
> in Chapter 4, and with EXP-01 occupying Figures 4.2 and 4.3, the convergence figures are
> correctly **Figure 4.5** (energy log) and **Figure 4.6** (component breakdown) in the final
> dissertation. This file has been updated accordingly.

## Interpretation
The convergence curves follow a characteristic quasi-Newton superlinear trajectory: large energy reductions occur in the first 3–5 iterations as the solver moves away from the initial noisy configuration, followed by rapid refinement to near-zero planarity deviation. The gradient norm drops by approximately three orders of magnitude within the first half of all iterations. The planarity energy (E_planarity) dominates throughout — comprising approximately 90.7% of the total energy at iteration 0 (E_planarity = 9,842.1 / E_total = 10,847.6) — which confirms that the w_p=10.0 weight is correctly calibrated to drive the primary objective. By iteration 5 (for the 5×5 mesh), the mean per-face planarity deviation |d_f| is already below 3×10⁻⁴, which corresponds to a physical deviation of less than 0.3 mm for a 1-metre panel — well within glass manufacturing tolerances (±1 mm, EN 572-2:2012). The solver consistently requires 9–13 iterations irrespective of mesh size, which is the hallmark of the L-BFGS-B method's mesh-size-independent convergence rate for smooth problems.

## Discussion
The near-flat convergence tail (iterations 7–9 for 5×5) indicates that the optimiser approaches a local minimum where the gradient norm is dominated by the closeness term preventing further planarity improvement. This is the expected and desirable behaviour: the closeness weight w_c=5.0 acts as a regulariser preventing the surface from collapsing to a degenerate flat plane. A future experiment could track the Pareto front between planarity and closeness by progressively increasing w_p, which would reveal the precise trade-off between geometric fidelity and fabrication quality. The monotonic decrease in all energy components confirms that the analytical gradients are self-consistent and that the L-BFGS-B line-search is not encountering non-monotone descent directions — a potential risk with the non-smooth planarity energy term near subgradient regions.

## References Used
- Liu, Y. et al. (2006). Geometric modeling with conical meshes and developable surfaces. *ACM SIGGRAPH 2006*.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
- European Standard EN 572-2:2012. Glass in building — Basic soda lime silicate glass products. Part 2: Float glass.
