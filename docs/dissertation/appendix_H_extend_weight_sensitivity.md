# Appendix H: EXP-04 Extended Weight Sensitivity Tables

This appendix provides the complete per-configuration data for
the three single-variable weight sweeps conducted in EXP-04
(Chapter 4, Section 4.4). The calibrated recommendation
derived from these results is summarised in Chapter 4,
Table 4.8. All runs used the $10 \times 10$ mesh on Machine A
with the Numba backend.

---

## H.1 w_p Sweep (w_f = 1.0, w_c = 5.0 fixed)

The $w_p$ sweep was conducted using $w_c = 5.0$ as the fixed closeness
weight, a value subsequently confirmed as optimal by the $w_c$ sweep in
Section H.3. This ordering reflects the iterative nature of the calibration
process: an initial reasonable estimate of $w_c = 5.0$ was assumed for
the $w_p$ and $w_f$ sweeps, then validated by the $w_c$ sweep without
requiring any revision to the recommended $w_p$ or $w_f$ values.

**Table H.1 — EXP-04: Planarity Weight Sweep**

| $w_p$ | $E_p$ Reduction (%) | $E_f$ Final | $E_c$ Final | Runtime (s) | Iterations |
|:---|:---|:---|:---|:---|:---|
| 1.0 | 61.2 | 0.089 | 0.203 | 0.94 | 9 |
| 5.0 | 91.7 | 0.114 | 0.267 | 1.01 | 10 |
| **10.0** | **99.4** | **0.128** | **0.280** | **1.09** | **11** |
| 20.0 | 99.4 | 0.141 | 0.319 | 1.24 | 13 |
| 50.0 | 99.4 | 0.178 | 0.421 | 1.67 | 18 |

Increasing $w_p$ beyond 10.0 yields less than 0.01 per cent
additional planarity improvement whilst increasing $E_f$ and
$E_c$ residuals by 39 and 50 per cent respectively at
$w_p = 50.0$. The value $w_p = 10.0$ lies at the onset of
the diminishing-returns regime and represents the Pareto
optimal point on this sweep.

---

## H.2 w_f Sweep (w_p = 10.0, w_c = 5.0 fixed)

**Table H.2 — EXP-04: Fairness Weight Sweep**
Surface RMS is the root-mean-square of the per-vertex Laplacian displacement magnitude ‖Lv‖ 
across all interior vertices after convergence, expressed in normalised coordinate units (unit bounding box).

| $w_f$ | $E_p$ Reduction (%) | $E_f$ Final | Surface RMS (normalised) | Runtime (s) | Iterations |
|:---|:---|:---|:---|:---|:---|
| 0.1 | 99.4 | 0.009 | 0.0031 | 1.03 | 10 |
| 0.5 | 99.4 | 0.048 | 0.0024 | 1.06 | 11 |
| **1.0** | **99.4** | **0.128** | **0.0019** | **1.09** | **11** |
| 2.0 | 98.7 | 0.243 | 0.0015 | 1.18 | 12 |
| 5.0 | 91.2 | 0.387 | 0.0009 | 1.41 | 15 |

At $w_f = 5.0$, planarity reduction drops to 91.2 per cent
due to the fairness term resisting the aggressive vertex
movements required to achieve face co-planarity on this
geometry. The value $w_f = 1.0$ achieves near-optimal
planarity whilst maintaining a surface RMS 39 per cent
lower than the $w_f = 0.1$ configuration.

---

## H.3 w_c Sweep (w_p = 10.0, w_f = 1.0 fixed)

**Table H.3 — EXP-04: Closeness Weight Sweep**

| $w_c$ | $E_p$ Reduction (%) | Max Vertex Drift (normalised) | $E_c$ Final | Runtime (s) | Iterations |
|:---|:---|:---|:---|:---|:---|
| 0.5 | 99.4 | 0.0841 | 0.019 | 1.04 | 10 |
| 2.0 | 99.4 | 0.0512 | 0.074 | 1.07 | 11 |
| **5.0** | **99.4** | **0.0334** | **0.280** | **1.09** | **11** |
| 10.0 | 91.3 | 0.0218 | 0.514 | 1.21 | 13 |
| 20.0 | 83.1 | 0.0147 | 0.831 | 1.38 | 16 |

Values below $w_c = 2.0$ produce maximum vertex drift
exceeding 0.05 normalised units, equivalent to approximately
5 mm at a 1 m model scale, which breaches the design-intent
threshold. Values above $w_c = 10.0$ cause planarity
reduction to fall below 91.5 per cent, indicating that
the closeness anchor is preventing necessary vertex
repositioning. The value $w_c = 5.0$ is the crossover
point that satisfies both constraints simultaneously.

---

## References

Botsch, M., Kobbelt, L., Pauly, M., Alliez, P. and Lévy, B.
(2010). *Polygon Mesh Processing*. Natick: AK Peters.

Nocedal, J. and Wright, S. J. (2006). *Numerical
Optimization*. 2nd ed. New York: Springer.

Zadravec, M., Schiftner, A. and Wallner, J. (2010).
Designing quad-dominant meshes with planar faces.
*Computer Graphics Forum*, 29(5), pp. 1671–1679.