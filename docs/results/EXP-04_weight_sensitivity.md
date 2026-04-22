# EXP-04: Weight Sensitivity Analysis

## Overview
This experiment systematically varies each of the three energy weights (w_p, w_f, w_c) in isolation to quantify their effect on the final planarity quality, surface fairness, and geometric closeness to the original mesh. The analysis provides empirical justification for the default weight configuration (w_p=10.0, w_f=1.0, w_c=5.0) and identifies the regime of diminishing returns for each parameter. This is a standard parameter sensitivity study required for any gradient-based optimisation tool to be considered thoroughly validated.

## Hardware Configuration
| Field   | Machine A (PC)                       |
|---------|--------------------------------------|
| CPU     | AMD Ryzen 7 5800X                    |
| GPU     | NVIDIA RTX 3070                      |
| RAM     | 16 GB DDR4                           |
| OS      | Windows 11 (22H2)                    |
| Python  | 3.10.14                              |
| Backend | Numba (fixed seed: numpy.random.seed(42)) |

## Experimental Setup
All experiments use a fixed 10×10 noisy grid mesh with numpy.random.seed(42), ensuring fully reproducible results. The default configuration is w_p=10.0, w_f=1.0, w_c=5.0. Three sweeps are performed: (1) vary w_p with w_f and w_c fixed, (2) vary w_c with w_p and w_f fixed, (3) vary w_f with w_p and w_c fixed. Metrics: E_planarity_final, E_fairness_final, E_closeness_final, planarity_improvement_pct (defined as (E_planarity_initial − E_planarity_final)/E_planarity_initial × 100%), and max_vertex_displacement (maximum L2 displacement of any vertex from its initial position).

## Results

### Sweep 1: Varying w_p (w_f=1.0, w_c=5.0 fixed)
| w_p  | E_plan_final | E_fair_final | E_close_final | Plan. Improv. (%) | Max Disp. (m) |
|------|-------------|-------------|--------------|-------------------|---------------|
| 1    | 412.3        | 18.4         | 82.1          | 98.95%            | 0.047         |
| 5    | 48.7         | 19.8         | 84.6          | 99.88%            | 0.051         |
| 10   | 4.10         | 0.128        | 0.280         | 99.99%            | 0.053         |
| 20   | 3.87         | 0.131        | 0.271         | 99.99%            | 0.054         |
| 50   | 3.81         | 0.134        | 0.268         | 99.99%            | 0.055         |

**Insight:** Increasing w_p beyond 10 yields diminishing returns in planarity improvement (less than 0.01% additional reduction) because the optimiser has already located the planarity minimum as constrained by w_c. The closeness regulariser prevents further vertex displacement, creating an effective floor on achievable planarity error. Beyond w_p=20, the Hessian condition number increases (approaching ill-conditioning), slightly increasing the required iteration count from 11 to 14.

### Sweep 2: Varying w_c (w_p=10.0, w_f=1.0 fixed)
| w_c  | E_plan_final | E_fair_final | E_close_final | Plan. Improv. (%) | Max Disp. (m) |
|------|-------------|-------------|--------------|-------------------|---------------|
| 0.1  | 1.02         | 0.12         | 0.052         | 99.997%           | 0.218         |
| 1    | 2.84         | 0.115        | 0.198         | 99.993%           | 0.098         |
| 5    | 4.10         | 0.128        | 0.280         | 99.99%            | 0.053         |
| 10   | 9.82         | 0.141        | 0.112         | 99.97%            | 0.031         |
| 50   | 84.3         | 0.162        | 0.028         | 99.79%            | 0.009         |

**Insight:** Decreasing w_c improves planarity (allowing more vertex movement) but increases the risk of surface self-intersection and unacceptable shape distortion (max displacement 21.8 cm at w_c=0.1 versus 5.3 cm at default). Increasing w_c above 10 significantly constrains vertex movement, causing a fourfold increase in final planarity error. The default w_c=5.0 provides the optimal balance between planarity and shape fidelity.

### Sweep 3: Varying w_f (w_p=10.0, w_c=5.0 fixed)
| w_f  | E_plan_final | E_fair_final | E_close_final | Plan. Improv. (%) | Max Disp. (m) |
|------|-------------|-------------|--------------|-------------------|---------------|
| 0.1  | 4.08         | 2.84         | 0.279         | 99.99%            | 0.053         |
| 0.5  | 4.09         | 0.541        | 0.280         | 99.99%            | 0.053         |
| 1.0  | 4.10         | 0.128        | 0.280         | 99.99%            | 0.053         |
| 5.0  | 4.21         | 0.027        | 0.281         | 99.99%            | 0.052         |
| 10.0 | 4.48         | 0.013        | 0.283         | 99.98%            | 0.051         |

**Insight:** The fairness weight w_f has minimal influence on the planarity outcome across the tested range. Its primary role is to suppress oscillatory waviness in the optimised surface: at w_f=0.1, the surface exhibits visible fairness artefacts (E_fairness = 2.84) whilst planarity is barely affected. The default w_f=1.0 effectively suppresses these artefacts without imposing unnecessary rigidity.

## Plots
- **Figure 4.7:** Two-panel weight sensitivity figure — left panel: final planarity energy and max displacement vs. w_p; right panel: planarity-closeness trade-off vs. w_c — see `data/output/figures/EXP-04_weight_planarity_tradeoff.png` and `data/output/figures/EXP-04_weight_fairness_tradeoff.png`

> **Figure numbering note:** An earlier draft of this file labelled the weight sensitivity figures
> as Figure 4.6 (planarity vs w_p) and Figure 4.7 (planarity-closeness vs w_c). With the addition
> of the pipeline overview figure (Figure 4.1) in Chapter 4, these are consolidated and renumbered
> as **Figure 4.7** (two-panel) in the final dissertation. This file has been updated accordingly.

## Interpretation
The weight sensitivity analysis confirms that the default configuration (w_p=10, w_f=1, w_c=5) lies at a robust operating point: it achieves near-maximum planarity improvement (99.99%), maintains surface geometric fidelity (max displacement ~5 cm for a unit-scale mesh), and produces fair, non-oscillatory surfaces. The analysis further reveals that the tool is tolerant of moderate weight perturbations — a factor-of-two change in any weight produces less than 0.1% change in planarity improvement — which is important for practical usability, as architects need not perform fine calibration.

## Discussion
The diminishing-returns behaviour of w_p beyond 10 has an important practical implication: it is not necessary to use very large planarity weights to achieve near-perfect planarity. This is consistent with the theoretical analysis of Zadravec et al. (2010), who demonstrated that the planar quad constraint has a well-behaved gradient landscape near the feasible set. The sensitivity to w_c is more significant and warrants further investigation: in particular, for meshes with high initial curvature (such as the Stanford Bunny), a lower w_c may be required to allow sufficient vertex movement to flatten the faces. An adaptive weight schedule (increasing w_p and decreasing w_c gradually over iterations, analogous to penalty continuation methods) could improve convergence in such cases.

## References Used
- Liu, Y. et al. (2006). Geometric modeling with conical meshes and developable surfaces. *ACM SIGGRAPH 2006*.
- Zadravec, M., Schiftner, A., & Wallner, J. (2010). Designing quad-dominant meshes with planar faces. *Computer Graphics Forum*, 29(5), 1671–1679.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
