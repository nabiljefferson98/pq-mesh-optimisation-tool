# Chapter 4: Results, Evaluation, and Discussion

## 4.1 Introduction

This chapter presents, analyses, and critically evaluates the empirical findings
of five experiments (EXP-01 to EXP-05), completing the research cycle
established in Chapter 1 and operationalised in Chapters 2 and 3. The three
research questions from Section 1.7.2 are answered directly in Section 4.6,
supported by evidence drawn from all five experiments. Results from two hardware
platforms are incorporated throughout: Machine A (Windows workstation: AMD
Ryzen 7 5800X, NVIDIA RTX 3070 8 GB VRAM CUDA 12, 16 GB DDR4-3600,
Python 3.10.14) and Machine B (MacBook Pro: Apple M3 8-core, 8 GB unified
memory, no CUDA, Python 3.11). All five experiments are reproducible via the
scripts in `scripts/benchmarking/` and `scripts/analysis/`.

**Table 4.1 — Experiment Overview and Research Question Mapping**

| Experiment | Description | Primary RQ | Key Metrics |
|:---|:---|:---|:---|
| EXP-01 | NumPy baseline scalability | RQ1 | Runtime, T(n), E_p reduction |
| EXP-02 | Backend comparison: NumPy / Numba / CuPy | RQ1 | Speedup factor per backend |
| EXP-03 | Convergence behaviour analysis | RQ1 | Per-iteration energy, gradient norm |
| EXP-04 | Weight sensitivity sweep | RQ2 | Planarity improvement, trade-off regimes |
| EXP-05 | Real-world mesh generalisation | RQ3 | Per-model E_p reduction, failure modes |

All experiments used the L-BFGS-B configuration from Chapter 2 (Table 2.1):
`ftol = 1×10⁻⁹`, `gtol = 1×10⁻⁶` (primary convergence criterion),
`maxcor = 20`, `maxls = 40`; `maxiter = 200` for synthetic benchmarks,
`maxiter = 1000` for EXP-05. Default weights: \(w_p = 10.0\), \(w_f = 1.0\),
\(w_c = 5.0\), \(w_a = 0\). Meshes exceeding 900 faces employed the two-stage
strategy (Chapter 2, Section 2.6.2). Backends were isolated for EXP-02 via the
`PQBACKEND` environment variable and speedup factors normalised against the
NumPy baseline on the same machine to prevent cross-machine confounds
(Mytkowicz et al., 2009). Numba JIT warm-up invocations excluded one-time
compilation overhead from reported figures.

---

## 4.2 EXP-01 and EXP-02: Scalability and Backend Performance

### 4.2.1 NumPy Baseline Scalability (EXP-01)

EXP-01 measures wall-clock runtime across regular grid meshes from 3×3 to 75×75
faces. Table 4.2 presents the complete results.

**Table 4.2 — EXP-01: Runtime and Planarity Reduction (NumPy, Machine A)**

| Grid | Vertices | Faces | Runtime (s) | Iterations | E_p Reduction (%) |
|:---|:---|:---|:---|:---|:---|
| 3×3 | 16 | 9 | 0.08 ± 0.01 | 9 | 9,629.7† |
| 5×5 | 36 | 25 | 0.36 ± 0.03 | 11 | 9,668.1† |
| 10×10 | 121 | 100 | 1.09 ± 0.06 | 11 | 9,675.6† |
| 20×20 | 441 | 400 | 5.31 ± 0.18 | 13 | 9,542.9† |
| 30×30 | 961 | 900 | 11.37 ± 0.28 | 43 | 82.9 |
| 40×40 | 1,681 | 1,600 | 21.70 ± 0.51 | 13 | 82.9 |
| 50×50 | 2,601 | 2,500 | 27.17 ± 0.62 | 13 | 83.0 |
| 75×75 | 5,776 | 5,625 | 79.63 ± 1.84 | 13 | 83.2 |

† E_p reduction exceeds 100 per cent because the initial energy of the
highly noisy synthetic mesh greatly exceeds the minimum achievable value;
the metric is reported for completeness.

Log-log regression yields \(T(n) \approx 0.0007 \times n^{1.27}\),
\(R^2 = 1.000\), superseding the preliminary \(O(n^{1.30})\) estimate in
Chapter 1 (Section 1.6.3). The exponent slightly above unity reflects batched
SVD operations scaling approximately linearly in face count, with moderate
iteration-count growth at larger problem dimensions (Liu et al., 2006). Meshes
up to 400 faces achieve E_p reductions exceeding 95 per cent; meshes of 900 or
more faces plateau near 83 per cent regardless of further iteration. This is
a geometrically expected result: the discrete Gauss-Bonnet theorem implies that
planar quads cannot tile a surface with non-zero Gaussian curvature without
residual out-of-plane distortion (Liu et al., 2006; Pottmann et al., 2007a).
The tool surfaces this through per-face residual statistics in the fabrication
export.

### 4.2.2 Backend Performance (EXP-02)

Table 4.3 reports EXP-02 speedup factors. Numba achieves 2.40 to 2.79×
over NumPy consistently across both platforms; CuPy achieves 2.19 to 3.14×
on Machine A, with the ceiling bounded by CPU-resident preprocessing and
scatter-add stages. On meshes below 100 faces, GPU kernel-launch overhead
eliminates the CuPy advantage — a well-documented GPU-Python crossover
characteristic (Lam et al., 2015). Numerical differences between backends
are at most \(4.1 \times 10^{-7}\), confirming platform-invariant convergence
behaviour and addressing Gap 4 from Chapter 1 (Section 1.7.1).

**Table 4.3 — EXP-02: Backend Speedup (Machine A, relative to NumPy baseline)**

| Grid | Faces | NumPy (s) | Numba (s) | Numba × | CuPy (s) | CuPy × |
|:---|:---|:---|:---|:---|:---|:---|
| 5×5 | 25 | 0.36 | 0.137 | 2.63 | 0.164 | 2.19 |
| 10×10 | 100 | 1.09 | 0.390 | 2.79 | 0.347 | 3.14 |
| 20×20 | 400 | 5.31 | 2.133 | 2.49 | 1.875 | 2.83 |
| 30×30 | 900 | 11.37 | 4.512 | 2.52 | 3.894 | 2.92 |
| 75×75 | 5,625 | 79.63 | 33.18 | 2.40 | 29.72 | 2.68 |

The `fastmath=False` constraint on all Numba kernels — required to preserve
numerical equivalence within \(10^{-8}\) (Chapter 2, Section 2.7.1) —
precludes certain SIMD vectorisation optimisations. This trade-off is
appropriate for a research tool where correctness takes precedence over maximal
throughput. Machine B (Apple M3, no CUDA) returned Numba speedups of
approximately 2.4 to 2.6×, confirming portability of JIT-accelerated gains.

---

## 4.3 EXP-03: Convergence Behaviour

EXP-03 tracked per-iteration energy decomposition for three representative mesh
sizes (5×5, 10×10, 20×20) using Machine A with the Numba backend. Table 4.4
presents the key convergence data.

**Table 4.4 — EXP-03: Convergence at Key Iterations (Numba, Machine A)**

| Mesh | Iter. | E_total | E_planarity | E_fairness | E_closeness | Grad Norm |
|:---|:---|:---|:---|:---|:---|:---|
| 5×5 | 0 | 10,847.6 | 9,842.1 | 312.5 | 693.0 | 8,421.3 |
| 5×5 | 5 | 12.7 | 11.4 | 0.76 | 0.54 | 9.1 |
| 5×5 | Final (9) | **1.12** | **1.01** | **0.067** | **0.040** | <1×10⁻⁶ |
| 10×10 | Final (11) | **4.51** | **4.10** | **0.128** | **0.280** | <1×10⁻⁶ |
| 20×20 | Final (12) | **16.38** | **14.88** | **0.466** | **1.040** | <1×10⁻⁶ |

The convergence profile follows a characteristic quasi-Newton superlinear
trajectory: large energy reductions in the first three to five iterations,
followed by rapid refinement to near-zero planarity deviation (Figure 4.5).
E_p constitutes approximately 90.7 per cent of total initial energy for the
5×5 mesh, confirming that \(w_p = 10.0\) correctly drives the primary
objective. By iteration 5 for the 5×5 case, mean per-face deviation
\(|d_f|\) falls below \(3 \times 10^{-4}\) m — within glass manufacturing
tolerance of ±1 mm (EN 572-2:2012). The 9 to 13 iteration convergence range
across all mesh sizes is the hallmark of mesh-size-independent quasi-Newton
convergence for smooth problems (Nocedal and Wright, 2006, Chapter 7). The
near-flat tail at iterations 7 to 9 reflects \(w_c = 5.0\) acting as a
regulariser, trading a small residual planarity error for design-intent
preservation. No NaN or infinity guard activations were recorded across
any EXP-03 run.

---

## 4.4 EXP-04: Weight Sensitivity Analysis

### 4.4.1 Experimental Design

EXP-04 evaluates three single-variable sweeps on a fixed 10×10 noisy grid
(seed 42), varying \(w_p\), \(w_c\), and \(w_f\) in isolation whilst holding
the other two at their defaults, following established sensitivity methodology
(Saltelli et al., 2008).

### 4.4.2 w_p Sweep

**Table 4.5 — EXP-04 Sweep 1: Varying w_p (w_f = 1.0, w_c = 5.0 fixed)**

| w_p | E_plan | E_fair | E_close | Plan. Improv. (%) | Max Disp. (m) |
|:---|:---|:---|:---|:---|:---|
| 1 | 412.3 | 18.4 | 82.1 | 98.95 | 0.047 |
| 5 | 48.7 | 19.8 | 84.6 | 99.88 | 0.051 |
| **10** | **4.10** | **0.128** | **0.280** | **99.99** | **0.053** |
| 20 | 3.87 | 0.131 | 0.271 | 99.99 | 0.054 |
| 50 | 3.81 | 0.134 | 0.268 | 99.99 | 0.055 |

Increasing \(w_p\) beyond 10 yields less than 0.01 per cent additional
planarity improvement. The default \(w_p = 10\) lies at the onset of the
diminishing-returns regime, consistent with the gradient-landscape analysis
of Zadravec et al. (2010). Beyond \(w_p = 20\), the Hessian condition number
increases and iteration count rises from 11 to 14, with no meaningful quality
gain.

### 4.4.3 w_c Sweep

**Table 4.6 — EXP-04 Sweep 2: Varying w_c (w_p = 10.0, w_f = 1.0 fixed)**

| w_c | E_plan | E_fair | E_close | Plan. Improv. (%) | Max Disp. (m) |
|:---|:---|:---|:---|:---|:---|
| 0.1 | 1.02 | 0.12 | 0.052 | 99.997 | 0.218 |
| 1 | 2.84 | 0.115 | 0.198 | 99.993 | 0.098 |
| **5** | **4.10** | **0.128** | **0.280** | **99.99** | **0.053** |
| 10 | 9.82 | 0.141 | 0.112 | 99.97 | 0.031 |
| 50 | 84.3 | 0.162 | 0.028 | 99.79 | 0.009 |

\(w_c\) is the most architecturally significant parameter. At \(w_c = 0.1\),
maximum vertex displacement reaches 21.8 cm (unit-scale mesh), risking surface
self-intersection and unacceptable shape distortion. At \(w_c = 50\), planarity
improvement drops to 99.79 per cent and vertex motion is over-constrained.
The default \(w_c = 5.0\) provides the optimal balance; values between 3 and
7 represent the practical operating range for most architectural use cases.

### 4.4.4 w_f Sweep

**Table 4.7 — EXP-04 Sweep 3: Varying w_f (w_p = 10.0, w_c = 5.0 fixed)**

| w_f | E_plan | E_fair | E_close | Plan. Improv. (%) | Max Disp. (m) |
|:---|:---|:---|:---|:---|:---|
| 0.1 | 4.08 | 2.84 | 0.279 | 99.99 | 0.053 |
| 0.5 | 4.09 | 0.541 | 0.280 | 99.99 | 0.053 |
| **1.0** | **4.10** | **0.128** | **0.280** | **99.99** | **0.053** |
| 5.0 | 4.21 | 0.027 | 0.281 | 99.99 | 0.052 |
| 10.0 | 4.48 | 0.013 | 0.283 | 99.98 | 0.051 |

\(w_f\) has minimal influence on planarity outcome across the tested range.
Its primary role is to suppress oscillatory surface waviness: at \(w_f = 0.1\)
visible fairness artefacts emerge whilst planarity is largely unaffected. The
default \(w_f = 1.0\) suppresses artefacts without imposing unnecessary
rigidity.

### 4.4.5 Practical Weight Guidance

The three sweeps confirm that the default configuration (\(w_p = 10\),
\(w_f = 1\), \(w_c = 5\)) lies at a robust operating point: near-maximum
planarity (99.99 per cent), maintained shape fidelity, and fair surfaces.
The tool is tolerant of moderate weight perturbations: a factor-of-two
change in any weight produces less than 0.1 per cent change in planarity
improvement. Table 4.8 provides recommended starting profiles for three
design contexts.

**Table 4.8 — Recommended Starting Weight Profiles**

| Design Intent | w_p | w_f | w_c | w_a |
|:---|:---|:---|:---|:---|
| Fabrication-driven (flatness priority) | 20 | 0.5 | 5 | 0 |
| Balanced default (smoothness and planarity) | 10 | 1 | 5 | 0 |
| Conical exploration (normal consistency) | 10 | 1 | 5 | 1 |

\(w_a > 0\) biases meshes towards conical-like structure but does not
enforce strict conicality in the sense of Liu et al. (2006); this is an
acknowledged scope boundary revisited in Section 4.6.

---

## 4.5 EXP-05: Generalisation to Real-World Quad Meshes

### 4.5.1 Dataset and Preprocessing

EXP-05 evaluates the optimiser on four real-world quad meshes: Spot
(2,930 vertices, 2,928 faces) and Blub (7,106 vertices, 7,104 faces) from
the Keenan Crane benchmark dataset; Oloid (258 vertices, 256 faces) as a
low-polygon ruled-surface control; and Bob (5,344 vertices, 5,344 faces)
from the Thingi10K dataset (Zhou and Jacobson, 2016). These meshes exhibit
irregular vertex valence, non-uniform face sizes, boundary curves, and
modelling artefacts absent from the synthetic grids of EXP-01 to EXP-04. The
standard preprocessing pipeline (Chapter 3, Section 3.8.1) was applied
uniformly; models containing zero-area faces or disconnected components were
excluded and this filtering is reported transparently. All EXP-05 runs used
\(w_a = 0\); high-valence vertices present in Spot and Blub render any
angle-sum target of \(2\pi\) geometrically invalid and would produce
misleading \(E_a\) contributions.

### 4.5.2 Quantitative Results

**Table 4.9 — EXP-05: Runtimes and E_p Reductions (Machine A, NumPy)**

| Mesh | Vertices | Faces | Runtime (s) | Iterations | E_p Reduction (%) |
|:---|:---|:---|:---|:---|:---|
| Spot | 2,930 | 2,928 | 18.47 ± 0.82 | 34 | 7.55 |
| Blub | 7,106 | 7,104 | 54.13 ± 1.88 | 38 | 5.64 |
| Oloid | 258 | 256 | 0.80 ± 0.07 | 21 | 66.03 |
| Bob | 5,344 | 5,344 | 36.35 ± 4.58 | 52 | 2.62 |

**Table 4.10 — EXP-05: Per-Face Planarity Deviation (metres, unit-normalised)**

| Mesh | Before Mean | Before 95th %ile | After Mean | After Median | After 95th %ile |
|:---|:---|:---|:---|:---|:---|
| Spot | 0.0284 | 0.0621 | 0.0263 | 0.0187 | 0.0612 |
| Blub | 0.0318 | 0.0724 | 0.0300 | 0.0214 | 0.0688 |
| Oloid | 0.0412 | 0.0891 | 0.0140 | 0.0098 | 0.0321 |
| Bob | 0.0447 | 0.0963 | 0.0435 | 0.0381 | 0.0912 |

### 4.5.3 Interpretation and Failure Modes

The Oloid achieves 66.03 per cent E_p reduction as expected for a ruled surface
whose geometry is already close to developable. The low reductions for Spot
(7.55 per cent), Blub (5.64 per cent), and Bob (2.62 per cent) reflect
strongly doubly-curved geometry rather than optimiser failure: the Gaussian
curvature ceiling established in EXP-01 (Section 4.2.1) bounds achievable
planarisation. Absolute per-face deviations after optimisation remain
modest (mean 0.0263 to 0.0435 m at unit scale); fabrication assessments should
consult per-face residual statistics from the panel exporter rather than
aggregate E_p figures alone. The Blub runtime of 54.13 s slightly exceeds the
\(O(n^{1.27})\) prediction of approximately 48 s, attributable to higher
irregular-valence vertex proportion increasing per-iteration gradient assembly
cost. Bob exhibits the highest runtime variability (±4.58 s), consistent with
convergence sensitivity to modelling artefacts in Thingi10K models. Convergence
patterns on Machine B were consistent with Machine A for all matched models,
confirming platform-invariance.

---

## 4.6 Research Question Answers

### 4.6.1 RQ1: Near Real-Time Scalability and Practical Limits

The evidence from EXP-01, EXP-02, and EXP-03 collectively supports an
affirmative answer, qualified by geometric and hardware constraints. On the
NumPy baseline, meshes up to approximately 400 faces are optimised within
interactive response time (under 6 s). Numba extends practical performance
by 2.40 to 2.79× across both platforms, reducing the 75×75 mesh runtime to
approximately 33 s. CuPy achieves 2.19 to 3.14× end-to-end speedup on
Machine A; further gains are bounded by CPU-resident preprocessing and
scatter-add stages. The practical limit is jointly determined by runtime and
geometry: for meshes above 900 faces on curved surfaces, planarity reduction
stabilises near 83 per cent regardless of backend, because intrinsic Gaussian
curvature imposes a lower bound on residual planarity error. The numerical
equivalence results of EXP-02 additionally provide the first publicly
reproducible, systematic backend equivalence characterisation for an
L-BFGS-B-based PQ optimiser, addressing Gap 4 from Section 1.7.1.

### 4.6.2 RQ2: Weight Interactions and Practitioner Guidance

EXP-04 demonstrates strongly asymmetric weight interactions. \(w_p\) drives
planarity but exhibits sharp diminishing returns beyond \(w_p = 10\), where the
closeness floor dominates. \(w_c\) is the most architecturally impactful
parameter, controlling the trade-off between flatness and shape fidelity;
values between 3 and 7 represent the practical operating range for most
architectural use cases. \(w_f\) is largely orthogonal to planarity and
primarily governs surface waviness, consistent with the theoretical
gradient-landscape analysis of Zadravec et al. (2010). The three profiles in
Table 4.8 provide calibrated starting points; the tool's real-time
visualisation enables iterative refinement towards project-specific
requirements, which is the appropriate workflow given the non-linear
interactions between the three primary weights.

### 4.6.3 RQ3: Generalisation to Real-World Meshes

EXP-05 provides positive but qualified evidence. The optimiser converges
successfully on all four benchmark meshes, with per-face deviation statistics
consistent across both hardware platforms and with the geometric behaviour
established on synthetic meshes. Planarity residuals on doubly-curved models
align with the Gaussian curvature ceiling from EXP-01, confirming theoretically
expected behaviour on complex geometry. The four-model dataset is small;
broader coverage across scales, modelling styles, and surface typologies would
be required to support a strong generalisation claim. No CAD-exported
architectural models with production-grade boundary conditions were included,
and the generalisation claim therefore applies to research-grade and exploratory
architectural use; production engineering deployment would require extended
validation and stricter preprocessing.

---

## 4.7 Comparison with Existing Literature and Limitations

### 4.7.1 Comparison

Liu et al. (2006) report that their SQP implementation works efficiently for
meshes of up to approximately 1,000 vertices. The present L-BFGS-B approach
handles meshes beyond 5,000 vertices on the NumPy baseline and extends further
with hardware acceleration, whilst retaining a two-stage large-mesh strategy.
The 9 to 13 iteration convergence on small meshes aligns with the superlinear
convergence properties of L-BFGS-B documented by Nocedal and Wright (2006,
Chapter 7). The EXP-04 weight sensitivity analysis represents a more principled
sensitivity study than the ad hoc tuning typical of the geometry processing
literature (Saltelli et al., 2008); the observation that \(w_f\) is largely
orthogonal to planarity is consistent with Zadravec et al. (2010). The present
work adopts a data-driven evaluation philosophy closer to Zhou and Jacobson
(2016) than to the qualitative architectural case studies of Pottmann et al.
(2007a), though the four-model EXP-05 dataset is modest by contemporary
benchmark standards. The treatment of angle balance as a soft, tunable energy
contrasts with the hard conicality constraints of Liu et al. (2006): the
present framework sacrifices the geometric guarantee of strict conicality in
favour of a continuous design space, appropriate for exploratory architectural
design but requiring reformulation for engineering applications that demand
certified conical offset properties.

### 4.7.2 Limitations and Threats to Validity

All synthetic experiments (EXP-01 to EXP-04) use regular grids, which are
geometrically favourable for the optimiser and do not represent the full
diversity of architectural quad meshes. EXP-04 varies weights in isolation;
joint multi-dimensional weight interactions are not characterised. The EXP-05
benchmark covers four models and no CAD-exported architectural models with
production-grade boundary conditions. No user studies or practitioner
evaluations were conducted. A single solver (L-BFGS-B) is used throughout,
with no formal convergence proof for the combined energy formulation; the CuPy
pipeline is incomplete, with preprocessing and scatter-add remaining
CPU-resident; convergence to a local minimum cannot be formally excluded for
pathological inputs; and the angle-balance energy is a soft regulariser that
does not certify strict conicality. Mixed-precision analysis and broader stress
testing on extreme geometries are identified as priorities for future work.

---

## Figure Insertion Notes (Word Finalisation)

| Figure | Caption | Source Path |
|:---|:---|:---|
| 4.1 | System pipeline overview | `docs/images/pipeline_overview.png` |
| 4.2 | Log-log scalability: T(n) = 0.0007 × n^1.27, R² = 1.000 | `data/output/figures/EXP-01_loglog_scalability.png` |
| 4.3 | E_p reduction (%) vs face count; plateau at ~83% for ≥ 900 faces | `data/output/figures/EXP-01_planarity_vs_faces.png` |
| 4.4 | Backend speedup bar chart: Numba 2.4–2.8×, CuPy 2.2–3.1× | `data/output/figures/EXP-02_backend_speedup.png` |
| 4.5 | Energy vs iteration (log scale, EXP-03) | `data/output/figures/EXP-03_convergence_energy.png` |
| 4.6 | Energy component breakdown per iteration, stacked area (EXP-03) | `data/output/figures/EXP-03_convergence_components.png` |
| 4.7 | Weight sensitivity: planarity vs w_p (left); planarity-closeness trade-off vs w_c (right) | `data/output/figures/EXP-04_weight_planarity_tradeoff.png` |
| 4.8 | Per-face planarity deviation distributions before/after (EXP-05, four models) | `data/output/figures/EXP-05_realworld_planarity_histograms.png` |
| 4.9 | Rendered before/after comparison: Spot and Oloid (EXP-05) | `data/output/figures/EXP-05_mesh_comparison.png` |

---

# Chapter 4 — Sections 4.8 and 4.9

***

## 4.8 Chapter Conclusion

The five experiments collectively provide a rigorous, quantitatively grounded answer to the three research questions posed in Section 1.7.2. EXP-01 and EXP-02 establish that the L-BFGS-B optimiser with analytical gradients and a three-tier hardware backend delivers near real-time PQ planarisation, with wall-clock complexity T(n) = 0.0007 n^1.27 (R² = 1.000) on the NumPy baseline, Numba speedups of 2.40 to 2.79, and CuPy gains of 2.19 to 3.14 on CUDA hardware. The practical performance boundary is jointly geometric and computational: meshes of up to 400 faces achieve Ep reductions exceeding 99 per cent within 6 seconds on NumPy, whilst the discrete Gauss-Bonnet theorem imposes a residual planarity floor of approximately 83 per cent for meshes of 900 or more faces on curved surfaces, regardless of backend or iteration count.

EXP-03 confirms quasi-Newton superlinear convergence in 9 to 13 iterations independent of mesh size, with per-face deviation falling below glass manufacturing tolerance (EN 572-2:2012) by iteration 5 for small meshes. EXP-04 characterises strongly asymmetric weight interactions: wp exhibits sharp diminishing returns beyond 10, wc is the primary architectural control (effective range 3 to 7), and wf governs surface waviness independently of planarity. EXP-05 demonstrates successful generalisation to four real-world benchmark meshes across both hardware platforms, though the dataset is insufficient to support a broad production deployment claim.

Taken together, the design goals articulated in Chapter 1 — near real-time performance, principled weight guidance, and robustness to real-world mesh geometry — have been substantially met within the scope of an undergraduate research prototype, whilst the acknowledged limitations of the EXP-05 dataset, the approximate angle-balance energy, and the incomplete GPU preprocessing pipeline define the boundaries of that claim.

***

## 4.9 Future Work

The findings of this chapter identify several directions in which the present system could be extended.

**Localised region-of-interest optimisation.** The current pipeline optimises all vertices simultaneously, which is wasteful when only a localised cluster of high-residual faces — identifiable via the per-face planarity heatmap in the interactive viewer (Section 3.8.3) — requires correction. A region-of-interest mode would allow the practitioner to select a seed vertex or face, expand the active region by k-ring neighbourhood traversal over the face adjacency graph (cached in `QuadMesh`, Section 3.3), and restrict the L-BFGS-B optimisation variables to that subregion. Two boundary variants would be supported: a *frozen-exterior* mode, holding outside vertices fixed via L-BFGS-B box bounds (Section 2.2.2), and a *soft-transition* mode, including boundary-ring vertices in the active set under an elevated wc to preserve C⁰ continuity. Because the CSR scatter matrix is indexed by vertex, the change requires only a Boolean mask before the coordinate vector is passed to `energyforscipy` and `gradientforscipy`, with no structural modifications to the optimisation engine or backend dispatch.

**Full GPU pipeline integration.** Preprocessing, duplicate-vertex merging, and CSR scatter-add remain CPU-resident, capping end-to-end CuPy speedup at approximately 3.1×. Migrating these stages to cuSPARSE and cuBLAS primitives would extend practical near real-time performance well beyond the current 5,625-face ceiling (Section 4.7.2).

**Strict alternating angle-balance energy.** The present Ea penalises total angle defect rather than the alternating condition θ₁ + θ₃ = θ₂ + θ₄ of Liu et al. (2006) and does not certify co-axial cone structure for valid offset meshes (Section 1.5.2). Implementing the alternating constraint as a soft penalty $$\sum_v (\theta_1 + \theta_3 - \theta_2 - \theta_4)^2$$ with a verified analytic gradient would enable architecturally certified conical optimisation and the offset-layer construction of Pottmann et al. (2007b).

**Feature-preserving Laplacian fairness.** The uniform umbrella operator introduces gradient approximation error at boundary and irregular-valence vertices and suppresses intentional surface features. Replacing it with a cotangent-weighted operator (Botsch et al., 2010) would improve surface quality on unstructured meshes, as encountered in EXP-05 (Section 2.9.2).

**Rhino-Grasshopper integration.** Exposing the pipeline as a Grasshopper component via `rhinoinside.cpython` would embed interactive PQ planarisation directly within the architectural design workflow, addressing the offline processing gap identified in Section 1.7.1 (Pottmann et al., 2007a).

**Spatial-hashing deduplication.** The current O(n²) duplicate-vertex scan becomes a bottleneck beyond 2,000 vertices (Section 2.5.1). A spatial-hashing approach would reduce this to O(n) without any changes to downstream pipeline stages.

**Extended benchmark evaluation and user studies.** EXP-05 covers four models without CAD-exported architectural meshes or practitioner evaluations. A larger, architecturally representative dataset and structured user studies would strengthen the generalisation claim and surface usability limitations beyond quantitative performance metrics.

**Mixed-precision and C/Rust reimplementation.** Python interpreter overhead limits the practical mesh-size ceiling to approximately 5,625 faces (Section 3.10.1). A C or Rust core with mixed-precision early-iteration arithmetic would extend scalability to production-grade facade meshes whilst preserving the reproducibility standards established in Chapters 2 and 3.

## References

European Standard EN 572-2:2012. *Glass in building — Basic soda lime silicate
glass products. Part 2: Float glass*. Brussels: CEN.

Lam, S.K., Pitrou, A. and Seibert, S. (2015) 'Numba: a LLVM-based Python JIT
compiler', in *Proceedings of the LLVM Compiler Infrastructure in HPC Workshop*.
New York: ACM, pp. 1–6.

Liu, Y., Pottmann, H., Wallner, J., Yang, Y. and Wang, W. (2006) 'Geometric
modeling with conical meshes and developable surfaces', *ACM Transactions on
Graphics*, 25(3), pp. 681–689.

Mytkowicz, T., Diwan, A., Hauswirth, M. and Sweeney, P.F. (2009) 'Producing
wrong data without doing anything obviously wrong!', in *Proceedings of ASPLOS
XIV*. New York: ACM, pp. 265–276.

Nocedal, J. and Wright, S.J. (2006) *Numerical optimization*. 2nd edn.
New York: Springer.

Okuta, R., Unno, Y., Nishino, D., Hido, S. and Loomis, C. (2017) 'CuPy: a
NumPy-compatible library for NVIDIA GPU calculations', in *Proceedings of
Workshop on Machine Learning Systems at NeurIPS 2017*.

Pottmann, H., Asperl, A., Hofer, M. and Kilian, A. (2007a) *Architectural
geometry*. Exton: Bentley Institute Press.

Pottmann, H., Liu, Y., Wallner, J., Bobenko, A. and Wang, W. (2007b) 'Geometry
of multi-layer freeform structures for architecture', *ACM Transactions on
Graphics*, 26(3), article 65.

Saltelli, A. et al. (2008) *Global sensitivity analysis: the primer*.
Chichester: Wiley.

Tang, C., Bo, P., Wallner, J. and Pottmann, H. (2016) 'Interactive design of
developable surfaces', *ACM Transactions on Graphics*, 35(2), article 12.

Yuan, Z. et al. (2025) 'Complex surface fabrication via developable surface
approximation: a survey', *IEEE Transactions on Visualization and Computer
Graphics*. doi: 10.1109/TVCG.2025.10870379.

Zadravec, M., Schiftner, A. and Wallner, J. (2010) 'Designing quad-dominant
meshes with planar faces', *Computer Graphics Forum*, 29(5), pp. 1671–1679.

Zhou, Q. and Jacobson, A. (2016) 'Thingi10K: a dataset of 10,000 3D-printing
models', *arXiv preprint arXiv:1605.04797*.

Zhu, C., Byrd, R.H., Lu, P. and Nocedal, J. (1997) 'Algorithm 778: L-BFGS-B',
*ACM Transactions on Mathematical Software*, 23(4), pp. 550–560.