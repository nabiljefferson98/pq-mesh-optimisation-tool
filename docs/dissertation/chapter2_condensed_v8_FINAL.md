# Chapter 2: Methodology

## 2.1 Overview and Scope

This chapter formalises the methodological approach adopted to address the PQ mesh
optimisation problem established in Chapter 1. Its purpose is distinct from that of Chapter 3,
which documents the software architecture and implementation details: the present chapter
presents the abstract mathematical and algorithmic design, justifies each decision with
reference to established literature and project requirements, and defines the quality standard
governing the experimental evaluation in Chapter 4.

---

## 2.2 Problem Formulation and Data Representation

### 2.2.1 Discrete Mesh Representation

The input to the optimisation pipeline is the quad mesh $\mathcal{M} = (V, F)$ defined in
Section 1.3.1, where $V \in \mathbb{R}^{n \times 3}$ is the array of $n$ vertex positions and
$F \subseteq \{0, \ldots, n-1\}^4$ encodes face connectivity. The quadrilateral face structure
is a precondition of the fabrication workflow: each face corresponds to a manufacturable flat
panel and the planarity energy is derived specifically for a quad (Section 1.2.2). Manifoldness
and the absence of self-intersections are assumed at the modelling stage, and practical
violations are addressed by the pre-processing pipeline in Section 2.5.

### 2.2.2 Data Structures and Optimisation Variables

The `QuadMesh` class stores vertex positions as a floating-point matrix of shape $(n, 3)$,
face connectivity as an integer array of shape $(|F|, 4)$, lazily cached edge-adjacency
structures, and a precomputed CSR scatter matrix $S$ of shape $(n, |F| \times 4)$ that enables
BLAS-backed gradient accumulation two-to-four times faster than `numpy.add.at`; full
construction details are given in Chapter 3 (Section 3.2). The optimisation variables are
the flattened vertex coordinates $\mathbf{x} = \text{vec}(V) \in \mathbb{R}^{3n}$, reshaped
to $(n, 3)$ at each function evaluation. Optional box constraints restrict each coordinate to
a prescribed interval around its reference position and are passed directly to L-BFGS-B,
consistent with Zhu et al. (1997). Vertex-position variables are preferred over conjugate
direction fields or edge-length representations because they provide direct geometric feedback,
simplify analytic gradient derivation, and export to fabrication formats without any
additional projection steps.

---

## 2.3 Energy-Based Objective Function

The total energy:

$$E_{\text{total}} = w_p E_p + w_f E_f + w_c E_c + w_a E_a$$

was introduced in Section 1.2.2, and its four constituent terms were defined in Sections 1.3.2,
1.3.3, and 1.5.2. The current section focuses on the methodological architecture of the
weighted sum, the design rationale for each term, and the weight-configuration strategy that
frames the experimental design of Chapter 4.

### 2.3.1 Weight Architecture and Trade-Off Design

Each scalar weight $w_p, w_f, w_c, w_a \geq 0$ is tuneable at runtime and controls one
dimension of the four-way trade-off between fabrication feasibility, surface regularity,
design fidelity, and conical-like vertex structure. This architecture generalises the
formulation of Liu et al. (2006), in which conicality is either enforced as a hard SQP
equality constraint or entirely absent, with no continuous intermediate available to a
practitioner wishing to blend PQ planarity and conical structure simultaneously. The ability
to assign any non-negative value to $w_a$ without modifying the algorithm or data structures
is a specific contribution of the current project, identified in Section 1.7.1.

From a multi-objective perspective, the four weights parameterise a trajectory across
multi-dimensional trade-off surfaces. The weight-sensitivity experiment in Chapter 4 (EXP-04)
surveys this surface across fifteen configurations (three single-variable sweeps of five values
each) to characterise these trade-offs quantitatively.

### 2.3.2 Planarity Energy: Methodological Justification

Two methodological consequences of the SVD planarity formulation, established in Section 1.3.2,
merit emphasis. Firstly, the sum-of-squares form yields a globally smooth objective with a
continuous analytic gradient at every vertex configuration, which is a prerequisite for the
stable inverse-Hessian approximation maintained by L-BFGS-B (Nocedal and Wright, 2006).
Secondly, by driving the SVD residuals of every face towards zero, the planarity energy
enforces discrete developability at the strip level without requiring an explicit ruling
representation: each row of planar quad faces constitutes a discrete developable strip, with
column edges serving as the rulings of the strip's tangent developable surface (Section 1.4.1).
This level of developability is adequate for panel fabrication; full isometric flattenability
would require additional intrinsic metric constraints beyond the current scope (Tang et al.,
2016).

Throughout this dissertation, the quantity $d_{i,f}$ — the signed perpendicular distance of
vertex $i$ of the face $f$ from the SVD best-fit plane of that face — is referred to as the
**planarity deviation**, and its root-mean-square across all face vertices is the primary
fabrication quality metric reported in Chapter 4.

### 2.3.3 Fairness, Closeness, and Angle-Balance: Key Design Decisions

Two methodological decisions for $E_f$ and $E_c$ require emphasis beyond what is stated in
Chapter 1. Firstly, reference positions for $E_c$ are the post-normalisation vertex
coordinates, not the raw CAD input: anchoring the closeness energy to raw coordinates at
typical CAD scale factors causes $E_c$ to dominate the line search (Section 2.5.2). Secondly,
the uniform umbrella operator in $E_f$ penalises curvature variation uniformly; for surfaces
with intentional sharp features, $w_f$ requires careful calibration to avoid suppressing them,
and feature-preserving alternatives (Botsch et al., 2010) are identified as future work.

The critical methodological decision for $E_a$ is the treatment of $w_a$ as a first-class,
runtime-tuneable parameter throughout the entire codebase, even though its default value is
zero in all PQ-only experiments. Activating joint PQ and conical optimisation requires only
an interactive weight adjustment, without modifying the algorithm or data structures, which
directly supports the design-exploration workflow of Section 1.2.2. As established in Section
1.5.2, satisfying $E_a = 0$ does not imply the alternating balance condition
$\alpha_1 + \alpha_3 = \alpha_2 + \alpha_4$ required for exact conicality (Liu et al., 2006).
This limitation is a deliberate scope boundary and is critically evaluated in Section 2.9.2.

---

## 2.4 Analytical Gradient Derivation and Verification

Analytic gradients are essential because central finite differences would require approximately
34,656 energy evaluations per gradient step for the largest benchmarked mesh of $(n = 5,776)$
vertices, rendering near real-time performance infeasible and introducing SVD-driven numerical
noise into the L-BFGS-B inverse-Hessian approximation (Nocedal and Wright, 2006).

### 2.4.1 Planarity Gradient and Scatter-Add Assembly

Let $\mathbf{n}_f \in \mathbb{R}^3$ denote the unit normal of face $f$, defined as the last
right singular vector of the centred vertex matrix $M_f \in \mathbb{R}^{4 \times 3}$ — the
eigenvector corresponding to the smallest singular value of $M_f$. The vector points
perpendicular to the SVD best-fit plane of face $f$ and is well-defined whenever the centred
vertex matrix has a unique smallest singular value, which holds for all non-collinear quads;
the degenerate case is removed by the pre-processing pipeline (Section 2.5.1).

Differentiating $E_p = \sum_f \sum_{i=1}^4 d_{i,f}^2$, where
$d_{i,f} = (v_i - c_f) \cdot \mathbf{n}_f$ is the planarity deviation of vertex $i$ on face
$f$ and $c_f = \frac{1}{4}\sum_{i \in f} v_i$, and treating $\mathbf{n}_f$ as fixed at the
current iterate, yields:

$$\frac{\partial E_p}{\partial v_i} = 2 \sum_{f \ni i} d_{i,f} \cdot \mathbf{n}_f - \frac{1}{4} \sum_{j \in f} d_{j,f} \cdot \mathbf{n}_f$$

The centroid-adjustment term cancels across all four vertices of each face because
$\sum_{i \in f} d_{i,f} = 0$ by construction, and the first-order stationarity condition for
$\mathbf{n}_f$ eliminates the SVD derivative contribution. The gradient assembly exploits the
precomputed scatter matrix:

$$\nabla_V E_p = S \cdot g_{fc}$$

where $g_{fc} \in \mathbb{R}^{(|F| \times 4) \times 3}$ holds per-face, per-vertex
contributions computed via batched SVD over the $(|F|, 4, 3)$ face-vertex tensor, accumulated
in one BLAS-backed sparse matrix multiplication. Backend-specific kernel variants and
numerical equivalence are documented in Chapter 3 (Section 3.5.2).

### 2.4.2 Supplementary Gradient Derivations

The closeness gradient $\partial E_c / \partial v_i = 2(v_i - v_{0,i})$ is an inexpensive
Euclidean difference. The fairness gradient is a linear function of the current Laplacian
displacement, which under the uniform-valence assumption holds exactly for interior vertices
of the regular meshes used in the evaluation but introduces approximation error at boundary
and irregular-valence vertices — a limitation discussed in Section 2.9.2. Full derivations
for $\partial E_f / \partial v_i$ and $\partial E_a / \partial v_i$, including the per-face
angle-contribution tensor and the `_ANGLE_SIGNS` type constraints required for Numba JIT
compilation, are provided in Appendix D.

### 2.4.3 Gradient Verification and Numerical Safeguards

All four analytic gradients are verified against central finite differences to a relative
error below $10^{-4}$ across all degrees of freedom, implemented in
`tests/test_gradients.py` and validated on flat, sinusoidally perturbed, and cylindrically
curved mesh geometries. A separate suite in `tests/test_numerical_equivalence.py` validates
Numba-versus-NumPy parity to $10^{-10}$ for meshes up to $10 \times 10$ faces and $10^{-8}$
for $20 \times 20$ meshes. Step-by-step instructions to run the gradient verification and 
numerical equivalence test suites independently are provided in 
Appendix G, Sections G.7 and G.8. The observed divergence at the $10^{-8}$ level is
well below the convergence criterion $\texttt{gtol} = 10^{-4}$ and three orders of magnitude
below the gradient verification tolerance of $10^{-4}$, and has no material effect on
convergence (Higham, 2002). Two numerical safeguards in the SciPy interface return a sentinel
energy of $10^{300}$ and zero gradient entries on NaN or infinity, which are never activated
on well-formed inputs but ensure graceful degradation on degenerate cases.

---

## 2.5 Pre-processing Pipeline

Three categories of pre-processing execute before any mesh reaches the optimiser: mesh
cleaning and degeneracy handling (Section 2.5.1), scale normalisation and reference
configuration (Section 2.5.2), and automatic weight suggestion (Section 2.5.3). The complete
pipeline is illustrated in Figure 2.1.

**Figure 2.1:** Pre-processing pipeline executed before each mesh reaches the L-BFGS-B
optimiser. Steps 1–3 (Section 2.5.1) perform mesh cleaning and degeneracy handling; Steps
4–5 (Section 2.5.2) perform scale normalisation and set reference positions $v_0$; Step 6
(Section 2.5.3) computes heuristic suggested weights. Yellow side-exits indicate recoverable
warnings; the red side-exit at Step 3 indicates a hard abort via `ValueError`; the green node
denotes the successful pipeline output.

### 2.5.1 Mesh Cleaning and Degeneracy Handling

Three cleaning steps execute before any mesh reaches the optimiser. Firstly, duplicate
vertices are merged via an $O(n^2)$ pairwise distance scan; unmerged duplicates fragment the
Laplacian adjacency graph and produce unphysical vertex movement under $E_f$. A warning is
emitted for inputs exceeding 2,000 vertices, and a spatial-hashing approach is identified as
a future efficiency improvement. Secondly, faces with zero or near-zero area (preprocessing
threshold $10^{-8}$) are removed, as near-degenerate SVD matrices produce unreliable best-fit
normals and propagate NaN through the gradient computation (Botsch et al., 2010); a secondary
validation check in `validate_mesh()` applies the tighter threshold $10^{-10}$ before
optimisation begins (Section 2.6.1). Thirdly, the `QuadMesh` constructor raises a `ValueError`
on negative face indices, preventing silent incorrect vertex lookups via NumPy's wrap-around
indexing semantics.

### 2.5.2 Scale Normalisation and Reference Configuration

The mesh is centred at the origin and uniformly scaled so that the longest bounding box axis
has unit length. This is numerically essential: a CAD model in millimetres carries
$E_c \approx 10^6$ relative to unit scale, a discrepancy of approximately ten orders of
magnitude that causes $E_c$ to dominate the total energy from the first iteration and
prevents convergence. After normalisation, reference positions $V_0$ in `vertices_original`
are set to the cleaned and normalised coordinates, ensuring that $E_c$ penalises deviation
from the pre-processed design intent rather than from raw CAD metric artefacts.

### 2.5.3 Automatic Weight Suggestion

A heuristic weight suggestion procedure in preprocessor.py targets the initial normalised
component-energy ratio $E_p : E_f : E_c \approx 10 : 1 : 5$ by evaluating the initial term
magnitudes at the starting vertex configuration and computing proportional weights, with a
guard for near-planar meshes, where $E_p$ is negligibly small (guard threshold: $E_p < 10^{-10}$).
This provides a reproducible starting point for interactive tuning rather than a guaranteed
optimum (Botsch et al., 2010). All fifteen weight configurations in Chapter 4 (EXP-04) depart from 
these suggested weights, validating their role as a practical baseline; 
the full per-configuration results are tabulated in Appendix H.

---

## 2.6 Optimisation Strategy

### 2.6.1 Algorithm Selection and Hyperparameter Configuration

The selection of L-BFGS-B over gradient descent, conjugate gradient, SQP, and augmented
Lagrangian methods was justified in Section 1.6.2 on grounds of superlinear convergence,
modest memory footprint, native box-constraint support, and robustness to ill-conditioned
objectives. Table 2.1 specifies the hyperparameter configuration for each stage of the
two-stage optimisation strategy described in Section 2.6.2.

**Table 2.1:** L-BFGS-B hyperparameter configuration by stage.

| Parameter                     | Stage 1 (Rapid Planarity)     | Stage 2 (Balanced Refinement) | Rationale                                                                                                                                                                                                                      |
|-------------------------------| ----------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ftol`                        | $10^{-7}$                     | $10^{-9}$                     | Stage 1 uses a loose energy-change criterion to terminate quickly once faces are approximately flat; Stage 2 tightens this to avoid premature termination on flat energy plateaux during fine convergence                       |
| `gtol`                        | $10^{-4}$                     | $10^{-5}$                     | Stage 1 tolerates a larger residual gradient; Stage 2 tightens to confirm genuine stationarity; the Stage 2 value is one order of magnitude below the gradient verification tolerance of $10^{-4}$ established in §2.4.3       |
| `maxcor`                      | 10                            | 20                            | Stage 1 uses a shorter history to reduce per-iteration memory cost; Stage 2 doubles the L-BFGS-B correction history for a smoother inverse-Hessian approximation during refinement                                             |
| `maxls`                       | 20                            | 40                            | Stage 1 uses the SciPy default of 20 backtracking steps; Stage 2 doubles this to prevent early line-search termination on large or nearly-converged meshes                                                                     |
| `maxiter`                     | min(200, max_iterations // 3) | remainder of max_iterations   | Stage 1 is allocated at most one-third of the total budget, capped at 200 iterations; Stage 2 receives all remaining iterations; the combined total is strictly bounded by max_iterations (default 1,000)                      |
| `stage1_planarity_multiplier` | 5.0                           | n/a                           | Planarity weight is multiplied by this factor in Stage 1; fairness and closeness are reduced to 10% of their standard values                                                                                                   |

`gtol` is the primary convergence indicator because a small energy change is indistinguishable
via `ftol` alone from a near-zero line-search step, whereas a small gradient norm provides an
unambiguous stationarity signal. The Stage 2 value of $\texttt{gtol} = 10^{-5}$ is one order of
magnitude below the $10^{-4}$ gradient verification bound established in §2.4.3, ensuring that
the solver halts only when the gradient is genuinely negligible relative to the verified precision
of the analytic derivatives. The Stage 1 value of $\texttt{gtol} = 10^{-4}$ matches the gradient
verification tolerance exactly, which is intentional: Stage 1 is considered complete as soon as
the gradient magnitude falls within the verified precision of the derivative computation, leaving
fine convergence to Stage 2.

### 2.6.2 Two-Stage Strategy and Expected Convergence

For meshes exceeding approximately 900 faces, a two-stage strategy applies an elevated $w_p$
in stage one for rapid planarity reduction, then rebalances weights in stage two to recover
fairness and closeness. Without staging, the fairness term can resist aggressive planarity
moves on high-curvature meshes, stalling the solver at a local minimum with poor surface
regularity. The principal limitation is the introduction of empirically set stage-one
hyperparameters; a fixed profile is employed for all large-mesh experiments in Chapter 4 to
maintain comparability. Both stages share the same `MeshOptimiser` and
`OptimisationConfig` objects; only the weight vector changes between stages.

Observed convergence: meshes of at most 400 faces converge in 9 to 13 iterations with
planarity reduction exceeding 99 per cent; meshes of 900 or more faces converge in 50 to 200
iterations with approximately 83 per cent reduction, consistent with the intrinsic Gaussian
curvature of the target surface imposing a geometric lower bound on planarity error (as
established in Appendix C, Section C.3).

---

## 2.7 Software Quality, Robustness, and Testing

### 2.7.1 Hardware Backend and Numerical Consistency

The three-tier hardware backend, introduced in Section 1.6.3, preserves strict
floating-point associativity across all tiers via `fastmath=False` on every Numba kernel.
Enabling fast floating-point reassociation would break scatter-add associativity, producing
gradients diverging from the NumPy reference by more than the $10^{-8}$ verification
tolerance. All Numba `try`-blocks employ `except Exception` handling rather than
`except ImportError` because Numba's `@njit` decorator raises `TypingError`,
`LoweringError`, and LLVM compilation errors on platforms where JIT compilation fails; the
narrower handler causes a `NameError` at runtime (Lam, Pitrou and Seibert, 2015). Each
`except` block emits a `warnings.warn` message, ensuring transparency. Full kernel code and
compilation flags are documented in Chapter 3 (Section 3.5.2).

### 2.7.2 Testing Infrastructure

The test suite comprises 321 tests across four categories — unit and integration, gradient
verification, numerical equivalence, and robustness and regression — with zero failures and
one skip for the GUI-dependent smoke test. Coverage stands at 79 per cent across the
complete `src/` tree (excluding `interactive_optimisation.py`), rising to 81 per cent for
the non-interactive modules alone. Continuous integration runs on Windows 11 and macOS
with Python 3.10 through 3.12, incorporating `mypy`, `bandit`, and pre-commit hooks for
style and security. Version control follows atomic feature-branch development with
pull-request-based merges. Full CI configuration and the test module inventory are detailed
in Chapter 3 (Section 3.7). The complete hardware and software environment under which all 321 
tests were validated is recorded in Appendix G, Table G.1.

---

## 2.8 Visualisation and Fabrication Export

The interactive viewer couples the optimisation engine to a Polyscope-based real-time
rendering layer: weight sliders trigger full optimisation re-runs, and per-face planarity and
per-vertex conical-defect heatmaps provide immediate fabrication-quality feedback. This
directly addresses the interactive feedback gap identified in Section 1.7.1 (Pottmann et al.,
2007a). The software architecture of the viewer is detailed in Chapter 3 (Section 3.6).

For fabrication export, a local two-dimensional frame $\{u_f, v_f\}$ is constructed in the
SVD best-fit plane of each face, and the four vertices are projected onto it; the maximum
absolute out-of-plane residual is computed and reported per panel with a warning threshold of
5 mm at model scale. Validated panels are exported to DXF for CNC workflows and SVG for
vector graphics preview using atomic writes (`tempfile.mkstemp` + `os.replace`). The
following metrics are logged per run and constitute the quantitative basis for all five
experiments in Chapter 4: $E_p$ reduction percentage, $E_f$, $E_c$ values, mean and maximum
angle defect, iteration count, and wall-clock time.

---

## 2.9 Critical Synthesis

### 2.9.1 Methodological Strengths

The principal strength is the unified four-term energy framework with exact analytic
gradients, runtime-tuneable weights, and a first-class angle-balance term accessible without
algorithm changes. The planarity gradient derivation of Section 2.4.1 is exact rather than
approximate, and systematic central-finite-difference verification to a relative error below
$10^{-4}$ addresses the reproducibility gap identified in Section 1.7.1. The pre-processing
anchored reference configuration ensures numerical conditioning at all mesh scales, and the
three-tier backend with enforced associativity delivers near real-time performance across
diverse hardware configurations without sacrificing numerical equivalence. The 321-test suite
with continuous integration provides ongoing correctness assurance across software revisions.

### 2.9.2 Methodological Limitations

Three limitations require explicit acknowledgement. Firstly, $E_a$ approximates conicality
as a total angle defect rather than the alternating balance condition
$\alpha_1 + \alpha_3 = \alpha_2 + \alpha_4$ of Liu et al. (2006): these are geometrically
distinct, and $E_a = 0$ does not guarantee the coaxial cone structure required for valid
offset meshes and orthogonal support strips in multi-layer envelopes (Section 1.5.2). Any
future application requiring offset-mesh compatibility would necessitate implementing the
alternating constraint either as a hard SQP equality or as a soft penalty
$\sum(\alpha_1 + \alpha_3 - \alpha_2 - \alpha_4)^2$ with an analytic gradient compatible
with the existing L-BFGS-B pipeline. Secondly, the uniform umbrella operator in $E_f$
introduces gradient approximation error at boundary and irregular-valence vertices;
feature-preserving Laplacians (Botsch et al., 2010) are identified as future work for
unstructured mesh workflows. Thirdly, L-BFGS-B assumes a smooth energy landscape; SVD
instabilities on near-degenerate faces in early iterations can induce effectively non-smooth
behaviour, forcing very small line-search steps that technically violate the method's
smoothness assumptions — although the NaN-guarding safeguards ensure that these paths are
never activated on any well-formed input in the experimental evaluation.

### 2.9.3 Positioning Relative to Existing Work

The current methodology is situated within, and builds directly upon, two landmark
contributions to computational architectural geometry. Liu et al. (2006) establish the
rigorous theoretical foundation for conical meshes and developable surfaces, introducing hard
SQP constraint enforcement and exact conicality conditions that continue to set the benchmark
for planarity precision in the field; the angle-balance energy term adopted in the current
work is directly informed by their geometric analysis of per-vertex angle sums. Tang et al.
(2016) make the seminal contribution of demonstrating that interactive optimisation is
achievable within a developable surface design pipeline, and it is precisely this interactive
design paradigm that motivates the current project's central emphasis on runtime-tuneable
weights and near real-time performance.

The current work extends both contributions by unifying PQ planarity, fairness, closeness,
and conicality as continuous, simultaneously tuneable energies within a hardware-accelerated
framework, augmented by exact analytic gradients, systematic gradient verification, and a
structured experimental evaluation on standardised benchmark datasets — a combination not
present in any single surveyed tool (Section 1.6). The methodological scope is appropriately bounded for an undergraduate 
research prototype, where reproducibility and developer productivity 
are prioritised over production-grade scalability; joint fabrication 
modelling and cost optimisation are identified as natural directions 
for future interdisciplinary work (Appendix I, Section I.4.3).
