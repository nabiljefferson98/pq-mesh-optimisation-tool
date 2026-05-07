# Chapter 1: Introduction and Background Research

## 1.1 Introduction

The design of contemporary freeform building envelopes presents a fundamental challenge at the
boundary of geometry, computation, and fabrication. Architects specify surfaces of arbitrary
curvature using smooth digital representations, yet the physical panels, beams, and joints
from which such envelopes are assembled impose strict constraints that smooth models do not
automatically satisfy. Among the most significant is the requirement for planar panels: glass,
structural sheet metal, and composite cladding are manufactured flat and cannot economically
be formed into double-curved shapes at the scale of individual facade elements (Pottmann et
al., 2007a). This constraint motivates the central object of study in this dissertation: the
**planar quad (PQ) mesh** — a quadrilateral mesh in three-dimensional space in which every
face lies in a single plane. PQ meshes have emerged as the canonical discrete representation
for fabrication-aware architectural geometry, and their optimisation constitutes an active area
of computational geometry research with significant practical implications.

Sections 1.2 to 1.5 introduce the motivating problem and the core geometric concepts, Section
1.6 surveys optimisation methods, and Section 1.7 identifies four gaps in the existing
literature and states the three research questions addressed in Chapter 4. The aim of this
dissertation is to design, implement and empirically evaluate a real-time PQ mesh optimisation
and visualisation tool that brings interactive geometric feedback to a class of
fabrication-aware design problems that existing approaches address only in offline,
batch-processing contexts.

---

## 1.2 Motivation: Freeform Architecture and the Fabrication Challenge

### 1.2.1 Planar Quadrilateral Panels and Fabrication Advantages

Glymph et al. (2002) argued that planar quadrilateral facets are preferable to triangular or
non-planar quadrilateral alternatives and demonstrated several simple practical strategies for
constructing such meshes. Liu et al. (2006) subsequently provided the first systematic
geometry processing treatment, as illustrated in Figure 1.1. Planar panels confer substantial
fabrication advantages: they can be cut directly from flat sheets without bending, permit
simpler support-structure detailing, and reduce production costs compared with double-curved
alternatives (Pottmann et al., 2007a). An arbitrary quadrangulation of a smooth freeform
surface will rarely yield exactly planar faces; a nonlinear optimisation step is therefore
required to achieve or approximate planarity whilst preserving the intended design form
(Alliez et al., 2003).

**Figure 1.1:** Freeform building envelopes tessellated with planar quadrilateral panels.
Reproduced from Liu et al. (2006, Fig. 1).

### 1.2.2 The Case for Near Real-Time Optimisation

Existing PQ planarisation methods are implemented predominantly as offline batch processes,
requiring the designer to submit a mesh, wait for convergence, and inspect the result (Liu et
al., 2006). This workflow is poorly suited to the exploratory nature of architectural design,
in which a practitioner may wish to adjust constraint weightings, experiment with mesh
resolution, or negotiate competing objectives with immediate visual feedback. The architectural
geometry community has articulated the need for tools integrating optimisation into the design
process (Pottmann et al., 2007a), but this need remains largely unmet by open, general-purpose
implementations.

This dissertation responds to that gap with a tool that minimises the weighted sum:

$$E_{\text{total}} = w_p E_p + w_f E_f + w_c E_c + w_a E_a$$

where the weights $w_p$, $w_f$, $w_c$, and $w_a$ are tuneable at runtime. **Near real-time
performance** is defined here as convergence ranging from sub-second for 9-face meshes to
approximately 80 seconds for 5,625-face meshes on the NumPy baseline, with planarity
reductions exceeding 99 per cent for meshes of at most 400 faces and approximately 83 per
cent for meshes of 900 or more faces. The mathematical foundations of each energy term are
established in Sections 1.3 to 1.5.

---

## 1.3 Planar Quad Meshes: Definition, Properties, and Quality Criteria

### 1.3.1 Definition and Discrete Differential Geometry Background

A planar quad mesh is a quadrilateral mesh embedded in three-dimensional Euclidean space in
which every face is contained within a single plane. In discrete differential geometry, PQ
meshes are interpreted as discrete analogues of conjugate curve networks on smooth surfaces
(do Carmo, 1976, p. 150; Liu et al., 2006). Within the quad-net framework of Bobenko and
Suris (2008), PQ meshes are compatible with the principle of discrete integrability, which
underpins the regularity of their geometric features under refinement and justifies enforcing
planarity via smooth, differentiable energies without degeneracy — a property critical for
stable gradient-based optimisation. The historical development of this framework, including
the foundational contributions of Sauer (1970) and the SIGGRAPH survey of Desbrun et al.
(2005), is detailed in Appendix C (Section C.1).

### 1.3.2 Planarity Energy: SVD Formulation

Liu et al. (2006) employ an angle-sum condition requiring interior quad angles to sum to
$2\pi$, supplemented by determinant-based coplanarity conditions for degenerate
configurations. The current project adopts an SVD-based planarity energy measuring the sum
of squared perpendicular distances of each face's vertices from their best-fit plane:

$$E_p = \sum_{f \in F} \sum_{i=1}^{4} d_{i,f}^2$$

where $d_{i,f}$ — referred to throughout this dissertation as the **planarity deviation** of
vertex $i$ of face $f$ — are the signed perpendicular distances from the SVD-derived
best-fit plane, as illustrated in Figure 1.2.

**Figure 1.2:** Planar versus non-planar quad face. Panel (a): all four vertices lie on a
single plane, $d_{i,f} = 0$, $E_p = 0$. Panel (b): vertex $v_4$ is displaced out of the
SVD best-fit plane (orange); non-zero distances $d_{i,f}$ (red dashed lines) contribute
directly to $E_p > 0$.

This energy is globally smooth, zero if and only if all faces are exactly planar, and yields
a well-defined gradient at every vertex configuration without requiring supplementary
determinant conditions. The root-mean-square of $d_{i,f}$ across all face vertices is the
primary fabrication quality metric reported in Chapter 4. The higher per-face cost (one SVD
per face) is addressed through hardware-accelerated batched computation in Chapter 2, with
the full gradient derivation in Chapter 2 (Section 2.4.1).

### 1.3.3 Fairness and Closeness Energies

Planarity optimisation in isolation can introduce irregular configurations that are visually
unacceptable and structurally inappropriate (Liu et al., 2006). Two additional energy terms
are therefore incorporated. The **fairness energy** penalises irregular vertex positions using
the discrete Laplacian umbrella operator:

$$E_f = \sum_{v} \left\| v - \frac{1}{|N_v|} \sum_{u \in N_v} u \right\|^2$$

where $N_v$ is the edge-adjacent neighbour set of vertices $v$. In contexts with intentional
surface features, $w_f$ must be calibrated carefully (Chapter 2, Section 2.3.3). The
**closeness energy** penalises deviations from the pre-processed input positions $V_0$:

$$E_c = \sum_{v} \| v - v_0 \|^2$$

Reference positions are anchored to post-normalisation vertex coordinates to ensure
comparable energy magnitudes across all four terms from the first iteration (Chapter 2,
Section 2.5.2).

---

## 1.4 Developable Surfaces and PQ Strip Discretisation

### 1.4.1 Classical Theory and Discrete PQ Strips

A developable surface has zero Gaussian curvature and is the envelope of a one-parameter
family of tangent planes, each touching the surface along a straight ruling (do Carmo, 1976),
as illustrated in Figure 1.3.

**Figure 1.3:** Developable surface decomposing into ruled patches with constant tangent
planes along each ruling. Reproduced from Tang et al. (2016, Fig. 3).

Liu et al. (2006) establish the fundamental discrete counterpart: a **PQ strip** — a single
row of planar quadrilateral faces constitutes a discrete analogue of the classical tangent
surface construction. In a full PQ mesh, each row of faces is therefore discrete-developable,
and driving the SVD planarity residuals of every face towards zero enforces this row-level
developability without requiring any explicit ruling representation. Full isometric
flattenability would require additional intrinsic metric constraints such as the geodesic
arc-length conditions studied by Chu and Séquin (2002), which is beyond the current scope
(Tang et al., 2016).

### 1.4.2 Limitation of Existing Developable Design Workflows

Strictly enforcing developability constraints typically restricts achievable shapes and may
produce significant deviations from the intended design for surfaces with complex curvature
(Yuan et al., 2025). Most existing workflows treat developability as a sequential offline
process, which prevents any interactive exploration of design alternatives, and survey
literature consistently highlights the challenge of communicating geometric error to
non-specialist stakeholders (Yuan et al., 2025). The current project contributes to
addressing these limitations by providing a real-time environment in which planarity and
conical energy terms continuously inform design choices.

---

## 1.5 Conical Meshes: Principal Curvature Discretisation

### 1.5.1 Definitions, Angle-Balance Criterion, and Geometric Advantages

Liu et al. (2006) introduce conical meshes as PQ meshes constituting a discrete analogue of
principal curvature line networks. A valence-four vertex is conical if the four oriented face
planes are tangent to a common orientated cone of revolution; the defining algebraic condition
(Wang et al., 2006) is the alternating angle-balance criterion:

$$\alpha_1 + \alpha_3 = \alpha_2 + \alpha_4$$

where $\alpha_1, \alpha_2, \alpha_3, \alpha_4$ are the interior angles of the four incident
faces in cyclic order, as illustrated in Figure 1.4.

**Figure 1.4:** Conical mesh vertex configuration. The four oriented face planes meet at
vertex $v$ and are tangent to a common oriented cone $\Gamma$; angles
$\alpha_1, \ldots, \alpha_4$ satisfy $\alpha_1 + \alpha_3 = \alpha_2 + \alpha_4$. The cone
axis $G$ constitutes the discrete surface normal at $v$. Reproduced from Liu et al.
(2006, Fig. 10a).

The primary architectural motivation is the **consistent offset property**: offsetting all
face planes by a fixed distance yields a new conical PQ mesh of the same connectivity (Liu et
al., 2006), ensuring that every layer of a multi-layer envelope is geometrically well-defined
and connected by planar orthogonal support strips (Pottmann et al., 2007b). The broader
geometric context — including the Laguerre/Möbius duality between conicality and circularity
— is discussed in Appendix C (Section C.1).

### 1.5.2 The Angle-Balance Energy as a First-Class Term

The current project introduces a differentiable energy term $E_a$ that encourages a
conical-like vertex structure:

$$E_a = \sum_{v} \left( \sum_{f \sim v} \theta_{f,v} - 2\pi \right)^2$$

where $\theta_{f,v}$ is the interior angle of face $f$ at vertex $v$. This total
angle-defect formulation penalises departure from $2\pi$ at each vertex, a discrete Gaussian
curvature measure related to local developability. It is geometrically distinct from the
alternating condition of Liu et al. (2006): $E_a = 0$ does not guarantee the coaxial cone
structure required for valid offset meshes. This is a deliberately acknowledged scope
boundary; implementing the alternating condition as a soft penalty
$\sum(\alpha_1 + \alpha_3 - \alpha_2 - \alpha_4)^2$ is identified as future work in Chapter
4. $E_a$ is treated as a first-class term throughout the codebase with default weight
$w_a = 0$; setting $w_a > 0$ enables joint PQ and angle-defect optimisation without any
algorithmic changes.

---

## 1.6 Optimisation Methods

### 1.6.1 Prior Works

Liu et al. (2006) propose both an SQP method and a penalty method for PQ and conical mesh
optimisation, solving the coefficient system with TAUCS and UMFPACK. Their two-stage pipeline
— penalty method for rapid initial approach, then SQP for high-accuracy refinement — works
efficiently for meshes of up to approximately 1,000 vertices but was designed without GPU
acceleration, limiting scalability and reproducibility for an interactive architectural
workflow.

### 1.6.2 L-BFGS-B: Algorithm Selection Rationale

L-BFGS-B (Nocedal and Wright, 2006; Zhu et al., 1997) was selected over SQP,
Gauss-Newton, and augmented Lagrangian alternatives for three reasons: all four energy terms
are smooth with closed-form analytic gradients; the limited-memory Hessian approximation uses
only $O(mn)$ memory rather than $O(n^2)$ (Nocedal and Wright, 2006); and L-BFGS-B is robust
to ill-conditioned energy landscapes arising when weights differ by several orders of
magnitude. The analytic gradient for all four terms is verified against finite differences as
part of the test suite — a reproducibility safeguard absent from most published PQ
implementations (Chapter 2, Section 2.4.3). A full four-method comparison is provided in
Appendix C (Table C.1).

### 1.6.3 Hardware Acceleration

A three-tier computational backend dispatches energy and gradient computations to CuPy
(CUDA GPU), Numba (CPU-parallel JIT), or vectorised NumPy in order of hardware availability.
Empirical benchmarking (Chapter 4) confirms wall-clock complexity
$T(n) \approx 0.0007 \times n^{1.07}$, $R^2 = 1.000$, across the full test range. Full
per-backend speedup factors and planarity reduction figures are reported in Chapter 4
(EXP-01 and EXP-02).

---

## 1.7 Gaps in the Literature and Research Questions

### 1.7.1 Identified Limitations

Four principal gaps motivate the current project. Firstly, PQ and conical optimisation tools
are predominantly offline and batch-orientated, with no support for interactive parameter
adjustment or real-time visual feedback (Pottmann et al., 2007a). Secondly, the conical
angle-balance condition is treated as a binary constraint — either fully enforced within the
SQP or entirely absent — preventing exploration of intermediate geometries (Liu et al., 2006).
Thirdly, the literature provides limited quantitative benchmarking on standardised datasets,
as most published results are qualitative and implementations are rarely made publicly
available. Fourthly, existing implementations do not document systematic gradient
verification or numerical safeguards against ill-conditioned energy landscapes.

### 1.7.2 Research Questions

In response to these gaps, this dissertation investigates the following three research
questions:

1. Can an L-BFGS-B-based optimiser with an analytic gradient and hardware-accelerated
   backends deliver near real-time PQ planarisations on meshes of architectural scale, and
   what are the practical limits of this approach in terms of mesh size, convergence quality,
   and planarity reduction?

2. How do the relative weightings of the planarity, fairness, closeness, and angle-balance
   energies interact in practice, and what guidance can be offered to practitioners seeking
   to navigate the trade-offs between geometric fidelity, surface smoothness, and
   conical-like structure during interactive design?

3. Does the SVD-based planarity energy generalise robustly to real-world quad meshes from
   benchmark geometry datasets and CAD-exported models, beyond the synthetic test cases used
   in development?

These questions are addressed through five documented experiments in Chapter 4.

---

## 1.8 Dissertation Structure

Chapter 2 presents the mathematical formulation of the four energy terms, their analytic
gradients, the pre-processing pipeline, the optimisation algorithm with hyperparameter
choices, and the fabrication export workflow. Chapter 3 describes the software architecture,
the three-tier hardware backend, data flow, and the testing infrastructure. Chapter 4 reports
five experiments covering scalability, backend performance, convergence behaviour, weight
sensitivity, and generalisation to real-world meshes, and identifies future directions
including full GPU pipeline integration, Rhino-Grasshopper extension, and direct
implementation of the Liu et al. (2006) alternating angle-balance condition.