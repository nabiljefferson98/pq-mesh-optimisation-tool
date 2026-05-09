# Appendix I: Future Work and Development Roadmap

This appendix elaborates the four-tier future work programme
summarised in Chapter 4, Section 4.9. Each direction is
described with its implementation pathway, dependency on
existing infrastructure, and estimated scope relative to
the current project baseline.

---

## I.1 Tier 1: Algorithmic Extensions (Highest Impact)

### I.1.1 Strict Alternating Angle-Balance Constraint

As established in Chapter 1 (Section 1.5.2) and critically
evaluated in Chapter 2 (Section 2.9.2), the current angle-
balance term $E_a$ approximates conicality as a total
angle-defect penalty rather than enforcing the alternating
condition $\alpha_1 + \alpha_3 = \alpha_2 + \alpha_4$ of
Liu et al. (2006). Implementing the alternating constraint
as a soft penalty:

$$E_{\text{alt}} = \sum_v \left(
  \alpha_{1,v} + \alpha_{3,v} - \alpha_{2,v} - \alpha_{4,v}
\right)^2$$

would require: (i) a face-ordering convention assigning
consistent alternating labels to the four incident faces at
each interior vertex; (ii) an analytic gradient compatible
with the existing L-BFGS-B interface; and (iii) extension of
the `test_gradients_extended.py` suite to verify the new term
against central finite differences. The existing energy
framework accommodates this as a fifth term without any
change to the optimiser, data structures, or backend dispatch
logic. This is the highest-priority single algorithmic
extension because it removes the principal theoretical
limitation identified in Section 2.9.2 and would enable
the pipeline to produce offset-mesh-compatible results for
multi-layer glazing applications.

### I.1.2 Full GPU Pipeline

EXP-02 demonstrates that CuPy provides 2.19× to 3.14×
speedup for meshes of 400 faces or more, but incurs a
0.74× slowdown at 25 faces due to PCIe host-to-device
transfer overhead dominating at small problem sizes. A
full GPU pipeline would eliminate this overhead by
maintaining the vertex array and gradient buffers
permanently on device throughout the optimisation loop,
transferring results to host only at convergence or for
visualisation updates. The principal implementation
challenge is the angle-balance gradient, which currently
has no Tier 1 path due to irregular per-vertex adjacency
structure (Appendix E, Section E.2.5). A CUDA-compatible
padded adjacency kernel, analogous to the existing
`vertex_face_ids_padded` structure, would resolve this.
Target speedup at 5,625 faces: 5× to 8× over NumPy,
reducing the 79.63 s EXP-01 benchmark to approximately
10 to 16 seconds.

### I.1.3 Cotangent Laplacian Fairness

The uniform umbrella operator in $E_f$ introduces
gradient approximation error at boundary and
irregular-valence vertices because it weights all adjacent
vertices equally regardless of edge length or face area
(Botsch et al., 2010). Replacing it with the cotangent
Laplacian:

$$L_{ij}^{\cot} = \frac{1}{2}\left(
  \cot \alpha_{ij} + \cot \beta_{ij}
\right)$$

would preserve sharp features and produce geometrically
consistent smoothing on unstructured meshes. The gradient
derivation requires updating the sparse Laplacian
construction in `src/core/mesh.py` and the gradient
accumulation in `src/optimisation/gradients.py`. The
cotangent weights introduce a dependency on face angles
that must be recomputed when vertex positions change,
adding moderate overhead per iteration. This extension is
particularly relevant for the Bob benchmark mesh, where
non-uniform re-meshed edge lengths reduce the quality of
the uniform Laplacian approximation (Appendix F, §F.5).

---

## I.2 Tier 2: Software Infrastructure

### I.2.1 Spatial-Hashing Duplicate-Vertex Removal

The current $O(n^2)$ pairwise-distance deduplication in
`preprocessor.py` emits a warning for inputs exceeding
2,000 vertices (Chapter 2, Section 2.5.1). A spatial
hash grid with bucket size equal to the merge threshold
($10^{-8}$) would reduce this to $O(n)$ average-case
complexity, removing the practical ceiling on mesh size
for the preprocessing stage. Implementation requires
approximately 50 lines of additional code and a new
`test_preprocessor.py` test case covering the hash-grid
path. This is a low-risk, moderate-priority improvement.

### I.2.2 Rhino-Grasshopper Integration

The current interactive interface requires Polyscope,
a standalone Python-native viewer that operates outside
the standard architectural design workflow. A
Grasshopper component exposing `MeshOptimiser` as a
parametric node would allow architects to drive
optimisation weight sliders directly from within their
existing Rhino models, receiving per-face planarity
heatmaps as Grasshopper colour outputs. Implementation
requires wrapping the Python pipeline in a `ghpythonlib`
component and resolving the Rhino SDK dependency chain.
This extension directly addresses the practitioner
workflow gap identified in Section 1.7.1.

### I.2.3 Mixed-Precision and C/Rust Re-implementation

The NumPy baseline processes 5,625 faces in approximately
80 seconds, which is adequate for interactive design
iteration on meshes typical of academic study but falls
short of production requirements for large facade meshes
(>10,000 faces). A C or Rust re-implementation of the
gradient kernels — retaining the Python orchestration
layer and SciPy L-BFGS-B interface — is estimated to
provide 10× to 20× speedup over the NumPy baseline,
approaching the 10,000-face regime at under 30 seconds
on equivalent hardware. Mixed-precision (float32 forward
pass, float64 gradient accumulation) is an intermediate
option within the existing Python codebase.

---

## I.3 Tier 3: Evaluation Extensions

### I.3.1 Architectural Case-Study Benchmark

The four EXP-05 models (Spot, Blub, Oloid, Bob) are
geometry-processing benchmarks rather than architectural
geometry. Evaluation on three to five real architectural
facade meshes — specifically gridshell structures,
free-form canopies, and planar-quad panellised facades
drawn from published projects (e.g. Zaha Hadid Architects,
SANAA) — would provide direct evidence of practical
applicability and strengthen the claims of Section 4.7.3.
This requires acquiring or constructing quad-dominant
OBJ meshes of these structures, which is a non-trivial
data-acquisition task.

### I.3.2 Formal Practitioner User Study

A structured user study (n ≥ 8 participants, think-aloud
protocol, standardised design task) measuring
design-intent preservation under interactive weight
adjustment would provide empirical evidence for the
usability claims of Section 1.7.1 and Chapter 2,
Section 2.8. Outcome metrics would include: percentage
of participants achieving target planarity within five
minutes; subjective design-intent preservation score
(5-point Likert); and number of weight adjustment steps
to convergence. This study would constitute a natural
Chapter 5 for a Master's-level extension of the project.

### I.3.3 Cross-Platform GPU Validation

EXP-02 measured CuPy performance on Machine A (NVIDIA
RTX 3070, CUDA 12). Validation on Machine B (Apple M3,
macOS) using Metal Performance Shaders via `mlx` or
`torch.mps` would confirm that the speedup findings
generalise across GPU architectures and are not specific
to CUDA implementations. This is a low-effort extension
requiring approximately one additional benchmark run
per mesh size.

---

## I.4 Tier 4: Theoretical Extensions

### I.4.1 Ruling-Direction Joint Optimisation

Tang et al. (2016) demonstrate that jointly optimising
vertex positions and ruling directions in a single
variational formulation produces surfaces with guaranteed
geometric developability rather than the approximate
fabrication-grade planarity achieved by the present
pipeline. Integrating a ruling-direction field as an
additional optimisation variable — and coupling it to
$E_p$ via a soft orthogonality constraint — would
move the pipeline from the current fabrication-grade
regime to the geometric-developability regime, enabling
applications requiring exact unrolling such as sheet-metal
fabrication and aerospace skin panelling.

### I.4.2 Intrinsic Metric Constraints for Exact Flattenability

Full isometric flattenability requires that the Gaussian
curvature is identically zero everywhere, not merely that
face vertices are co-planar (Stein et al., 2018). Adding
an intrinsic metric constraint — penalising deviation of
edge lengths from their reference values — would prevent
the optimiser from achieving planarity through stretching
rather than bending. This constraint introduces a strong
coupling between $E_p$ and $E_c$ that would require
careful weight re-calibration beyond the EXP-04 results.

### I.4.3 Fabrication Cost Integration

The current energy function treats all configurations
with equivalent $E_p$ as equally optimal. A fabrication
cost term integrating material waste (proportional to
panel bounding-box area minus face area) and CNC path
length would steer the optimiser towards configurations
that are simultaneously planar and economically
manufacturable. This connects the geometric optimisation
problem to the production cost modelling literature
(Eigensatz et al., 2010) and would constitute a natural
interdisciplinary contribution at the intersection of
computational geometry and architectural economics.

---

## References

Botsch, M., Kobbelt, L., Pauly, M., Alliez, P. and
Lévy, B. (2010). *Polygon Mesh Processing*.
Natick: AK Peters.

Eigensatz, M., Kilian, M., Schiftner, A., Mitra, N. J.,
Pottmann, H. and Pauly, M. (2010). Panelling
architectural freeform surfaces. *ACM Transactions on
Graphics*, 29(4), Article 45.

Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L. and
Wang, W. (2006). Geometric modeling with conical meshes
and developable surfaces. *ACM Transactions on
Graphics*, 25(3), pp. 681–689.

Stein, O., Grinspun, E. and Crane, K. (2018).
Developability of triangle meshes. *ACM Transactions
on Graphics*, 37(4), Article 77.

Tang, C., Bo, P., Wallner, J. and Pottmann, H. (2016).
Interactive design of developable surfaces.
*ACM Transactions on Graphics*, 35(2), Article 12.