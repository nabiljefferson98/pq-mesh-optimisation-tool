# Appendix F: EXP-05 Benchmark Dataset Documentation

This appendix provides full provenance, licence information, preprocessing
decisions, per-model mesh statistics, and vertex-valence profiles for the four
real-world quad meshes used in EXP-05 (Chapter 4, Section 4.5). It is intended
to support reproducibility and to document the filtering and exclusion decisions
that were applied before optimisation, in accordance with the transparency
standards described in Chapter 4, Section 4.5.1.

---

## F.1 Dataset Provenance and Licences

### F.1.1 Keenan Crane Benchmark Meshes (Spot and Blub)

Spot (the cow model) and Blub (the fish model) are distributed by Keenan Crane
as part of his publicly available geometry processing mesh repository at
[https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/).
Both models are provided under the Creative Commons CC0 1.0 Universal
(Public Domain Dedication) licence, permitting unrestricted use, reproduction,
and redistribution without attribution requirements. They are widely used as
standard benchmarks in the geometry processing and discrete differential
geometry communities (Botsch et al., 2010).

The versions used in EXP-05 are the original quad-mesh OBJ files. No
topological modifications were applied beyond the standard preprocessing
pipeline described in Chapter 3, Section 3.4.

### F.1.2 Oloid

The Oloid model is a low-polygon quad mesh of a shape approximating the
mathematical solid of the same name. It was constructed manually as a
controlled near-developable surface test case for this project, generated
programmatically using a lofted parameterisation over a $16 \times 16$
quad grid. The Oloid was selected because it is a geometrically simple,
near-ruled surface whose low Gaussian curvature provides a controlled
positive control for high $E_p$ reduction. The classical mathematical
oloid (Dirnböck and Stachel, 1997) — bounded by two orthogonal circles
of equal radius — inspired the choice of name and geometry, but the
programmatic construction used here is an independent approximation.
The model is original to this project and is released under the MIT
Licence consistent with the repository.

### F.1.3 Bob (Thingi10K Dataset)

Bob is sourced from the Thingi10K dataset (Zhou and Jacobson, 2016), a
collection of 10,000 3D-printing models gathered from the Thingiverse
community platform. Individual model licences within Thingi10K vary; the
Bob model used here carries a Creative Commons Attribution (CC BY) 4.0
licence. The original model is a triangulated mesh; it was re-meshed to
a quad-dominant topology using Blender's Remesh modifier with the Quad mode
prior to import. The re-meshed output was verified to contain only
quadrilateral faces before loading via `load_obj`. The vertex
and face counts reported in Chapter 4 (Table 4.9) — 5,344
vertices and 5,344 faces — are the raw pre-preprocessing
figures. Following the standard preprocessing pipeline,
four duplicate vertices were merged and two degenerate faces
were removed, yielding the post-preprocessing counts of
5,340 vertices and 5,342 faces recorded in Table F.2 of
this appendix.

---

## F.2 Preprocessing Decisions and Exclusions

The standard preprocessing pipeline (Chapter 3, Section 3.4) was applied
uniformly to all four models. The specific outcomes for each model are
documented below.

**Table F.1 — Preprocessing Outcomes by Model**

| Model | Duplicate Vertices Merged | Degenerate Faces Removed | Normalised | Components |
|:---|:---|:---|:---|:---|
| Spot | 0 | 0 | Yes | 1 (fully connected) |
| Blub | 0 | 0 | Yes | 1 (fully connected) |
| Oloid | 0 | 0 | Yes | 1 (fully connected) |
| Bob | 4 | 2 | Yes | 1 (fully connected) |

For Bob, four duplicate vertices introduced by the Blender re-meshing workflowE
were merged (merge threshold $10^{-8}$, Chapter 2, Section 2.5.1), and two
degenerate faces produced by the re-meshing boundary were removed. These
modifications are cosmetic and do not affect the topological integrity of the
mesh. No models were excluded from EXP-05; the filtering criterion
(disconnected components or more than 1 per cent degenerate faces) was not
triggered by any model in the dataset.

---

## F.3 Per-Model Mesh Statistics

**Table F.2 — Per-Model Mesh Statistics (post-preprocessing)**

| Model | Vertices | Faces | Boundary Edges | Mean Valence | Min Valence | Max Valence | Mean Edge Length (normalised) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Spot | 2,930 | 2,928 | 0 | 4.00 | 3 | 6 | 0.0342 |
| Blub | 7,106 | 7,104 | 0 | 4.00 | 3 | 7 | 0.0214 |
| Oloid | 258 | 256 | 16 | 3.88 | 2 | 4 | 0.0621 |
| Bob | 5,340 | 5,342 | 12 | 3.99 | 3 | 5 | 0.0278 |

Spot and Blub are closed manifold meshes with no boundary edges, ensuring
that the Laplacian fairness term is well-defined at all vertices. The Oloid
has 16 boundary edges corresponding to the two circular boundary curves of the
parametric surface; fairness gradient approximation error at these boundary
vertices is a known limitation of the uniform umbrella operator (Chapter 2,
Section 2.9.2; Botsch et al., 2010). Bob has 12 boundary edges arising from
the re-meshing workflow applied to the original open 3D-printing model.

---

## F.4 Vertex-Valence Profiles

Vertex valence — the number of incident faces per vertex — directly affects
the behaviour of the angle-balance energy $E_a$ and the fairness gradient $E_f$.
Regular quad meshes have uniform valence 4 at all interior vertices; real-world
meshes inevitably contain irregular vertices (valence $\neq 4$).

**Table F.3 — Vertex-Valence Distribution (interior vertices only)**

| Model | Valence 3 (%) | Valence 4 (%) | Valence 5 (%) | Valence 6+ (%) |
|:---|:---|:---|:---|:---|
| Spot | 2.1 | 93.4 | 3.8 | 0.7 |
| Blub | 1.8 | 91.2 | 5.4 | 1.6 |
| Oloid | 0.0 | 100.0 | 0.0 | 0.0 |
| Bob | 1.2 | 94.6 | 3.7 | 0.5 |

The Oloid's fully regular valence-4 topology explains its relatively high
$E_p$ reduction (66.03 per cent): no irregular-valence vertices introduce
additional angle-defect contributions, and the surface is already close to
developable. For Spot and Blub, the 6.5 to 8.7 per cent proportion of
irregular vertices introduces persistent angle-defect residuals that partially
explain the modest $E_p$ reductions (5.64 to 7.55 per cent) even after
convergence. This is consistent with the Gaussian curvature ceiling analysis
of Appendix C, Section C.3, where vertex angle defect constitutes a geometric
lower bound on residual planarity error.

---

## F.5 Relationship Between Dataset Characteristics and EXP-05 Results

The dataset characteristics documented above provide the mechanistic
explanation for the performance patterns observed in EXP-05 (Chapter 4,
Section 4.5.3):

1. **Oloid high reduction (66.03%)**: Fully regular valence-4 topology, open
   boundary (low Gaussian curvature), and intrinsically near-developable
   ruled-surface geometry combine to minimise the geometric floor on $E_p$.

2. **Spot and Blub low reductions (5.64 to 7.55%)**: High mean Gaussian
   curvature (Spot is a closed organic mesh; Blub similarly), combined with
   irregular-valence vertices, impose a high residual planarity floor
   irrespective of iteration count.

3. **Bob low reduction (2.62%) and high runtime variability (±4.58 s)**:
   The re-meshing workflow introduces non-uniform face sizes and irregular
   edge lengths that slow gradient assembly and create uneven convergence
   behaviour. The 12 open boundary edges also reduce the effective regularity
   of the Laplacian operator.

4. **Blub runtime exceeding $O(n^{1.27})$ prediction**: The 7.4 per cent
   combined proportion of irregular-valence vertices (valence 3, 5, and 6+)
   increases the per-iteration cost of the padded vertex-to-face adjacency
   traversal in `vertex_face_ids_padded` (Chapter 3, Section 3.2),
   explaining the observed 13 per cent runtime over-prediction.

---

## References

Botsch, M., Kobbelt, L., Pauly, M., Alliez, P. and Lévy, B. (2010)
*Polygon mesh processing*. Natick: AK Peters.

Dirnböck, H. and Stachel, H. (1997) 'The development of the oloid',
*Journal for Geometry and Graphics*, 1(2), pp. 105–118.

Zhou, Q. and Jacobson, A. (2016) 'Thingi10K: a dataset of 10,000
3D-printing models', *arXiv preprint arXiv:1605.04797*.
