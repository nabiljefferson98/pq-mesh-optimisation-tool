# Appendix C: Extended Theoretical Background

This appendix supplements Chapter 1 with material that, whilst academically important, is not
essential to following the main argument of the dissertation. Section C.1 surveys the
historical development of PQ mesh theory and its discrete integrability foundations, including
the Laguerre/Möbius duality between conicality and circularity. Section C.2 provides the full
four-way optimisation method comparison referenced in Chapter 1 (Section 1.6.2) as Table C.1.
Section C.3 discusses the relationship between PQ meshes and developable surface theory and
provides the geometric interpretation of the planarity reduction ceiling observed in the
experimental results of Chapter 4.

---

## C.1 Historical Development of PQ Mesh Theory and Discrete Integrability

The systematic geometric study of quadrilateral nets may be traced to Sauer (1970), who
examined difference-geometric analogues of classical surface theory and identified the
significance of conjugate nets for discrete surface representations. The contemporary
framework of discrete differential geometry, surveyed by Desbrun, Grinspun and Schröder
(2005) in a landmark SIGGRAPH course and developed comprehensively by Bobenko and Suris
(2008), treats PQ meshes as **quad nets** — first-class geometric objects rather than mere
surface approximations. Within this framework, PQ meshes are compatible with the principle
of discrete integrability: they belong to a class of nets whose cross-ratio at every face is
real, a condition equivalent to the planarity constraint (Bobenko and Suris, 2008).

This integrability underpins two practically important properties. Firstly, it justifies the
use of smooth, differentiable energy functions for optimisation: the planarity constraint
surface is a real algebraic variety of well-understood local geometry, so gradient descent is
stable and well-conditioned away from degenerate configurations. Secondly, it connects PQ
meshes directly to the classical theory of conjugate curve networks on smooth surfaces, where
the tangent directions at each surface point are conjugate with respect to the second
fundamental form (do Carmo, 1976, p. 150), providing the rigorous theoretical basis for
interpreting the conical mesh structure of Liu et al. (2006) as a discrete principal
curvature network.

Liu et al. (2006) additionally establish that conicality and circularity, in which each quad 
has a circumscribed circle are the two known discrete analogues of principal curvature networks 
on smooth surfaces. Bobenko and Suris (2008) establish that these two types stand in precise 
duality under Laguerre and Möbius transformations respectively, a deep structural correspondence 
that connects the Laguerre-geometric and Möbius-geometric perspectives on discrete curvature 
line networks. Conicality is the architecturally preferred condition because it supports the 
consistent offset property (Chapter 1, Section 1.5.1), while circularity is the computationally 
more tractable dual. This duality is not exploited in the current implementation but represents 
a theoretically motivated direction for future work.

---

## C.2 Full Comparison of Candidate Optimisation Methods

**Table C.1:** Comparison of candidate optimisation methods for PQ mesh planarisation.
Memory cost, gradient requirements, constraint support, GPU scalability, and principal 
limitations are assessed for SQP (Liu et al., 2006), Gauss-Newton, augmented Lagrangian, 
and the selected L-BFGS-B method (referenced in Chapter 1, Section 1.6.2). GPU scalability is 
assessed with respect to energy and gradient computation; the L-BFGS-B solver loop executes on 
the CPU in all configurations.

| Method                 | Hessian Memory                    | Gradient Required          | Box Constraints | GPU Scalable                                         | Key Limitation                                                                            |
| ---------------------- | --------------------------------- | -------------------------- | --------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| SQP (Liu et al., 2006) | O(n2)O(n^2)O(n2) KKT system       | Jacobian + KKT factors     | Not native      | No                                                   | Approx. 1,000-vertex limit; no public implementation                                      |
| Gauss-Newton           | O(n2)O(n^2)O(n2) normal equations | Jacobian (approx. Hessian) | Not native      | Partial                                              | Ill-conditioned when weights differ by orders of magnitude                                |
| Augmented Lagrangian   | O(n)O(n)O(n) per iteration        | Gradient + penalty terms   | Via penalty     | Partial                                              | Penalty parameter tuning required; slower near optimum                                    |
| L-BFGS-B (this work)   | O(mn)O(mn)O(mn) limited memory    | Analytic (closed form)     | Native          | Partial (energy/gradient on GPU; solver loop on CPU) | Smooth-landscape assumption; cannot enforce exact planarity as a hard equality constraint |

The SQP approach of Liu et al. (2006) achieves the highest planarity accuracy for small meshes 
because it enforces the planarity constraint as a hard equality; however, its $O(n^2)$ KKT memory 
requirement and the absence of a public implementation prevent direct comparison at scale. L-BFGS-B 
is the only method in this comparison that natively supports box constraints and maintains $O(mn)$ 
memory with $m \ll n$. Although the SciPy solver loop executes on the CPU, the energy and gradient 
computations are GPU-accelerated via CuPy, making the implementation hardware-portable and practically 
scalable to large architectural meshes. The inability to enforce planarity as a hard equality constraint 
is accepted as a design trade-off: the soft penalty formulation permits simultaneous optimisation 
of planarity, fairness, and closeness within a single unconstrained objective, which would not be 
possible under a hard-constraint formulation.

---

## C.3 PQ Meshes, Developable Surfaces, and the Scope of Planar-Strip Developability

The connection between PQ meshes and developable surfaces, introduced in Chapter 1 (Section
1.4.1), merits further elaboration. A **developable surface** is locally isometric to the
plane: every point has a neighbourhood that can be unrolled without stretching or tearing
(do Carmo, 1976). The three classical species — cylinders, cones, and tangent developables —
are all characterised by a one-parameter family of rulings along which the tangent plane is
constant.

Liu et al. (2006) demonstrate that a single PQ strip — one row of planar quads — is a
discrete tangent developable: the column edges act as discrete rulings, and the shared
tangent-plane condition along those edges is precisely the planarity condition $E_p = 0$.
Enforcing $E_p = 0$ for every row of a PQ mesh therefore makes the mesh **row-developable**,
a weaker condition than full isometric flattenability.

Full isometric flattenability requires, in addition to planar faces, that the intrinsic metric
be flat, entailing geodesic arc-length conditions across face boundaries equivalent to the
constraints studied by Chu and Séquin (2002) for Bézier patches and by Tang et al. (2016) in
the context of interactive developable surface design. These additional constraints are beyond
the scope of the current project. Practitioners requiring full isometric flattenability — for
example, for textile patterning or sheet-metal unfolding with zero stretch — should extend the
current pipeline with the geodesic constraints of Tang et al. (2016).

The Gaussian curvature interpretation provides the geometric explanation for the approximately
83 per cent planarity reduction ceiling observed in EXP-01 (Chapter 4, Section 4.2.1). By the
discrete Gauss-Bonnet theorem, a closed surface has a fixed total Gaussian curvature
determined by its topology (Liu et al., 2006; Pottmann et al., 2007a). For a surface of
non-zero Gaussian curvature, enforcing exact planarity on every face would require the mesh to
be topologically equivalent to the plane — which a closed or doubly curved surface is not.
The residual planarity error observed on the 900-face and larger meshes is therefore a
geometric lower bound imposed by the target surface topology, not a limitation of the
optimiser.