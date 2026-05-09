# Appendix A: Self-Appraisal

## A.1 Critical Self-Evaluation

### A.1.1 Achievement Against Objectives

This project set out to design, implement, and empirically evaluate
a real-time PQ mesh optimisation and visualisation tool addressing
four gaps identified in the existing literature (Chapter 1,
Section 1.7.1). All three primary research questions were answered
affirmatively through five structured experiments (Chapter 4),
and all planned software deliverables were completed:

- A four-term weighted energy function ($E_p$, $E_f$, $E_c$, $E_a$)
  with analytic gradients verified against central finite differences
  to $< 10^{-4}$ relative error (Appendix D, Table D.1).
- A three-tier hardware-accelerated backend (CuPy, Numba, NumPy)
  with confirmed speedups of 2.40× to 3.14× over the NumPy baseline
  for meshes of practical size (Chapter 4, Table 4.3).
- A Polyscope-based interactive visualisation interface with runtime
  weight adjustment and per-face planarity heatmap.
- A reproducible test suite of 229 tests across 20 modules achieving
  ≥79% source coverage (Appendix E, Table E.3).
- An OBJ/JSON fabrication export pipeline for downstream manufacturing
  workflows (Chapter 2, Section 2.7).
- Full experiment reproducibility documentation (Appendix G).

The most significant technical achievement was the confirmation of
sub-quadratic wall-clock complexity $T(n) \approx 0.0007 \times n^{1.27}$
($R^2 = 1.000$) across two orders of magnitude in face count, with
9 to 13 iteration quasi-Newton convergence consistent with the
theoretical predictions of Nocedal and Wright (2006). The real-world
generalisation experiment (EXP-05) demonstrated that the pipeline
converges without manual intervention on all four benchmark meshes,
confirming robustness beyond synthetic test cases.

### A.1.2 Limitations and Shortcomings

The principal technical limitation is that the angle-balance term
$E_a$ implements a total angle-defect penalty rather than the strict
alternating condition $\alpha_1 + \alpha_3 = \alpha_2 + \alpha_4$
of Liu et al. (2006). This means the pipeline does not produce
offset-mesh-compatible conical results, a scope boundary that was
consciously defined at the project outset but represents the most
significant gap between the current implementation and production-grade
architectural geometry tools. Implementing the alternating condition
with an analytic gradient is the highest-priority future direction
(Appendix I, Section I.1.1).

A secondary limitation is the $O(n^2)$ deduplication step in the
preprocessor, which emits a performance warning for inputs exceeding
2,000 vertices (Chapter 2, Section 2.5.1). Whilst this does not affect
correctness, it creates a practical ceiling for mesh preprocessing
that a spatial hash grid would resolve in $O(n)$ average-case time
(Appendix I, Section I.2.1).

The EXP-05 real-world benchmarks showed modest planarity reductions
of 2.62 to 7.55 per cent on organic closed-manifold meshes (Spot,
Blub, Bob), consistent with theoretical predictions from Gaussian
curvature analysis but potentially disappointing from a practitioner
standpoint. The pipeline's effectiveness on these meshes is bounded
below by intrinsic curvature, not by algorithmic failure, and the
results were communicated with this qualification in Chapter 4,
Section 4.7.3.

The weight calibration in EXP-04 was conducted exclusively on the
$10 \times 10$ synthetic mesh. Whilst the recommended configuration
($w_p = 10.0$, $w_f = 1.0$, $w_c = 5.0$) is theoretically motivated,
its transferability to irregular or architectural meshes was not
formally validated. A practitioner user study, identified in
Appendix I, Section I.3.2, would be required to confirm this.

### A.1.3 How Problems Encountered Could Be Avoided

The most significant implementation problem was the silent Numba
JIT type inference failure caused by the `_ANGLE_SIGNS` constant
being declared as a Python list rather than a typed `numpy.float64`
array (Appendix D, Section D.3). This silently routed execution to
the NumPy fallback without any error message, causing anomalous
benchmark results that initially appeared as a correctness problem.
The root cause was identified through systematic isolation of the
backend dispatch logic. In future projects, silent fallback
mechanisms should include a mandatory warning log entry, and all
JIT-compiled functions should be accompanied by a smoke test that
verifies the expected backend is active before any benchmarking run.

A related problem was the discovery that `ftol` in the Stage 1
L-BFGS-B call was initially set to $10^{-6}$ rather than the
intended $10^{-7}$, causing Stage 1 to terminate earlier than the
two-stage design required (Chapter 2, Section 2.6.1). This was
identified during the EXP-03 convergence analysis when iteration
counts were lower than expected for large meshes. In future
projects, all hyperparameter values should be locked in a
configuration dataclass from the outset, with the configuration
file itself version-controlled and included in the test suite as
an invariant check.

The EXP-01 scalability experiment originally stored timing results
without documenting the Numba warm-up exclusion procedure, making
initial results non-reproducible across machines. This was resolved
by adding the explicit warm-up exclusion protocol to Appendix G,
Section G.3. Future benchmarking should treat warm-up exclusion as
a mandatory first step, documented in the experiment script itself
rather than in post-hoc documentation.

---

## A.2 Personal Reflection and Lessons Learned

### A.2.1 Technical Development

This project represented the first substantial independent experience
with numerical optimisation at scale, and the most significant
personal learning was understanding the practical gap between
algorithmic correctness and numerical stability. It was not
sufficient to verify that the analytic gradient formula was
mathematically correct; it was equally important to verify
numerically against finite differences, to confirm that the
gradient remained well-behaved on non-synthetic geometries, and
to ensure that the L-BFGS-B interface received gradients in the
exact memory layout and dtype that SciPy expected. The discipline
of writing gradient verification tests before benchmarking, rather
than after, would have prevented several days of debugging.

The three-tier backend architecture taught an important lesson about
premature abstraction: the CuPy backend was the most technically
ambitious component, but EXP-02 demonstrated that it is
counter-productive below 200 faces due to PCIe transfer overhead.
Building the NumPy baseline to full correctness first, before
attempting hardware acceleration, would have been a more efficient
development path. The abstraction layer that now makes backend
switching transparent was designed only after the NumPy baseline
was stable, and this ordering proved to be the right choice.

Working with the Polyscope interactive viewer introduced challenges
around the separation of optimisation state from rendering state.
Early prototypes embedded the optimiser loop inside the Polyscope
callback, causing blocking behaviour when convergence was slow. The
refactoring to a separate optimiser thread with a shared state lock
was the correct architectural decision, but it introduced
concurrency edge cases that required careful testing. This
reinforced the importance of separating computational and
presentation concerns from the beginning of implementation, a
principle that is well-established in software engineering but
easy to neglect when prototyping quickly.

### A.2.2 Research and Academic Development

Engaging critically with the Liu et al. (2006) paper — rather than
simply implementing their method — was the most valuable academic
exercise of the project. Understanding precisely why their SQP
formulation requires explicit constraint Jacobians, and why this
limits scalability, was essential for justifying the L-BFGS-B
choice. This experience reinforced the importance of reading primary
sources in full rather than relying on survey characterisations.

The experience of writing and iterating the dissertation itself
provided a clear lesson about the relationship between writing and
thinking. Several inconsistencies in the mathematical formulation
of Chapter 2 were only identified during the process of writing
the gradient derivations for Appendix D, because the act of
writing forced precision that informal notes had obscured. Future
projects should integrate technical writing into the development
process from an early stage, rather than treating it as a
post-implementation activity.

### A.2.3 Project Management

The project was managed using a Kanban-style board with weekly
milestones. The most significant schedule challenge was the EXP-04
weight calibration phase, which required fifteen separate experimental
runs and took approximately twice as long as initially estimated due
to the need to re-run sweeps after the `ftol` correction described in
Section A.1.3. In future projects, parameter sensitivity experiments
should be allocated at least double the initially estimated time,
and a configuration lock point should be established before
commencing multi-run experiments.

The use of Git version control for both the codebase and the
dissertation markdown files was essential for recovering from several
documentation errors discovered during the audit process. The
practice of committing dissertation files alongside code changes
meant that the state of the documentation at any experimental
milestone could be reconstructed precisely. This is strongly
recommended for all future dissertation projects.

---

## A.3 Legal, Social, Ethical and Professional Issues

### A.3.1 Legal Issues

All four real-world benchmark meshes used in EXP-05 are released
under permissive licences that explicitly permit use in academic
research and redistribution. Specifically:

- **Spot, Blub, Bob** — released by Keenan Crane under Creative
  Commons Zero (CC0) licence, placing them in the public domain
  with no attribution requirement, though attribution is provided
  as best practice (Appendix F, Section F.1).
- **Oloid** — sourced from Keenan Crane's 3D Model Repository
  under a CC0 public domain dedication, permitting unrestricted
  use and redistribution without attribution requirement, though
  attribution is provided as best practice (Appendix F,
  Section F.1.2).

All open-source software libraries used in this project are released
under permissive licences (MIT, BSD-3, or Apache 2.0) that impose
no restrictions on academic use or redistribution (Appendix B,
Section B.1). No proprietary software, data, or intellectual
property was incorporated into the project deliverables.

The source code repository is publicly accessible at
https://github.com/nabiljefferson98/pq-mesh-optimisation-tool
under the MIT Licence, permitting unrestricted use, modification,
and redistribution with attribution. No patent claims are made
on any component of the pipeline.

### A.3.2 Social Issues

The primary application domain of this project is architectural
fabrication: the optimisation of facade panels and structural
envelopes for buildings. The social implications of more efficient
planar panellisation tools include potential reductions in material
waste and manufacturing cost for complex building envelopes,
which may make freeform architectural geometry more economically
accessible to smaller-budget projects and in lower-income contexts.
There are no negative social implications identified for this project.

This project was conducted entirely with synthetic and publicly
available benchmark data. No user data, personal information, or
observations of human subjects were collected at any stage. No
formal ethical approval was required.

### A.3.3 Ethical Issues

No human participants were involved in any aspect of this project.
No data collection involving personal or sensitive information was
conducted. No user study was performed; the practitioner user study
described in Appendix I, Section I.3.2 is identified as future work
only and was not executed as part of this project.

The use of AI-assisted tools during development and writing is
disclosed fully in Appendix B, Section B.5. All AI-generated content was
critically reviewed, verified against primary sources, and
substantially revised before inclusion in the dissertation. No
AI-generated text was included verbatim in the final report.

### A.3.4 Professional Issues

This project was conducted in accordance with the BCS Code of
Conduct (British Computer Society, 2022), specifically:

- **Public interest:** The tool is released as open-source software
  under the MIT Licence, maximising accessibility and enabling peer
  verification of results, consistent with the BCS principle of
  acting in the public interest through transparent, reproducible
  research practice.
- **Professional competence and integrity:** All experimental
  claims are supported by documented, reproducible experiments
  (Appendix G). No results were selectively reported; limitations
  identified by the experiments are disclosed fully in Chapter 4,
  Section 4.7, and Section A.1.2 of this appendix.
- **Duty to the profession:** The dissertation follows the
  University of Leeds academic integrity policy. All external
  materials are attributed (Appendix B). The GenAI usage log is
  disclosed in full (Appendix B, Section B.5).

No conflicts of interest exist in relation to this project.

---

## References

British Computer Society (2022). *BCS Code of Conduct*. Swindon:
BCS. Available at: https://www.bcs.org/membership/become-a-member/bcs-code-of-conduct/

Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L. and Wang, W.
(2006). Geometric modeling with conical meshes and developable
surfaces. *ACM Transactions on Graphics*, 25(3), pp. 681–689.

Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*.
2nd ed. New York: Springer.