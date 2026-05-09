# Summary

This dissertation presents the design, implementation, and empirical
evaluation of an interactive planar quad (PQ) mesh optimisation and
visualisation tool for fabrication-aware architectural geometry. PQ
meshes — quadrilateral meshes in which every face lies in a single
plane — are the canonical discrete representation for glass and
sheet-metal building envelopes, yet existing optimisation tools
operate exclusively as offline batch processes with no support for
interactive parameter adjustment or real-time visual feedback.

The pipeline minimises a weighted sum of four energy terms:
planarity ($E_p$), fairness ($E_f$), closeness ($E_c$), and
angle-balance ($E_a$), using the L-BFGS-B quasi-Newton optimiser
with fully analytic gradients verified against central finite
differences. A three-tier hardware-accelerated backend dispatches
computations to a CUDA GPU (CuPy), CPU-parallel JIT (Numba), or
vectorised NumPy in order of hardware availability. An interactive
Polyscope-based GUI provides real-time per-face planarity
heatmaps and runtime weight adjustment. A fabrication export
pathway produces OBJ and JSON outputs compatible with downstream
manufacturing workflows.

Five experiments evaluated the pipeline against three research
questions. Scalability experiments confirmed sub-quadratic
wall-clock complexity $T(n) \approx 0.0007 \times n^{1.27}$
($R^2 = 1.000$) across meshes from 9 to 5,625 faces, with Numba
and CuPy backends delivering speedups of up to 2.79× and 3.14×
respectively over the NumPy baseline. Convergence experiments
confirmed quasi-Newton superlinear convergence in 9 to 13
iterations to gradient norm below $10^{-5}$, with mean per-face
deviations within glass manufacturing tolerance of ±1 mm at
physical scale. A fifteen-configuration weight sensitivity
analysis identified $w_p = 10.0$, $w_f = 1.0$, $w_c = 5.0$ as
the Pareto-optimal calibration. Generalisation experiments on
four real-world benchmark meshes from Keenan Crane's 3D Model
Repository achieved 2.62 to 66.03 per cent planarity energy
reduction, with performance bounded below by intrinsic Gaussian
curvature in a manner consistent with classical surface theory.

The pipeline is released as open-source software under the MIT
Licence at https://github.com/nabiljefferson98/pq-mesh-optimisation-tool,
with a 321-test reproducible test suite, full experiment
reproducibility documentation, and all dissertation files
version-controlled in the same repository. The primary limitation
is that the angle-balance term does not enforce the strict
alternating conicality condition required for offset-mesh
compatibility; implementing this condition constitutes the
highest-priority direction for future work.