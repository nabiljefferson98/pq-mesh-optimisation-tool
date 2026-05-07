# Appendix D: Supplementary Gradient Derivations

This appendix provides the full analytic gradient derivations for the fairness energy $E_f$
and the angle-balance energy $E_a$, referenced in Chapter 2 (Section 2.4.2). It also
documents the `_ANGLE_SIGNS` constant and its Numba JIT type constraint, documented in Chapter 3
(Section 3.6.2).

---

## D.1 Fairness Gradient

The fairness energy penalises deviation of each vertex from the centroid of its edge-adjacent
neighbours:

$$E_f = \sum_{v} \left\| v - \frac{1}{|N_v|} \sum_{u \in N_v} u \right\|^2$$

Let $L_v = v - \frac{1}{|N_v|} \sum_{u \in N_v} u$ denote the Laplacian displacement at
vertex $v$. The gradient with respect to vertex $v$ is:

$$\frac{\partial E_f}{\partial v} = 2 L_v - \frac{2}{|N_w|} \sum_{w : v \in N_w} L_w$$

The first term is the direct contribution from vertex $v$'s own Laplacian displacement.
The second term accumulates the indirect contributions from all vertices $w$ for which $v$
is a neighbour. Under the uniform-valence assumption (all interior vertices of a regular
grid have the same valence $|N_v| = |N_w| = k$), this simplifies to a purely linear
function of current vertex positions and is assembled via the Laplacian adjacency matrix.
At boundary and irregular-valence vertices, the uniform assumption introduces a small
approximation error proportional to valence deviation, as noted in Chapter 2 (Section 2.4.2).

---

## D.2 Angle-Balance Gradient

The angle-balance energy penalises departure from $2\pi$ total angle sum at each vertex:

$$E_a = \sum_{v} \left( \sum_{f \sim v} \theta_{f,v} - 2\pi \right)^2$$

where $\theta_{f,v}$ is the interior angle of face $f$ at vertex $v$. For a triangulated
neighbourhood (or a quad face treated as two triangles), the interior angle at vertex $v_i$
of a face with vertices $v_i, v_j, v_k$ is:

$$\theta_{f,v_i} = \arccos \left( \frac{(v_j - v_i) \cdot (v_k - v_i)}{\|v_j - v_i\| \|v_k - v_i\|} \right)$$

Differentiating with respect to $v_i$, $v_j$, and $v_k$ yields the per-face
angle-contribution tensor, assembled into the gradient via the `_ANGLE_SIGNS` sign assignments.
For vertex $v_i$ (the apex of the angle):

$$\frac{\partial \theta_{f,v_i}}{\partial v_i} = -\frac{1}{\sin \theta_{f,v_i}} \cdot \nabla_{v_i} \cos \theta_{f,v_i}$$

The full chain-rule expansion over all faces incident to $v$ gives:

$$\frac{\partial E_a}{\partial v} = 2 \left( \sum_{f \sim v} \theta_{f,v} - 2\pi \right) \sum_{f \sim v} \frac{\partial \theta_{f,v}}{\partial v}$$

This gradient is exact at all non-degenerate vertex configurations. At vertices where
$\sin \theta_{f,v} \to 0$ (nearly flat or collinear faces), numerical instability is
possible; such cases are excluded by the pre-processing degeneracy check of Section 2.5.1.

---

## D.3 The _ANGLE_SIGNS Constant and Numba JIT Type Constraint

The angle-balance gradient requires sign assignments for each of the four corner
contributions per quad face, encoded in the module-level constant `_ANGLE_SIGNS`. The critical
implementation constraint is that `_ANGLE_SIGNS` must be declared as a typed
`numpy.float64` array:

```python
import numpy as np
_ANGLE_SIGNS = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
```

A plain Python list or `numpy.int32` array causes Numba's closure type-inference to fail
at JIT compilation time with a `TypingError` (Lam, Pitrou and Seibert, 2015). Combined
with the broad `except Exception` handler of Chapter 3 (Section 3.4.2), such a failure
would silently route execution to the NumPy fallback, losing the expected CPU acceleration
without any visible error. The fix was applied in March 2026 and is regression-tested in
`tests/test_robustness.py` (Chapter 3, Section 3.7).

---

## D.4 Gradient Verification Summary

All four analytic gradients are verified against central finite differences in
`tests/test_gradients.py` and `tests/test_gradients_extended.py`. Table D.1 summarises the
observed maximum relative errors across all test geometries.

**Table D.1:** Gradient verification summary. Maximum relative error between analytic and
central-finite-difference gradients, evaluated over flat, sinusoidally perturbed, and
cylindrically curved mesh geometries.

| Energy Term | Max Relative Error | Test Module |
|---|--------------------|---|
| $E_p$ (planarity) | $< 10^{-4}$        | `test_gradients.py` |
| $E_f$ (fairness) | $< 10^{-4}$        | `test_gradients.py` |
| $E_c$ (closeness) | $< 10^{-4}$        | `test_gradients.py` |
| $E_a$ (angle-balance) | $< 10^{-4}$        | `test_gradients_extended.py` |

All four terms pass verification across all tested geometries. The Numba-versus-NumPy
numerical equivalence is separately validated in `tests/test_numerical_equivalence.py` to
$10^{-10}$ for $10 \times 10$ meshes and $10^{-8}$ for $20 \times 20$ meshes (Chapter 2,
Section 2.4.3; Higham, 2002). `tests/test_gradients_extended.py` additionally verifies the
SciPy interface functions `energy_for_scipy` and `gradient_for_scipy`, confirming that the
non-finite sentinel guards ($10^{300}$ and `nan_to_num` replacement) behave correctly on
degenerate inputs without activating on well-formed meshes.