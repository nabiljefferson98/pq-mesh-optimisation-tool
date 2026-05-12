# Methodology

**PQ Mesh Optimisation — Mathematical and Algorithmic Methodology**

---

## 1. Problem Statement

An architectural surface is typically designed as a smooth freeform shape but must
be fabricated from flat panels. **Planar Quad (PQ) meshes** enable this by
requiring every quad face to be perfectly (or near-perfectly) planar, so each
face can be cut from a flat sheet.

Given an initial quad mesh $\mathcal{M} = (V, F)$ with vertex positions
$V \in \mathbb{R}^{n \times 3}$ and face connectivity $F$, the goal is to find
new vertex positions $V^*$ that minimise planarity deviation while remaining
close to the original design and preserving surface smoothness.

---

## 2. Energy Formulation

The total energy is a weighted sum of four geometric penalty terms:

$$E(V) = w_p\, E_p(V) + w_f\, E_f(V) + w_c\, E_c(V) + w_a\, E_a(V)$$

### 2.1 Planarity Energy $E_p$

The planarity energy penalises the deviation of each vertex from its face's
best-fit plane.

**Derivation:**
For face $f$ with vertices $\{v_i\}_{i=0}^{3}$:
1. Compute centroid $c_f = \frac{1}{4}\sum_i v_i$.
2. Centre the vertices: $\tilde{v}_i = v_i - c_f$.
3. Find the best-fit plane normal $\hat{n}_f$ as the left singular vector
   corresponding to the smallest singular value of the $4 \times 3$ matrix
   $[\tilde{v}_0,\, \tilde{v}_1,\, \tilde{v}_2,\, \tilde{v}_3]^T$ (SVD).
4. The signed distance of vertex $i$ from the plane is $d_{f,i} = \tilde{v}_i \cdot \hat{n}_f$.

$$E_p(V) = \sum_{f \in F} \sum_{i=0}^{3} d_{f,i}^2$$

This is $E_p = 0$ if and only if all faces are perfectly planar.
The sum-of-squares formulation gives a smooth energy landscape with a dense,
continuous gradient — critical for stable L-BFGS-B convergence.

**Implementation:** Fully vectorised on the face-vertex tensor `(F, 4, 3)` using
batched `np.linalg.svd`, with no Python loop over faces. When `cupy` is
installed, computations are dispatched to the NVIDIA GPU for ~10× speedup on
large meshes; `numba` provides parallel CPU acceleration otherwise.

**Backend dispatch (added 15 Mar 2026):**

The planarity energy computation uses a three-tier backend hierarchy to exploit
available hardware acceleration:

| Tier | Backend | Mechanism |
|------|---------|-----------|
| 1 | CuPy (GPU) | Batched CuPy SVD on device; sparse GPU matmul for scatter-add |
| 2 | Numba (CPU parallel) | `@njit(parallel=True, cache=True, fastmath=False)` kernels `_planarity_energy_numba` and `_planarity_per_face_numba`; `prange` over faces |
| 3 | NumPy (baseline) | Vectorised batched `np.linalg.svd`; validated reference path |

The Numba kernel preserves `fastmath=False` to maintain float associativity
and guarantee numerical equivalence with the NumPy baseline. This equivalence
is verified by `tests/test_numerical_equivalence.py`. The `cache=True` flag
persists JIT-compiled artefacts to `__pycache__`, eliminating compile overhead
on all subsequent runs.

All Numba try/except blocks use `except Exception` (not merely `except ImportError`)
because Numba's `@njit` decorator can raise `numba.core.errors.TypingError`,
`LoweringError`, and LLVM backend errors on platforms where Numba is installed
but LLVM compilation fails. Narrowing to `ImportError` causes a `NameError`
at runtime when the decorated symbol is referenced after a failed compilation.
Each except block emits a `warnings.warn` message to inform the user that
the NumPy fallback path is in use.

---

### 2.2 Fairness Energy $E_f$

Fairness maintains surface regularity by penalising irregular vertex positioning
relative to its neighbours via the discrete Laplacian (umbrella operator):

$$E_f(V) = \sum_{v \in V} \left\|\Delta v\right\|^2, \qquad \Delta v = v - \frac{1}{|\mathcal{N}(v)|}\sum_{u \in \mathcal{N}(u)} u$$

where $\mathcal{N}(v)$ is the set of vertices sharing an edge with $v$.

This prevents the optimiser from producing wrinkled or irregular surfaces while
minimising planarity energy.

---

### 2.3 Closeness Energy $E_c$

Closeness maintains fidelity to the original design by penalising deviation from
the initial vertex positions $V^0$:

$$E_c(V) = \sum_{v \in V} \left\|v - v^0\right\|^2$$

$V^0$ is set to the post-preprocessing (post-normalisation) positions, so the
penalty is relative to the preprocessed shape rather than the raw input.

---

### 2.4 Angle Balance Energy $E_a$ (conical constraint)

Conical meshes generalise PQ meshes and are particularly useful for
panelisation. A vertex is *conical* if all face angles at that vertex sum to
$2\pi$. The angle balance energy penalises the deviation from this condition:

$$E_a(V) = \sum_{v \in V} \left(\sum_{f \ni v} \theta_{f,v} - 2\pi\right)^2$$

where $\theta_{f,v}$ is the interior angle at vertex $v$ in face $f$.

For most optimisation runs $w_a = 0$ (PQ-only mode). Setting $w_a > 0$
encourages additional conical structure.

---

## 3. Analytical Gradients

All four energy terms are differentiable with respect to $V$. Analytical
gradients are critical for efficient quasi-Newton optimisation — numerical
finite differences would increase function evaluations by a factor of $2n$
(one perturbation per degree of freedom).

### 3.1 Planarity Gradient

Differentiating $E_p$ with respect to vertex $v_i$:

$$\frac{\partial E_p}{\partial v_i} = \sum_{f \ni i} 2\, d_{f,i}\, \hat{n}_f$$

That is, each vertex receives a gradient contribution from every face it
belongs to, proportional to its signed distance from that face's best-fit
plane, directed along the face normal.

**Assembly:** The per-face contributions are accumulated into the
per-vertex gradient via a **sparse scatter-add**:

```python
grad = scatter_matrix @ per_face_contributions   # (n_verts, 3)
```

where `scatter_matrix` is a precomputed `csr_matrix` of shape
`(n_vertices, n_faces × 4)` encoding face–vertex membership. This replaces
`np.add.at` and achieves the accumulation as a single BLAS-backed sparse
matrix multiplication.

**Backend dispatch (added 15 Mar 2026):**

`compute_planarity_gradient` uses the same three-tier dispatch as the energy:

| Tier | Backend | Kernel | Notes |
|------|---------|--------|-------|
| 1 | CuPy (GPU) | `_planarity_gradient_gpu` | GPU SVD + CuPy sparse scatter-add |
| 2 | Numba (CPU parallel) | `_planarity_gradient_contributions_numba` | `@njit(parallel=True, cache=True, fastmath=False)`; returns `(F, 4, 3)` tensor |
| 3 | NumPy (baseline) | inline batched SVD + einsum | Validated reference; always available |

Numerical equivalence validated by `TestPlanarityGradientNumbaEquivalence`
(10 tests). Tolerance: 1e-10 for meshes ≤10×10; 1e-8 for 20×20 meshes —
the observed relative error of ~2.16e-09 is consistent with deterministic LAPACK/LLVM
floating-point rounding divergence in the least-singular-value direction,
which is well understood from standard numerical analysis, and is not a race condition.
Impact on L-BFGS-B: abs_err ~9.7e-08 << gtol=1e-05.

### 3.2 Fairness Gradient

$$\frac{\partial E_f}{\partial v_i} = 2\,\Delta v_i - \frac{2}{|\mathcal{N}(i)|} \sum_{u \in \mathcal{N}(i)} \Delta u$$

### 3.3 Closeness Gradient

$$\frac{\partial E_c}{\partial v_i} = 2\,(v_i - v_i^0)$$

### 3.4 Gradient Verification

All gradients are validated against central finite differences using
`scripts/diagnostics/gradient_verification.py`. Relative error $< 10^{-5}$
is required for the analytical gradient to be accepted. This check runs as
part of the test suite (`test_gradients.py`).

---

## 4. Optimisation Algorithm

### 4.1 L-BFGS-B

The total energy is minimised using **L-BFGS-B** (Limited-memory
Broyden-Fletcher-Goldfarb-Shanno with Bounds) from `scipy.optimize.minimize`.

L-BFGS-B was chosen because:
- It is a **quasi-Newton method** — builds a low-rank approximation to the
  Hessian from $m$ recent gradient pairs, avoiding the $O(n^2)$ cost of
  storing the full Hessian.
- It handles **optional box constraints** on vertex positions
  (via `bounds_scale` parameter).
- It is robust to **ill-conditioned** objectives, which arise naturally
  when energy-term magnitudes differ by several orders of magnitude.

**Hyperparameters** (tuned empirically):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `ftol` | 1×10⁻⁹ | Tight energy-change criterion to prevent premature termination on large meshes |
| `gtol` | 1×10⁻⁵ | Primary criterion: gradient norm ≪ 1 signals genuine convergence, not plateau |
| `maxcor` | 20 | History size for Hessian approximation (doubled for smoother refinement) |
| `maxls` | 40 | Backtracking steps (doubled for flat energy regions on complex meshes) |
| `maxiter` | 1000 | Hard cap |

### 4.2 Variable Space

The optimiser works in the **flattened vertex space** $x \in \mathbb{R}^{3n}$,
where $x = V.\text{ravel}()$. The reshape to $(n, 3)$ is performed inside every
call to `energy_for_scipy` and `gradient_for_scipy`.

### 4.3 Convergence Behaviour

- **Small meshes** (≤ 400 faces): typically converge in 9–13 L-BFGS-B
  iterations, reducing planarity energy by > 99%.
- **Large meshes** (≥ 900 faces): converge in 50–200 iterations; planarity
  reduction ~83% (energy floor from shape curvature).
- **Wall-clock time**: $T(n) \approx O(n^{1.27})$ with $R^2 = 1.000$ across
  benchmarked mesh sizes (9 to 5,625 faces).

---

## 5. Preprocessing Pipeline

Real-world meshes from Blender, Rhino, and OpenSCAD require cleaning before
optimisation:

| Stage | Problem addressed | Method |
|-------|-------------------|--------|
| Duplicate vertex merge | Separate vertex copies at shared edges break the Laplacian adjacency graph | O(n²) pairwise distance scan (warns if n > 2000) |
| Degenerate face removal | Zero-area faces cause NaN in batched SVD of planarity energy | Area threshold 10⁻¹⁰ |
| Scale normalisation | A CAD model in millimetres has planarity energy ~10⁸ vs ~10⁻² for unit scale, causing L-BFGS-B line-search failure on iteration 1 | Centre at origin; scale longest bounding-box axis to `target_scale` (default 1.0) |
| Weight auto-suggestion | Optimal weights depend on mesh scale and initial energy magnitude | Target $E_p : E_f : E_c \approx 10 : 1 : 5$ at start; guard for near-PQ meshes |

---

## 6. Fabrication Export

Once optimised, each face of the PQ mesh is a planar quad that can be
manufactured from a flat sheet. The export pipeline:

1. **Unfold**: For each face, compute the best-fit plane via SVD. Construct
   a local 2D frame $\{\hat{u}, \hat{v}\}$ in the plane. Project the four
   vertices onto this frame to obtain 2D panel coordinates.

2. **Quality check**: The planarity residual (max out-of-plane component
   discarded during projection) is computed and reported. If residual
   exceeds a tolerance (default 5 mm at model scale), a warning is emitted.

3. **Export** to DXF (CNC/laser cutting) or SVG (browser/Illustrator
   preview). Both use atomic writes via `tempfile.mkstemp` + `os.replace`
   to prevent partial output files.

---

## 7. Error Handling and Robustness

Identified failure modes and their mitigations (implemented as part of the
security and robustness audit, March 2026):

| Failure mode | Risk | Mitigation |
|---|---|---|
| Negative face indices in OBJ | Silent numpy wrap-around gives wrong vertex positions | `QuadMesh.__init__` raises `ValueError` |
| NaN/Inf propagated by scipy | Divergent optimisation, silent wrong result | `energy_for_scipy` returns 1×10³⁰⁰ fallback; `gradient_for_scipy` zeros non-finite entries |
| Mesh state corruption in callback | If callback raises, original vertex positions lost | `try/finally` in `_create_callback` always restores `verts_backup` |
| Path traversal in `save_obj` | Write to arbitrary filesystem locations | `Path.resolve()` + check for `..` in path parts |
| DXF layer name injection | Special characters break DXF format or allow injection | Sanitise with `re.sub(r"[^A-Za-z0-9_\-]", "_", name)[:255]` |
| Partial export file on crash | Corrupt DXF/SVG undetected by downstream tools | Atomic write: write to tempfile, rename on success, delete on failure |
| Empty mesh in `suggest_weights` | `vertices.max(axis=0)` raises on zero-row array | Early return with safe defaults |
| Malformed OBJ tokens | `float()` / `int()` crash halts entire load | Per-token `try/except`, skip + warn, cap warnings at 5 |
| Dead `isinstance(x, tuple)` on energy snapshots | Future API change to `compute_total_energy` silently returns tuple; optimiser uses wrong scalar value without raising | Replaced with `assert isinstance(_initial_raw / _final_raw, float)` guards (added 15 Mar 2026) |
| Numba `@njit` compilation failure (non-ImportError) | `except ImportError` too narrow — `TypingError` / LLVM errors escape the handler; decorated name referenced before assignment causes `NameError`, failing 7 tests | Broadened to `except Exception as _numba_*_exc` with `warnings.warn` fallback in all three Numba try blocks (added 15 Mar 2026) |

---

## 8. References

1. Liu, Y., Pottmann, H., Wallner, J., Yang, Y., & Wang, W. (2006).
   *Geometric modeling with conical meshes and developable surfaces.*
   ACM Transactions on Graphics, 25(3), 681–689.

2. Pottmann, H., Asperl, A., Hofer, M., & Kilian, A. (2007).
   *Architectural geometry.* Bentley Institute Press.

3. Pottmann, H., Liu, Y., Wallner, J., Bobenko, A., & Wang, W. (2007).
   *Geometry of multi-layer freeform structures for architecture.*
   ACM Transactions on Graphics, 26(3), Article 65.

4. Nocedal, J., & Wright, S. J. (2006).
   *Numerical Optimization* (2nd ed.). Springer.

5. Zhu, C., Byrd, R. H., Lu, P., & Nocedal, J. (1997).
   Algorithm 778: L-BFGS-B — Fortran subroutines for large-scale bound-constrained
   optimisation. *ACM Transactions on Mathematical Software*, 23(4), 550–560.

6. Crane, K., de Goes, F., Desbrun, M., & Schröder, P. (2013).
   *Digital geometry processing with discrete exterior calculus.*
   ACM SIGGRAPH Courses.

7. Lam, S. K., Pitrou, A., & Seibert, S. (2015).
   Numba: A LLVM-based Python JIT compiler.
   *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*, 1–6.

8. Higham, N. J. (2002).
   *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
   [Forward error bounds for floating-point arithmetic chains; used to derive
   the 1e-8 tolerance in `TestPlanarityGradientNumbaEquivalence`.]
