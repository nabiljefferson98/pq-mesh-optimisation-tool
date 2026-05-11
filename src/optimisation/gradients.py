"""
src/optimisation/gradients.py

Gradient computation for PQ mesh optimisation.

This module computes the analytical gradient of the total energy function
with respect to the position of every vertex in the mesh. A gradient is a
vector that points in the direction of the steepest energy increase for each
vertex. The optimiser uses the negative of this gradient to move vertices
towards lower energy, which means flatter, smoother, and more developable
geometry.

Computing gradients analytically (by mathematical derivation) is far more
efficient than estimating them numerically, because numerical estimation
requires evaluating the full energy function twice for every single vertex
coordinate. For a mesh with n vertices, that is 6n energy evaluations per
iteration compared to just one with analytical gradients.

Four individual gradient functions are provided, one for each energy term:

  Planarity gradient -- direction to move each vertex to reduce the
                             non-planarity of its adjacent faces.

  Fairness gradient -- direction to move each vertex to improve
                             smoothness relative to its neighbours.

  Closeness gradient -- direction to move each vertex back towards
                             its original position (pulls against drift).

  Angle balance gradient -- direction to move each vertex to bring the
                             alternating face angles towards equality,
                             improving developability.

The combined gradient is the weighted sum of all four individual gradients,
matching the same weights used in the total energy function.

Backend dispatch
----------------
Every gradient function automatically selects the fastest available
computation path in the following order of priority:

  Tier 1 CuPy (NVIDIA GPU) -- batched SVD and sparse matrix multiply
                                    on GPU via CuPy. Requires HAS_CUDA.
  Tier 2 Numba (CPU, parallel) -- custom @njit kernels compiled to native
                                    machine code, parallelled across faces
                                    or vertices using prange.
  Tier 3 NumPy (CPU, serial) -- fully vectorised NumPy operations.
                                    Always available as a fallback.

'Src/backends.py' determines backend availability.

Changelog
---------
  15 March 2026:
    Added '_planarity_gradient_contributions_numba' Tier-2 kernel for the
    planarity gradient (previously missing).
    Added '_HAS_NUMBA_PLANARITY_GRAD' flag and '_planarity_grad_kern' symbol.
    Broadened 'except ImportError' to 'except Exception' throughout all
    Numba dispatch blocks, as Numba compilation errors are not always
    ImportError instances.
    Fixed '_planarity_gradient_gpu' to fall back to Numba (Tier 2) before
    NumPy (Tier 3) on GPU failure.

References
----------
Nocedal, J. and Wright, S. J. (2006).
  Numerical Optimization. 2nd ed. Springer.

Crane, K., de Goes, F., Desbrun, M., and Schroder, P. (2013).
  "Digital geometry processing with discrete exterior calculus."
  ACM SIGGRAPH 2013 Courses.

Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.

Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.

Author: Muhammad Nabil
Date: 2 February 2026 | Updated: 15 March 2026
"""

import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import numpy as _np_const

from src.optimisation.mesh_geometry import (  # noqa: F401 (kept for external use)
    compute_all_planarity_deviations,
)

logger = logging.getLogger(__name__)

"""
Minimum sine value used during angle gradient computation.
Prevents the gradient from becoming arbitrarily large near degenerate
face angles (angles very close to 0 or 180 degrees). In practice,
no valid architectural quad mesh will have face angles outside the
range this threshold implies, so clamping here is safe.
"""
_SIN_MIN_ANGLE_GRAD = 1e-2

"""
Sign pattern for the alternating-angle imbalance formula.
The conical condition is (alpha_0 + alpha_2) - (alpha_1 + alpha_3) == 0.
When differentiating with respect to each angle in turn, the signs
alternate as +1, -1, +1, -1. Storing this as a typed NumPy array
is required for safe type inference inside Numba compiled kernels.
"""
_ANGLE_SIGNS = _np_const.array([1.0, -1.0, 1.0, -1.0], dtype=_np_const.float64)

"""
============================================================================
 TIER-2 NUMBA KERNEL: PLANARITY GRADIENT CONTRIBUTIONS
============================================================================

 Why a separate Numba kernel for the planarity gradient?
 -------------------------------------------------------
 Computing the planarity gradient requires scattering contributions from
 faces back to vertices: each vertex accumulates gradient contributions
 from every face it belongs to. A naive parallel loop over faces would
 cause write conflicts when two threads try to update the same vertex
 simultaneously.

 This is resolved by splitting the computation into two stages:

   Stage A (face-parallel, this kernel):
     For each face, compute the per-vertex gradient contribution as a
     local (4 x 3) block. Each face writes only to its own block in the
     output array, so there are no write conflicts and full parallelism
     is achieved using prange.

   Stage B (scatter-add, caller):
     Accumulate all per-face contributions into the vertex gradient array
     using a precomputed sparse scatter matrix (mesh.scatter_matrix). This
     is a single BLAS-backed sparse matrix multiply with no Python loop.

 How the face normal is computed inside the kernel
 -------------------------------------------------
 The normal direction for each face is the eigenvector corresponding to
 the smallest eigenvalue of the 3x3 covariance matrix formed from the
 centred vertex positions. Instead of calling SVD (which is expensive
 inside a Numba kernel), the smallest eigenvalue is obtained using
 Cardano's closed-form formula and the eigenvector is recovered using
 the cross-product method. This is numerically equivalent to NumPy SVD
 and is validated by the test suite.

 Compiler settings
 -----------------
 fastmath=False: preserves floating-point associativity, ensuring results
                 match the NumPy baseline in numerical equivalence tests.
 cache=True:     compiled binary is saved to disk and reused on subsequent
                 runs, eliminating 2-5 seconds of compile time.
 parallel=True:  Numba spawns a thread pool; each face is processed by
                 one thread. Thread count defaults to os.cpu_count().
"""

_HAS_NUMBA_PLANARITY_GRAD = False
_planarity_grad_kern = None  # set to the @njit function if compilation succeeds

try:
    import numpy as _np_pg
    from numba import njit as _njit_pg
    from numba import prange as _prange_pg

    @_njit_pg(parallel=True, cache=True, fastmath=False)  # pragma: no cover
    def _planarity_gradient_contributions_numba(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """
        Numba-parallel planarity gradient contribution's kernel.

        For each quad face, computes the gradient contribution of each of
        its four vertices with respect to the planarity energy of that face.
        The contribution for vertex i in face f is 2 multiplied by the signed
        distance of that vertex from the face's best-fit plane, multiplied by
        the face normal vector. These per-face contributions are returned in
        a three-dimensional array and then scattered to vertices by the caller
        using the precomputed sparse scatter matrix.

        This function is never called directly. It is compiled to native machine
        code by Numba and dispatched by 'compute_planarity_gradient' when
        Numba is available.

        Parameters
        ----------
        vertices : numpy.ndarray
            Vertex position array of shape (n_vertices, 3), dtype float64.
        faces : numpy.ndarray
            Face index array of shape (n_faces, 4), dtype int64.

        Returns
        -------
        numpy.ndarray
            Gradient contribution array of shape (n_faces, 4, 3), dtype
            float64. To get the final vertex gradient, the caller must
            compute: scatter_matrix @ contrib.reshape(-1, 3).
        """
        n_faces = faces.shape[0]
        contrib = _np_pg.zeros((n_faces, 4, 3), dtype=_np_pg.float64)

        for fi in _prange_pg(n_faces):
            v0 = faces[fi, 0]
            v1 = faces[fi, 1]
            v2 = faces[fi, 2]
            v3 = faces[fi, 3]

            # ── Centroid ────────────────────────────────────────────────────
            cx = (
                vertices[v0, 0] + vertices[v1, 0] + vertices[v2, 0] + vertices[v3, 0]
            ) * 0.25
            cy = (
                vertices[v0, 1] + vertices[v1, 1] + vertices[v2, 1] + vertices[v3, 1]
            ) * 0.25
            cz = (
                vertices[v0, 2] + vertices[v1, 2] + vertices[v2, 2] + vertices[v3, 2]
            ) * 0.25

            # ── Centred positions (4×3) ────────────────────────────────────
            t = _np_pg.empty((4, 3), dtype=_np_pg.float64)
            t[0, 0] = vertices[v0, 0] - cx
            t[0, 1] = vertices[v0, 1] - cy
            t[0, 2] = vertices[v0, 2] - cz
            t[1, 0] = vertices[v1, 0] - cx
            t[1, 1] = vertices[v1, 1] - cy
            t[1, 2] = vertices[v1, 2] - cz
            t[2, 0] = vertices[v2, 0] - cx
            t[2, 1] = vertices[v2, 1] - cy
            t[2, 2] = vertices[v2, 2] - cz
            t[3, 0] = vertices[v3, 0] - cx
            t[3, 1] = vertices[v3, 1] - cy
            t[3, 2] = vertices[v3, 2] - cz

            # ── A = M^T M (3×3 symmetric covariance) ───────────────────────
            a00 = (
                t[0, 0] * t[0, 0]
                + t[1, 0] * t[1, 0]
                + t[2, 0] * t[2, 0]
                + t[3, 0] * t[3, 0]
            )
            a01 = (
                t[0, 0] * t[0, 1]
                + t[1, 0] * t[1, 1]
                + t[2, 0] * t[2, 1]
                + t[3, 0] * t[3, 1]
            )
            a02 = (
                t[0, 0] * t[0, 2]
                + t[1, 0] * t[1, 2]
                + t[2, 0] * t[2, 2]
                + t[3, 0] * t[3, 2]
            )
            a11 = (
                t[0, 1] * t[0, 1]
                + t[1, 1] * t[1, 1]
                + t[2, 1] * t[2, 1]
                + t[3, 1] * t[3, 1]
            )
            a12 = (
                t[0, 1] * t[0, 2]
                + t[1, 1] * t[1, 2]
                + t[2, 1] * t[2, 2]
                + t[3, 1] * t[3, 2]
            )
            a22 = (
                t[0, 2] * t[0, 2]
                + t[1, 2] * t[1, 2]
                + t[2, 2] * t[2, 2]
                + t[3, 2] * t[3, 2]
            )

            # ── Smallest eigenvector of A ────────────────────────────────────
            # Step 1: Cardano's closed-form eigenvalue λ_min
            p1 = a01 * a01 + a02 * a02 + a12 * a12

            if p1 == 0.0:
                # A is already diagonal, eigenvalue/vector trivial
                if a00 <= a11 and a00 <= a22:
                    nx, ny, nz = 1.0, 0.0, 0.0
                elif a11 <= a00 and a11 <= a22:
                    nx, ny, nz = 0.0, 1.0, 0.0
                else:
                    nx, ny, nz = 0.0, 0.0, 1.0
            else:
                q = (a00 + a11 + a22) / 3.0
                p2 = (
                    (a00 - q) * (a00 - q)
                    + (a11 - q) * (a11 - q)
                    + (a22 - q) * (a22 - q)
                    + 2.0 * p1
                )
                p = (p2 / 6.0) ** 0.5

                b00 = (a00 - q) / p
                b01 = a01 / p
                b02 = a02 / p
                b11 = (a11 - q) / p
                b12 = a12 / p
                b22 = (a22 - q) / p

                det_half = (
                    b00 * (b11 * b22 - b12 * b12)
                    - b01 * (b01 * b22 - b12 * b02)
                    + b02 * (b01 * b12 - b11 * b02)
                ) * 0.5

                # Clamp argument of arccos to [-1, 1]
                if det_half < -1.0:
                    det_half = -1.0
                elif det_half > 1.0:
                    det_half = 1.0

                phi = _np_pg.arccos(det_half) / 3.0
                # +2π/3 shift selects the smallest of the three eigenvalues
                lam_min = q + 2.0 * p * _np_pg.cos(phi + 2.0943951023931953)

                # Step 2: Cross-product method, find the most stable
                # column of null((A − λ_min I)) by taking pairwise
                # cross-products of the three rows and selecting the one
                # with the largest magnitude.
                r00 = a00 - lam_min
                r11 = a11 - lam_min
                r22 = a22 - lam_min

                # Row0 × Row1  [r00,a01,a02] × [a01,r11,a12]
                cx0 = a01 * a12 - a02 * r11
                cy0 = a02 * a01 - r00 * a12
                cz0 = r00 * r11 - a01 * a01
                m0 = (cx0 * cx0 + cy0 * cy0 + cz0 * cz0) ** 0.5

                # Row0 × Row2  [r00,a01,a02] × [a02,a12,r22]
                cx1 = a01 * r22 - a02 * a12
                cy1 = a02 * a02 - r00 * r22
                cz1 = r00 * a12 - a01 * a02
                m1 = (cx1 * cx1 + cy1 * cy1 + cz1 * cz1) ** 0.5

                # Row1 × Row2  [a01,r11,a12] × [a02,a12,r22]
                cx2 = r11 * r22 - a12 * a12
                cy2 = a12 * a02 - a01 * r22
                cz2 = a01 * a12 - r11 * a02
                m2 = (cx2 * cx2 + cy2 * cy2 + cz2 * cz2) ** 0.5

                if m0 >= m1 and m0 >= m2:
                    inv_m = 1.0 / (m0 + 1e-300)
                    nx, ny, nz = cx0 * inv_m, cy0 * inv_m, cz0 * inv_m
                elif m1 >= m0 and m1 >= m2:
                    inv_m = 1.0 / (m1 + 1e-300)
                    nx, ny, nz = cx1 * inv_m, cy1 * inv_m, cz1 * inv_m
                else:
                    inv_m = 1.0 / (m2 + 1e-300)
                    nx, ny, nz = cx2 * inv_m, cy2 * inv_m, cz2 * inv_m

            # ── Gradient contributions  2·d_i·n̂_f ─────────────────────────
            for li in range(4):
                d = t[li, 0] * nx + t[li, 1] * ny + t[li, 2] * nz
                contrib[fi, li, 0] = 2.0 * d * nx
                contrib[fi, li, 1] = 2.0 * d * ny
                contrib[fi, li, 2] = 2.0 * d * nz

        return contrib

    _HAS_NUMBA_PLANARITY_GRAD = True
    _planarity_grad_kern = _planarity_gradient_contributions_numba

except Exception as _numba_planarity_grad_exc:
    warnings.warn(
        f"Numba planarity gradient kernel unavailable "
        f"({type(_numba_planarity_grad_exc).__name__}: "
        f"{_numba_planarity_grad_exc}). "
        "NumPy baseline will be used for the planarity gradient.",
        RuntimeWarning,
        stacklevel=1,
    )

# ============================================================================
# ANALYTICAL GRADIENTS FOR INDIVIDUAL ENERGY TERMS
# ============================================================================


def compute_planarity_gradient(mesh) -> np.ndarray:
    """
    Compute the gradient of planarity energy with respect to vertex positions.

    For each vertex, this gradient points in the direction that would most
    increase the total planarity energy if the vertex moved that way. The
    optimiser moves in the opposite direction to reduce planarity error.

    The gradient for vertex i is the sum of contributions from every face
    that contains vertex i. For each such face, the contribution is twice
    the signed distance of vertex i from the face's best-fit plane,
    multiplied by the face's unit normal vector. This derivation follows
    directly from differentiating the SVD-based planarity energy expression.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh whose planarity gradient is to be computed. Must have
        'vertices', 'faces', and 'scatter_matrix' attributes.

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (n_vertices, 3), dtype float64. Each row
        gives the gradient vector for the corresponding vertex. Always
        returned as a CPU NumPy array regardless of which backend is used.

    Notes
    -----
    Dispatches to three tiers: CuPy GPU (Tier 1), Numba parallel CPU
    (Tier 2), or NumPy batched SVD (Tier 3). All tiers use the same
    sparse scatter matrix for the final accumulation step and produce
    numerically equivalent results.

    References
    ----------
    Liu et al. (2006). "Geometric modelling with conical meshes and
    developable surfaces." ACM Transactions on Graphics, 25(3), 681-689.
    """
    # ── Tier 1: CuPy GPU ────────────────────────────────────────────────────
    try:
        from src.backends import HAS_CUDA

        if HAS_CUDA:
            return _planarity_gradient_gpu(mesh)
    except ImportError:
        pass

    # ── Tier 2: Numba CPU-parallel ───────────────────────────────────────────
    try:
        from src.backends import HAS_NUMBA

        if HAS_NUMBA and _HAS_NUMBA_PLANARITY_GRAD and _planarity_grad_kern is not None:
            contrib = _planarity_grad_kern(
                mesh.vertices.astype(np.float64),
                mesh.faces.astype(np.int64),
            )  # (F, 4, 3)
            # Scatter-add: single BLAS-backed sparse matmul (no Python loop).
            # scatter_matrix shape: (n_verts, F*4), which built once, cached.
            return mesh.scatter_matrix @ contrib.reshape(-1, 3)
    except Exception as _numba_pg_dispatch_exc:
        warnings.warn(
            f"Numba planarity gradient dispatch failed "
            f"({type(_numba_pg_dispatch_exc).__name__}): "
            f"{_numba_pg_dispatch_exc}. Falling back to NumPy.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Tier 3: NumPy baseline ───────────────────────────────────────────────
    # Vectorised batched SVD, no Python loop over faces.
    face_verts = mesh.vertices[mesh.faces]  # (F, 4, 3)
    centroids = face_verts.mean(axis=1, keepdims=True)  # (F, 1, 3)
    centered = face_verts - centroids  # (F, 4, 3)

    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)  # Vt: (F, 3, 3)
        normals = Vt[:, -1, :]  # (F, 3)

        signed_dists = np.einsum("fvd,fd->fv", centered, normals)  # (F, 4)
        contributions = (  # (F, 4, 3)
            2.0 * signed_dists[:, :, None] * normals[:, None, :]
        )

        # Scatter-add via precomputed sparse matrix (BLAS matmul, no Python loop).
        return mesh.scatter_matrix @ contributions.reshape(-1, 3)
    except np.linalg.LinAlgError:
        # SVD convergence failure (e.g., degenerate faces with NaN/Inf vertices)
        return np.zeros((mesh.n_vertices, 3), dtype=np.float64)


def compute_fairness_gradient(mesh) -> np.ndarray:
    """
    Compute the gradient of fairness energy with respect to vertex positions.

    Fairness energy measures how much each vertex deviates from the average
    of its neighbours. Differentiating this with respect to vertex positions
    gives a gradient that can be expressed compactly using the discrete
    Laplacian matrix L. The gradient for the entire mesh is 2 multiplied by
    L transposed multiplied by (L multiplied by the vertex matrix). Because
    L is symmetric for the combinatorial Laplacian used here, this simplifies
    to 2 times L squared times the vertex matrix.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. The Laplacian matrix is accessed via
        'mesh.laplacian' (a sparse matrix, cached at construction).

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (n_vertices, 3), dtype float64.

    Notes
    -----
    There is no Numba tier for this gradient because sparse matrix multiply
    is already highly optimised on CPU through SciPy's BLAS backend. The
    two computation paths are CuPy GPU (Tier 1) and SciPy sparse CPU (Tier 3).

    References
    ----------
    Crane et al. (2013). "Digital geometry processing with discrete
    exterior calculus." ACM SIGGRAPH 2013 Courses.
    """
    # ── Tier 1: CuPy GPU ────────────────────────────────────────────────────
    try:
        from src.backends import HAS_CUDA

        if HAS_CUDA:
            return _fairness_gradient_gpu(mesh)
    except ImportError:
        pass

    # ── Tier 3: NumPy/SciPy baseline (Tier 2 not applicable) ────────────────
    L = mesh.laplacian
    return 2.0 * (L.T @ (L @ mesh.vertices))


def compute_closeness_gradient(mesh) -> np.ndarray:
    """
    Compute the gradient of closeness energy with respect to vertex positions.

    Closeness energy is the sum of squared distances between each vertex and
    its original position. Differentiating this with respect to vertex i gives
    simply 2 times the displacement vector (current position minus original
    position). This is the simplest of the four gradients: it is exact,
    requires no approximation, and can be computed as a single array
    subtraction.

    Physically, this gradient acts like a spring: the further a vertex has
    moved from its original position, the stronger the force pulling it back.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. Must have 'vertices' (current positions)
        and 'vertices_original' (positions at the start of optimisation).

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (n_vertices, 3), dtype float64.

    Notes
    -----
    Dispatches to CuPy GPU (Tier 1) if available, otherwise computes
    directly in NumPy (Tier 3). No Numba tier is needed because the
    computation is a single vectorised array operation.
    """
    try:
        from src.backends import HAS_CUDA

        if HAS_CUDA:
            from src.backends import to_device, to_numpy

            v_gpu = to_device(mesh.vertices)
            v0_gpu = to_device(mesh.vertices_original)
            return to_numpy(2.0 * (v_gpu - v0_gpu))
    except (ImportError, Exception):
        pass
    return 2.0 * (mesh.vertices - mesh.vertices_original)


def compute_angle_balance_gradient(mesh) -> np.ndarray:
    """
    Compute the analytical gradient of the angle balance (conical) energy.

    The angle balance energy penalises 4-valent interior vertices where
    the alternating face angles do not sum to equal values. At such a
    vertex, four faces meet with corner angles alpha_0, alpha_1, alpha_2,
    alpha_3 in cyclic order. The imbalance is (alpha_0 + alpha_2) minus
    (alpha_1 + alpha_3). The energy is the square of this imbalance.

    Differentiating the energy with respect to vertex positions gives a
    gradient that depends on all vertices involved in the four surrounding
    faces: the central vertex itself, plus its eight nearest neighbours
    (two per face). The gradient is computed analytically using the standard
    formula for the derivative of the angle between two edge vectors.

    For each face angle at vertex v, three gradient contributions arise:
    one for v itself, one for the previous vertex in the face, and one
    for the next vertex in the face. These are accumulated by sign according
    to which angles are being added versus subtracted in the imbalance
    formula.

    Vertices with fewer or more than four incident faces contribute zero
    to the gradient, as the conical condition is not defined for them.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. Must have 'vertices', 'faces',
        'vertex_face_ids_padded', and 'angle_balance_scratch' attributes.

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (n_vertices, 3), dtype float64.

    Notes
    -----
    Dispatches to Numba parallel CPU (Tier 2) if available, otherwise
    uses a serial NumPy loop (Tier 3). There is no GPU tier for this
    gradient due to the irregular vertex-to-face adjacency structure.

    The analytical computation replaces a previous numerical finite-difference
    approximation, reducing the computational complexity from O(6n squared)
    energy evaluations to O(n) arithmetic operations per iteration.

    References
    ----------
    Liu et al. (2006). "Geometric modelling with conical meshes and
    developable surfaces." ACM Transactions on Graphics, 25(3), 681-689.
    """
    # ── Tier 2: Numba CPU-parallel ───────────────────────────────────────────
    try:
        from src.backends import HAS_NUMBA

        if HAS_NUMBA:
            vf = mesh.vertex_face_ids_padded
            scratch = mesh.angle_balance_scratch
            return _angle_balance_gradient_numba(
                mesh.vertices, mesh.faces, vf, *scratch
            )
    except Exception as _numba_angle_exc:
        warnings.warn(
            f"Numba angle-balance gradient kernel failed "
            f"({type(_numba_angle_exc).__name__}: {_numba_angle_exc}); "
            "falling back to NumPy implementation.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Tier 3: NumPy baseline ───────────────────────────────────────────────
    grad = np.zeros_like(mesh.vertices)  # (n_vertices, 3)

    for vid in range(mesh.n_vertices):
        incident = mesh.get_vertex_faces(vid)

        # Conical condition is only defined for 4-valent interior vertices.
        # Boundary or irregular-valence vertices contribute zero gradient.
        if len(incident) != 4:
            continue

        v = mesh.vertices[vid]

        angles: list = []  # α_i for i=0..3
        grad_v: list = []  # ∂α_i/∂v (shape 3)
        grad_v_prev: list = []  # ∂α_i/∂v_prev (shape 3)
        grad_v_next: list = []  # ∂α_i/∂v_next (shape 3)
        prev_ids: list = []  # vertex index of v_prev in face i
        next_ids: list = []  # vertex index of v_next in face i

        for face_id in incident:
            face = mesh.faces[face_id]
            n_f = len(face)
            local_idx = int(np.where(face == vid)[0][0])

            vp_id = int(face[(local_idx - 1) % n_f])
            vn_id = int(face[(local_idx + 1) % n_f])
            v_prev = mesh.vertices[vp_id]
            v_next = mesh.vertices[vn_id]

            e1 = v_prev - v  # (3,)
            e2 = v_next - v  # (3,)
            l1 = np.linalg.norm(e1)
            l2 = np.linalg.norm(e2)

            if l1 < 1e-12 or l2 < 1e-12:
                angles.append(0.0)
                grad_v.append(np.zeros(3))
                grad_v_prev.append(np.zeros(3))
                grad_v_next.append(np.zeros(3))
                prev_ids.append(vp_id)
                next_ids.append(vn_id)
                continue

            e1h = e1 / l1
            e2h = e2 / l2
            c = float(np.clip(np.dot(e1h, e2h), -1.0 + 1e-8, 1.0 - 1e-8))
            s = float(max(np.sqrt(1.0 - c * c), _SIN_MIN_ANGLE_GRAD))

            alpha = float(np.arccos(c))
            inv_s = 1.0 / s

            dv = inv_s * ((e2h - c * e1h) / l1 + (e1h - c * e2h) / l2)
            dvp = (-inv_s) * (e2h - c * e1h) / l1
            dvn = (-inv_s) * (e1h - c * e2h) / l2

            angles.append(alpha)
            grad_v.append(dv)
            grad_v_prev.append(dvp)
            grad_v_next.append(dvn)
            prev_ids.append(vp_id)
            next_ids.append(vn_id)

        if len(angles) < 4:
            continue

        delta = (angles[0] + angles[2]) - (angles[1] + angles[3])
        signs = [+1.0, -1.0, +1.0, -1.0]
        two_delta = 2.0 * delta

        for i in range(4):
            si = signs[i]
            grad[vid] += two_delta * si * grad_v[i]
            grad[prev_ids[i]] += two_delta * si * grad_v_prev[i]
            grad[next_ids[i]] += two_delta * si * grad_v_next[i]

    return grad


def compute_angle_balance_energy_scalar(mesh) -> float:
    """
    Return the angle balance energy as a scalar float.

    This is a thin wrapper around 'compute_angle_balance_energy' from
    'energy_terms.py'. It exists solely to provide a consistent callable
    interface for the numerical gradient verifier, which requires a function
    that accepts only a mesh and returns a single float.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate.

    Returns
    -------
    float
        Total angle balance energy.
    """
    from src.optimisation.energy_terms import compute_angle_balance_energy

    return compute_angle_balance_energy(mesh)


# ============================================================================
# COMBINED GRADIENT
# ============================================================================


def compute_total_gradient(
    mesh, weights: Dict[str, float], use_numerical: bool = False
) -> np.ndarray:
    """
    Compute the gradient of the total weighted energy with respect to
    all vertex positions.

    The total gradient is the weighted sum of the four individual gradients.
    Because differentiation is a linear operation, the gradient of a weighted
    sum equals the weighted sum of individual gradients. This means each
    gradient term is computed independently and then combined with its weight.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate.
    weights : dict
        Dictionary mapping energy term names to their scalar weights.
        Required keys: 'planarity', 'fairness', 'closeness'.
        Optional key: 'angle_balance' (defaults to 0.0 if absent).
    use_numerical : bool, optional
        If True, compute the gradient using central finite differences
        instead of the analytical formulae. This is intended only for
        verification and debugging. Numerical gradients are significantly
        slower and should never be used during optimisation.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Total gradient array of shape (n_vertices, 3), dtype float64.
        Each row is the gradient vector for the corresponding vertex.

    Raises
    ------
    ValueError
        If any of the required weight keys are absent from 'weights'.
    """
    if use_numerical:
        from src.optimisation.energy_terms import compute_total_energy

        return compute_numerical_gradient(
            mesh, lambda m: compute_total_energy(m, weights), epsilon=1e-6
        )

    for key in ("planarity", "fairness", "closeness"):
        if key not in weights:
            raise ValueError(
                f"compute_total_gradient: missing required weight '{key}'. "
                f"Received keys: {list(weights.keys())}"
            )

    grad_planarity = compute_planarity_gradient(mesh)
    grad_fairness = compute_fairness_gradient(mesh)
    grad_closeness = compute_closeness_gradient(mesh)

    grad_total = (
        weights["planarity"] * grad_planarity
        + weights["fairness"] * grad_fairness
        + weights["closeness"] * grad_closeness
    )

    if weights.get("angle_balance", 0.0) > 0:
        grad_total += weights["angle_balance"] * compute_angle_balance_gradient(mesh)

    return grad_total


# ============================================================================
# NUMERICAL GRADIENT VERIFICATION
# ============================================================================


def compute_numerical_gradient(mesh, energy_func, epsilon: float = 1e-6) -> np.ndarray:
    """
    Estimate the gradient using central finite differences.

    For each vertex coordinate, the gradient is approximated by perturbing
    that coordinate by a small amount in both directions and measuring the
    change in energy. The formula is:

      gradient[i, j] = (E(v_ij + epsilon) - E(v_ij - epsilon)) / (2 * epsilon)

    This approach requires two energy evaluations for every scalar coordinate,
    giving a total of 2 * n_vertices * 3 evaluations per gradient call. It
    is accurate to the second order in epsilon and is used exclusively to verify
    that the analytical gradients are correctly derived.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to differentiate. Vertex positions are temporarily modified
        during computation and always restored to their original values, even
        if an exception occurs.
    energy_func : callable
        A function that accepts a QuadMesh and returns scalar float energy.
    epsilon : float, optional
        The finite difference step size. The default value of 1e-6 gives
        a good balance between truncation error and floating-point cancellation
        for double-precision arithmetic. The default is 1e-6.

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (n_vertices, 3), dtype float64.

    Notes
    -----
    This function should never be used during actual optimisation. It is
    intended only for testing and verification. Use 'compute_total_gradient'
    with 'use_numerical=False' for all production use.
    """
    n_verts = mesh.n_vertices
    gradient = np.zeros((n_verts, 3))

    for i in range(n_verts):
        for j in range(3):
            original = mesh.vertices[i, j]
            try:
                mesh.vertices[i, j] = original + epsilon
                E_plus = energy_func(mesh)
                mesh.vertices[i, j] = original - epsilon
                E_minus = energy_func(mesh)
                gradient[i, j] = (E_plus - E_minus) / (2.0 * epsilon)
            finally:
                mesh.vertices[i, j] = original

    return gradient


def compute_numerical_gradient_term(
    mesh, energy_func, epsilon: float = 1e-6
) -> np.ndarray:
    """
    Convenience wrapper around 'compute_numerical_gradient'.

    Provided for backwards compatibility. All parameters and return values
    are identical to 'compute_numerical_gradient'. Prefer calling
    'compute_numerical_gradient' directly in new code.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to differentiate.
    energy_func : callable
        A function that accepts a QuadMesh and returns a scalar float.
    epsilon : float, optional
        Finite difference step size. The default is 1e-6.

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (n_vertices, 3), dtype float64.
    """
    return compute_numerical_gradient(mesh, energy_func, epsilon)


def verify_gradient(
    mesh, weights: Dict[str, float], tolerance: float = 1e-4, verbose: bool = True
) -> Tuple[bool, float]:
    """
    Verify the analytical gradient against a numerical finite-difference estimate.

    Computes both the analytical total gradient and the numerical total
    gradient, then measures their relative difference. If the relative
    error is below the given tolerance, the analytical gradient is considered
    correct. This check is run during testing to guard against errors in the
    mathematical derivation.

    The relative error is computed as the Euclidean norm of the difference
    divided by the norm of the numerical gradient. A small relative error
    confirms that the analytical gradient closely matches the numerical
    estimate.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to test. Should be a small mesh for speed.
    weights : dict
        Weight dictionary with keys 'planarity', 'fairness', 'closeness',
        and optionally 'angle_balance'.
    tolerance : float, optional
        Maximum acceptable relative error. Values below 1e-4 are considered
        acceptable for gradient-based optimisation. Default is 1e-4.
    verbose : bool, optional
        If True, print a formatted comparison table to standard output.
        Default is True.

    Returns
    -------
    tuple
        A two-element tuple (is_correct, relative_error) where 'is_correct'
        is a bool and 'relative_error' is a float.
    """
    grad_analytical = compute_total_gradient(mesh, weights, use_numerical=False)
    grad_numerical = compute_total_gradient(mesh, weights, use_numerical=True)

    diff = grad_analytical - grad_numerical
    error_abs = np.linalg.norm(diff)
    norm_numerical = np.linalg.norm(grad_numerical)

    relative_error = error_abs / norm_numerical if norm_numerical > 1e-10 else error_abs
    is_correct = bool(relative_error < tolerance)

    if verbose:
        print("=" * 70)
        print("GRADIENT VERIFICATION")
        print("=" * 70)
        print(f"Analytical gradient norm : {np.linalg.norm(grad_analytical):.6e}")
        print(f"Numerical  gradient norm : {np.linalg.norm(grad_numerical):.6e}")
        print(f"Absolute difference      : {error_abs:.6e}")
        print(f"Relative error           : {relative_error:.6e}")
        print(f"Tolerance                : {tolerance:.6e}")
        print()
        if is_correct:
            print("✓ PASSED: Gradient verification successful")
        else:
            print("✗ FAILED: Gradient error exceeds tolerance")
            print("  Consider: reduce step size, check analytical derivation")
        print("=" * 70)

    return is_correct, float(relative_error)


# ============================================================================
# GRADIENT-BASED DIAGNOSTICS
# ============================================================================


def compute_gradient_statistics(gradient: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics about the size of a gradient array.

    Useful for monitoring gradient health during optimisation: very large
    gradients suggest numerical instability, while very small gradients
    near zero suggest convergence.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array of shape (n_vertices, 3).

    Returns
    -------
    dict
        Dictionary with the following keys and float values:
        'norm' -- Frobenius norm of the full gradient matrix.
        'max_magnitude' -- Largest per-vertex gradient magnitude.
        'mean_magnitude' -- Mean per-vertex gradient magnitude.
        'std_magnitude' -- Standard deviation of per-vertex magnitudes.
        'max_component' -- Largest absolute value of any single coordinate.
        'min_magnitude' -- Smallest per-vertex gradient magnitude.
    """
    vertex_magnitudes = np.linalg.norm(gradient, axis=1)
    return {
        "norm": float(np.linalg.norm(gradient)),
        "max_magnitude": float(np.max(vertex_magnitudes)),
        "mean_magnitude": float(np.mean(vertex_magnitudes)),
        "std_magnitude": float(np.std(vertex_magnitudes)),
        "max_component": float(np.max(np.abs(gradient))),
        "min_magnitude": float(np.min(vertex_magnitudes)),
    }


def print_gradient_analysis(mesh, weights: Dict[str, float]):
    """
    Print a breakdown of individual and combined gradient norms to the console.

    A developer-facing diagnostic function. Computes all four individual
    gradients and the total gradient, then prints their norms alongside
    the corresponding weights. Useful for confirming that each gradient
    term is contributing meaningfully to the overall optimisation direction
    and that no single term is dominating or vanishing unexpectedly.

    This function uses 'print()' intentionally and writes to standard output.
    It is not part of the optimisation hot-path.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to analyse.
    weights : dict
        Weight dictionary with keys 'planarity', 'fairness', 'closeness',
        and optionally 'angle_balance'.

    Returns
    -------
    None
        Results are printed to standard output only.
    """
    grad_planarity = compute_planarity_gradient(mesh)
    grad_fairness = compute_fairness_gradient(mesh)
    grad_closeness = compute_closeness_gradient(mesh)
    grad_total = compute_total_gradient(mesh, weights)
    stats = compute_gradient_statistics(grad_total)

    print("=" * 70)
    print("GRADIENT ANALYSIS")
    print("=" * 70)
    print("Total gradient:")
    print(f"  Norm           : {stats['norm']:.6f}")
    print(f"  Max magnitude  : {stats['max_magnitude']:.6f}")
    print(f"  Mean magnitude : {stats['mean_magnitude']:.6f}")
    print(f"  Std deviation  : {stats['std_magnitude']:.6f}")
    print()
    print("Component gradients:")
    print(
        f"  Planarity : norm = {np.linalg.norm(grad_planarity):.6f}"
        f"  (weight: {weights['planarity']})"
    )
    print(
        f"  Fairness  : norm = {np.linalg.norm(grad_fairness):.6f}"
        f"  (weight: {weights['fairness']})"
    )
    print(
        f"  Closeness : norm = {np.linalg.norm(grad_closeness):.6f}"
        f"  (weight: {weights['closeness']})"
    )
    print("=" * 70)


# ============================================================================
# PRIVATE GPU GRADIENT IMPLEMENTATIONS
# ============================================================================


def _planarity_gradient_gpu(mesh) -> np.ndarray:
    """
    CuPy GPU implementation of the planarity gradient (Tier 1).

    Transfers vertex and face arrays to GPU memory, performs batched SVD
    across all faces using CuPy, computes per-face gradient contributions,
    and scatters them back to vertex positions using a GPU sparse matrix.
    The GPU scatter matrix is built from the CPU scatter matrix on the first
    call and cached on the mesh object for reuse across iterations.

    If a GPU out-of-memory error occurs, the function falls back through
    the full three-tier hierarchy: Numba (Tier 2) if available, then
    NumPy (Tier 3).

    This function is never called directly. It is dispatched by
    'compute_planarity_gradient' when 'HAS_CUDA' is True.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. Must have 'vertices', 'faces', and
        'scatter_matrix' attributes.

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (n_vertices, 3), dtype float64. Always
        a CPU NumPy array, regardless of which tier performed the
        computation.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    from src.backends import gpu_memory_guard, to_device, to_numpy

    _gpu_succeeded = False
    result = None

    with gpu_memory_guard():
        verts_gpu = to_device(mesh.vertices)
        faces_gpu = to_device(mesh.faces)
        face_verts = verts_gpu[faces_gpu]
        centroids = face_verts.mean(axis=1, keepdims=True)
        centered = face_verts - centroids

        _, _, Vt = cp.linalg.svd(centered, full_matrices=False)
        normals = Vt[:, -1, :]

        signed_dists = cp.einsum("fvd,fd->fv", centered, normals)
        contributions = 2.0 * signed_dists[:, :, None] * normals[:, None, :]

        # Build and cache the GPU scatter matrix (SciPy CSR → CuPy CSR).
        if not hasattr(mesh, "_scatter_matrix_gpu") or mesh._scatter_matrix_gpu is None:
            mesh._scatter_matrix_gpu = cpsp.csr_matrix(mesh.scatter_matrix)

        grad_gpu = mesh._scatter_matrix_gpu @ contributions.reshape(-1, 3)
        result = to_numpy(grad_gpu)
        _gpu_succeeded = True

    if not _gpu_succeeded or result is None:
        # GPU failed, which will fall through to Numba (Tier 2) or NumPy (Tier 3).
        # Respect the full three-tier hierarchy rather than bypassing Tier 2.
        try:
            from src.backends import HAS_NUMBA

            if (
                HAS_NUMBA
                and _HAS_NUMBA_PLANARITY_GRAD
                and _planarity_grad_kern is not None
            ):
                contrib = _planarity_grad_kern(
                    mesh.vertices.astype(np.float64),
                    mesh.faces.astype(np.int64),
                )
                return mesh.scatter_matrix @ contrib.reshape(-1, 3)
        except Exception:
            pass
        # Final fallback: NumPy baseline
        fv = mesh.vertices[mesh.faces]
        c_ = fv - fv.mean(axis=1, keepdims=True)
        _, _, V = np.linalg.svd(c_, full_matrices=False)
        n_ = V[:, -1, :]
        d_ = np.einsum("fvd,fd->fv", c_, n_)
        cont_ = 2.0 * d_[:, :, None] * n_[:, None, :]
        return mesh.scatter_matrix @ cont_.reshape(-1, 3)

    return result


def _fairness_gradient_gpu(mesh) -> np.ndarray:
    """
    CuPy GPU implementation of the fairness gradient (Tier 1).

    Converts the mesh Laplacian to a CuPy sparse matrix on the first call
    (cached as 'mesh._laplacian_gpu') and computes the sparse matrix
    product 2 * L_transposed * (L * V) entirely on GPU. Falls back to
    the SciPy CPU computation if a GPU error occurs.

    This function is never called directly. It is dispatched by
    'compute_fairness_gradient' when 'HAS_CUDA' is True.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. GPU Laplacian cached as 'mesh._laplacian_gpu'.

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (n_vertices, 3), dtype float64. Always
        a CPU NumPy array.
    """
    import cupyx.scipy.sparse as cpsp

    from src.backends import gpu_memory_guard, to_device, to_numpy

    _gpu_succeeded = False
    result = None

    with gpu_memory_guard():
        if not hasattr(mesh, "_laplacian_gpu") or mesh._laplacian_gpu is None:
            mesh._laplacian_gpu = cpsp.csr_matrix(mesh.laplacian)
        L = mesh._laplacian_gpu
        V = to_device(mesh.vertices)
        result = to_numpy(2.0 * (L.T @ (L @ V)))
        _gpu_succeeded = True

    if not _gpu_succeeded or result is None:
        L_cpu = mesh.laplacian
        return 2.0 * (L_cpu.T @ (L_cpu @ mesh.vertices))

    return result


# ============================================================================
# TIER-2 NUMBA KERNEL: ANGLE BALANCE GRADIENT
# ============================================================================

try:
    import numpy as _np_grad
    from numba import njit
    from numba import prange as _prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover
    def _angle_balance_gradient_numba(
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_face_ids: np.ndarray,
        scratch_gvp: np.ndarray,
        scratch_gvn: np.ndarray,
        scratch_prev: np.ndarray,
        scratch_next: np.ndarray,
        scratch_active: np.ndarray,
    ) -> np.ndarray:
        """
        Numba-parallel angle balance gradient kernel.

        Computes the angle balance gradient in two passes to avoid write
        conflicts in the parallel stage:

          Pass 1 (parallel over vertices):
            Each active 4-valent vertex computes its own central gradient
            contribution and stores the neighbour contributions in scratch
            buffers. Because each vertex writes only to its own rows in the
            scratch arrays, no writing conflicts occur.

          Pass 2 (serial scatter):
            A single thread reads from the scratch buffers and accumulates
            neighbour contributions into the gradient array.

        The scratch buffers are allocated once per mesh topology by
        'QuadMesh.angle_balance_scratch' and reused across all optimisation
        iterations to avoid repeated memory allocation.

        This function is never called directly. It is compiled by Numba and
        dispatched by 'compute_angle_balance_gradient' when 'HAS_NUMBA' is
        True.

        Parameters
        ----------
        vertices : numpy.ndarray
            Vertex position array of shape (n_vertices, 3), dtype float64.
        faces : numpy.ndarray
            Face index array of shape (n_faces, 4), dtype int32.
        vertex_face_ids : numpy.ndarray
            Padded adjacency array of shape (n_vertices, max_valence),
            dtype int32. Unused padding slots are filled with -1.
        scratch_gvp : numpy.ndarray
            Pre-allocated buffer of shape (n_vertices, 4, 3), dtype float64,
            for storing previous-vertex gradient contributions.
        scratch_gvn : numpy.ndarray
            Pre-allocated buffer of shape (n_vertices, 4, 3), dtype float64,
            for storing next-vertex gradient contributions.
        scratch_prev : numpy.ndarray
            Pre-allocated buffer of shape (n_vertices, 4), dtype int32,
            for storing previous-vertex indices.
        scratch_next : numpy.ndarray
            Pre-allocated buffer of shape (n_vertices, 4), dtype int32,
            for storing next-vertex indices.
        scratch_active : numpy.ndarray
            Pre-allocated buffer of shape (n_vertices,), dtype int8,
            marking which vertices are active 4-valent vertices.

        Returns
        -------
        numpy.ndarray
            Gradient array of shape (n_vertices, 3), dtype float64.
        """
        n_verts = vertices.shape[0]
        grad = _np_grad.zeros((n_verts, 3), dtype=_np_grad.float64)
        SIN_MIN = 1e-2

        # ── Reset scratch buffers in-place ───────────────────────────────────
        for vid in _prange(n_verts):
            scratch_active[vid] = 0
            for k in range(4):
                scratch_prev[vid, k] = -1
                scratch_next[vid, k] = -1
                for d in range(3):
                    scratch_gvp[vid, k, d] = 0.0
                    scratch_gvn[vid, k, d] = 0.0

        # ── Pass 1: parallel — central vertex only ────────────────────────────
        for vid in _prange(n_verts):
            max_val = vertex_face_ids.shape[1]
            n_valid = 0
            for k in range(max_val):
                if vertex_face_ids[vid, k] != -1:
                    n_valid += 1
            if n_valid != 4:
                continue

            angles = _np_grad.empty(4, dtype=_np_grad.float64)
            gv = _np_grad.empty((4, 3), dtype=_np_grad.float64)
            gvp_loc = _np_grad.empty((4, 3), dtype=_np_grad.float64)
            gvn_loc = _np_grad.empty((4, 3), dtype=_np_grad.float64)
            prev_ids = _np_grad.empty(4, dtype=_np_grad.int64)
            next_ids = _np_grad.empty(4, dtype=_np_grad.int64)

            for i in range(4):
                angles[i] = 0.0
                prev_ids[i] = -1
                next_ids[i] = -1
                for d in range(3):
                    gv[i, d] = 0.0
                    gvp_loc[i, d] = 0.0
                    gvn_loc[i, d] = 0.0

            for k in range(4):
                fid = int(vertex_face_ids[vid, k])
                n_f = faces.shape[1]
                local_idx = -1
                for j in range(n_f):
                    if faces[fid, j] == vid:
                        local_idx = j
                        break
                if local_idx == -1:
                    prev_ids[k] = -1
                    next_ids[k] = -1
                    continue

                vp_id = int(faces[fid, (local_idx - 1) % n_f])
                vn_id = int(faces[fid, (local_idx + 1) % n_f])
                prev_ids[k] = vp_id
                next_ids[k] = vn_id

                e1x = vertices[vp_id, 0] - vertices[vid, 0]
                e1y = vertices[vp_id, 1] - vertices[vid, 1]
                e1z = vertices[vp_id, 2] - vertices[vid, 2]
                e2x = vertices[vn_id, 0] - vertices[vid, 0]
                e2y = vertices[vn_id, 1] - vertices[vid, 1]
                e2z = vertices[vn_id, 2] - vertices[vid, 2]
                l1 = _np_grad.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
                l2 = _np_grad.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)

                if l1 < 1e-12 or l2 < 1e-12:
                    continue

                e1hx = e1x / l1
                e1hy = e1y / l1
                e1hz = e1z / l1
                e2hx = e2x / l2
                e2hy = e2y / l2
                e2hz = e2z / l2
                c = e1hx * e2hx + e1hy * e2hy + e1hz * e2hz
                c = max(-1.0 + 1e-8, min(1.0 - 1e-8, c))
                s = max(_np_grad.sqrt(1.0 - c * c), SIN_MIN)
                angles[k] = _np_grad.arccos(c)

                inv_s = 1.0 / s
                gv[k, 0] = inv_s * ((e2hx - c * e1hx) / l1 + (e1hx - c * e2hx) / l2)
                gv[k, 1] = inv_s * ((e2hy - c * e1hy) / l1 + (e1hy - c * e2hy) / l2)
                gv[k, 2] = inv_s * ((e2hz - c * e1hz) / l1 + (e1hz - c * e2hz) / l2)
                gvp_loc[k, 0] = -inv_s * (e2hx - c * e1hx) / l1
                gvp_loc[k, 1] = -inv_s * (e2hy - c * e1hy) / l1
                gvp_loc[k, 2] = -inv_s * (e2hz - c * e1hz) / l1
                gvn_loc[k, 0] = -inv_s * (e1hx - c * e2hx) / l2
                gvn_loc[k, 1] = -inv_s * (e1hy - c * e2hy) / l2
                gvn_loc[k, 2] = -inv_s * (e1hz - c * e2hz) / l2

            delta = (angles[0] + angles[2]) - (angles[1] + angles[3])
            two_delta = 2.0 * delta

            for i in range(4):
                si = _ANGLE_SIGNS[i]
                grad[vid, 0] += two_delta * si * gv[i, 0]
                grad[vid, 1] += two_delta * si * gv[i, 1]
                grad[vid, 2] += two_delta * si * gv[i, 2]

            for i in range(4):
                si = _ANGLE_SIGNS[i]
                td_si = two_delta * si
                scratch_gvp[vid, i, 0] = td_si * gvp_loc[i, 0]
                scratch_gvp[vid, i, 1] = td_si * gvp_loc[i, 1]
                scratch_gvp[vid, i, 2] = td_si * gvp_loc[i, 2]
                scratch_gvn[vid, i, 0] = td_si * gvn_loc[i, 0]
                scratch_gvn[vid, i, 1] = td_si * gvn_loc[i, 1]
                scratch_gvn[vid, i, 2] = td_si * gvn_loc[i, 2]
                scratch_prev[vid, i] = prev_ids[i]
                scratch_next[vid, i] = next_ids[i]
                scratch_active[vid] = 1

        # ── Pass 2: serial — neighbour scatter ───────────────────────────────
        for vid in range(n_verts):
            if scratch_active[vid] == 0:
                continue
            for i in range(4):
                p = scratch_prev[vid, i]
                n_ = scratch_next[vid, i]
                if p < 0 or n_ < 0:
                    continue
                grad[p, 0] += scratch_gvp[vid, i, 0]
                grad[p, 1] += scratch_gvp[vid, i, 1]
                grad[p, 2] += scratch_gvp[vid, i, 2]
                grad[n_, 0] += scratch_gvn[vid, i, 0]
                grad[n_, 1] += scratch_gvn[vid, i, 1]
                grad[n_, 2] += scratch_gvn[vid, i, 2]

        return grad

except Exception as _numba_angle_grad_exc:
    _numba_angle_grad_error = str(_numba_angle_grad_exc)
    warnings.warn(
        f"Numba angle-balance gradient kernel unavailable "
        f"({_numba_angle_grad_error}); falling back to NumPy implementation.",
        RuntimeWarning,
        stacklevel=1,
    )

    def _angle_balance_gradient_numba(
        vertices,
        faces,
        vertex_face_ids,
        scratch_gvp,
        scratch_gvn,
        scratch_prev,
        scratch_next,
        scratch_active,
    ):
        raise RuntimeError(
            f"_angle_balance_gradient_numba unavailable: "
            f"{_numba_angle_grad_error}. "
            "backends.py should have set HAS_NUMBA=False in this case."
        )


# ============================================================================
# SCIPY INTERFACE
# ============================================================================


def gradient_for_scipy(
    x_flat: np.ndarray, mesh, weights: Dict[str, float]
) -> np.ndarray:
    """
    Gradient function in the format required by scipy.optimize.minimize.

    SciPy's L-BFGS-B solver passes and receives flat one-dimensional arrays
    rather than the two-dimensional vertex matrices used internally. This
    wrapper reshapes the incoming flat array back into vertex positions,
    computes the total analytical gradient, and returns it as a flat array.

    If any gradient values are non-finite (for example, due to a degenerate
    face with zero area), they are replaced with zero and a warning is
    printed. This prevents the optimiser from crashing on problematic meshes
    while still signalling that something may need investigation.

    Parameters
    ----------
    x_flat : numpy.ndarray
        Flat array of shape (n_vertices * 3, ) representing the current
        vertex positions as provided by SciPy.
    mesh : QuadMesh
        The mesh being optimised. Its 'vertices' attribute is updated
        in-place before computing the gradient.
    weights : dict
        Weight dictionary with keys 'planarity', 'fairness', 'closeness',
        and optionally 'angle_balance'.

    Returns
    -------
    numpy.ndarray
        Flat gradient array of shape (n_vertices * 3,), dtype float64,
        compatible with SciPy L-BFGS-B.
    """
    mesh.vertices = x_flat.reshape(-1, 3)
    grad = compute_total_gradient(mesh, weights)

    if not np.isfinite(grad).all():
        n_bad = int(np.sum(~np.isfinite(grad)))
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  ⚠️  gradient_for_scipy: {n_bad} non-finite gradient component(s) "
            "replaced with 0 — check mesh for degenerate faces."
        )

    return grad.flatten()


def energy_for_scipy(x_flat: np.ndarray, mesh, weights: Dict[str, float]) -> float:
    """
    Energy function in the format required by scipy.optimize.minimize.

    SciPy's L-BFGS-B solver calls this function at each trial point during
    the line search, passing the current vertex positions as a flat array.
    This wrapper reshapes the array, updates the mesh vertices, computes
    the total weighted energy, and returns it as a scalar float.

    If the computed energy is non-finite (for example, due to a degenerate
    configuration), a very large fallback value of 1e300 is returned. This
    causes SciPy to reject the trial point and backtrack, preventing the
    optimiser from diverging.

    Parameters
    ----------
    x_flat : numpy.ndarray
        Flat array of shape (n_vertices * 3), representing the current
        vertex positions as provided by SciPy.
    mesh : QuadMesh
        The mesh being optimised. Its 'vertices' attribute is updated
        in-place before computing the energy.
    weights : dict
        Weight dictionary with keys 'planarity', 'fairness', 'closeness',
        and optionally 'angle_balance'.

    Returns
    -------
    float
        Total weighted energy, or 1e300 if the energy is non-finite.
    """
    from src.optimisation.energy_terms import compute_total_energy

    mesh.vertices = x_flat.reshape(-1, 3)

    res = compute_total_energy(mesh, weights)
    energy = float(res[0]) if isinstance(res, tuple) else float(res)

    if not np.isfinite(energy):
        print(
            f"  ⚠️  energy_for_scipy: non-finite energy {energy!r} — "
            "returning large fallback value to abort line search."
        )
        energy = 1e300

    return energy
