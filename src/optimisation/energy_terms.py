"""
src/optimisation/energy_terms.py

Energy terms for planar quad (PQ) mesh optimisation.

This module defines the four geometric penalty functions that together
form the total optimisation objective. Each function measures how far
the current mesh deviates from a desired geometric property. The
optimiser minimises the weighted sum of all four penalties simultaneously.

The four energy terms are:

  Planarity energy -- penalises quad faces whose four corners do not
                         lie on a common plane. A face is perfectly planar
                         when its smallest singular value is zero.

  Fairness energy -- penalises uneven vertex spacing by measuring how
                         much each vertex deviates from the average position
                         of its neighbours (discrete Laplacian smoothness).

  Closeness energy -- penalises displacement from the original mesh
                         positions, acting as a regularity that prevents
                         the optimiser from distorting the overall shape.

  Angle balance energy -- penalises vertices where opposite face angles do
                          not sum to the same value, which is the defining
                          property of a conical mesh and is necessary for
                          the surface to be developable (unfoldable flat).

Backend dispatch
----------------
Each energy function automatically selects the fastest available
computation path at runtime:

  1. CuPy (NVIDIA GPU) -- used when a CUDA-capable GPU is detected.
  2. Numba (CPU, parallel) -- used when Numba is installed and no GPU
     is present; compiles to native machine code on the first call.
  3. NumPy (CPU, serial) -- always available; used as the fallback when
     neither CuPy nor Numba is installed.

'Src/backends.py' handles the backend selection transparently.
Callers do not need to choose or configure a backend explicitly.

References
----------
Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.

Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.

Nocedal, J. and Wright, S. J. (2006).
  Numerical Optimization. 2nd ed. Springer.

Author: Muhammad Nabil
Date: 2 February 2026
"""
import logging
import warnings
from typing import Dict

import numpy as np

from src.optimisation.mesh_geometry import compute_conical_angle_imbalance

logger = logging.getLogger(__name__)

# ============================================================================
# INDIVIDUAL ENERGY COMPONENTS
# ============================================================================


def compute_planarity_energy(
    mesh, planarity_deviations=None
) -> float:
    """
    Compute the total planarity energy of the mesh.

    For each quad face, this function measures how far the four corner
    vertices deviate from a common plane. The deviation is quantified
    using Singular Value Decomposition (SVD): the four vertices are
    assembled into a centred matrix, and its smallest singular value
    represents the thickness of the face in the direction perpendicular
    to its best-fit plane. The energy for that face is the square of
    this value. A face lying perfectly flat contributes zero energy.

    The total planarity energy is the sum of per-face energies across
    all faces. Lower values indicate a flatter, more planar mesh.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh whose planarity is to be measured. Must have 'vertices'
        (shape n_vertices x 3, float64) and 'faces' (shape n_faces x 4,
        integer) attributes.
    planarity_deviations : numpy.ndarray or None, optional
        Pre-computed per-face deviations. Reserved for future use;
        currently unused and may be passed as None.

    Returns
    -------
    float
        Total planarity energy. Zero means all faces are perfectly flat.
        Larger values indicate more non-planar faces.

    Notes
    -----
    The computation dispatches to the fastest available backend:
    CuPy GPU first, then Numba parallel CPU, then NumPy CPU baseline.
    All three backends produce numerically equivalent results.

    References
    ----------
    Liu et al. (2006). "Geometric modelling with conical meshes and
    developable surfaces." ACM Transactions on Graphics, 25(3), 681-689.
    """
    try:
        from src.backends import HAS_CUDA, HAS_NUMBA

        if HAS_CUDA:
            return _planarity_energy_gpu(mesh)

        if HAS_NUMBA and mesh.faces.ndim == 2 and mesh.faces.shape[1] == 4:
            return float(_planarity_energy_numba(mesh.vertices, mesh.faces))

    except (ImportError, ModuleNotFoundError) as exc:
        warnings.warn(
            f"Accelerated planarity backend unavailable ({exc}); "
            "falling back to NumPy implementation.",
            RuntimeWarning,
        )

    face_verts = mesh.vertices[mesh.faces]
    centroids = face_verts.mean(axis=1, keepdims=True)
    centered = face_verts - centroids
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normals = Vt[:, -1, :]
    signed_dists = np.einsum("fvd,fd->fv", centered, normals)

    return float(np.sum(signed_dists**2))


def compute_planarity_per_face(mesh) -> np.ndarray:
    """
    Compute the planarity deviation for each face individually.

    Unlike 'compute_planarity_energy', which returns a single scalar
    for the whole mesh, this function returns one value per face. Each
    value is the smallest singular value of the centred face-vertex
    matrix, representing how far that face deviates from being flat.
    A value of zero means the face is perfectly planar.

    This function is primarily used for diagnostics, visualisation, and
    per-face heatmap rendering rather than during optimisation itself.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to analyse. Must have 'vertices' and 'faces' attributes.

    Returns
    -------
    numpy.ndarray
        A one-dimensional array of shape (n_faces,) containing the
        planarity deviation for each face. All values are non-negative.
        Values close to zero indicate flat faces.

    Notes
    -----
    Dispatches to Numba if available, otherwise uses NumPy. The Numba
    path computes the full per-face SVD in parallel across all faces.
    """
    try:
        from src.backends import HAS_NUMBA

        if HAS_NUMBA and mesh.faces.ndim == 2 and mesh.faces.shape[1] == 4:
            return _planarity_per_face_numba(mesh.vertices, mesh.faces)

    except (ImportError, ModuleNotFoundError):
        pass

    face_verts = mesh.vertices[mesh.faces]
    centroids = face_verts.mean(axis=1, keepdims=True)
    centered = face_verts - centroids
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    return S[:, -1]


def compute_fairness_energy(mesh) -> float:
    """
    Compute the fairness (smoothness) energy of the mesh.

    Fairness energy measures how evenly the vertices are distributed
    across the surface. For each vertex, the discrete Laplacian
    operator computes the difference between that vertex's position
    and the average position of its direct neighbours. When all
    neighbours are evenly spaced, this difference is zero. The
    fairness energy is the sum of squared Laplacian vectors across
    all vertices.

    A mesh with low-fairness energy has smoothly varying curvature
    and evenly spaced vertices, which is desirable for both aesthetic
    and fabrication reasons.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. The Laplacian matrix is accessed via
        'mesh.laplacian' and is cached on first use.

    Returns
    -------
    float
        Total fairness energy. Zero means all vertices lie exactly at
        the average of their neighbours (perfectly regular spacing).

    Notes
    -----
    The discrete Laplacian used here is the combinatorial (unweighted)
    Laplacian, consistent with the formulation in Crane et al. (2013).
    Dispatches to CuPy GPU if available, otherwise uses NumPy sparse
    matrix multiplication.

    References
    ----------
    Crane, K., de Goes, F., Desbrun, M., and Schroder, P. (2013).
    "Digital geometry processing with discrete exterior calculus."
    ACM SIGGRAPH 2013 Courses.
    """
    try:
        from src.backends import HAS_CUDA

        if HAS_CUDA:
            return _fairness_energy_gpu(mesh)
    except ImportError:
        logger.warning("CUDA backend unavailable; falling back to CPU.")

    L = mesh.laplacian
    laplacian_coords = L @ mesh.vertices
    return float(np.sum(laplacian_coords**2))


def compute_closeness_energy(mesh) -> float:
    """
    Compute the closeness (shape fidelity) energy of the mesh.

    Closeness energy measures how far the current vertex positions have
    moved from their original positions at the start of optimisation.
    For each vertex, the squared Euclidean distance between its current
    position and its original position is computed. The total closeness
    energy is the sum of these squared distances across all vertices.

    This term acts as a regularity: without it, the optimiser could
    satisfy the planarity and fairness constraints by collapsing the
    mesh to a degenerate shape. By penalising large displacements, it
    ensures the optimised mesh retains the designer's original intent.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. Must have both 'vertices' (current
        positions) and 'vertices_original' (original positions stored
        at initialisation time) attributes.

    Returns
    -------
    float
        Total closeness energy. Zero means the mesh is identical to its
        original. Larger values indicate greater overall displacement.

    Notes
    -----
    Closeness energy is always zero at the start of optimisation and
    increases as the optimiser moves vertices. A high-closeness weight
    restricts how freely vertices can move, trading geometric quality
    for shape fidelity.
    """
    try:
        from src.backends import HAS_CUDA

        if HAS_CUDA:
            from src.backends import to_device

            v_gpu = to_device(mesh.vertices)
            v0_gpu = to_device(mesh.vertices_original)
            d = v_gpu - v0_gpu
            return float((d * d).sum().get())
    except (ImportError, Exception):
        pass

    displacement = mesh.vertices - mesh.vertices_original
    energy = np.sum(displacement**2)
    return energy


def compute_angle_balance_energy(mesh) -> float:
    """
    Compute the angle balance energy for conical mesh developability.

    A conical mesh is one in which, at every interior vertex, the
    angles of the surrounding quad faces satisfy a specific balance
    condition: the sum of alternating face angles must be equal.
    Formally, for four faces meeting at a vertex with corner angles
    alpha_1, alpha_2, alpha_3, alpha_4 in cyclic order, the condition
    is (alpha_1 + alpha_3) == (alpha_2 + alpha_4). When this holds at
    every vertex, the mesh is conical, which is a necessary condition
    for it to be developable (i.e. unrollable flat without cutting).

    The angle balance energy is the sum of squared imbalances across
    all interior vertices with exactly four incident faces. Vertices
    with fewer or more than four incident faces are skipped, as the
    conical condition is defined only for 4-valent vertices.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. Must have 'vertices', 'faces', and
        'vertex_face_ids_padded' attributes.

    Returns
    -------
    float
        Total angle balance energy. Zero means all 4-valent vertices
        satisfy the conical mesh condition exactly.

    Notes
    -----
    This term is optional. It should only be included (via a non-zero
    weight) when the target application specifically requires a
    developable or near-developable surface. For general PQ mesh
    optimisation without developability constraints, set its weight
    to zero.

    References
    ----------
    Liu et al. (2006). "Geometric modelling with conical meshes and
    developable surfaces." ACM Transactions on Graphics, 25(3), 681-689.
    """
    try:
        from src.backends import HAS_CUDA, HAS_NUMBA

        if HAS_CUDA:
            return _angle_balance_energy_gpu(mesh)

        if HAS_NUMBA:
            vf = mesh.vertex_face_ids_padded
            return float(_angle_balance_numba(mesh.vertices, mesh.faces, vf))

    except (ImportError, AttributeError):
        logger.warning(
            "Accelerated angle balance backend unavailable; "
            "falling back to Python implementation.",
            exc_info=True,
        )

    energy = 0.0
    for vertex_id in range(mesh.n_vertices):
        imbalance = compute_conical_angle_imbalance(mesh, vertex_id)
        energy += imbalance**2
    return energy


# ============================================================================
# COMBINED ENERGY
# ============================================================================


def compute_total_energy(
    mesh,
    weights,
    planarity_deviations=None,
    return_components=False,
):
    """
    Compute the weighted total energy for mesh optimisation.

    This is the main objective function that the optimiser minimises.
    It combines all four individual energy terms into a single scalar
    value by multiplying each by its corresponding weight and summing
    the results. The optimiser calls this function (indirectly via
    'energy_for_scipy') at every iteration.

    The total energy is: (w_planarity * E_planarity) + (w_fairness *
    E_fairness) + (w_closeness * E_closeness) + (w_angle * E_angle).

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate.
    weights : dict
        A dictionary mapping energy term names to their scalar weights.
        Required keys: 'planarity', 'fairness', 'closeness'.
        Optional key: 'angle_balance' (defaults to 0.0 if absent).
        Typical starting values: planarity=10.0, fairness=1.0,
        closeness=1.0, angle_balance=0.0 (disabled by default).
    planarity_deviations : numpy.ndarray or None, optional
        Pre-computed planarity values. Currently unused; pass None.
    return_components : bool, optional
        If True, also returns a dictionary containing each
        energy value alongside its weighted contribution. Useful for
        diagnostics and the optimiser's result summary.
        Default is False.

    Returns
    -------
    float or tuple
        If 'return_components' is False: a single float representing
        the total weighted energy. Lower is better.
        If 'return_components' is True: a tuple (total_energy, dict)
        where the dictionary contains keys 'E_planarity', 'E_fairness',
        'E_closeness', 'E_angle_balance', 'weighted_planarity',
        'weighted_fairness', 'weighted_closeness', and
        'weighted_angle_balance'.

    Notes
    -----
    Weight selection affects the optimiser's
    behaviour. A suggested starting point is to run
    'suggest_weight_scaling(mesh)' before optimisation to get
    weights that balance the magnitudes of each term automatically.
    """
    E_planar = compute_planarity_energy(mesh, planarity_deviations)
    E_fair = compute_fairness_energy(mesh)
    E_close = compute_closeness_energy(mesh)

    w_angle = weights.get("angle_balance", 0.0)
    if w_angle > 0:
        E_angle = compute_angle_balance_energy(mesh)
    else:
        E_angle = 0.0

    E_total = (
        weights["planarity"] * E_planar
        + weights["fairness"] * E_fair
        + weights["closeness"] * E_close
        + w_angle * E_angle
    )

    if return_components:
        components = {
            "E_planarity": E_planar,
            "E_fairness": E_fair,
            "E_closeness": E_close,
            "E_angle_balance": E_angle,
            "weighted_planarity": weights["planarity"] * E_planar,
            "weighted_fairness": weights["fairness"] * E_fair,
            "weighted_closeness": weights["closeness"] * E_close,
            "weighted_angle_balance": w_angle * E_angle,
        }
        return E_total, components

    return E_total


# ============================================================================
# DIAGNOSTIC UTILITIES
# (interactive developer tools, print() is intentional here;
#  these functions are not called by the optimisation hot-path)
# ============================================================================


def analyse_energy_components(mesh, weights) -> None:
    """
    Print a detailed breakdown of all energy components to the console.

    This is a developer-facing diagnostic function. It computes all
    four energy terms and prints both the raw (unweighted) values and
    the weighted contributions in a formatted table. It is intended for
    interactive exploration before or after running the optimiser to
    understand which terms dominate the total energy.

    This function uses 'print()' intentionally and writes to standard
    output. It is not called during optimisation. To suppress output
    and access the values programmatically, use
    'compute_total_energy(mesh, weights, return_components=True)'
    directly.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to analyse.
    weights : dict
        Weight dictionary with keys 'planarity', 'fairness',
        'closeness', and optionally 'angle_balance'.

    Returns
    -------
    None
        Results are printed to standard output only.

    Examples
    --------
    To inspect energy components before optimisation:

        from src.optimisation.energy_terms import analyse_energy_components
        analyse_energy_components(mesh, weights={"planarity": 10.0,
                                                  "fairness": 1.0,
                                                  "closeness": 1.0})
    """
    E_total, components = compute_total_energy(  # type: ignore[misc]
        mesh, weights, return_components=True
    )

    print("=" * 60)
    print("ENERGY ANALYSIS")
    print("=" * 60)
    print("Raw Components (unweighted):")
    print(f"  Planarity:      {components['E_planarity']:.6f}")
    print(f"  Fairness:       {components['E_fairness']:.6f}")
    print(f"  Closeness:      {components['E_closeness']:.6f}")
    print(f"  Angle balance:  {components['E_angle_balance']:.6f}")
    print()
    print("Weighted Components:")
    print(
        f"  Planarity:      {components['weighted_planarity']:.6f}"
        f"  (weight: {weights['planarity']})"  # noqa: E501
    )
    print(
        f"  Fairness:       {components['weighted_fairness']:.6f}"
        f"  (weight: {weights['fairness']})"  # noqa: E501
    )
    print(
        f"  Closeness:      {components['weighted_closeness']:.6f}"
        f"  (weight: {weights['closeness']})"  # noqa: E501
    )
    if weights.get("angle_balance", 0) > 0:
        print(
            f"  Angle balance:  {components['weighted_angle_balance']:.6f}"
            f"  (weight: {weights['angle_balance']})"  # noqa: E501
        )
    print()
    print(f"Total Energy: {E_total:.6f}")
    print("=" * 60)


def suggest_weight_scaling(mesh, verbose: bool = True) -> Dict[str, float]:
    """
    Suggest appropriate weights based on the mesh's current energy magnitudes.

    Different meshes will have very different raw energy values depending
    on their size, complexity, and initial geometry. If all weights are
    set to 1.0 on a large mesh, the planarity term may be thousands of
    times larger than the closeness term, causing the optimiser to
    effectively ignore the weaker terms. This function measures all four
    raw energies at unit weight and recommends weights that bring each
    contribution to a similar order of magnitude.

    The suggested weights are a starting point, not a definitive answer.
    They should be adjusted based on the specific goals of the
    optimisation (for example, increasing the planarity weight if
    flatness is the primary concern).

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to analyse. Must be the mesh as loaded before any
        optimisation is applied, so that closeness energy is zero and
        does not distort the weight recommendations.
    verbose : bool, optional
        If True, print a formatted recommendation table to the console.
        If False, return the weights silently without printing.
        Default is True.

    Returns
    -------
    dict
        A dictionary with keys 'planarity', 'fairness', 'closeness',
        and 'angle_balance', each mapped to a suggested float weight.

    Examples
    --------
    To get suggested weights before optimisation:

        from src.optimisation.energy_terms import suggest_weight_scaling
        weights = suggest_weight_scaling(mesh, verbose=True)
        result = optimiser.optimise(mesh, config=OptimisationConfig(weights=weights))
    """
    unit_weights = {
        "planarity": 1.0,
        "fairness": 1.0,
        "closeness": 1.0,
        "angle_balance": 1.0,
    }

    _, components = compute_total_energy(  # type: ignore[misc]
        mesh, unit_weights, return_components=True
    )

    E_p = components["E_planarity"]
    E_f = components["E_fairness"]
    E_c = components["E_closeness"]
    E_a = components["E_angle_balance"]

    if E_p > 1e-10:
        w_p = 100.0
        w_f = (10.0 / E_f) if E_f > 1e-10 else 1.0
        w_c = (50.0 / E_c) if E_c > 1e-10 else 10.0
        w_a = (10.0 / E_a) if E_a > 1e-10 else 0.0
    else:
        w_p = 10.0
        w_f = 1.0
        w_c = 10.0
        w_a = 0.0

    suggested = {
        "planarity": w_p,
        "fairness": w_f,
        "closeness": w_c,
        "angle_balance": w_a,
    }

    if verbose:
        print("=" * 60)
        print("WEIGHT RECOMMENDATIONS")
        print("=" * 60)
        print("Raw energy magnitudes:")
        print(f"  E_planarity:    {E_p:.6f}")
        print(f"  E_fairness:     {E_f:.6f}")
        print(f"  E_closeness:    {E_c:.6f}")
        print(f"  E_angle_balance: {E_a:.6f}")
        print()
        print("Suggested weights:")
        print(f"  planarity:      {w_p:.2f}")
        print(f"  fairness:       {w_f:.2f}")
        print(f"  closeness:      {w_c:.2f}")
        print(f"  angle_balance:  {w_a:.2f}")
        print()
        print("These weights aim to balance energy contributions.")
        print("Adjust based on desired emphasis (e.g., increase")
        print("planarity weight if flatness is critical).")
        print("=" * 60)

    return suggested


# ============================================================================
# PRIVATE BACKEND IMPLEMENTATIONS
# (called only via the dispatch blocks above, never imported by external code)
# ============================================================================


# ── Numba planarity kernels ──────────────────────────────────────────────────

try:
    import numpy as _np_planarity
    from numba import njit
    from numba import prange as _prange

    @njit(parallel=True, cache=True, fastmath=False) # pragma: no cover
    def _planarity_energy_numba(vertices: np.ndarray, faces: np.ndarray) -> float:
        """
        Numba-parallel planarity energy kernel for quad meshes.

        Computes the same result as the NumPy fallback in
        'compute_planarity_energy' but runs in parallel across all
        faces using Numba's prange. For each face, the four vertex
        positions are centred about their mean, assembled into a 4x3
        matrix, and decomposed by SVD. The square of the smallest
        singular value is the per-face planarity energy.

        This function is compiled to native machine code on the first call
        (cache=True means later calls reuse the compiled binary).
        It is never called directly; dispatch is handled by
        'compute_planarity_energy'.

        Parameters
        ----------
        vertices : numpy.ndarray
            Vertex position array of shape (n_vertices, 3), dtype float64.
        faces : numpy.ndarray
            Face index array of shape (n_faces, 4), dtype int64.

        Returns
        -------
        float
            Total planarity energy (sum of squared minimum singular
            values across all faces).
        """
        n_faces = faces.shape[0]
        per_face_energy = _np_planarity.zeros(n_faces, dtype=_np_planarity.float64)

        for fid in _prange(n_faces):
            M = _np_planarity.empty((4, 3), dtype=_np_planarity.float64)

            cx = 0.0
            cy = 0.0
            cz = 0.0

            for k in range(4):
                vid = faces[fid, k]
                cx += vertices[vid, 0]
                cy += vertices[vid, 1]
                cz += vertices[vid, 2]

            cx *= 0.25
            cy *= 0.25
            cz *= 0.25

            for k in range(4):
                vid = faces[fid, k]
                M[k, 0] = vertices[vid, 0] - cx
                M[k, 1] = vertices[vid, 1] - cy
                M[k, 2] = vertices[vid, 2] - cz

            _, S, _ = _np_planarity.linalg.svd(M, full_matrices=False)
            s_min = S[2]
            per_face_energy[fid] = s_min * s_min

        total = 0.0
        for fid in range(n_faces):
            total += per_face_energy[fid]

        return total

    @njit(parallel=True, cache=True, fastmath=False) # pragma: no cover
    def _planarity_per_face_numba(
        vertices: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """
        Numba-parallel per-face planarity deviation kernel.

        Computes the smallest singular value of the centred face-vertex
        matrix for every face in parallel. This is the per-face version
        of '_planarity_energy_numba': instead of summing the squared
        values, it returns the raw (unsquared) singular value for each
        face individually.

        Used by 'compute_planarity_per_face' for diagnostic output and
        heatmap visualisation. Never called directly by external code.

        Parameters
        ----------
        vertices : numpy.ndarray
            Vertex position array of shape (n_vertices, 3), dtype float64.
        faces : numpy.ndarray
            Face index array of shape (n_faces, 4), dtype int64.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_faces,) containing the smallest singular
            value for each face. Values near zero indicate flat faces.
        """
        n_faces = faces.shape[0]
        out = _np_planarity.empty(n_faces, dtype=_np_planarity.float64)

        for fid in _prange(n_faces):
            M = _np_planarity.empty((4, 3), dtype=_np_planarity.float64)

            cx = 0.0
            cy = 0.0
            cz = 0.0

            for k in range(4):
                vid = faces[fid, k]
                cx += vertices[vid, 0]
                cy += vertices[vid, 1]
                cz += vertices[vid, 2]

            cx *= 0.25
            cy *= 0.25
            cz *= 0.25

            for k in range(4):
                vid = faces[fid, k]
                M[k, 0] = vertices[vid, 0] - cx
                M[k, 1] = vertices[vid, 1] - cy
                M[k, 2] = vertices[vid, 2] - cz

            _, S, _ = _np_planarity.linalg.svd(M, full_matrices=False)
            out[fid] = S[2]

        return out

except Exception as e:
    _numba_plan_error = str(e)
    warnings.warn(
        f"Numba planarity energy kernel unavailable ({_numba_plan_error}); "
        "falling back to NumPy implementation.",
        RuntimeWarning,
        stacklevel=1,
    )

    def _planarity_energy_numba(vertices, faces):  # type: ignore[misc]
        raise RuntimeError(
            f"_planarity_energy_numba unavailable: {_numba_plan_error}. "
            "backends.py should have set HAS_NUMBA=False in this case."
        )

    def _planarity_per_face_numba(vertices, faces):  # type: ignore[misc]
        raise RuntimeError(
            f"_planarity_per_face_numba unavailable: {_numba_plan_error}. "
            "backends.py should have set HAS_NUMBA=False in this case."
        )


# ── Numba angle-balance energy kernel ───────────────────────────────────────

try:
    import numpy as _np_numba
    from numba import njit
    from numba import prange as _prange  # noqa: F811

    @njit(parallel=True, cache=True, fastmath=False) # pragma: no cover
    def _angle_balance_numba(
        vertices: np.ndarray,  # (n_verts, 3)  float64
        faces: np.ndarray,  # (n_faces, 4)  int32
        vertex_face_ids: np.ndarray,  # (n_verts, max_valence)  int32
    ) -> float:
        """
        Numba-parallel angle balance energy kernel.

        Computes the conical mesh angle balance energy in parallel across
        all vertices. For each vertex with exactly four incident faces,
        the four corner angles at that vertex are computed from the dot
        product of the two edge vectors meeting at the vertex within each
        face. The imbalance is (alpha_1 + alpha_3) - (alpha_2 + alpha_4),
        and the contribution to the energy is the square of this value.

        Vertices with fewer or more than four incident faces contribute
        zero (they are skipped). This matches the serial NumPy fallback
        in 'compute_angle_balance_energy' exactly.

        Parameters
        ----------
        vertices : numpy.ndarray
            Vertex position array of shape (n_vertices, 3), dtype float64.
        faces : numpy.ndarray
            Face index array of shape (n_faces, 4), dtype int32.
        vertex_face_ids : numpy.ndarray
            Padded adjacency array of shape (n_vertices, max_valence),
            dtype int32. Entries of -1 indicate unused padding slots.

        Returns
        -------
        float
            Total angle balance energy across all 4-valent vertices.
        """
        n_verts = vertices.shape[0]
        max_val = vertex_face_ids.shape[1]

        per_vertex_energy = _np_numba.zeros(n_verts, dtype=_np_numba.float64)

        for vid in _prange(n_verts):
            n_valid = 0
            for k in range(max_val):
                if vertex_face_ids[vid, k] != -1:
                    n_valid += 1
            if n_valid != 4:
                continue

            angles_local = _np_numba.empty(4, dtype=_np_numba.float64)

            for k in range(4):
                fid = int(vertex_face_ids[vid, k])
                n_f = faces.shape[1]

                local_idx = -1
                for j in range(n_f):
                    if faces[fid, j] == vid:
                        local_idx = j
                        break
                if local_idx == -1:
                    angles_local[k] = 0.0
                    continue

                vp_id = int(faces[fid, (local_idx - 1) % n_f])
                vn_id = int(faces[fid, (local_idx + 1) % n_f])

                e1x = vertices[vp_id, 0] - vertices[vid, 0]
                e1y = vertices[vp_id, 1] - vertices[vid, 1]
                e1z = vertices[vp_id, 2] - vertices[vid, 2]
                e2x = vertices[vn_id, 0] - vertices[vid, 0]
                e2y = vertices[vn_id, 1] - vertices[vid, 1]
                e2z = vertices[vn_id, 2] - vertices[vid, 2]

                l1 = _np_numba.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
                l2 = _np_numba.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)

                if l1 < 1e-12 or l2 < 1e-12:
                    angles_local[k] = 0.0
                    continue

                cos_a = (e1x * e2x + e1y * e2y + e1z * e2z) / (l1 * l2)
                if cos_a > 1.0 - 1e-8:
                    cos_a = 1.0 - 1e-8
                elif cos_a < -1.0 + 1e-8:
                    cos_a = -1.0 + 1e-8
                angles_local[k] = _np_numba.arccos(cos_a)

            imbalance = (angles_local[0] + angles_local[2]) - (
                angles_local[1] + angles_local[3]
            )
            per_vertex_energy[vid] = imbalance * imbalance

        energy = 0.0
        for i in range(n_verts):
            energy += per_vertex_energy[i]

        return energy

except Exception as e:
    _numba_angle_error = str(e)
    warnings.warn(
        f"Numba angle-balance energy kernel unavailable ({_numba_angle_error}); "
        "falling back to NumPy implementation.",
        RuntimeWarning,
        stacklevel=1,
    )

    def _angle_balance_numba(vertices, faces, vertex_face_ids):  # type: ignore[misc]
        raise RuntimeError(
            f"_angle_balance_numba unavailable: {_numba_angle_error}. "
            "backends.py should have set HAS_NUMBA=False in this case."
        )


# ── CuPy GPU planarity kernel ────────────────────────────────────────────────


def _planarity_energy_gpu(mesh) -> float:
    """
    CuPy GPU implementation of planarity energy.

    Transfers vertex and face arrays to GPU memory, performs batched
    SVD across all faces using CuPy, and returns the total planarity
    energy as a CPU float. If a GPU out-of-memory error occurs, the
    'gpu_memory_guard' context manager catches it, switches the backend
    to NumPy, and the function falls back to a CPU computation before
    returning.

    This function is never called directly. It is dispatched by
    'compute_planarity_energy' when 'HAS_CUDA' is True.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. Vertex and face arrays are transferred to
        GPU memory internally.

    Returns
    -------
    float
        Total planarity energy computed on GPU (or CPU on OOM fallback).
    """
    import cupy as cp

    from src.backends import gpu_memory_guard, to_device

    _gpu_succeeded = False
    result = 0.0

    with gpu_memory_guard():
        verts_gpu = to_device(mesh.vertices)
        faces_gpu = to_device(mesh.faces)
        face_verts = verts_gpu[faces_gpu]
        centroids = face_verts.mean(axis=1, keepdims=True)
        centered = face_verts - centroids
        _, _, Vt = cp.linalg.svd(centered, full_matrices=False)
        normals = Vt[:, -1, :]
        signed_dists = cp.einsum("fvd,fd->fv", centered, normals)
        result = float(cp.sum(signed_dists**2).get())
        _gpu_succeeded = True

    if not _gpu_succeeded:
        face_verts = mesh.vertices[mesh.faces]
        centroids = face_verts.mean(axis=1, keepdims=True)
        centered = face_verts - centroids
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normals = Vt[:, -1, :]
        signed_dists = np.einsum("fvd,fd->fv", centered, normals)
        return float(np.sum(signed_dists**2))

    return result


# ── CuPy GPU fairness kernel ─────────────────────────────────────────────────


def _fairness_energy_gpu(mesh) -> float:
    """
    CuPy GPU implementation of fairness energy via sparse Laplacian.

    Converts the mesh Laplacian to a CuPy sparse matrix on the first call
    (cached on 'mesh._laplacian_gpu' for reuse across iterations), then
    performs the sparse matrix-vector product on GPU. Falls back to
    NumPy sparse multiplication if a GPU error occurs.

    This function is never called directly. It is dispatched by
    'compute_fairness_energy' when 'HAS_CUDA' is True.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. The GPU Laplacian cache is stored as
        'mesh._laplacian_gpu'.

    Returns
    -------
    float
        Total fairness energy computed on GPU (or CPU on OOM fallback).
    """
    import cupyx.scipy.sparse as cpsp

    from src.backends import gpu_memory_guard, to_device

    _gpu_succeeded = False
    result = 0.0

    with gpu_memory_guard():
        if not hasattr(mesh, "_laplacian_gpu") or mesh._laplacian_gpu is None:
            mesh._laplacian_gpu = cpsp.csr_matrix(mesh.laplacian)
        L = mesh._laplacian_gpu
        V = to_device(mesh.vertices)
        lv = L @ V
        result = float((lv * lv).sum().get())
        _gpu_succeeded = True

    if not _gpu_succeeded:
        # Direct NumPy fallback, avoids it re-entering GPU dispatch
        L_cpu = mesh.laplacian
        laplacian_coords = L_cpu @ mesh.vertices
        return float(np.sum(laplacian_coords**2))

    return result


def _angle_balance_energy_gpu(mesh) -> float:
    """
    CuPy GPU implementation of angle balance energy.

    Computes the conical mesh angle balance energy fully on GPU using
    vectorised CuPy operations. The algorithm:

      1. Identifies all 4-valent interior vertices via a validity mask.
      2. Gathers the four incident face indices per active vertex.
      3. For each face, locates the position of the vertex within the
         face and derives the two adjacent vertices (previous and next
         in the face's vertex order).
      4. Computes edge vectors and their dot products to get corner
         angles via arccos.
      5. Computes the squared alternating-angle imbalance per vertex
         and sums to a scalar.

    Falls back to Numba (if available) or serial NumPy if a GPU error
    occurs during computation.

    This function is never called directly. It is dispatched by
    'compute_angle_balance_energy' when 'HAS_CUDA' is True.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate. Must have 'vertex_face_ids_padded',
        'vertices', and 'faces' attributes.

    Returns
    -------
    float
        Total angle balance energy computed on GPU (or CPU fallback).

    References
    ----------
    Liu et al. (2006). "Geometric modelling with conical meshes and
    developable surfaces." ACM Transactions on Graphics, 25(3), 681-689.
    """
    import cupy as cp

    from src.backends import HAS_NUMBA, gpu_memory_guard, to_device

    _gpu_succeeded = False
    result = 0.0

    with gpu_memory_guard():
        vf = mesh.vertex_face_ids_padded  # (n_verts, max_val) int32
        faces_gpu = to_device(mesh.faces)  # (n_faces, 4)    int32
        verts_gpu = to_device(mesh.vertices)  # (n_verts, 3)    float64
        vf_gpu = to_device(vf)  # (n_verts, max_val)

        n_verts, max_val = vf_gpu.shape
        n_f_cols = faces_gpu.shape[1]

        # Build a boolean mask of 4-valent vertices
        valid_counts = cp.sum(vf_gpu != -1, axis=1)  # (n_verts,)
        is_4valent = valid_counts == 4  # (n_verts,)
        active_ids = cp.where(is_4valent)[0]  # (n_active,)
        n_active = int(active_ids.shape[0])

        if n_active == 0:
            result = 0.0
            _gpu_succeeded = True
        else:
            # Gather the 4 face ids per active vertex: (n_active, 4)
            face_ids_active = vf_gpu[active_ids, :4]  # (n_active, 4)

            # Gather faces for those face ids: (n_active, 4, 4) [vertex indices]
            faces_for_active = faces_gpu[face_ids_active]  # (n_active, 4, n_f_cols)

            # For each active vertex v and each of its 4 faces, find local_idx
            # by broadcasting: compare faces_for_active against active_ids
            vid_broadcast = active_ids[:, None, None]  # (n_active, 1, 1)
            local_match = faces_for_active == vid_broadcast  # (n_active, 4, n_f_cols)
            local_idx = cp.argmax(local_match.astype(cp.int32), axis=2)  # (n_active, 4)

            # Gather prev/next vertex indices using modular indexing
            prev_local = (local_idx - 1) % n_f_cols  # (n_active, 4)
            next_local = (local_idx + 1) % n_f_cols

            # Build flat gather indices into faces_for_active
            face_row = (
                cp.arange(n_active, dtype=cp.int32)[:, None]
                * cp.ones(4, dtype=cp.int32)[None, :]
            )  # (n_active, 4)
            face_col = (
                cp.arange(4, dtype=cp.int32)[None, :]
                * cp.ones(n_active, dtype=cp.int32)[:, None]
            )  # (n_active, 4)

            # Flat-index into faces_for_active: (n_active, 4, n_f_cols)
            fa_flat = faces_for_active.reshape(n_active * 4, n_f_cols)
            pl_flat = prev_local.ravel()
            nx_flat = next_local.ravel()

            row_idx = cp.arange(n_active * 4, dtype=cp.int32)
            vp_ids = fa_flat[row_idx, pl_flat].reshape(n_active, 4)  # (n_active, 4)
            vn_ids = fa_flat[row_idx, nx_flat].reshape(n_active, 4)

            # Gather 3D positions: (n_active, 4, 3)
            v_pos = verts_gpu[active_ids]  # (n_active, 3)
            vp_pos = verts_gpu[vp_ids.ravel()].reshape(n_active, 4, 3)
            vn_pos = verts_gpu[vn_ids.ravel()].reshape(n_active, 4, 3)

            e1 = vp_pos - v_pos[:, None, :]  # (n_active, 4, 3)
            e2 = vn_pos - v_pos[:, None, :]

            l1 = cp.linalg.norm(e1, axis=2, keepdims=True).clip(min=1e-12)
            l2 = cp.linalg.norm(e2, axis=2, keepdims=True).clip(min=1e-12)

            e1h = e1 / l1
            e2h = e2 / l2

            cos_a = cp.sum(e1h * e2h, axis=2).clip(-1.0 + 1e-8, 1.0 - 1e-8)
            angles = cp.arccos(cos_a)  # (n_active, 4)

            imbalance = (angles[:, 0] + angles[:, 2]) - (angles[:, 1] + angles[:, 3])
            result = float(cp.sum(imbalance**2).get())
            _gpu_succeeded = True

    if not _gpu_succeeded:
        # Fallback: Numba if available, else serial NumPy
        try:
            from src.backends import HAS_NUMBA

            if HAS_NUMBA:
                vf = mesh.vertex_face_ids_padded
                return float(_angle_balance_numba(mesh.vertices, mesh.faces, vf))
        except Exception:
            pass
        energy = 0.0
        for vertex_id in range(mesh.n_vertices):
            imbalance = compute_conical_angle_imbalance(mesh, vertex_id)
            energy += imbalance**2
        return energy

    return result
