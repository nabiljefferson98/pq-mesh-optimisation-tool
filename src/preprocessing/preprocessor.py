"""
src/preprocessing/preprocessor.py

Pre-processing pipeline for real-world and architectural quad mesh datasets.

Real-world architectural meshes imported from CAD tools such as Blender,
Rhino, and ArchiCAD differ from the synthetic unit-scale flat grids used
during development in three critical ways that must be addressed before any
mesh reaches the optimiser:

  1. Arbitrary scale:
     A CAD model in millimetres has vertex coordinates of order 1e3 to 1e4.
     The squared-distance planarity energy scales as the fourth power of the
     coordinate size, producing energy values of order 1e8 to 1e16 for
     unscaled models. This causes L-BFGS-B line-search failure on the very
     first iteration, as the gradient norm far exceeds the step-size bounds
     of the SciPy solver. Scale normalisation to a unit bounding box
     (the longest axis = 1.0) reduces the energy to order 1e-2, well within the
     solver's operating range.

  2. Duplicate vertices:
     Most mesh exporters write separate vertex copies at each shared edge,
     producing a "polygon soup" rather than a connected mesh. Duplicate
     vertices break the vertex-to-face adjacency graph used by the
     Laplacian fairness term and the conical angle balance gradient, causing
     incorrect gradient computations. Merging vertices within a configurable
     distance threshold (default 1e-8) restores full connectivity.

  3. Degenerate faces:
     Zero-area quads arise from coincident vertices or degenerate topology.
     They cause division by zero inside the batched SVD used for planarity
     energy computation and produce infinite values in the gradient, causing
     immediate optimiser failure. Removing faces below an area threshold
     (default 1e-10) eliminates this failure mode.

Pipeline stages
---------------
  Stage 1:  Record bounding box of original mesh for diagnostics.
  Stage 2:  Merge duplicate vertices using a cKDTree-based O(n log n)
            union-find algorithm.
  Stage 3:  Remove degenerate (zero-area) faces.
  Stage 4:  Normalise: centre at origin, scale the longest axis to target_scale.
  Stage 5:  Rebuild a 'QuadMesh' with 'vertices_original' set to the
            post-normalised positions.
  Stage 6:  Auto-suggest energy weights calibrated to this mesh's initial
            energy magnitudes.

Usage
-----
::

    from src.io.obj_handler import load_obj
    from src.preprocessing.preprocessor import preprocess_mesh

    raw = load_obj("data/input/architectural/saddle_8x8.obj")
    mesh, info = preprocess_mesh(raw, normalise=True, verbose=True)

After this call, 'mesh' is ready to pass directly to 'MeshOptimiser'.
The 'closeness' energy term will measure vertex displacement relative to
the post-normalised positions stored in 'mesh.vertices_original', which is
the correct baseline for preserving design intent during optimisation.

References
----------
Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.

Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.

Author: Muhammad Nabil
Date: March 2026
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.mesh import QuadMesh

# ============================================================================
# DATA CLASS: PREPROCESSING RECORD
# ============================================================================


@dataclass
class PreprocessingInfo:
    """
    Records every transformation applied during pre-processing and mesh statistics.

    An instance of this class is returned alongside the processed mesh by
    'preprocess_mesh()'. It provides full traceability of what was done to
    the mesh, which is important for dissertation documentation and for
    reproducing results.

    Attributes
    ----------
    original_vertices : int
        Vertex count of the input mesh before any processing.
    original_faces : int
        Face count of the input mesh before any processing.
    final_vertices : int
        Vertex count of the processed mesh ready for optimisation.
    final_faces : int
        Face count of the processed mesh ready for optimisation.
    was_normalised : bool
        True if scale normalisation was applied.
    scale_factor : float
        The multiplier applied to all vertex coordinates during normalisation.
        A value of 1.0 means no scaling was applied (either normalisation was
        disabled or the mesh was already unit scale).
    centroid_offset : numpy.ndarray
        The translation vector subtracted from all vertices to centre the
        mesh at the origin, shape (3, ). Zero if normalisation was disabled.
    removed_degenerate : int
        Number of degenerate (zero-area) faces removed.
    removed_duplicate : int
        Number of duplicate vertex pairs merged.
    bounding_box_size : tuple of float
        The (dx, dy, dz) bounding box dimensions of the original (pre-normalised)
        mesh, in the original coordinate system. Useful for understanding the
        model's original scale.
    suggested_weights : dict or None
        Auto-tuned energy weight dictionary with keys 'planarity', 'fairness',
        'closeness', and 'angle_balance', or None if weight suggestion failed.
    warnings : list of str
        Non-fatal warnings emitted during processing (for example, reporting
        the number of merged duplicates or removed degenerate faces).
    """
    original_vertices: int = 0
    original_faces: int = 0
    final_vertices: int = 0
    final_faces: int = 0
    was_normalised: bool = False
    scale_factor: float = 1.0
    centroid_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    removed_degenerate: int = 0
    removed_duplicate: int = 0
    bounding_box_size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    suggested_weights: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# PUBLIC API
# ============================================================================
def preprocess_mesh(
    mesh: QuadMesh,
    normalise: bool = True,
    target_scale: float = 1.0,
    remove_degenerates: bool = True,
    merge_duplicates: bool = True,
    duplicate_threshold: float = 1e-8,
    verbose: bool = True,
) -> Tuple[QuadMesh, PreprocessingInfo]:
    """
    Apply the full pre-processing pipeline to a raw quad mesh.

    Executes the six pipeline stages in order (see module docstring) and
    returns a new 'QuadMesh' that is clean, unit-scale, and ready for
    optimisation. The input mesh is not modified in place.

    Parameters
    ----------
    mesh : QuadMesh
        The raw input mesh as returned by 'load_obj()'. Not modified.
    normalise : bool, optional
        Apply scale normalisation (stage 4). Strongly recommended for
        real-world datasets. Default is True.
    target_scale : float, optional
        Desired length of the longest bounding-box axis after normalisation.
        The default is 1.0 (unit scale).
    remove_degenerates : bool, optional
        Remove zero-area faces (stage 3). Default is True.
    merge_duplicates : bool, optional
        Merge coincident vertices (stage 2). Default is True.
    duplicate_threshold : float, optional
        Distance threshold for duplicate vertex detection in world units
        before normalisation. Default is 1e-8.
    verbose : bool, optional
        Print a preprocessing report to standard output. Default is True.

    Returns
    -------
    processed_mesh : QuadMesh
        The cleaned, normalised mesh. Its 'vertices_original' attribute
        is set to the post-normalised positions so that the closeness energy
        measures displacement from the pre-optimisation shape.
    info : PreprocessingInfo
        Full record of all transformations applied and mesh statistics.

    Raises
    ------
    TypeError
        If 'mesh' is not a 'QuadMesh' instance.
    ValueError
        If no valid faces remain after duplicate merging and degenerate
        removal. This indicates a fundamentally broken input mesh.
    """
    if not isinstance(mesh, QuadMesh):
        raise TypeError(
            f"preprocess_mesh expects a QuadMesh, got {type(mesh).__name__}"
        )

    info = PreprocessingInfo(
        original_vertices=mesh.n_vertices,
        original_faces=mesh.n_faces,
    )

    vertices = mesh.vertices.copy()
    faces = mesh.faces.tolist()  # List[List[int]] for flexible manipulation

    # Stage 1 — record original bounding box
    bb_min = vertices.min(axis=0)
    bb_max = vertices.max(axis=0)
    bb_size = bb_max - bb_min
    info.bounding_box_size = (float(bb_size[0]), float(bb_size[1]), float(bb_size[2]))

    # Stage 2 — merge duplicate vertices
    if merge_duplicates:
        vertices, faces, n_merged = _merge_duplicate_vertices(
            vertices, faces, threshold=duplicate_threshold
        )
        info.removed_duplicate = n_merged
        if n_merged > 0:
            info.warnings.append(
                f"Merged {n_merged} duplicate vertex pair(s) — "
                "common in Blender/Rhino OBJ exports."
            )

    # Stage 3 — remove degenerate faces
    if remove_degenerates:
        faces, n_removed = _remove_degenerate_faces(vertices, faces)
        info.removed_degenerate = n_removed
        if n_removed > 0:
            info.warnings.append(f"Removed {n_removed} degenerate (zero-area) face(s).")

    if len(faces) == 0:
        raise ValueError(
            "No valid faces remain after preprocessing.\n"
            "Check input mesh for degenerate geometry "
            "(all faces may be zero-area or out-of-bounds)."
        )

    # Stage 4 — normalise
    face_array = np.array(faces, dtype=np.int32)
    if normalise:
        vertices, scale_factor, centroid = _normalise_vertices(
            vertices, target_scale=target_scale
        )
        info.was_normalised = True
        info.scale_factor = float(scale_factor)
        info.centroid_offset = centroid

    # Stage 5 — build cleaned mesh
    processed_mesh = QuadMesh(vertices, face_array)
    # Critical: reset baseline for closeness energy to post-normalised shape
    processed_mesh.vertices_original = vertices.copy()

    info.final_vertices = processed_mesh.n_vertices
    info.final_faces = processed_mesh.n_faces

    # Stage 6 — auto-suggest weights
    info.suggested_weights = suggest_weights_for_mesh(processed_mesh)

    if verbose:
        _print_report(info)

    return processed_mesh, info


def suggest_weights_for_mesh(mesh: QuadMesh) -> Dict[str, float]:
    """
    Auto-tune energy weights calibrated to the given mesh's initial energies.

    Computes the initial planarity and fairness energies of the mesh and
    scales the weights so that the weighted energies are of comparable
    magnitude at the start of optimisation. The target ratio is:

        w_p * E_p : w_f * E_f : w_c * E_c ≈ 10 : 1 : 5

    This prevents any single energy term from dominating the gradient
    entirely, which would cause the optimiser to make progress on one
    goal while ignoring the others.

    A special case applies when the initial planarity energy is below
    1e-8 (the mesh is already near-planar, as is typical for meshes
    parameterised along principal curvature lines). In this case,
    setting w_p = 10 / E_p would produce astronomically large weights
    that amplify numerical noise rather than driving useful convergence.
    Instead, weights are calibrated to the fairness energy alone, and a
    diagnostic message is printed recommending noise injection for testing.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh for which to compute suggested weights. Must have at least
        one vertex and one face.

    Returns
    -------
    dict
        Dictionary with keys 'planarity', 'fairness', 'closeness', and
        'angle_balance'. All values are non-negative floats rounded to
        four decimal places. The 'angle_balance' weight is always 0.0
        in the auto-suggested configuration; it should be set manually
        if the conical mesh condition is desired.
    """
    from src.optimisation.energy_terms import (
        compute_fairness_energy,
        compute_planarity_energy,
    )

    if mesh.n_vertices == 0 or mesh.n_faces == 0:
        return {
            "planarity": 100.0,
            "fairness": 1.0,
            "closeness": 10.0,
            "angle_balance": 0.0,
        }

    E_p = float(compute_planarity_energy(mesh))
    E_f = float(compute_fairness_energy(mesh)) + 1e-12

    # Proxy for expected closeness energy at first meaningful step
    bb = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    disp_proxy = float(bb.max()) * 0.01
    E_c_proxy = disp_proxy**2 * mesh.n_vertices + 1e-12

    NEAR_ZERO_PLANARITY = 1e-8

    if E_p < NEAR_ZERO_PLANARITY:
        # Mesh is already near-PQ. Use moderate, fairness-calibrated weights.
        # Setting planarity weight proportional to 1/fairness keeps the
        # gradient well-conditioned without amplifying near-zero noise.
        w_p = float(np.clip(1.0 / E_f, 1.0, 1e4))
        w_f = 1.0
        # Before: closeness at 50% of planarity target = too competitive
        # After: closeness at 10% of planarity target = complementary regularity
        w_c = float(np.clip(1.0 / E_c_proxy, 1e-2, 1e5))
        print(
            f"  ℹ️  Initial planarity ≈ 0 ({E_p:.2e}) — mesh is near-PQ by "
            f"construction.\n"
            f"     Using fairness-calibrated weights. To test planarity "
            f"optimisation,\n"
            f"     regenerate with noise:  "
            f"python scripts/mesh_generation/generate_test_meshes.py"
        )
    else:
        # Normal case: planarity is non-trivial, scale all weights accordingly
        w_p = float(np.clip(10.0 / E_p, 1.0, 1e6))
        w_f = float(np.clip(1.0 / E_f, 1e-4, 1e3))
        w_c = float(np.clip(5.0 / E_c_proxy, 1e-2, 1e5))

    return {
        "planarity": round(w_p, 4),
        "fairness": round(w_f, 4),
        "closeness": round(w_c, 4),
        "angle_balance": 0.0,
    }


# ============================================================================
# PRIVATE STAGE IMPLEMENTATIONS
# ============================================================================
def _normalise_vertices(
    vertices: np.ndarray,
    target_scale: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Centre a vertex array at the origin and scale to a unit bounding box.

    Translates all vertices so that their centroid is at the origin, then
    scales uniformly so that the longest bounding-box axis equals
    'target_scale'. If the mesh is degenerate (all vertices coincident),
    the array is centred but not scaled.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertex positions, shape (n, 3), dtype float64.
    target_scale : float, optional
        Desired length of the longest bounding-box axis. Default is 1.0.

    Returns
    -------
    normalised_vertices : numpy.ndarray
        Transformed vertex positions, shape (n, 3).
    scale_factor : float
        The multiplier applied: normalised = (original - centroid) * scale.
    centroid : numpy.ndarray
        The centroid that was subtracted, shape (3, ).
    """
    centroid = vertices.mean(axis=0)
    v = vertices - centroid
    bb = v.max(axis=0) - v.min(axis=0)
    longest = float(bb.max())
    if longest < 1e-12:
        return v, 1.0, centroid
    scale = target_scale / longest
    return v * scale, scale, centroid


def _merge_duplicate_vertices(
    vertices: np.ndarray,
    faces: List[List[int]],
    threshold: float = 1e-8,
) -> Tuple[np.ndarray, List[List[int]], int]:
    """
    Merge all vertices within 'threshold' distance using a cKDTree union-find.

    Replaces the original O(n^2) pairwise distance scan with an O(n log n)
    spatial query. The algorithm proceeds in four steps:

      Step 1: Build a cKDTree over all vertex positions.
      Step 2: Query all pairs of vertices within 'threshold' distance using
              'query_pairs()'.
      Step 3: Apply union-find with path compression to cluster all coincident
              vertices into equivalence classes. The representative of each
              class is the vertex with the smallest index, ensuring
              deterministic output.
      Step 4: Remap all face indices to the representative vertex index of
              their cluster. Discard duplicate representative mappings.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertex positions, shape (n, 3), dtype float64.
    faces : list of list of int
        Face connectivity as a list of vertex-index lists.
    threshold : float, optional
        Distance threshold for merging. Default is 1e-8.

    Returns
    -------
    new_vertices : numpy.ndarray
        Compacted vertex array with duplicate rows removed.
    new_faces : list of list of int
        Face connectivity remapped to the compacted vertex indices.
    n_merged : int
        Number of vertex pairs that were merged.
    """
    from scipy.spatial import cKDTree

    n = len(vertices)
    if n == 0:
        return vertices, faces, 0

    tree = cKDTree(vertices)
    # Query all pairs within threshold — returns list of pairs
    pairs = tree.query_pairs(r=threshold, output_type="ndarray")

    # Union-Find: each vertex starts as its own root
    parent = np.arange(n, dtype=np.int32)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            # Always merge to the lower index for deterministic output
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for a, b in pairs:
        _union(int(a), int(b))

    # Flatten all roots
    roots = np.array([_find(i) for i in range(n)], dtype=np.int32)

    unique_roots = sorted(set(int(r) for r in roots))
    old_to_new = {old: new for new, old in enumerate(unique_roots)}

    new_vertices = vertices[unique_roots]
    remap = np.array([old_to_new[int(roots[i])] for i in range(n)], dtype=np.int32)

    new_faces = [[int(remap[v]) for v in face] for face in faces]
    n_merged = n - len(unique_roots)
    return new_vertices, new_faces, n_merged


def _remove_degenerate_faces(
    vertices: np.ndarray,
    faces: List[List[int]],
    area_threshold: float = 1e-10,
) -> Tuple[List[List[int]], int]:
    """
    Remove faces whose area falls below 'area_threshold'.

    Quad area is approximated using the cross-product of the two diagonal
    vectors: area ≈ 0.5 * ||d1 × d2|| where d1 = v_2 - v_0 and
    d2 = v_3 - v_1. Triangle area uses the standard cross-product formula.
    N-gon faces (five or more vertices) are always kept as they are passed
    through to the OBJ handler.

    Faces with any out-of-bounds vertex index are also removed, as they
    indicate corrupt mesh connectivity that would cause index errors in
    the optimiser.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertex positions, shape (n, 3), dtype float64.
    faces : list of list of int
        Face connectivity as a list of vertex-index lists.
    area_threshold : float, optional
        Minimum face area to keep. Default is 1e-10.

    Returns
    -------
    keep : list of an int
        Filtered face list containing only non-degenerate faces with valid
        vertex indices.
    n_removed : int
        Number of faces that were removed.
    """
    keep: List[List[int]] = []
    n_verts = len(vertices)

    for face in faces:
        # Skip faces with out-of-bounds indices (common in corrupt exports)
        if any(v < 0 or v >= n_verts for v in face):
            continue

        verts = vertices[face]
        area: float
        if len(face) == 4:
            d1 = verts[2] - verts[0]
            d2 = verts[3] - verts[1]
            area = float(0.5 * np.linalg.norm(np.cross(d1, d2)))
        elif len(face) == 3:
            area = float(
                0.5 * np.linalg.norm(np.cross(verts[1] - verts[0], verts[2] - verts[0]))
            )
        else:
            area = 1.0  # keep n-gons (they pass through to obj_handler)

        if area > area_threshold:
            keep.append(face)

    n_removed = len(faces) - len(keep)
    return keep, n_removed


# ============================================================================
# REPORTING
# ============================================================================
def _print_report(info: PreprocessingInfo) -> None:
    """
    Print a formatted preprocessing report to standard output.

    Displays vertex and face counts before and after processing, the
    bounding box of the original mesh, normalisation parameters,
    duplicate and degenerate removal counts, any warnings, and the
    auto-suggested energy weights. Called by 'preprocess_mesh()' when
    'verbose=True'.

    Parameters
    ----------
    info : PreprocessingInfo
        The preprocessing record produced by 'preprocess_mesh()'.
    """
    print("=" * 60)
    print("PREPROCESSING REPORT")
    print("=" * 60)
    print(f"  Vertices : {info.original_vertices:>6} → {info.final_vertices}")
    print(f"  Faces    : {info.original_faces:>6} → {info.final_faces}")
    bb = info.bounding_box_size
    print(f"  Bounding box (original): " f"{bb[0]:.4f} × {bb[1]:.4f} × {bb[2]:.4f}")
    if info.was_normalised:
        print(f"  Normalised  : scale factor = {info.scale_factor:.6e}")
    if info.removed_duplicate:
        print(f"  Duplicates removed : {info.removed_duplicate}")
    if info.removed_degenerate:
        print(f"  Degenerate faces removed : {info.removed_degenerate}")
    for w in info.warnings:
        print(f"  ⚠️  {w}")
    if info.suggested_weights:
        sw = info.suggested_weights
        print(
            f"  Suggested weights → "
            f"planarity={sw['planarity']:.2f}, "
            f"fairness={sw['fairness']:.4f}, "
            f"closeness={sw['closeness']:.4f}"
        )
    print("=" * 60 + "\n")
