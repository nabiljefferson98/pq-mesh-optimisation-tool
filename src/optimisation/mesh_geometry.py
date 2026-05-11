"""
src/optimisation/mesh_geometry.py

Geometric metric functions for planar quad mesh analysis.

This module provides per-face and per-vertex geometric measurements that
are used both during optimisation (for convergence monitoring) and in the
post-optimisation analysis tools. The functions in this module operate on
individual faces or vertices rather than the full mesh, making them
straightforward to understand and test in isolation.

For computational efficiency during optimisation, the energy functions in
'energy_terms.py' compute equivalent quantities in fully vectorised form
over all faces simultaneously. The functions here are provided as clear,
readable reference implementations and are used in the visualisation and
diagnostic tools where per-face or per-vertex granularity is needed.

Functions provided
------------------
  compute_face_planarity_deviation -- deviation of a single face from its
                                      best-fit plane (planarity metric).
  compute_all_planarity_deviations -- deviation for every face in a mesh.
  compute_angle_at_vertex_in_face -- interior angle at one vertex of a face.
  compute_conical_angle_imbalance -- angle balance imbalance at one vertex
                                     (measure of the conical mesh condition).

References
----------
Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.

Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.
"""

from typing import Any

import numpy as np
from numpy import dtype, ndarray


def compute_face_planarity_deviation(face_vertices: np.ndarray) -> float:
    """
    Compute the planarity deviation of a single quad face.

    A quad face is planar if all four of its vertices lie exactly on a
    common plane. In practice, optimised faces are only approximately
    planar. This function measures the deviation from planarity as the
    maximum absolute distance of any vertex from the face's best-fit plane.

    The best-fit plane is found by computing the singular value
    decomposition of the matrix of centred vertex positions. The right
    singular vector corresponding to the smallest singular value is the
    normal to the plane that minimises the sum of squared vertex distances.
    This is the standard approach for fitting a plane to a set of points
    in three dimensions.

    For a triangle face, the deviation is always zero because three points
    are always coplanar, and 0.0 is returned immediately without computation.

    Parameters
    ----------
    face_vertices : numpy.ndarray
        Vertex positions of the face, shape (n, 3) where n is 3 or 4.
        dtype float64.

    Returns
    -------
    float
        Maximum distance from any vertex to the best-fit plane. Zero for
        a perfectly planar face or for a triangle face. Always non-negative.

    References
    ----------
    Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
    "Architectural geometry." Computers and Graphics, 47, pp. 145-164.
    """
    if face_vertices.shape[0] == 3:
        # Triangle is always planar
        return 0.0

    # Centre the points
    centroid = np.mean(face_vertices, axis=0)
    centred = face_vertices - centroid

    # Compute normal via SVD (smallest singular vector)
    U, S, Vt = np.linalg.svd(centred)
    normal = Vt[-1, :]  # Last row = normal to best-fit plane

    # Compute distances from each vertex to the plane
    distances = np.abs(np.dot(centred, normal))

    # Planarity deviation = max distance (or RMS)
    return np.max(distances)


def compute_all_planarity_deviations(mesh) -> np.ndarray:
    """
    Compute the planarity deviation for every face in a mesh.

    Calls 'compute_face_planarity_deviation' for each face in the sequence and
    returns the results as a NumPy array. This function is used in the
    visualisation and reporting tools to produce per-face planarity maps
    and summary statistics.

    For large meshes during active optimisation, use the vectorised
    planarity energy computation in 'energy_terms.py' instead, as it is
    significantly faster. This function is intended for post-optimisation
    analysis and visualisation where per-face granularity is required.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to analyse.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_faces,) containing the planarity deviation for
        each face, in the same order as 'mesh.faces'. Values are
        non-negative floats; zero indicates a perfectly planar face.
    """
    deviations = np.zeros(mesh.n_faces)

    for face_id in range(mesh.n_faces):
        face_verts = mesh.get_face_vertices(face_id)
        deviations[face_id] = compute_face_planarity_deviation(face_verts)

    return deviations


def compute_angle_at_vertex_in_face(
    face_vertices: np.ndarray, vertex_index: int
) -> ndarray[tuple[Any, ...], dtype[Any]]:
    """
    Compute the interior angle at one vertex of a face.

    The interior angle at a vertex is the angle between the two edges of
    the face that meet at that vertex. It is computed from the dot product
    of the two normalised edge direction vectors. A small epsilon is added
    to the edge lengths before normalisation to prevent division by zero
    for degenerate edges.

    The vertex at 'vertex_index' is the current vertex; the previous and
    next vertices are determined by the cyclic ordering of 'face_vertices',
    wrapping around at the first and last positions.

    Parameters
    ----------
    face_vertices : numpy.ndarray
        Vertex positions of the face, shape (n, 3) where n is 3 or 4.
        dtype float64.
    vertex_index : int
        Index of the vertex within the face at which to compute the angle.
        Must be in the range [0, n).

    Returns
    -------
    float
        Interior angle at the specified vertex, in radians. Always in the
        range [0, pi].
    """
    n = len(face_vertices)

    # Get the three points: prev, current, next
    v_prev = face_vertices[(vertex_index - 1) % n]
    v_curr = face_vertices[vertex_index]
    v_next = face_vertices[(vertex_index + 1) % n]

    # Vectors from current vertex
    edge1 = v_prev - v_curr
    edge2 = v_next - v_curr

    # Normalise
    edge1_norm = edge1 / (np.linalg.norm(edge1) + 1e-12)
    edge2_norm = edge2 / (np.linalg.norm(edge2) + 1e-12)

    # Angle via dot product
    cos_angle = np.clip(np.dot(edge1_norm, edge2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return angle


def compute_conical_angle_imbalance(mesh, vertex_id: int) -> float:
    """
    Compute the conical angle imbalance at a single vertex.

    The conical mesh condition requires that for every 4-valent interior
    vertex, the alternating face angles sum to equal values:

        alpha_0 + alpha_2 == alpha_1 + alpha_3

    where alpha_0 to alpha_3 are the four face angles at the vertex, in
    cyclic order around it. When this condition holds exactly at every
    vertex, the mesh is a conical mesh: all face planes at each vertex are
    tangent to a common cone. This is the discrete analogue of the condition
    for a surface to be developable.

    The imbalance at a vertex is the absolute difference between the two
    alternating sums:

        imbalance = |(alpha_0 + alpha_2) - (alpha_1 + alpha_3)|

    A value of zero means the vertex satisfies the conical condition exactly.
    Larger values indicate greater deviation from a conical configuration.

    Vertices with fewer or more than four incident faces do not satisfy the
    standard conical condition and contribute zero imbalance by convention.
    These include boundary vertices and irregular-valence interior vertices.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to analyse. Must have 'get_vertex_faces' and
        'get_face_vertices' methods.
    vertex_id : int
        Index of the vertex to evaluate. Must be in the range
        [0, mesh.n_vertices).

    Returns
    -------
    float
        Conical angle imbalance in radians. Zero for non-4-valent vertices
        or for vertices that satisfy the conical condition exactly.

    References
    ----------
    Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
    "Geometric modelling with conical meshes and developable surfaces."
    ACM Transactions on Graphics, 25(3), pp. 681-689.
    """
    # Get faces around this vertex
    incident_faces = mesh.get_vertex_faces(vertex_id)

    if len(incident_faces) != 4:
        # Conical condition typically requires 4-valent vertices
        # For boundary or irregular vertices, return a penalty
        return 0.0  # Or np.inf, depending on how you want to handle this

    # Compute angles at this vertex in each face
    angles = []
    for face_id in incident_faces:
        face_verts = mesh.get_face_vertices(face_id)
        # Find which index in the face corresponds to vertex_id
        local_index = np.where(mesh.faces[face_id] == vertex_id)[0][0]
        angle = compute_angle_at_vertex_in_face(face_verts, local_index)
        angles.append(angle)

    # Check angle balance: w1 + w3 = w2 + w4
    # (Assuming angles are ordered around the vertex)
    if len(angles) == 4:
        imbalance = abs((angles[0] + angles[2]) - (angles[1] + angles[3]))
    else:
        imbalance = 0.0  # pragma: no cover

    return imbalance
