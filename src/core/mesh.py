"""
src/core/mesh.py

Core mesh data structure for planar quad mesh optimisation.

This module defines the 'QuadMesh' class, which is the central data structure
of the entire project. Every other module -- the energy functions, gradient
functions, optimiser, preprocessor, and exporters -- operates on a 'QuadMesh'
instance.

A QuadMesh stores:

  Vertices -- a two-dimensional array of 3D coordinates, one row per vertex.
  Faces -- a two-dimensional array of vertex indices, one row per face,
               where each row lists the four vertices that form a quad face.

In addition to these two core arrays, the class maintains a collection of
cached derived quantities. Computing these from scratch at every optimisation
iteration would be prohibitively expensive, so they are built once from the
mesh topology on first access and then reused for the lifetime of the object.
The caches include the sparse Laplacian matrix, the scatter matrix for gradient
accumulation, the vertex-to-face adjacency table, and the pre-allocated scratch
buffers used by the Numba gradient kernels.

Because face connectivity never changes during optimisation (only vertex
positions change), all topology caches remain valid from construction until
the mesh is explicitly replaced. Callers should only call
'reset_topology_cache()' if they modify face connectivity after construction,
which is not part of the normal optimisation workflow.

Supported face types
--------------------
The class is designed for quad meshes (four vertices per face) but also accepts
triangular faces (three vertices per face) for compatibility with mesh files
that contain a mix of face types. The angle balance energy and its gradient are
only defined for 4-valent interior vertices on quad meshes.

References
----------
Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.

Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.
"""
from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix


class QuadMesh:
    """
    Core mesh data structure for quad-dominant surfaces.

    Stores vertex positions and face connectivity for a quad mesh and
    provides lazy-cached access to all derived topological and algebraic
    quantities needed by the optimisation pipeline. All caches are built
    on first access and reused across iterations.

    Parameters
    ----------
    vertices : array-like
        Vertex coordinates. Must be convertible to a NumPy array of shape
        (n_vertices, 3) containing finite float64 values.
    faces : array-like
        Face connectivity. Must be convertible to a NumPy array of shape
        (n_faces, verts_per_face) containing non-negative int32 indices
        into the vertex array. All indices must be less than n_vertices.
        Typically, verts_per_face is 4 for quad meshes.

    Attributes
    ----------
    vertices : numpy.ndarray
        Current vertex positions, shape (n_vertices, 3), dtype float64.
        Updated in-place by the optimiser at each iteration.
    faces : numpy.ndarray
        Face connectivity, shape (n_faces, verts_per_face), dtype int32.
        Fixed at construction; never modified during optimisation.
    vertices_original : numpy.ndarray
        Copy of the vertex positions at construction time. Used by the
        closeness energy term to measure how far vertices have drifted
        from the original design. Shape (n_vertices, 3), dtype float64.

    Raises
    ------
    ValueError
        If 'vertices' is not a (n, 3) array, if 'faces' is not a 2D
        array, or if any face index is negative or out of bounds.

    Examples
    --------
    Creating a simple quad mesh with four vertices and one face:

        import numpy as np
        from src.core.mesh import QuadMesh

        vertices = np.array([[0, 0, 0], [1, 0, 0],
                              [1, 1, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2, 3]])
        mesh = QuadMesh(vertices, faces)
        print(mesh.n_vertices, mesh.n_faces)
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Initialise the mesh and validate the input arrays.

        Converts both inputs to NumPy arrays of the required dtype, checks
        that their shapes are valid, confirms that all face indices are
        within bounds, and stores a copy of the initial vertex positions
        for the closeness energy term.

        Parameters
        ----------
        vertices : array-like
            Vertex coordinates of shape (n_vertices, 3).
        faces : array-like
            Face indices of shape (n_faces, verts_per_face). All values
            must be in the range [0, n_vertices).

        Raises
        ------
        ValueError
            If 'vertices' do not have a shape (n, 3), if 'faces' is not
            two-dimensional, if any face index is negative, or if any
            face index equals or exceeds the number of vertices.
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)

        # Validation
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be (n, 3), got {self.vertices.shape}")

        if self.faces.ndim != 2:
            raise ValueError(f"Faces must be 2D array, got {self.faces.ndim}D")

        if not np.all((self.faces.shape[1] == 3) | (self.faces.shape[1] == 4)):
            # Allow triangles for now, but warn
            print(f"Warning: Non-quad faces detected (shape: {self.faces.shape})")  # pragma: no cover

        # Validate face indices
        if self.faces.size > 0:  # Only check if faces exist
            if np.min(self.faces) < 0:
                raise ValueError(
                    f"Face indices must be non-negative: min index {np.min(self.faces)}"
                    " (negative indices are not supported)"
                )
            if np.max(self.faces) >= len(self.vertices):
                raise ValueError(
                    f"Face indices out of bounds: max index {np.max(self.faces)}"
                    f" >= {len(self.vertices)} vertices"
                )

        # Store original positions for closeness energy
        self.vertices_original = self.vertices.copy()

        # Topology caches (computed lazily)
        self._vertex_faces: Optional[List[List[int]]] = None
        self._vertex_neighbors: Optional[List[List[int]]] = None

        # Sparse scatter matrix: shape (n_verts, n_entries), where
        #    n_entries = faces.ravel().size = n_faces * verts_per_face
        # S[v, k] = 1 iff mesh.faces.ravel()[k] == v  →  gradient = S @ contrib
        self._scatter_matrix: Optional[csr_matrix] = None

        # Padded vertex-face adjacency for uniform face degree equal to the
        # maximum face degree in the mesh (typically 4 for quad-dominant meshes),
        self._vertex_face_ids_padded: Optional[np.ndarray] = None

        # GPU-side topology caches (used by faster backends)
        self._laplacian_gpu: Optional[csr_matrix] = None
        self._scatter_matrix_gpu: Optional[csr_matrix] = None

        # Numba-side scratch caches (used by faster backends)
        self._numba_scratch: Optional[tuple] = None

        self._laplacian_cpu: Optional[csr_matrix] = None
        self._vertex_face_overflow_mask: Optional[np.ndarray] = None

    @property
    def n_vertices(self) -> int:
        """
        Number of vertices in the mesh.

        Returns
        -------
        int
            The total number of vertices, equal to 'len(self.vertices)'.
        """
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        """
        Number of faces in the mesh.

        Returns
        -------
        int
            The total number of faces, equal to 'len(self.faces)'.
        """
        return len(self.faces)

    @property
    def scatter_matrix(self) -> csr_matrix:
        """
        Sparse scatter matrix for accumulating per-face gradient contributions
        back to vertex positions.

        The planarity gradient is computed as a (n_faces, verts_per_face, 3)
        array of per-face contributions. To sum these into a (n_vertices, 3)
        gradient array, each contribution must be added to the row
        corresponding to its vertex. Doing this with a Python loop over faces
        is slow. The scatter matrix encodes the same operation as a single
        sparse matrix multiplied, which is backed by BLAS and runs efficiently
        on both CPU and GPU.

        The matrix has shape (n_vertices, n_faces * verts_per_face). Entry
        (v, k) is 1 if the k-th entry of the flattened face array belongs to
        vertex v, and 0 otherwise. Multiplying by a (n_entries, 3) contribution
        array gives the correct (n_vertices, 3) gradient.

        Built once from the face connectivity on first access and cached for
        the lifetime of the mesh. Because faces never change during
        optimisation, this cache is always valid.

        Returns
        -------
        scipy.sparse.csr_matrix
            Scatter matrix of shape (n_vertices, n_faces * verts_per_face),
            dtype float64, stored in CSR format.
        """
        if self._scatter_matrix is None:
            flat = self.faces.ravel()  # vertex index per (face, local_vertex) entry
            n_entries = flat.size  # n_faces * verts_per_face (3 or 4)
            cols = np.arange(n_entries, dtype=np.int32)
            self._scatter_matrix = csr_matrix(
                (np.ones(n_entries), (flat, cols)), shape=(self.n_vertices, n_entries)
            )
        return self._scatter_matrix

    @property
    def laplacian(self) -> csr_matrix:
        """
        Cached combinatorial (uniform) Laplacian matrix for the mesh.

        The discrete Laplacian is a square matrix of shape (n_vertices,
        n_vertices). For each vertex i, the Laplacian encodes the
        difference between vertex i's position and the average position
        of its direct neighbours. Multiplying the Laplacian by the vertex
        matrix gives the Laplacian coordinates, which measure how far each
        vertex deviates from its neighbourhood average. This is used
        directly in the fairness energy and its gradient.

        The combinatorial (uniform-weight) Laplacian is used here: each
        neighbour contributes equally regardless of edge length or face
        area. Off-diagonal entries are -1 for connected vertex pairs and 0
        otherwise. Diagonal entries equal the number of neighbours (the
        vertex degree).

        Construction uses fully vectorised SciPy sparse operations and
        involves no Python loop over vertices or faces. The matrix is
        built once on first access and cached as a CSR sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Laplacian matrix of shape (n_vertices, n_vertices), dtype
            float64, stored in CSR format.

        References
        ----------
        Crane, K., de Goes, F., Desbrun, M., and Schroder, P. (2013).
        "Digital geometry processing with discrete exterior calculus."
        ACM SIGGRAPH 2013 Courses.
        """
        if self._laplacian_cpu is not None:
            return self._laplacian_cpu

        from scipy.sparse import csr_matrix as _csr

        n = self.n_vertices
        n_f, n_v = self.faces.shape

        # Generate all directed half-edges from each face
        # For a quad face [a, b, c, d]: edges (a,b), (b,c), (c,d), (d,a)
        src = self.faces.ravel()  # (n_f * n_v,)
        dst = np.roll(self.faces, -1, axis=1).ravel()  # (n_f * n_v,)

        # Combine both directions to form undirected edge pairs
        all_src = np.concatenate([src, dst])
        all_dst = np.concatenate([dst, src])

        # Deduplicate: keep unique (i, j) pairs with i != j
        mask = all_src != all_dst
        all_src = all_src[mask]
        all_dst = all_dst[mask]

        # Build sparse matrix: off-diagonal = -1, then fix diagonal
        data = np.full(len(all_src), -1.0, dtype=np.float64)
        L = _csr((data, (all_src, all_dst)), shape=(n, n), dtype=np.float64)

        # Eliminate duplicate edges (e.g. shared by two faces) by summing them,
        # then clamp off-diagonal to exactly -1 (uniform Laplacian).
        # Using sum_duplicates then normalise:
        L.sum_duplicates()
        L.data[:] = -1.0  # uniform weight regardless of duplicate count

        # Diagonal = degree (count of unique neighbours)
        # np.diff(L.indptr) gives number of stored entries per row.
        # After clamping all off-diagonal to -1, the diagonal = -row_sum.
        diag_vals = np.array(-L.sum(axis=1)).ravel()
        from scipy.sparse import diags as _diags

        D = _diags(diag_vals, 0, shape=(n, n), dtype=np.float64, format="csr")
        self._laplacian_cpu = (L + D).tocsr()
        return self._laplacian_cpu

    @property
    def vertex_face_ids_padded(self) -> np.ndarray:
        """
        Pre-computed vertex-to-face adjacency table for Numba acceleration.

        For each vertex, this table stores the indices of all faces that
        contain that vertex. The result is stored as a two-dimensional
        integer array of shape (n_vertices, max_valence), where max_valence
        is the largest number of faces incident to any single vertex in the
        mesh. Unused slots (for vertices with fewer than max_valence incident
        faces) are filled with -1 as a sentinel value.

        This table is required by the Numba gradient kernels because Numba
        compiled functions cannot use Python lists. The fixed-width padded
        format allows the kernels to iterate over all possible face slots
        with a simple inner loop and skip the -1 entry.

        The maximum stored valence is capped at 16 to prevent excessive
        memory use in the rare case of pathological high-valence vertices.
        Vertices with more than 16 incident faces have their extra faces
        truncated silently; because the angle balance computation only acts
        on vertices with exactly four incident faces, this truncation does
        not affect correctness for typical architectural quad meshes.

        Built once on first access and cached for the lifetime of the mesh.

        Returns
        -------
        numpy.ndarray
            Padded adjacency table of shape (n_vertices, max_valence),
            dtype int32. Unused entries are -1.
        """
        if self._vertex_face_ids_padded is not None:
            return self._vertex_face_ids_padded

        import warnings

        # Vectorised valence computation, no Python loop
        flat_faces = self.faces.ravel()
        valences = np.bincount(flat_faces, minlength=self.n_vertices)
        # Raw maximum valence from topology (maybe very large for pathological meshes)
        raw_max_valence = int(valences.max()) if self.n_vertices > 0 else 0

        """
        To avoid O(n_vertices * raw_max_valence) memory blowups from a single
        pathological vertex, cap the *stored* valence at a small fixed width.
        The Numba kernels only act when n_valid == 4, so any extra faces beyond
        this cap would be ignored anyway.
        """
        MAX_STORED_VALENCE = 16
        if self.n_vertices > 0:
            max_valence = max(4, min(raw_max_valence, MAX_STORED_VALENCE))
        else:
            max_valence = 4
        # Track vertices whose true valence exceeds the storage cap so callers
        # can optionally inspect/diagnose them.
        overflow_mask = valences > MAX_STORED_VALENCE
        self._vertex_face_overflow_mask = overflow_mask.astype(np.bool_)

        """
        Boundary vertices (low valence) are silently skipped, which is
        correct and expected for any finite mesh with a perimeter.
        Choose an expected interior valence based on face degree so that
        regular triangle meshes do not spuriously trigger "high-valence"
        warnings.
        """
        verts_per_face = int(self.faces.shape[1]) if self.n_faces > 0 else 4
        if verts_per_face == 4:
            expected_valence = 4
            mesh_type = "quad"
        elif verts_per_face == 3:
            # The regular interior valence for triangle meshes is typically ~6.
            expected_valence = 6
            mesh_type = "triangle"
        else:
            # Fallback: treat like a quad mesh to avoid surprising changes in
            # behaviour if other face degrees are introduced in the future.
            expected_valence = 4
            mesh_type = "polygonal"

        n_high_valence = int(np.sum(valences > expected_valence))
        n_overflow = int(np.sum(overflow_mask))
        if n_high_valence > 0:
            msg = (
                f"High-valence vertices detected (raw max valence = {raw_max_valence}, "
                f"stored max = {max_valence}, storage cap = {MAX_STORED_VALENCE}). "
                f"These are topologically non-standard for a {mesh_type} mesh and will "
                "be skipped in angle balance computation. Check mesh import for "
                "T-junctions or non-manifold geometry."
            )
            if n_overflow > 0:
                msg += (
                    f" {n_overflow} vertices exceed the storage cap and were truncated."
                )
            warnings.warn(msg, UserWarning, stacklevel=2)

        # Vectorised table construction using argsort-based grouping
        n_entries = flat_faces.size
        face_ids = np.repeat(
            np.arange(self.n_faces, dtype=np.int32), self.faces.shape[1]
        )
        sort_idx = np.argsort(flat_faces, kind="stable")
        sorted_verts = flat_faces[sort_idx]
        sorted_faces = face_ids[sort_idx]

        col_idx = np.zeros(n_entries, dtype=np.int32)
        boundaries = np.concatenate([[0], np.where(np.diff(sorted_verts))[0] + 1])

        # Compute the per-entry column index as position within each vertex's group
        # using a cumulative count that resets at group boundaries.
        group_ids = np.searchsorted(boundaries, np.arange(n_entries), side="right") - 1
        col_idx = np.arange(n_entries, dtype=np.int32) - boundaries[group_ids]

        table = np.full((self.n_vertices, max_valence), -1, dtype=np.int32)
        valid = col_idx < max_valence
        table[sorted_verts[valid], col_idx[valid]] = sorted_faces[valid]

        self._vertex_face_ids_padded = table
        return table

    @property
    def angle_balance_scratch(self) -> tuple:
        """
        Pre-allocated scratch buffers for the Numba angle balance gradient kernel.

        The Numba kernel '_angle_balance_gradient_numba' in 'gradients.py'
        requires several temporary arrays during its two-pass computation.
        Allocating these arrays inside a Numba compiled function at every
        iteration would cause repeated heap allocation overhead. Instead,
        they are allocated once here, cached on the mesh, and passed directly
        into the kernel as parameters.

        The buffers are automatically re-allocated if the number of vertices
        changes (which should not happen during normal optimisation).

        Returns
        -------
        tuple
            A five-element tuple containing the following pre-allocated
            NumPy arrays:

            scratch_gvp : numpy.ndarray, shape (n_vertices, 4, 3), float64
                Stores previous-vertex gradient contributions per vertex.
            scratch_gvn : numpy.ndarray, shape (n_vertices, 4, 3), float64
                Stores next-vertex gradient contributions per vertex.
            scratch_prev : numpy.ndarray, shape (n_vertices, 4), int32
                Stores previous-vertex indices for each incident face.
            scratch_next : numpy.ndarray, shape (n_vertices, 4), int32
                Stores next-vertex indices for each incident face.
            scratch_active : numpy.ndarray, shape (n_vertices,), int8
                Flags indicating which vertices are active 4-valent vertices.
        """
        n = self.n_vertices
        if self._numba_scratch is None or self._numba_scratch[0].shape[0] != n:
            self._numba_scratch = (
                np.zeros((n, 4, 3), dtype=np.float64),  # scratch_gvp
                np.zeros((n, 4, 3), dtype=np.float64),  # scratch_gvn
                np.full((n, 4), -1, dtype=np.int32),  # scratch_prev
                np.full((n, 4), -1, dtype=np.int32),  # scratch_next
                np.zeros(n, dtype=np.int8),  # scratch_active
            )
        return self._numba_scratch

    def reset_topology_cache(self) -> None:
        """
        Invalidate all cached topology-derived quantities.

        Forces all lazy-computed caches (Laplacian, scatter matrix,
        vertex-face adjacency, and Numba scratch buffers) to be rebuilt
        on their next access. This is only necessary if face connectivity
        is modified after construction.

        During normal optimisation, only vertex positions change. Because
        all caches depend solely on face connectivity, they remain valid
        throughout the entire optimisation and this method should not be
        called. It is provided for completeness and for workflows that
        involve mesh editing between optimisation runs.
        """
        self._vertex_faces = None
        self._vertex_neighbors = None
        self._scatter_matrix = None
        self._laplacian_cpu = None
        self._vertex_face_ids_padded = None
        self._vertex_face_overflow_mask = None
        self._laplacian_gpu = None
        self._scatter_matrix_gpu = None
        self._numba_scratch = None

    def get_vertex_faces(self, vertex_id: int) -> List[int]:
        """
        Return the indices of all faces incident to a given vertex.

        This is the one-ring face neighbourhood of the vertex: all faces
        that contain the vertex as one of their corners. The result is
        computed from the full face connectivity on the first call and cached
        for later queries.

        Parameters
        ----------
        vertex_id : int
            Index of the vertex to query. Must be in the range
            [0, n_vertices].

        Returns
        -------
        list of int
            List of face indices, each in the range [0, n_faces). The
            order matches the order in which the faces appear in
            'self.faces'.

        Raises
        ------
        IndexError
            If 'vertex_id' is negative or greater than or equal to
            'n_vertices'.
        """
        if not 0 <= vertex_id < self.n_vertices:
            raise IndexError(
                f"vertex_id {vertex_id} is out of range " f"[0, {self.n_vertices})"
            )
        if self._vertex_faces is None:
            self._compute_vertex_faces()
        assert self._vertex_faces is not None
        return self._vertex_faces[vertex_id]

    def _compute_vertex_faces(self):
        """
        Build the vertex-to-face adjacency list from the face connectivity.

        Iterates over all faces and appends each face index to the adjacency
        list of every vertex it contains. The result is stored in
        'self._vertex_faces' as a list of lists, one inner list per vertex.

        This method is called automatically by 'get_vertex_faces' on first
        access and should not be called directly by external code.
        """
        self._vertex_faces = [[] for _ in range(self.n_vertices)]
        assert self._vertex_faces is not None
        for face_id, face in enumerate(self.faces):
            for vertex_id in face:
                self._vertex_faces[vertex_id].append(face_id)

    def get_face_vertices(self, face_id: int) -> np.ndarray:
        """
        Return the 3D positions of all vertices forming a given face.

        Gathers the vertex coordinates for the requested face by indexing
        'self.vertices' with the face's vertex indices. For a quad face,
        returns a (4, 3) array; for a triangular face, a (3, 3) array.

        Parameters
        ----------
        face_id : int
            Index of the face to query. Must be in the range [0, n_faces).

        Returns
        -------
        numpy.ndarray
            Vertex positions array of shape (verts_per_face, 3), dtype
            float64. Each row is the 3D coordinate of one corner vertex.

        Raises
        ------
        IndexError
            If 'face_id' is negative or greater than or equal to 'n_faces'.
        """
        if not 0 <= face_id < self.n_faces:
            raise IndexError(f"face_id {face_id} is out of range [0, {self.n_faces})")
        return self.vertices[self.faces[face_id]]

    def update_vertices(self, new_positions: np.ndarray):
        """
        Replace the current vertex positions with a new set of positions.

        Validates that the new array has the correct shape and contains
        only finite values, then copies it into 'self.vertices'. This
        operation does not invalidate topology caches because only vertex
        positions change, not face connectivity.

        This method provides an explicit, validated interface for updating
        vertex positions. The optimiser also updates 'self.vertices'
        directly by assignment for performance reasons, but external code
        should use this method to benefit from the validation checks.

        Parameters
        ----------
        new_positions : numpy.ndarray
            New vertex positions of shape (n_vertices, 3). Must contain
            only finite values (no NaN or infinity).

        Raises
        ------
        ValueError
            If 'new_positions' does not match the shape of the current
            vertex array, or if it contains any NaN or infinite values.
        """
        if new_positions.shape != self.vertices.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.vertices.shape},"
                f" got {new_positions.shape}"
            )
        if not np.isfinite(new_positions).all():
            raise ValueError(
                "new_positions contains NaN or Inf values — "
                "check for degenerate geometry or numerical overflow"
            )
        self.vertices = new_positions.copy()

    def reset_to_original(self):
        """
        Reset vertex positions to the state recorded at construction time.

        Restores 'self.vertices' from 'self.vertices_original', which holds
        a copy of the vertex positions provided when the mesh was first
        created. This is useful for re-running optimisation from scratch
        or for comparing the optimised result against the original geometry.

        Topology caches are not invalidated because face connectivity has
        not changed.
        """
        self.vertices = self.vertices_original.copy()
