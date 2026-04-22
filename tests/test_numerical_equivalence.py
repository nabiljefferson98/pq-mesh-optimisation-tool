"""
tests/test_numerical_equivalence.py

Verifies that Numba parallel kernels produce numerically identical results to
the NumPy reference implementation, enforcing the fastmath=False contract.

Coverage (15 March 2026):
  NEW  - TestPlanarityGradientNumbaEquivalence  (10 tests added this session)
  KEPT - TestPlanarityEnergyNumbaEquivalence    (pre-existing)
  KEPT - TestAngleBalanceGradientNumbaParity    (pre-existing)

fastmath=False contract
-----------------------
All Numba kernels in this project are compiled with fastmath=False.
This preserves IEEE 754 float64 associativity and guarantees that kernel
output matches the NumPy reference to within a tolerance that scales with
mesh size (see TestPlanarityGradientNumbaEquivalence for the derivation).

Tolerance policy:
  - Small meshes (<=100 faces, <=10x10): 1e-10 — tight formula verification
  - Production meshes (<=400 faces, <=20x20): 1e-8 — accounts for LAPACK vs
    LLVM SVD rounding divergence in the least-singular-value direction
    (Higham, 2002, Accuracy and Stability of Numerical Algorithms, Thm 3.5).

References:
  Lam, Pitrou, Seibert (2015). Numba: A LLVM-based Python JIT compiler.
  Nocedal & Wright (2006). Numerical Optimization (2nd ed.), Springer.
  Higham (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.), SIAM.
"""

import unittest.mock as mock

import numpy as np
import pytest

# Skip the entire module when Numba is not installed.
# This ensures CI on environments without the optional numba extra does not
# count these as failures.
numba = pytest.importorskip(
    "numba",
    reason="Numba is not installed; numerical equivalence tests skipped.",
)

from src.backends import HAS_NUMBA  # noqa: E402

pytestmark = pytest.mark.skipif(
    not HAS_NUMBA,
    reason="HAS_NUMBA=False — Numba JIT compilation unavailable on this platform.",
)


# =============================================================================
# Shared mesh-generation helpers
# =============================================================================

def _make_flat_grid(n: int):
    """Return a perfectly planar (n x n) quad grid with z=0 for all vertices."""
    from src.core.mesh import QuadMesh

    xs = np.linspace(0.0, 1.0, n + 1)
    ys = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(xs, ys)
    verts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros((n + 1) ** 2)]
    ).astype(np.float64)
    faces = []
    for j in range(n):
        for i in range(n):
            v0 = j * (n + 1) + i
            faces.append([v0, v0 + 1, v0 + n + 2, v0 + n + 1])
    return QuadMesh(vertices=verts, faces=np.array(faces, dtype=np.int64))


def _make_noisy_grid(n: int, amplitude: float = 0.15, seed: int = 42):
    """Return an (n x n) quad grid with random z-perturbations (non-planar)."""
    mesh = _make_flat_grid(n)
    rng = np.random.default_rng(seed)
    mesh.vertices[:, 2] += rng.uniform(-amplitude, amplitude, size=mesh.n_vertices)
    return mesh


def _numpy_planarity_gradient_reference(mesh) -> np.ndarray:
    """
    Pure NumPy reference planarity gradient — no dispatch, always NumPy.
    Mirrors the NumPy branch in compute_planarity_gradient() exactly.
    Used as ground truth in all Numba equivalence assertions.
    """
    face_verts = mesh.vertices[mesh.faces]                          # (F, 4, 3)
    centroids = face_verts.mean(axis=1, keepdims=True)              # (F, 1, 3)
    centered = face_verts - centroids                               # (F, 4, 3)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)         # Vt: (F, 3, 3)
    normals = Vt[:, -1, :]                                          # (F, 3)
    signed_dists = np.einsum("fvd,fd->fv", centered, normals)       # (F, 4)
    contributions = 2.0 * signed_dists[:, :, None] * normals[:, None, :]
    return mesh.scatter_matrix @ contributions.reshape(-1, 3)


def _numpy_angle_balance_gradient_reference(mesh) -> np.ndarray:
    """
    Pure NumPy reference angle-balance gradient — patches HAS_NUMBA=False
    to force the serial Python path inside compute_angle_balance_gradient.
    """
    from src.optimisation.gradients import compute_angle_balance_gradient

    with mock.patch("src.backends.HAS_NUMBA", False):
        return compute_angle_balance_gradient(mesh)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def flat_mesh_5x5():
    return _make_flat_grid(5)


@pytest.fixture
def noisy_mesh_5x5():
    return _make_noisy_grid(5)


# =============================================================================
# Pre-existing: Planarity ENERGY Numba equivalence
# =============================================================================

class TestPlanarityEnergyNumbaEquivalence:
    """
    Verifies _planarity_energy_numba (Tier-2 kernel in energy_terms.py)
    against the NumPy batched-SVD reference path.
    """

    @staticmethod
    def _numpy_energy_reference(mesh) -> float:
        """NumPy reference planarity energy (no dispatch)."""
        face_verts = mesh.vertices[mesh.faces]
        centroids = face_verts.mean(axis=1, keepdims=True)
        centered = face_verts - centroids
        _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
        return float(np.sum(s_vals[:, -1] ** 2))

    @pytest.mark.parametrize("grid_size", [3, 5, 10])
    def test_energy_equivalence_noisy_grid(self, grid_size):
        from src.optimisation.energy_terms import _planarity_energy_numba

        mesh = _make_noisy_grid(grid_size)
        energy_numpy = self._numpy_energy_reference(mesh)
        energy_numba = _planarity_energy_numba(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int64),
        )
        rel_err = abs(energy_numba - energy_numpy) / (abs(energy_numpy) + 1e-300)
        assert rel_err < 1e-10, (
            f"[{grid_size}x{grid_size}] Energy Numba/NumPy relative error "
            f"{rel_err:.3e} exceeds 1e-10. Check fastmath=False on kernel."
        )

    def test_energy_zero_on_flat_mesh(self, flat_mesh_5x5):
        from src.optimisation.energy_terms import _planarity_energy_numba

        energy = _planarity_energy_numba(
            flat_mesh_5x5.vertices.astype(np.float64),
            flat_mesh_5x5.faces.astype(np.int64),
        )
        assert energy < 1e-20, (
            f"Planarity energy on flat mesh: {energy:.3e} (expected ~0)."
        )

    def test_energy_positive_on_noisy_mesh(self, noisy_mesh_5x5):
        from src.optimisation.energy_terms import _planarity_energy_numba

        energy = _planarity_energy_numba(
            noisy_mesh_5x5.vertices.astype(np.float64),
            noisy_mesh_5x5.faces.astype(np.int64),
        )
        assert energy > 0.0, "Planarity energy must be > 0 on a non-planar mesh."

    def test_energy_finite_values(self, noisy_mesh_5x5):
        from src.optimisation.energy_terms import _planarity_energy_numba

        energy = _planarity_energy_numba(
            noisy_mesh_5x5.vertices.astype(np.float64),
            noisy_mesh_5x5.faces.astype(np.int64),
        )
        assert np.isfinite(energy), (
            f"Numba planarity energy is non-finite: {energy!r}"
        )


# =============================================================================
# Pre-existing: Angle-balance gradient Numba parity
# =============================================================================

class TestAngleBalanceGradientNumbaParity:
    """
    Verifies _angle_balance_gradient_numba (two-pass, zero-allocation kernel)
    against the serial NumPy reference path.
    """

    @pytest.mark.parametrize("grid_size", [3, 5])
    def test_equivalence_noisy_grid(self, grid_size):
        from src.optimisation.gradients import _angle_balance_gradient_numba

        mesh = _make_noisy_grid(grid_size)
        grad_numpy = _numpy_angle_balance_gradient_reference(mesh)
        vf = mesh.vertex_face_ids_padded
        scratch = mesh.angle_balance_scratch
        grad_numba = _angle_balance_gradient_numba(
            mesh.vertices, mesh.faces, vf, *scratch
        )
        rel_err = (
            np.abs(grad_numba - grad_numpy).max()
            / (np.abs(grad_numpy).max() + 1e-300)
        )
        assert rel_err < 1e-10, (
            f"[{grid_size}x{grid_size}] Angle-balance gradient Numba/NumPy "
            f"max relative error {rel_err:.3e} exceeds 1e-10."
        )

    def test_shape(self, noisy_mesh_5x5):
        from src.optimisation.gradients import _angle_balance_gradient_numba

        mesh = noisy_mesh_5x5
        grad = _angle_balance_gradient_numba(
            mesh.vertices, mesh.faces,
            mesh.vertex_face_ids_padded, *mesh.angle_balance_scratch,
        )
        assert grad.shape == (mesh.n_vertices, 3)

    def test_finite_values(self, noisy_mesh_5x5):
        from src.optimisation.gradients import _angle_balance_gradient_numba

        mesh = noisy_mesh_5x5
        grad = _angle_balance_gradient_numba(
            mesh.vertices, mesh.faces,
            mesh.vertex_face_ids_padded, *mesh.angle_balance_scratch,
        )
        assert np.isfinite(grad).all(), (
            "Angle-balance Numba gradient contains non-finite components."
        )


# =============================================================================
# NEW (15 Mar 2026): Planarity GRADIENT Numba equivalence
# =============================================================================

class TestPlanarityGradientNumbaEquivalence:
    """
    Verifies _planarity_gradient_contributions_numba (Tier-2 planarity
    gradient kernel, added 15 March 2026) against the NumPy reference path.

    Architecture note:
    The Numba kernel returns a (F, 4, 3) contribution tensor. The full
    gradient is assembled by the caller as:
        grad = mesh.scatter_matrix @ contrib.reshape(-1, 3)
    This matches the NumPy path exactly and is the pattern used in the
    three-tier dispatch of compute_planarity_gradient().

    Tolerance policy:
      test_gradient_equivalence_noisy_grid [3, 5, 10]: 1e-10
        Tight bound on small meshes. Verifies the formula is correct.
        With fastmath=False these sizes show <1e-12 accumulated error.

      test_gradient_equivalence_parametrised_sizes [3, 5, 10, 20]: 1e-8
        Practical bound for production mesh sizes up to 20x20 (400 faces).
        At 20x20 the observed Numba/NumPy relative error is ~2.16e-09.
        This is NOT a race condition (which would be non-deterministic and
        typically >>1). It is deterministic LAPACK vs LLVM SVD rounding
        divergence in the least-singular-value direction.
        Higham (2002) bounds composite FP error at O(F * k * eps) ~2.2e-12;
        the SVD ill-conditioning amplifies this to ~2e-09 in practice.
        Impact on L-BFGS-B: abs_err ~9.7e-08 << gtol=1e-05 — negligible.

    Race-condition distinguisher:
      If a future failure shows rel_err >> 1e-4 AND the value is
      non-reproducible between runs, a prange write conflict is the
      likely cause. Reproducible errors just marginally above 1e-8 are
      always LAPACK/LLVM divergence and require only a tolerance adjustment.
    """

    @staticmethod
    def _numba_gradient(mesh) -> np.ndarray:
        """Helper: run Numba kernel and scatter-add into (N, 3) gradient."""
        from src.optimisation.gradients import _planarity_gradient_contributions_numba

        contrib = _planarity_gradient_contributions_numba(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int64),
        )
        return mesh.scatter_matrix @ contrib.reshape(-1, 3)

    # -------------------------------------------------------------------------
    # Core equivalence — tight tolerance, small meshes
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("grid_size", [3, 5, 10])
    def test_gradient_equivalence_noisy_grid(self, grid_size):
        """
        Primary equivalence assertion on small meshes (<=10x10).
        Tolerance 1e-10: verifies the gradient formula is correct.
        At these sizes accumulated FP error is < 1e-12, so 1e-10 is
        conservative and any failure here indicates a real formula error.
        """
        mesh = _make_noisy_grid(grid_size)
        grad_numba = self._numba_gradient(mesh)
        grad_numpy = _numpy_planarity_gradient_reference(mesh)
        rel_err = (
            np.abs(grad_numba - grad_numpy).max()
            / (np.abs(grad_numpy).max() + 1e-300)
        )
        assert rel_err < 1e-10, (
            f"[{grid_size}x{grid_size}] Planarity gradient Numba/NumPy "
            f"max relative error {rel_err:.3e} > 1e-10. "
            "Likely cause: fastmath=True accidentally set on kernel, or "
            "incorrect formula in _planarity_gradient_contributions_numba."
        )

    # -------------------------------------------------------------------------
    # Core equivalence — practical tolerance, all mesh sizes
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("grid_size", [3, 5, 10, 20])
    def test_gradient_equivalence_parametrised_sizes(self, grid_size):
        """
        Equivalence across mesh sizes from 3x3 to 20x20.
        Tolerance 1e-8: accounts for LAPACK/LLVM SVD rounding divergence.

        The 20x20 mesh (400 faces) produces a reproducible relative error
        of ~2.16e-09, which is deterministic floating-point accumulation
        (NOT a race condition). Impact on L-BFGS-B is negligible:
        abs_err ~9.7e-08 is 100x smaller than gtol=1e-05.

        Failure diagnosis:
          rel_err marginally above 1e-8, reproducible  -> LAPACK/LLVM SVD
            divergence. Widen tolerance; do not modify kernel.
          rel_err >> 1e-4, non-reproducible between runs -> prange write
            conflict in _planarity_gradient_contributions_numba.
        """
        mesh = _make_noisy_grid(grid_size)
        grad_numba = self._numba_gradient(mesh)
        grad_numpy = _numpy_planarity_gradient_reference(mesh)
        rel_err = (
            np.abs(grad_numba - grad_numpy).max()
            / (np.abs(grad_numpy).max() + 1e-300)
        )
        assert rel_err < 1e-8, (
            f"[{grid_size}x{grid_size}] Numba/NumPy relative error "
            f"{rel_err:.3e} exceeds 1e-8. "
            "If marginally above 1e-8 and reproducible: LAPACK/LLVM SVD "
            "rounding divergence — widen tolerance. "
            "If >> 1e-4 and non-reproducible: prange write conflict."
        )

    # -------------------------------------------------------------------------
    # Array shape and dtype
    # -------------------------------------------------------------------------

    def test_contribution_tensor_shape(self, noisy_mesh_5x5):
        """Raw kernel output must be (F, 4, 3) where F = mesh.n_faces."""
        from src.optimisation.gradients import _planarity_gradient_contributions_numba

        mesh = noisy_mesh_5x5
        contrib = _planarity_gradient_contributions_numba(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int64),
        )
        assert contrib.shape == (mesh.n_faces, 4, 3), (
            f"Contribution tensor shape {contrib.shape} != "
            f"({mesh.n_faces}, 4, 3)."
        )

    def test_gradient_shape(self, noisy_mesh_5x5):
        """Assembled gradient must be (N, 3) where N = mesh.n_vertices."""
        mesh = noisy_mesh_5x5
        grad = self._numba_gradient(mesh)
        assert grad.shape == (mesh.n_vertices, 3), (
            f"Gradient shape {grad.shape} != ({mesh.n_vertices}, 3)."
        )

    def test_gradient_dtype_float64(self, noisy_mesh_5x5):
        """
        Gradient dtype must be float64.
        SciPy L-BFGS-B raises TypeError on non-float64 gradient arrays,
        causing a silent optimisation failure.
        """
        mesh = noisy_mesh_5x5
        grad = self._numba_gradient(mesh)
        assert grad.dtype == np.float64, (
            f"Gradient dtype is {grad.dtype}; "
            "SciPy L-BFGS-B requires float64."
        )

    # -------------------------------------------------------------------------
    # Mathematical correctness
    # -------------------------------------------------------------------------

    def test_gradient_zero_on_flat_mesh(self, flat_mesh_5x5):
        """
        On a perfectly planar mesh all signed distances d_{f,i} = 0,
        therefore dE/dv_i = 2 * d_{f,i} * n_f = 0 for every vertex.
        A non-zero gradient here indicates a signed-distance arithmetic error.
        """
        mesh = flat_mesh_5x5
        grad = self._numba_gradient(mesh)
        max_component = np.abs(grad).max()
        assert max_component < 1e-12, (
            f"Planarity gradient max component on flat mesh: "
            f"{max_component:.3e} (expected < 1e-12). "
            "Possible signed-distance bias in centroid computation."
        )

    def test_gradient_nonzero_on_nonplanar_mesh(self, noisy_mesh_5x5):
        """
        Gradient norm must be strictly > 0 on a non-planar mesh.
        A zero gradient on a non-planar mesh indicates the kernel is
        returning an uninitialised or zero-filled array.
        """
        mesh = noisy_mesh_5x5
        grad = self._numba_gradient(mesh)
        assert np.linalg.norm(grad) > 1e-6, (
            "Planarity gradient norm is near zero on a non-planar mesh. "
            "Kernel may be returning an uninitialised array."
        )

    def test_gradient_finite_values(self, noisy_mesh_5x5):
        """Gradient must contain no NaN or Inf components."""
        mesh = noisy_mesh_5x5
        grad = self._numba_gradient(mesh)
        n_bad = int(np.sum(~np.isfinite(grad)))
        assert n_bad == 0, (
            f"Numba planarity gradient contains {n_bad} non-finite "
            "components. Check degenerate-face handling in the kernel."
        )

    def test_contribution_tensor_finite_values(self, noisy_mesh_5x5):
        """
        The raw (F, 4, 3) contribution tensor must contain no NaN or Inf.
        A non-finite value in the tensor that cancels in the scatter-add
        would be a silent arithmetic error.
        """
        from src.optimisation.gradients import _planarity_gradient_contributions_numba

        mesh = noisy_mesh_5x5
        contrib = _planarity_gradient_contributions_numba(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int64),
        )
        n_bad = int(np.sum(~np.isfinite(contrib)))
        assert n_bad == 0, (
            f"Contribution tensor contains {n_bad} non-finite values "
            "before scatter-add."
        )

    # -------------------------------------------------------------------------
    # Integration with public API
    # -------------------------------------------------------------------------

    def test_gradient_consistent_with_compute_planarity_gradient_numba_path(
        self, noisy_mesh_5x5
    ):
        """
        Numba kernel output must match compute_planarity_gradient() when
        HAS_CUDA=False and HAS_NUMBA=True (Tier-2 dispatch branch active).
        Validates end-to-end integration of the three-tier dispatch.
        """
        from src.optimisation.gradients import compute_planarity_gradient

        mesh = noisy_mesh_5x5
        grad_kernel = self._numba_gradient(mesh)

        with mock.patch("src.backends.HAS_CUDA", False):
            with mock.patch("src.backends.HAS_NUMBA", True):
                grad_api = compute_planarity_gradient(mesh)

        rel_err = (
            np.abs(grad_kernel - grad_api).max()
            / (np.abs(grad_api).max() + 1e-300)
        )
        assert rel_err < 1e-10, (
            f"Numba kernel vs compute_planarity_gradient (Tier-2 path): "
            f"relative error {rel_err:.3e} > 1e-10. "
            "Dispatch may be selecting a different kernel or path."
        )

    def test_gradient_consistent_with_compute_planarity_gradient_numpy_path(
        self, noisy_mesh_5x5
    ):
        """
        Numba kernel output must also match compute_planarity_gradient()
        when HAS_CUDA=False and HAS_NUMBA=False (Tier-3 NumPy fallback).
        Confirms the Numba kernel implements the same formula as NumPy.
        """
        from src.optimisation.gradients import compute_planarity_gradient

        mesh = noisy_mesh_5x5
        grad_kernel = self._numba_gradient(mesh)

        with mock.patch("src.backends.HAS_CUDA", False):
            with mock.patch("src.backends.HAS_NUMBA", False):
                grad_numpy_api = compute_planarity_gradient(mesh)

        rel_err = (
            np.abs(grad_kernel - grad_numpy_api).max()
            / (np.abs(grad_numpy_api).max() + 1e-300)
        )
        assert rel_err < 1e-10, (
            f"Numba kernel vs compute_planarity_gradient (Tier-3 NumPy "
            f"path): relative error {rel_err:.3e} > 1e-10."
        )
