"""
src/optimisation/optimiser.py

Main optimisation pipeline for planar quad mesh refinement.

This module ties together the energy functions, gradient functions, and
the SciPy L-BFGS-B solver into a single, self-contained pipeline. It is
the primary entry point for users who want to optimise a quad mesh towards
a flatter, smoother, and more developable geometry.

How the optimiser works
-----------------------
The optimiser treats vertex positions as unknowns and iteratively moves
them to reduce the total weighted energy. At each step, L-BFGS-B uses
the analytical gradient to decide which direction to move the vertices
and by how much. The process repeats until the energy stops improving
meaningfully or the iteration limit is reached.

Two-stage strategy
------------------
When 'two_stage=True' (the default), the optimiser runs in two sequential
passes over the same mesh:

  Stage 1 -- Rapid Planarity Pass:
    The planarity weight is temporarily boosted (by default, multiplied by
    five) while fairness and closeness weights are reduced to ten percent
    of their standard values. Looser convergence tolerances are used, so
    this stage terminates quickly. The goal is to drive all quad faces to
    approximately flat before the balanced pass begins, giving Stage 2 a
    much better starting point.

  Stage 2 -- Balanced Refinement:
    The original weights and tighter tolerances are restored. The solver
    warm-starts from the Stage 1 solution, which is already near-planar,
    so it converges faster than it would from the original mesh.

The combined iteration count from both stages is strictly bounded by
'max_iterations', so the two-stage approach does not cost extra iterations
compared to a single-stage run.

Key classes
-----------
  OptimisationConfig -- holds all settings for a single optimisation run.
  OptimisationResult -- holds all outputs, statistics, and history from
                        a completed optimisation run.
  MeshOptimiser -- the main class; call its 'optimise(mesh)' method.

Quick start
-----------
For most users, the convenience function 'optimise_mesh_simple' is the
simplest way to run an optimisation with sensible defaults:

    from src.optimisation.optimiser import optimise_mesh_simple
    result = optimise_mesh_simple(mesh, max_iter=500)
    print(result.summary())

References
----------
Nocedal, J. and Wright, S. J. (2006).
  Numerical Optimization. 2nd ed. Springer.

Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J. (1997).
  "Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale
  bound-constrained optimisation." ACM Transactions on Mathematical
  Software, 23(4), pp. 550-560.

Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from src.core.mesh import QuadMesh
from src.optimisation.energy_terms import (
    compute_closeness_energy,
    compute_fairness_energy,
    compute_planarity_energy,
    compute_total_energy,
)
from src.optimisation.gradients import (
    compute_total_gradient,
    energy_for_scipy,
    gradient_for_scipy,
)

try:
    from src.backends import print_backend_info as _print_backend_info

except ImportError:

    def _print_backend_info() -> None:
        """Fallback no-op when the optional backends module is unavailable."""
        return None


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

"""
Threshold below which an initial energy component is treated as zero for
the purposes of percentage-reduction reporting in OptimisationResult.summary().
This prevents division by values at floating-point noise level, which would
produce meaningless percentage improvements in the printed summary.
"""
_NEAR_ZERO_ENERGY: float = 1e-15


def _compute_angle_balance_component(mesh: QuadMesh) -> float:
    """
    Compute the raw angle balance energy for use in result reporting.

    Wraps 'compute_angle_balance_energy' with a broad exception handler so
    that the result summary degrades gracefully when the angle balance term
    is disabled (weight = 0.0) or unavailable. Returns 0.0 on any failure.

    This function is an internal helper and is never called by external code.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to evaluate.

    Returns
    -------
    float
        The raw angle balance energy, or 0.0 if computation fails.
    """
    try:
        from src.optimisation.energy_terms import compute_angle_balance_energy

        return float(compute_angle_balance_energy(mesh))
    except Exception:
        return 0.0


@dataclass
class OptimisationConfig:
    """
    Configuration settings for a single mesh optimisation run.

    All fields have sensible defaults suitable for most architectural quad
    meshes. Adjust them progressively rather than all at once: start with
    the default weights, observe the component energy breakdown in the
    result summary, then increase the weight of whichever term is not
    converging sufficiently.

    Attributes
    ----------
    weights : dict
        Energy term weights controlling the relative importance of each
        geometric goal. Required keys: 'planarity', 'fairness',
        'closeness'. Optional key: 'angle_balance' (defaults to 0.0 if
        absent, meaning the conical constraint is disabled by default).
        Typical starting values: planarity=100.0, fairness=1.0,
        closeness=10.0.
    max_iterations : int
        Maximum total number of L-BFGS-B iterations across both stages
        combined. The optimiser will stop earlier if it converges.
        Default is 1000.
    tolerance : float
        a Convergence threshold for the change in energy value between
        successive iterations. The optimiser stops when the change falls
        below this value. The default is 1e-6.
    gradient_tolerance : float
        Convergence threshold for the infinity norm of the gradient. The
        optimiser stops when the largest gradient component falls below
        this value. Default is 1e-4.
    verbose : bool
        If True, print a formatted progress table to the console during
        optimisation, including per-stage summaries and a final result
        breakdown. Default is True.
    history_tracking : bool
        If True, record the energy and gradient norm at every iteration.
        The recorded lists are stored in 'OptimisationResult.energy_history'
        and 'OptimisationResult.gradient_norm_history'. Useful for
        producing convergence plots. Default is True.
    bounds_scale : float or None
        If set, restricts each vertex coordinate to move no further than
        'bounds_scale' times its initial absolute value. For example,
        a value of 0.1 prevents any vertex from moving more than 10% of
        its initial distance from the origin. None means no bounds are
        applied. Default is None.
    two_stage : bool
        If True, run the two-stage strategy described in the module
        docstring. Stage 1 boosts planarity and uses loose tolerances;
        Stage 2 restores balanced weights and tightens tolerances.
        Recommended for most meshes. Default is True.
    stage1_planarity_multiplier : float
        Factor by which the planarity weight is multiplied during Stage 1.
        Higher values flatten faces more aggressively in Stage 1 but may
        temporarily worsen shape fidelity. Recommended range: 3.0 to 10.0.
        Default is 5.0.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {"planarity": 100.0, "fairness": 1.0, "closeness": 10.0}
    )
    max_iterations: int = 1000
    tolerance: float = 1e-6  # Relaxed convergence tolerance for Stage 1
    gradient_tolerance: float = 1e-4  # Relaxed gradient tolerance for Stage 1
    verbose: bool = True  # Whether to print progress during optimisation
    history_tracking: bool = (
        True  # Whether to track energy and gradient history for analysis
    )
    bounds_scale: Optional[float] = None

    """
    ── Two-stage strategy ──
    When True, a rapid planarity pass (Stage 1) precedes the balanced
    refinement (Stage 2).  Stage 1 uses a boosted planarity weight and
    looser tolerances to drive faces flat quickly; Stage 2 restores
    standard weights and tighter tolerances for final convergence.
    The combined iteration budget across both stages is strictly bounded
    by max_iterations.
    """
    two_stage: bool = True

    """
    Multiplier applied to the planarity weight in Stage 1.
    Recommended range: 3.0–10.0. Higher values flatten faster but
    risk over-correcting shape fidelity.
    """
    stage1_planarity_multiplier: float = 5.0

    def validate(self) -> None:
        """
        Validate all configuration parameters before optimisation begins.

        Checks that all numeric parameters are within meaningful ranges and
        that all required weight keys are present. Raises 'ValueError' with
        a descriptive message for the first invalid setting found.

        Raises
        ------
        ValueError
            If any parameter is invalid (for example, a negative weight,
            a non-positive tolerance, or a missing required weight key).
        """
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        if self.gradient_tolerance <= 0:
            raise ValueError("gradient_tolerance must be > 0")
        if self.stage1_planarity_multiplier <= 0:
            raise ValueError("stage1_planarity_multiplier must be > 0")
        for key in ["planarity", "fairness", "closeness"]:
            if key not in self.weights:
                raise ValueError(f"Missing weight for {key}")
            if self.weights[key] < 0:
                raise ValueError(f"Weight for {key} must be non-negative")


@dataclass
class OptimisationResult:
    """
    Complete record of a finished mesh optimisation run.

    Returned by 'MeshOptimiser.optimise()' and 'optimise_mesh_simple()'.
    Contains the optimised mesh, all summary statistics, per-component
    energy breakdowns, and optionally the full iteration history.

    Attributes
    ----------
    success : bool
        True if SciPy reported successful convergence. False if the
        iteration limit was reached or an error occurred. Note that
        'success=False' does not necessarily mean the result is unusable:
        a mesh that improved by 95% but hit the iteration limit is still
        a valid result.
    message : str
        The status message returned by 'scipy.optimize.minimize'. Useful
        for diagnosing why the optimiser stopped.
    optimised_mesh : QuadMesh
        The mesh with vertex positions updated to the optimised values.
        This is the same object passed in to 'optimise()', modified
        in-place, and also stored here for convenience.
    initial_energy : float
        Total weighted energy before optimisation.
    final_energy : float
        Total weighted energy after optimisation.
    n_iterations : int
        Total number of L-BFGS-B iterations across both stages.
    n_function_evaluations : int
        Total number of times the energy function was evaluated.
    n_gradient_evaluations : int
        Total number of times the gradient function was evaluated.
    execution_time : float
        Total wall-clock time in seconds for the entire optimisation,
        including both stages.
    energy_history : list of float or None
        Energy value recorded at every iteration if 'history_tracking'
        was True. None otherwise. Use this to produce a convergence plot.
    gradient_norm_history : list of float or None
        Gradient norm recorded at every iteration if 'history_tracking'
        was True. None otherwise.
    component_energies_initial : dict or None
        Raw (unweighted) energy for each term before optimisation, with
        keys 'planarity', 'fairness', 'closeness', 'angle_balance'.
    component_energies_final : dict or None
        Raw (unweighted) energy for each term after optimisation.
    """

    success: bool
    message: str
    optimised_mesh: QuadMesh
    initial_energy: float
    final_energy: float
    n_iterations: int
    n_function_evaluations: int
    n_gradient_evaluations: int
    execution_time: float
    energy_history: Optional[List[float]] = None
    gradient_norm_history: Optional[List[float]] = None
    component_energies_initial: Optional[Dict[str, float]] = None
    component_energies_final: Optional[Dict[str, float]] = None

    def energy_reduction(self) -> float:
        """
        Compute the relative reduction in total energy.

        Returns the fraction of the initial energy that was removed by
        optimisation. A value of 1.0 means the energy was eliminated
        entirely; a value of 0.5 means it was halved.

        Returns
        -------
        float
            Relative energy reduction in the range [0.0, 1.0]. Returns
            0.0 if the initial energy was zero.
        """
        if self.initial_energy == 0:
            return 0.0
        return (self.initial_energy - self.final_energy) / self.initial_energy

    def energy_reduction_percentage(self) -> float:
        """
        Compute the relative energy reduction as a percentage.

        Equivalent to 'energy_reduction() * 100'. A value of 80.0 means
        the total energy was reduced by 80%.

        Returns
        -------
        float
            Energy reduction percentage. Returns 0.0 if initial energy
            was zero.
        """
        return self.energy_reduction() * 100.0

    def summary(self) -> str:
        """
        Generate a human-readable summary of the optimisation result.

        Produces a formatted multi-line string covering: the convergence
        status (with a plain-English explanation), overall energy before
        and after, iteration and timing statistics, and a per-term energy
        breakdown showing how much each geometric goal improved. Trade-offs
        (where one term worsened slightly so others could improve) are
        identified and explained as normal behaviour.

        Returns
        -------
        str
            Formatted summary string. Call 'print(result.summary())' to
            display it or include it in a log file.
        """
        lines = [
            "=" * 70,
            "OPTIMISATION COMPLETE — RESULTS SUMMARY",
            "=" * 70,
        ]

        # ----------------------------------------------------------------
        # Status: distinguish convergence, iteration limit, and true failure
        # ----------------------------------------------------------------
        reduction_pct = self.energy_reduction_percentage()
        msg = self.message

        if self.success:
            status_str = (
                "FINISHED SUCCESSFULLY — the optimiser found the best solution it could"
            )
        elif "ITERATIONS REACHED LIMIT" in msg or "TOTAL NO. OF" in msg:
            status_str = (
                f"STEP LIMIT REACHED  "
                f"(score improved by {reduction_pct:.1f}% — "
                f"increase the maximum steps to continue improving)"
            )
        elif "ABNORMAL" in msg or "ERROR" in msg:
            status_str = (
                "FAILED — something went wrong (check that the mesh file is valid)"
            )
        elif reduction_pct > 95.0:
            status_str = (
                f"NEARLY THERE  "
                f"(score improved by {reduction_pct:.1f}% — good enough for most uses)"
            )
        else:
            status_str = (
                f"PARTIAL IMPROVEMENT  (score improved by {reduction_pct:.1f}%)"
            )

        lines.extend(
            [
                f"Result: {status_str}",
                f"(Technical message from solver: {msg})",
                "",
                "Overall score (lower = flatter, smoother, closer to original):",
                f"  Score at the start:            {self.initial_energy:.4f}",
                f"  Score at the end:              {self.final_energy:.4f}",
                f"  Total improvement:             {reduction_pct:.2f}%",
                "",
                "How hard did the optimiser work?",
                f"  Improvement steps taken:       {self.n_iterations}",
                f"  Times it checked the score:    {self.n_function_evaluations}",
                f"  Times it checked the direction:{self.n_gradient_evaluations}",
                f"  Total time:                    {self.execution_time:.2f} seconds",
                "",
            ]
        )

        if self.component_energies_initial and self.component_energies_final:
            lines.append("Per-goal score breakdown (how each individual goal changed):")
            _labels = {
                "planarity": "Panel flatness    ",
                "fairness": "Surface smoothness",
                "closeness": "Shape fidelity    ",
                "angle_balance": "Corner balance    ",
            }

            for key in self.component_energies_initial:
                initial = self.component_energies_initial[key]
                final = self.component_energies_final[key]
                label = _labels.get(key, f"{key:18s}")

                if initial > _NEAR_ZERO_ENERGY:
                    pct = (initial - final) / initial * 100
                    if pct >= 0:
                        lines.append(
                            f"  {label}: {initial:.4f} to {final:.4f}"
                            f"  ({pct:.1f}% better)"
                        )
                    else:
                        lines.append(
                            f"  {label}: {initial:.4f} to {final:.4f}"
                            f"  (+{abs(pct):.1f}% trade-off,"
                            f" normal when other goals improved)"
                        )
                elif initial <= _NEAR_ZERO_ENERGY and final <= _NEAR_ZERO_ENERGY:
                    lines.append(
                        f"  {label}: already at 0 -> still 0  (nothing to improve here)"
                    )
                elif initial <= _NEAR_ZERO_ENERGY:
                    lines.append(
                        f"  {label}: started at 0 -> {final:.4f}"
                        f"  (small increase — normal trade-off)"
                    )
                else:
                    lines.append(f"  {label}: {initial:.4f} -> {final:.4f}")

            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


class MeshOptimiser:
    """
    Main optimisation class for planar quad meshes.

    Wraps the SciPy L-BFGS-B solver with mesh-specific validation,
    two-stage execution logic, history tracking, and verbose progress
    reporting. Instantiate with an 'OptimisationConfig', then call
    'optimise(mesh)' to run the optimisation.

    The optimiser does not modify the mesh until the solver has found its
    best solution. If validation fails, an 'OptimisationResult' with
    'success=False' is returned immediately and the mesh is not changed.

    Examples
    --------
    Standard usage with default settings:

        config = OptimisationConfig(weights={"planarity": 100.0,
                                             "fairness": 1.0,
                                             "closeness": 10.0})
        optimiser = MeshOptimiser(config)
        result = optimiser.optimise(mesh)
        print(result.summary())

    Single-stage run with angle balance enabled:

        config = OptimisationConfig(
            weights={"planarity": 100.0, "fairness": 1.0,
                     "closeness": 10.0, "angle_balance": 50.0},
            two_stage=False,
            max_iterations=500,
        )
        result = MeshOptimiser(config).optimise(mesh)
    """

    def __init__(self, config: Optional[OptimisationConfig] = None):
        """
        Initialise the optimiser with a configuration object.

        Parameters
        ----------
        config : OptimisationConfig or None, optional
            Settings for this optimisation run. If None, a default
            'OptimisationConfig' is created with standard weights and
            parameters. Default is None.

        Raises
        ------
        ValueError
            If the provided configuration fails validation.
        """
        self.config = config if config is not None else OptimisationConfig()
        self.config.validate()

        # History tracking
        self._iteration_count: int = 0
        self._energy_history: list[float] = []
        self._gradient_norm_history: list[float] = []
        self._start_time: float = 0.0
        self._stage_label: str = "S1"  # set to "S1" / "S2" during optimise()

    def validate_mesh(self, mesh: QuadMesh) -> tuple[bool, str]:
        """
        Validate a mesh before optimisation begins.

        Checks that the mesh has enough vertices and faces,
        that all vertex coordinates are finite, that every face is a quad
        (exactly four vertices), that no face contains duplicate vertices,
        and that no face has a near-zero area. These checks catch the most
        common sources of silent failures or misleading results.

        Parameters
        ----------
        mesh : QuadMesh
            The mesh to validate.

        Returns
        -------
        tuple
            A two-element tuple (is_valid, message) where 'is_valid' is a
            bool and 'message' is a str. If 'is_valid' is True, 'message'
            is 'Valid'. If False, 'message' describes the first problem found.
        """
        if mesh.n_vertices < 4:
            return False, "Mesh must have at least 4 vertices"

        if mesh.n_faces < 1:
            return False, "Mesh must have at least 1 face"

        # Check for NaN or Inf values first, which guards all subsequent indexing
        if not np.isfinite(mesh.vertices).all():
            return False, "Mesh contains NaN or Inf vertex coordinates"

        """
        Guard: every face must be a quad (exactly 4 vertex indices).
        This check must precede any axis-1 indexing that assumes shape (F, 4, 3).
        Triangle meshes (size 3) and higher-order polygons (size > 4) are both
        rejected here with a clear diagnostic rather than an IndexError.
        """
        if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 4:
            return (
                False,
                f"Mesh faces must be quads (4 vertices per face); "
                f"got {mesh.faces.shape[1] if mesh.faces.ndim == 2 else '?'} "
                f"vertices per face.",
            )

        """
        Vectorised duplicate-vertex check across all faces simultaneously.
        np.sort along axis=1 makes each row canonical so set-size comparison
        is equivalent to checking for repeated indices within a face.
        """
        sorted_faces = np.sort(mesh.faces, axis=1)
        n_unique = np.apply_along_axis(
            lambda row: len(np.unique(row)), axis=1, arr=sorted_faces
        )
        bad_idx = np.where(n_unique < mesh.faces.shape[1])[0]
        if len(bad_idx) > 0:
            i = int(bad_idx[0])
            return False, f"Face {i} has duplicate vertices: {mesh.faces[i]}"

        """
        Vectorised near-zero area check using the diagonal cross-product
        of each quad face.  verts shape: (F, 4, 3).
        Safe to index [:, 2, :] and [:, 3, :] because the quad guard above
        has already confirmed faces.shape[1] == 4.
        """
        verts = mesh.vertices[mesh.faces]  # (F, 4, 3)
        diag1 = verts[:, 2, :] - verts[:, 0, :]  # (F, 3) — first diagonal
        diag2 = verts[:, 3, :] - verts[:, 1, :]  # (F, 3) — second diagonal
        areas = 0.5 * np.linalg.norm(np.cross(diag1, diag2), axis=1)  # (F,)
        bad_area = np.where(areas < 1e-10)[0]
        if len(bad_area) > 0:
            i = int(bad_area[0])
            return False, f"Face {i} has near-zero area: {areas[i]:.2e}"

        return True, "Valid"

    def optimise(self, mesh: QuadMesh) -> OptimisationResult:
        """
        Optimise the mesh to minimise the total weighted energy.

        Runs the configured optimisation strategy (one-stage or two-stage)
        using SciPy's L-BFGS-B solver. The mesh vertices are updated
        in place with the best solution found. All statistics, history,
        and component energies are recorded in the returned result object.

        The method follows this sequence:

          1. Validate the mesh. Return a failure result immediately if invalid.
          2. Snapshot the initial energy and per-component energies.
          3. If 'two_stage=True', run Stage 1 with boosted planarity weight.
          4. Run Stage 2 (or sole stage) with the original balanced weights.
          5. Update mesh vertices to the final optimised positions.
          6. Snapshot the final energies and build the result object.
          7. Print the result summary if 'verbose=True'.

        Parameters
        ----------
        mesh : QuadMesh
            The mesh to optimise. Modified in-place. Must pass 'validate_mesh'
            before optimisation proceeds.

        Returns
        -------
        OptimisationResult
            Complete record of the run, including the optimised mesh,
            energy statistics, iteration counts, timing, and history.

        Raises
        ------
        ValueError
            If the configuration is invalid (caught during '__init__').
        """
        # ── Mesh validation ────────────────────────────────────────────────────
        is_valid, error_msg = self.validate_mesh(mesh)
        if not is_valid:
            print(f"❌ Mesh validation failed: {error_msg}")
            return OptimisationResult(
                success=False,
                message=f"VALIDATION_ERROR: {error_msg}",
                optimised_mesh=mesh,
                n_iterations=0,
                n_function_evaluations=0,
                n_gradient_evaluations=0,
                initial_energy=0.0,
                final_energy=0.0,
                execution_time=0.0,
                component_energies_initial={},
                component_energies_final={},
            )

        # ── Closeness reference ────────────────────────────────────────────────
        if not hasattr(mesh, "vertices_original") or mesh.vertices_original is None:
            mesh.vertices_original = mesh.vertices.copy()

        # ── Initial energy snapshot ────────────────────────────────────────────
        _initial_raw = compute_total_energy(mesh, self.config.weights)
        assert isinstance(_initial_raw, float), (
            f"compute_total_energy returned unexpected type {type(_initial_raw)!r}. "
            "Call with return_components=True only when the component dict is needed. "
            "This assert guards against accidental return-type contract violations."
        )
        initial_energy: float = _initial_raw
        component_energies_initial = {
            "planarity": float(compute_planarity_energy(mesh)),
            "fairness": float(compute_fairness_energy(mesh)),
            "closeness": float(compute_closeness_energy(mesh)),
            "angle_balance": float(_compute_angle_balance_component(mesh)),
        }

        # ── Resolve the Stage 1 iteration budget up front ─────────────────────
        # Stage 1 is allocated at most one-third of the total budget, capped at
        # 200.  Stage 2 then receives the remainder so the combined total never
        # exceeds max_iterations.
        stage1_max: int = (
            min(200, self.config.max_iterations // 3) if self.config.two_stage else 0
        )
        stage2_max: int = self.config.max_iterations - stage1_max

        # ── Verbose header ─────────────────────────────────────────────────────
        if self.config.verbose:
            _print_backend_info()
            print("\n" + "=" * 70)
            print("MESH OPTIMISATION — STARTING")
            print("=" * 70)
            print(
                f"Mesh loaded: {mesh.n_vertices} corner points, "
                f"{mesh.n_faces} panels"
            )
            print(
                f"Priority settings — "
                f"Flatness: {self.config.weights['planarity']}, "
                f"Smoothness: {self.config.weights['fairness']}, "
                f"Shape fidelity: {self.config.weights['closeness']}"
            )
            if self.config.two_stage:
                print(
                    f"Strategy: two-stage  "
                    f"(Stage 1 planarity x"
                    f"{self.config.stage1_planarity_multiplier:.0f}, "
                    f"then balanced refinement)"
                )
            print()
            print(
                "Starting scores (lower is better — the optimiser will reduce these):"
            )
            print(f"  Overall combined score:     {initial_energy:.4f}")
            print(
                f"  Panel flatness score:       "
                f"{component_energies_initial['planarity']:.4f}"
                f"  (how uneven the panels are)"
            )
            print(
                f"  Surface smoothness score:   "
                f"{component_energies_initial['fairness']:.4f}"
                f"  (how bumpy the surface is)"
            )
            print(
                f"  Shape fidelity score:       "
                f"{component_energies_initial['closeness']:.4f}"
                f"  (how far vertices have moved from the original design)"
            )
            print("=" * 70)
            print(
                "Progress will be printed every 10 improvement steps.\n"
                "Each line shows: step number, combined score, "
                "rate of change (lower = nearly done), and time elapsed."
            )
            print()

        # ── History / state reset ──────────────────────────────────────────────
        self._iteration_count = 0
        self._energy_history = []
        self._gradient_norm_history = []

        # ── Shared setup ───────────────────────────────────────────────────────
        x0 = mesh.vertices.flatten()

        bounds = None
        if self.config.bounds_scale is not None:
            delta = np.abs(x0) * self.config.bounds_scale
            bounds = list(zip(x0 - delta, x0 + delta))

        callback = self._create_callback(mesh) if self.config.history_tracking else None

        self._start_time = time.perf_counter()

        # Stage 2 options — budget is the remainder after Stage 1.
        stage2_options = {
            "maxiter": stage2_max,
            "ftol": 1e-9,  # much tighter energy convergence
            "gtol": 1e-5,  # tighter gradient convergence
            "maxcor": 20,  # smoother descent in refinement
            "maxls": 40,  # more backtracking in flat regions
        }

        # ── Counters for combined Stage 1 + Stage 2 totals ────────────────────
        n_iter_s1: int = 0
        n_fev_s1: int = 0
        n_jev_s1: int = 0

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 1 — Rapid Planarity Pass
        # ══════════════════════════════════════════════════════════════════════
        if self.config.two_stage:
            stage1_weights = {
                "planarity": (
                    self.config.weights["planarity"]
                    * self.config.stage1_planarity_multiplier
                ),
                "fairness": self.config.weights["fairness"] * 0.1,
                "closeness": self.config.weights["closeness"] * 0.1,
                # Angle balance carries over unchanged, it is either 0 or
                # already small, and boosting it in Stage 1 can destabilise
                # the topology before faces are planar.
                "angle_balance": self.config.weights.get("angle_balance", 0.0),
            }

            # Stage 1 uses the pre-computed budget and looser stopping
            # criteria so the pass terminates quickly once faces are
            # approximately flat.
            stage1_options = {
                "maxiter": stage1_max,
                "ftol": 1e-7,
                "gtol": 1e-4,
                "maxcor": 10,
                "maxls": 20,
            }

            if self.config.verbose:
                print("─" * 70)
                print("Stage 1 of 2 — Rapid Planarity Pass")
                print(
                    f"  Planarity weight: "
                    f"{stage1_weights['planarity']:.1f}  "
                    f"(x{self.config.stage1_planarity_multiplier:.0f})"
                )
                print(
                    f"  Fairness / closeness: "
                    f"{stage1_weights['fairness']:.3f} / "
                    f"{stage1_weights['closeness']:.3f}  (x0.1 each)"
                )
                print(
                    f"  Max steps: {stage1_options['maxiter']} "
                    f"(Stage 2 will receive the remaining "
                    f"{stage2_max} steps)"
                )
                print("─" * 70)
                print()

            self._stage_label = "S1"
            stage1_scipy = minimize(
                fun=lambda x: energy_for_scipy(x, mesh, stage1_weights),
                x0=x0,
                jac=lambda x: gradient_for_scipy(x, mesh, stage1_weights),
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options=stage1_options,
            )  # type: ignore[call-overload]

            n_iter_s1 = stage1_scipy.nit
            n_fev_s1 = stage1_scipy.nfev
            n_jev_s1 = stage1_scipy.njev
            x0_stage2 = stage1_scipy.x  # warm start for Stage 2

            if self.config.verbose:
                # Report using the *balanced* weights so the score is
                # comparable with the initial and final scores printed
                # in the surrounding verbose blocks.
                s1_energy = float(
                    energy_for_scipy(x0_stage2, mesh, self.config.weights)
                )
                elapsed_s1 = time.perf_counter() - self._start_time
                print()
                print("─" * 70)
                print(
                    f"Stage 1 complete — {n_iter_s1} steps, "
                    f"balanced score = {s1_energy:.4f}  "
                    f"(elapsed: {elapsed_s1:.2f}s)"
                )
                print()
                print("─" * 70)
                print("Stage 2 of 2 — Balanced Refinement")
                print(
                    f"  Restoring standard weights — "
                    f"Flatness: {self.config.weights['planarity']}, "
                    f"Smoothness: {self.config.weights['fairness']}, "
                    f"Shape fidelity: {self.config.weights['closeness']}"
                )
                print("  Warm start from Stage 1 solution.")
                print(
                    f"  Stage 2 budget: {stage2_max} steps "
                    f"(total cap: {self.config.max_iterations})"
                )
                print("─" * 70)
                print()

        else:
            # Single-stage fallback — behaves identically to the original code.
            x0_stage2 = x0

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 2 — Balanced Refinement (or sole stage if two_stage=False)
        # ══════════════════════════════════════════════════════════════════════
        self._stage_label = "S2" if self.config.two_stage else ""

        scipy_result = minimize(
            fun=lambda x: energy_for_scipy(x, mesh, self.config.weights),
            x0=x0_stage2,
            jac=lambda x: gradient_for_scipy(x, mesh, self.config.weights),
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options=stage2_options,
        )  # type: ignore[call-overload]

        execution_time = time.perf_counter() - self._start_time

        # ── Finalise mesh ──────────────────────────────────────────────────────
        mesh.vertices = scipy_result.x.reshape(-1, 3)

        # ── Final energy snapshot ──────────────────────────────────────────────
        _final_raw = compute_total_energy(mesh, self.config.weights)
        assert isinstance(_final_raw, float), (
            f"compute_total_energy returned unexpected type {type(_final_raw)!r}. "
            "Call with return_components=True only when the component dict is needed. "
            "This assert guards against accidental return-type contract violations."
        )
        final_energy: float = _final_raw
        component_energies_final = {
            "planarity": float(compute_planarity_energy(mesh)),
            "fairness": float(compute_fairness_energy(mesh)),
            "closeness": float(compute_closeness_energy(mesh)),
            "angle_balance": float(_compute_angle_balance_component(mesh)),
        }

        # ── Build result (combined Stage 1 + Stage 2 counts) ──────────────────
        result = OptimisationResult(
            success=scipy_result.success,
            message=scipy_result.message,
            optimised_mesh=mesh,
            initial_energy=initial_energy,
            final_energy=final_energy,
            n_iterations=scipy_result.nit + n_iter_s1,
            n_function_evaluations=scipy_result.nfev + n_fev_s1,
            n_gradient_evaluations=scipy_result.njev + n_jev_s1,
            execution_time=execution_time,
            energy_history=(
                self._energy_history if self.config.history_tracking else None
            ),
            gradient_norm_history=(
                self._gradient_norm_history if self.config.history_tracking else None
            ),
            component_energies_initial=component_energies_initial,
            component_energies_final=component_energies_final,
        )

        if self.config.verbose:
            print("\n" + result.summary())

        return result

    def _create_callback(self, mesh: QuadMesh) -> Callable:
        """
        Create the iteration callback function passed to SciPy.

        SciPy calls the callback at the end of each L-BFGS-B
        iteration with the current vertex positions as a flat array. It
        records the energy and gradient norm for history tracking, and
        prints a progress line every ten iterations when 'verbose=True'.

        Progress lines are prefixed with '[S1]' or '[S2]' to identify
        which stage is currently running, making it easy to see where
        Stage 1 ends and Stage 2 begins in the console output.

        Parameters
        ----------
        mesh : QuadMesh
            The mesh being optimised. Vertex positions are temporarily
            updated to the callback's input and then restored, so that
            computing the energy and gradient in the callback does not
            interfere with the solver's internal state.

        Returns
        -------
        callable
            A function with signature 'callback(xk)' compatible with
            'scipy.optimize.minimize'.
        """

        def callback(xk):
            self._iteration_count += 1

            verts_backup = mesh.vertices.copy()
            try:
                mesh.vertices = xk.reshape(-1, 3)
                energy = compute_total_energy(mesh, self.config.weights)
                gradient = compute_total_gradient(mesh, self.config.weights)
                gradient_norm = np.linalg.norm(gradient)
                self._energy_history.append(energy)
                self._gradient_norm_history.append(gradient_norm)
            finally:
                mesh.vertices = verts_backup

            if self.config.verbose and self._iteration_count % 10 == 0:
                elapsed = time.perf_counter() - self._start_time
                label = self._stage_label
                prefix = f"[{label}] " if label else ""
                print(
                    f"{prefix}Step {self._iteration_count:4d}: "
                    f"score = {energy:.2f},  "
                    f"rate of change = {gradient_norm:.4f},  "
                    f"time elapsed = {elapsed:.2f}s"
                )
                print(
                    f"       (technical: iteration {self._iteration_count},  "
                    f"energy E = {energy:.6e},  "
                    f"|| gradient E || = {gradient_norm:.4e})"
                )

        return callback


def optimise_mesh_simple(
    mesh: QuadMesh,
    weights: Optional[Dict[str, float]] = None,
    max_iter: int = 1000,
    verbose: bool = True,
) -> OptimisationResult:
    """
    Simplified interface for running mesh optimisation with minimal setup.

    Creates an 'OptimisationConfig' with the provided settings and default
    values for everything else, then runs 'MeshOptimiser.optimise()'. This
    is the recommended entry point for users who do not need fine-grained
    control over the two-stage strategy or callback behaviour.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to optimise. Modified in-place.
    weights : dict or None, optional
        Energy term weights. If None, defaults to planarity=100.0,
        fairness=1.0, closeness=10.0 with angle balance disabled.
    max_iter : int, optional
        Maximum total number of L-BFGS-B iterations. Default is 1000.
    verbose : bool, optional
        If True, print progress and a final summary to the console.
        Default is True.

    Returns
    -------
    OptimisationResult
        Complete result of the optimisation run.

    Examples
    --------
    Basic usage:

        from src.optimisation.optimiser import optimise_mesh_simple
        from src.io.obj_handler import load_obj

        mesh = load_obj("my_mesh.obj")
        result = optimise_mesh_simple(mesh, max_iter=500)
        print(result.summary())
    """
    config = OptimisationConfig(
        weights=(
            weights
            if weights
            else {"planarity": 100.0, "fairness": 1.0, "closeness": 10.0}
        ),
        max_iterations=max_iter,
        verbose=verbose,
    )

    optimiser = MeshOptimiser(config)
    return optimiser.optimise(mesh)
