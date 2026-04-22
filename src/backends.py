"""
src/backends.py

Hardware backend detection, capability flags, and array-module routing for
PQ mesh optimisation.

This module is the single source of truth for which computation backend
(GPU, CPU-parallel, or CPU-serial) the project is currently using. Every
other module in the project imports the flags 'BACKEND', 'HAS_CUDA', and
'HAS_NUMBA' from here rather than performing their own detection. This
ensures consistent, predictable behaviour across all energy and gradient
functions.

The three backends, in priority order
--------------------------------------
  Tier 1 CuPy (NVIDIA GPU):
    Selected when CuPy is installed and a compatible NVIDIA CUDA GPU is
    available. Provides the fastest execution for large meshes by running
    batched SVD and sparse matrix operations on the GPU. Requires the
    CuPy package and CUDA Toolkit 13.x, including cuSOLVER.

  Tier 2 Numba (CPU, parallel):
    Selected when CuPy is unavailable, but Numba is installed. Custom
    '@njit(parallel=True)' kernels are compiled to native machine code
    by LLVM and run in parallel across CPU cores using Numba's thread
    pool. Faster than NumPy for large meshes on multicore machines.

  Tier 3 NumPy (CPU, serial):
    Always available. Used as the fallback when neither CuPy nor Numba
    is present. Fully vectorised NumPy operations ensure correctness and
    reasonable performance on any platform.

Overriding the automatic selection
------------------------------------
Set the 'PQ_BACKEND' environment variable before importing any project
module to force a specific backend:

    PQ_BACKEND=numpy Force the NumPy CPU baseline.
    PQ_BACKEND=numba Force Numba parallel CPU.
    PQ_BACKEND=cupy Force GPU (warns and falls back if CUDA is unavailable).

Module-level constants exposed
--------------------------------
  BACKEND: str -- 'cupy', 'numba', or 'numpy'.
  HAS_CUDA: bool -- True only when BACKEND == 'cupy'.
  HAS_NUMBA: bool -- True when Numba is installed and the backend is not
                         forced to 'numpy'.
  IS_APPLE_SILICON: bool -- True on Apple M-series machines (where CuPy
                             is not supported).

Self-containment
----------------
This module intentionally does not import any other project module at the
module level. It depends only on the Python standard library, NumPy, and
optionally CuPy and Numba. The one exception is 'warmup_numba_kernels()',
which uses lazy imports inside its function body to trigger Numba kernel
compilation without creating a circular import.

References
----------
Lam, S. K., Pitrou, A., and Seibert, S. (2015).
  "Numba: A LLVM-based Python JIT compiler." Proceedings of the Second
  Workshop on the LLVM Compiler Infrastructure in HPC, pp. 1-6.

Okuta, R., Unno, Y., Nishino, D., Hido, S., and Loomis, C. (2017).
  "CuPy: A NumPy-compatible library for NVIDIA GPU calculations."
  Proceedings of Workshop on Machine Learning Systems (LearningSys)
  at NeurIPS 2017.
"""
import os
import platform
import time
import warnings
from contextlib import contextmanager
from typing import Literal

import numpy as np

BackendType = Literal["numpy", "numba", "cupy"]


def _detect_backend() -> BackendType:
    """
    Detect and return the best available computation backend.

    Checks for CuPy and Numba availability in priority order, respecting
    the 'PQ_BACKEND' environment variable override. Always returns a valid
    backend name; never raises an exception.

    The CuPy probe is deliberately lightweight: it creates a small array and
    runs a simple arithmetic operation to verify that the CUDA runtime
    initialises correctly, without triggering the more expensive cuSOLVER
    or SVD initialisation that would occur on first GPU use.

    Returns
    -------
    str
        One of 'cupy', 'numba', or 'numpy'.

    Notes
    -----
    If 'PQ_BACKEND=cupy' is set but CuPy fails to initialise, a
    RuntimeWarning is emitted and detection continues automatically
    from 'numba', then 'numpy'. This ensures the optimiser always runs
    rather than failing at import time.
    """
    requested = os.environ.get("PQ_BACKEND", "auto").lower()

    if requested in ("cupy", "auto"):
        try:
            # pylint: disable=import-outside-toplevel
            import cupy as cp

            # Lightweight probe — exercises basic array creation and arithmetic
            # without forcing cuSOLVER/SVD initialisation at import time.
            _probe = cp.arange(4, dtype=cp.float32)
            _probe = (_probe**2).sum()
            cp.cuda.Stream.null.synchronize()
            # pylint: disable=c-extension-no-member
            if cp.cuda.runtime.getDeviceCount() > 0:    # pragma: no cover
                return "cupy"   # pragma: no cover
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if requested == "cupy": # pragma: no cover
            warnings.warn(
                "PQ_BACKEND=cupy requested but CuPy initialisation failed "
                "(missing CUDA runtime, nvrtc, cusolver, or no compatible GPU). "
                "Falling back to numba (or numpy if numba unavailable). "
                "To fix: reinstall CUDA Toolkit 13.x including cuSOLVER, "
                "then reinstall cupy-cuda13x.",
                RuntimeWarning,
                stacklevel=3,
            ) # pragma: no cover

            # After a forced CuPy failure, continue automatic backend
            # detection so that we actually attempt to use numba before
            # falling back to the NumPy baseline.
            requested = "auto" # pragma: no cover

    if requested in ("numba", "auto"):
        try:
            # pylint: disable=import-outside-toplevel,unused-import
            import numba  # noqa: F401

            return "numba"
        except ImportError:
            pass
        if requested == "numba":    # pragma: no cover
            warnings.warn(
                "PQ_BACKEND=numba requested but numba is not installed. "
                "Falling back to numpy. Install with: pip install numba",
                RuntimeWarning,
                stacklevel=3,
            )   # pragma: no cover

    return "numpy"


# ── Module-level constants, resolved once at import time ────────────────────
BACKEND: BackendType = _detect_backend()
HAS_CUDA: bool = BACKEND == "cupy"

# HAS_NUMBA is True if Numba is installed AND the backend is not forced to NumPy.
# pylint: disable=invalid-name
HAS_NUMBA: bool = False
if BACKEND in ("cupy", "numba"):
    try:
        # pylint: disable=import-outside-toplevel
        import numba as _numba  # noqa: F401

        HAS_NUMBA = True
    except ImportError:
        pass

IS_APPLE_SILICON: bool = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


# ── Array-module helpers ──────────────────────────────────────────────────────
def get_array_module():
    """
    Return the array module appropriate for the active backend.

    Returns CuPy when the GPU backend is active, otherwise returns NumPy.
    Use this when writing backend-agnostic array operations that should
    run on whichever device is currently selected.

    Returns
    -------
    module
        'cupy' if BACKEND == 'cupy', otherwise 'numpy'.

    Examples
    --------
    Backend-agnostic array creation:

        xp = get_array_module()
        arr = xp.zeros((100, 3), dtype=xp.float64)
    """
    if BACKEND == "cupy":
        # pylint: disable=import-outside-toplevel
        import cupy as cp

        return cp
    return np


def to_device(arr: np.ndarray):
    """
    Transfer a NumPy array to the active compute device.

    When the GPU backend is active, copies the array to GPU memory using
    'cupy.asarray()'. When the CPU backend is active, it returns the array
    unchanged (a no-op). This allows calling code to be written once and
    work correctly regardless of which backend is selected.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to transfer.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        The same array on the active device.
    """
    if BACKEND == "cupy":
        import cupy as cp

        return cp.asarray(arr)
    return arr


def to_numpy(arr) -> np.ndarray:
    """
    Transfer any array back to a CPU NumPy array.

    If the input is a CuPy GPU array, copies it from GPU memory to CPU
    memory and returns a NumPy array. If the input is already a NumPy
    array (or any other array-like), returns it as a NumPy array without
    copying. This is the standard way to retrieve a result from the GPU
    and return it to SciPy or any other CPU-only library.

    Parameters
    ----------
    arr : numpy.ndarray or cupy.ndarray
        The array to retrieve.

    Returns
    -------
    numpy.ndarray
        The array as a CPU NumPy array.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import cupy as cp  # type: ignore[import]
    except ImportError:
        cp = None  # type: ignore[assignment]
    else:
        if isinstance(arr, cp.ndarray): # pragma: no cover
            return cp.asnumpy(arr) # pragma: no cover
    return np.asarray(arr)


def get_sparse_module():
    """
    Return the sparse matrix module appropriate for the active backend.

    Returns 'cupyx.scipy.sparse' when the GPU backend is active, otherwise
    returns 'scipy.sparse'. Use this when building or multiplying sparse
    matrices (such as the Laplacian or scatter matrix) in a backend-agnostic
    way.

    Returns
    -------
    module
        'cupyx.scipy.sparse' if BACKEND == 'cupy', otherwise 'scipy.sparse'.
    """
    if BACKEND == "cupy":
        # pylint: disable=import-outside-toplevel
        import cupyx.scipy.sparse as cpsp

        return cpsp
    # pylint: disable=import-outside-toplevel
    from scipy import sparse

    return sparse


# ── GPU memory safety ─────────────────────────────────────────────────────────
@contextmanager
def gpu_memory_guard():
    """
    Context manager that catches GPU out-of-memory and CUDA runtime errors
    and automatically downgrades the backend to CPU rather than crashing.

    When a GPU operation inside this context raises a CuPy out-of-memory
    error or a CUDA runtime error, the context manager catches the exception,
    emits a 'RuntimeWarning' describing the failure, and updates the
    module-level 'BACKEND', 'HAS_CUDA', and 'HAS_NUMBA' flags to reflect
    the new CPU backend. Subsequent calls to the energy and gradient
    functions will then use the CPU path automatically.

    If Numba is available, the downgrade target is 'numba'; otherwise it
    is 'numpy'. If the exception is not a GPU error, it is re-raised as
    normal.

    Typical usage in an energy or gradient function:

        with gpu_memory_guard():
            result = _planarity_energy_gpu(mesh)

    If the GPU call succeeds, execution continues normally. If the GPU
    runs out of memory, the context manager degrades gracefully and the
    caller's fallback path takes over on the next call.
    """
    global BACKEND, HAS_CUDA, HAS_NUMBA  # pylint: disable=global-statement

    gpu_error_types = None
    try:
        # pylint: disable=import-outside-toplevel
        from cupy.cuda.memory import (
            OutOfMemoryError as _CuPyOutOfMemoryError,  # type: ignore[import]
        )
        from cupy.cuda.runtime import (
            CUDARuntimeError as _CuPyRuntimeError,  # type: ignore[import]
        )

        gpu_error_types = (_CuPyOutOfMemoryError, _CuPyRuntimeError)
    except ImportError:
        gpu_error_types = None
    try:
        yield
    except Exception as exc:  # pragma: no cover
        if gpu_error_types is not None and isinstance(exc, gpu_error_types): # pragma: no cover
            target_backend = "numpy"
            has_numba = False
            try:
                # pylint: disable=import-outside-toplevel,unused-import
                import numba  # noqa: F401
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            else:
                target_backend = "numba"
                has_numba = True

            warnings.warn(
                f"GPU error ({exc}). Switching to {target_backend} CPU backend "
                "for the remainder of this session.",
                RuntimeWarning,
                stacklevel=2,
            )
            BACKEND = target_backend
            HAS_CUDA = False
            HAS_NUMBA = has_numba
        else:
            raise


# ── Diagnostics ───────────────────────────────────────────────────────────────
def print_backend_info() -> None:
    """
    Print a summary of the active backend configuration to the console.

    Called automatically by 'MeshOptimiser.optimise()' at the start of each
    optimisation run (when 'verbose=True') and can also be called directly
    during interactive sessions to confirm which hardware is being used.

    Prints the active backend name and, where relevant:
      - GPU name and available VRAM (if CuPy is active).
      - A note about Apple Silicon (Metal) if running on an M-series Mac.
      - The installed Numba version (if Numba is available).
      - The installed NumPy version (always shown).

    Returns
    -------
    None
        Output is written to standard output only.
    """
    print(f"[PQ Mesh Optimiser] Backend: {BACKEND.upper()}")
    if HAS_CUDA:
        try:
            # pylint: disable=import-outside-toplevel
            import cupy as cp

            # pylint: disable=c-extension-no-member
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            name = name.decode() if isinstance(name, bytes) else name
            mem = cp.cuda.runtime.memGetInfo()
            print(f"  GPU : {name}")
            print(f"  VRAM: {mem[1]/1e9:.1f} GB total  |  {mem[0]/1e9:.1f} GB free")
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    if IS_APPLE_SILICON:
        print("  Platform: Apple Silicon (Metal — CuPy not supported)")
    if HAS_NUMBA:
        try:
            # pylint: disable=import-outside-toplevel
            import numba

            print(f"  Numba: {numba.__version__}")
        except (ImportError, Exception):  # pylint: disable=broad-exception-caught
            print("  Numba: [Incompatible with NumPy]")
    print(f"  NumPy: {np.__version__}")


# ── Numba kernel warm-up ──────────────────────────────────────────────────────
def warmup_numba_kernels() -> None:
    """
    Pre-compile all Numba JIT kernels before optimisation begins.

    Numba's '@njit(cache=True)' decorator defers LLVM compilation to the
    first call with a given set of argument types. On a typical machine,
    compiling the planarity energy and gradient kernels takes 30 to 90
    seconds on the first run. Because the first call happens inside the
    optimiser's main loop, the process would appear to freeze silently at
    the "STARTING OPTIMISATION" banner without this warm-up step.

    Calling this function once at application startup moves the compilation
    to a clearly labelled, timed step before any mesh is loaded. On all
    later runs in the same process, or in any future process, the
    compiled binaries are loaded from disk (from the '__pycache__' directory
    written by 'cache=True') and incur no additional compile time.

    Kernels compiled
    ----------------
    From 'src/optimisation/energy_terms.py':
      '_planarity_energy_numba(vertices, faces)' -- @njit(parallel=True),
      prange over faces; internally triggers compilation of the per-face
      helper '_planarity_per_face_numba' in the same call.

    From 'src/optimisation/gradients.py':
      '_planarity_gradient_contributions_numba(vertices, faces)' --
      @njit(parallel=True), returns a (n_faces, 4, 3) contribution tensor.

    A minimal single-face dummy mesh is used for the warm-up call. The
    dtypes of the dummy arrays (float64 for vertices, int64 for faces) match
    the dtypes passed at real call sites, ensuring that Numba compiles exactly
    the specialisation that will be reused during optimisation. A dtype
    mismatch would cause a silent second recompilation on the first live call.

    Lazy imports
    ------------
    Both 'energy_terms' and 'gradients' are imported inside this function
    body rather than at module level. This prevents a circular import:
    both of those modules import 'HAS_CUDA' and 'HAS_NUMBA' from this file
    at their own module level. A module-level import in the reverse direction
    would cause an 'ImportError' at startup.

    Error handling
    --------------
    Each kernel warm-up is wrapped in a broad 'except Exception' block,
    consistent with the project convention in all Numba dispatch paths.
    Numba can raise 'TypingError', 'LoweringError', or LLVM backend errors
    on platforms where Numba is installed, but compilation fails; none of
    these are 'ImportError' subclasses. A 'RuntimeWarning' is emitted for
    each failure, and the optimiser will fall back to NumPy automatically on
    the first live call.

    Returns
    -------
    None
        Compilation output and timing are printed to standard output.
        Nothing is returned.

    Notes
    -----
    This function does nothing if 'HAS_NUMBA' is False (for example, if
    Numba is not installed or if 'PQ_BACKEND=numpy' was set). In that case
    there are no Numba kernels to compile.

    To enable warm-up in the interactive visualisation tool, add the
    following two lines near the top of
    'src/visualisation/interactive_optimisation.py', after the call to
    'print_backend_info()':

        from src.backends import warmup_numba_kernels
        warmup_numba_kernels()
    """
    if not HAS_NUMBA:
        # Numba is not installed, or the backend was forced to numpy/cupy —
        # nothing to compile.
        return

    # Minimal single-face mesh: 4 coplanar vertices, one quad.
    # dtype=np.float64 and dtype=np.int64 must match the actual call-site
    # argument types used by energy_terms.py and gradients.py so that Numba
    # compiles exactly the specialisation that will be reused during
    # optimisation.  A dtype mismatch here would cause a *second* silent
    # recompilation on the first real call.
    _dummy_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    _dummy_faces = np.array([[0, 1, 2, 3]], dtype=np.int64)

    _t0 = time.perf_counter()
    print(
        "[PQ Mesh Optimiser] Numba backend detected — compiling JIT kernels.\n"
        "  This happens once only; compiled artefacts are cached in __pycache__.\n"
        "  Expected wait: 30–90 s on first run, <1 s on all subsequent runs."
    )

    # ── Energy kernel ──────────────────────────────────────────────────────────
    # Lazy relative import: resolved at call time, not at module import time.
    # This avoids the circular-import that would arise from a module-level
    # import because energy_terms.py imports HAS_CUDA, HAS_NUMBA from this file.
    _energy_ok = False
    try:
        from .optimisation.energy_terms import (  # noqa: PLC0415
            _planarity_energy_numba as _pen,
        )

        _pen(_dummy_vertices, _dummy_faces)
        _energy_ok = True
        print(
            f"  [✓] planarity_energy_numba compiled "
            f"({time.perf_counter() - _t0:.1f} s elapsed)"
        )
    except Exception as _exc:  # pylint: disable=broad-exception-caught
        warnings.warn(
            f"Numba energy kernel warm-up failed: {_exc}. "
            "The optimiser will attempt NumPy fallback on first call.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Gradient kernel ────────────────────────────────────────────────────────
    # planarity_gradient_contributions_numba is compiled independently of the
    # energy kernel; both must be warmed up to eliminate the silent pause.
    _gradient_ok = False
    try:
        from .optimisation.gradients import (  # noqa: PLC0415
            _planarity_gradient_contributions_numba as _pgn,
        )

        _pgn(_dummy_vertices, _dummy_faces)
        _gradient_ok = True
        print(
            f"  [✓] planarity_gradient_contributions_numba compiled "
            f"({time.perf_counter() - _t0:.1f} s elapsed)"
        )
    except Exception as _exc:  # pylint: disable=broad-exception-caught
        warnings.warn(
            f"Numba gradient kernel warm-up failed: {_exc}. "
            "The gradient path will attempt NumPy fallback on first call.",
            RuntimeWarning,
            stacklevel=2,
        )

    _elapsed = time.perf_counter() - _t0

    if _energy_ok and _gradient_ok:
        print(
            f"[PQ Mesh Optimiser] Numba kernels ready — "
            f"total compile time: {_elapsed:.1f} s. "
            "Future runs will be instant (cache=True)."
        )
    elif not _energy_ok and not _gradient_ok:   # pragma: no cover
        print(
            "[PQ Mesh Optimiser] WARNING — both Numba kernels failed to compile. "
            "Optimisation will use the NumPy baseline automatically."
        )   # pragma: no cover
    else:   # pragma: no cover
        print(
            f"[PQ Mesh Optimiser] Partial Numba warm-up completed in {_elapsed:.1f} s "
            f"(energy={'OK' if _energy_ok else 'FAILED'}, "
            f"gradient={'OK' if _gradient_ok else 'FAILED'}). "
            "See warnings above for details."
        )   # pragma: no cover
