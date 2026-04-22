"""
Tests for backend selection and data movement.

Tests verify:
- Default backend is valid (numpy, numba, or cupy)
- PQ_BACKEND environment variable correctly forces a backend
- Backend selection falls back gracefully when cupy is requested but CUDA is unavailable
- to_device and to_numpy roundtrip operations

Date: 14 March 2026
"""

import numpy as np

from src.backends import BACKEND, _detect_backend, to_device, to_numpy


def test_backend_is_valid():
    """Default backend must be one of the supported ones."""
    assert BACKEND in ("numpy", "numba", "cupy")


def test_forced_numpy(monkeypatch):
    """Setting PQ_BACKEND to 'numpy' must result in a 'numpy' backend."""
    monkeypatch.setenv("PQ_BACKEND", "numpy")
    assert _detect_backend() == "numpy"


def test_cupy_request_without_cuda_does_not_raise(monkeypatch):
    """
    If cupy is requested but CUDA is unavailable,
    it should fall back without error.
    """
    monkeypatch.setenv("PQ_BACKEND", "cupy")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = _detect_backend()
    assert result in ("cupy", "numba", "numpy")


def test_to_numpy_roundtrip():
    """Moving an array to the device and back should preserve the values."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    np.testing.assert_array_equal(to_numpy(to_device(arr)), arr)
