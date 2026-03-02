"""
ctypes wrapper for the C Lennard-Jones force kernel.

Build the shared library first:
    gcc -O2 -shared -fPIC -o forces_c.so forces_c.c    (Linux/macOS)
    gcc -O2 -shared -o forces_c.dll forces_c.c          (Windows)

Falls back to pure Python if the compiled library is not found.
"""

import ctypes
import os
import sys
import warnings

import numpy as np

_LIB = None
_dir = os.path.dirname(os.path.abspath(__file__))


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    if sys.platform == "win32":
        name = "forces_c.dll"
    elif sys.platform == "darwin":
        name = "forces_c.dylib"
    else:
        name = "forces_c.so"

    path = os.path.join(_dir, name)
    if not os.path.exists(path):
        return None

    lib = ctypes.CDLL(path)
    lib.compute_lj_forces.restype = None
    lib.compute_lj_forces.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # pos
        ctypes.POINTER(ctypes.c_double),  # forces
        ctypes.c_int,                     # n
        ctypes.c_double,                  # epsilon
        ctypes.c_double,                  # sigma
        ctypes.c_double,                  # max_force
    ]
    _LIB = lib
    return lib


def compute_forces(pos, forces, epsilon, sigma, max_force):
    """
    Compute LJ forces for all agents.

    pos:       (N, 2) float64 array of positions
    forces:    (N, 2) float64 array, overwritten with computed forces
    epsilon:   LJ well depth
    sigma:     LJ distance parameter
    max_force: saturation clamp
    """
    n = pos.shape[0]
    pos_flat = np.ascontiguousarray(pos.ravel(), dtype=np.float64)
    frc_flat = np.zeros(2 * n, dtype=np.float64)

    lib = _load_lib()

    if lib is not None:
        lib.compute_lj_forces(
            pos_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            frc_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_double(epsilon),
            ctypes.c_double(sigma),
            ctypes.c_double(max_force),
        )
    else:
        warnings.warn(
            "C force library not found, using pure Python fallback. "
            "Build with: gcc -O2 -shared -fPIC -o forces_c.so forces_c.c",
            RuntimeWarning,
        )
        frc_flat = _python_lj(pos_flat, n, epsilon, sigma, max_force)

    forces[:] = frc_flat.reshape(n, 2)


def _python_lj(pos, n, epsilon, sigma, max_force):
    """Pure Python fallback. Same algorithm, just slow."""
    forces = np.zeros(2 * n)
    min_dist = 0.1

    for i in range(n):
        xi, yi = pos[2 * i], pos[2 * i + 1]
        for j in range(i + 1, n):
            dx = pos[2 * j] - xi
            dy = pos[2 * j + 1] - yi
            r = max((dx**2 + dy**2) ** 0.5, min_dist)
            sr = sigma / r
            sr6 = sr**6
            fmag = 24.0 * epsilon * (2.0 * sr6**2 - sr6) / r
            fmag = max(-max_force, min(max_force, fmag))
            fx = fmag * dx / r
            fy = fmag * dy / r
            forces[2 * i] += fx
            forces[2 * i + 1] += fy
            forces[2 * j] -= fx
            forces[2 * j + 1] -= fy

    return forces
