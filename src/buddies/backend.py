"""Resolve a simulation's array backend: numpy on the CPU, cupy on the GPU.

The cupy import is deferred so cupy stays an optional dependency that is
only required when the GPU is actually requested (via simargs' ``--gpu``).
"""

import numpy


def get_backend(name):
    """Return the array module for ``name`` ("numpy"/"cpu" or "cupy"/"gpu")."""
    name = name.lower()
    if name in ("numpy", "np", "cpu"):
        return numpy
    if name in ("cupy", "gpu"):
        import cupy

        return cupy
    raise ValueError(f"unknown backend {name!r}; expected 'numpy' or 'cupy'")
