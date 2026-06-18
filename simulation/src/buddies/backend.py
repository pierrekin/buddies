"""Resolve a simulation's array backend: numpy on the CPU, cupy on the GPU.

The cupy import is deferred so cupy stays an optional dependency that is
only required when the GPU is actually requested (via simargs' ``--gpu``).
"""

import glob
import os

import numpy


def _configure_rocm_env():
    """Prepare the environment so cupy's runtime kernel compiler works on ROCm.

    Two ROCm-on-Arch quirks have to be handled before cupy is imported:

    - ``hipcc`` lives in ``/opt/rocm/bin``, which is not on PATH by default.
      cupy shells out to it to discover its default include directories, so a
      missing ``hipcc`` aborts every kernel compile.
    - cupy compiles kernels with hiprtc, which (unlike the hipcc driver) does
      not pull in the system libstdc++ headers, so device code that uses
      ``std::initializer_list`` fails to compile. Exposing the gcc C++ include
      dirs via ``CPLUS_INCLUDE_PATH`` makes hiprtc find them.

    Existing values are respected, so a user with a non-default ROCm install
    can override any of this from the shell.
    """
    rocm_home = (
        os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm"
    )
    os.environ.setdefault("ROCM_HOME", rocm_home)
    os.environ.setdefault("ROCM_PATH", rocm_home)

    rocm_bin = os.path.join(rocm_home, "bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if rocm_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([rocm_bin, *filter(None, path_parts)])

    # Locate the gcc libstdc++ headers (the dir tree holding <initializer_list>)
    # via the arch-specific bits/c++config.h, which pins the right version.
    for config in sorted(glob.glob("/usr/include/c++/*/*/bits/c++config.h")):
        triple_dir = os.path.dirname(os.path.dirname(config))  # .../<ver>/<triple>
        base = os.path.dirname(triple_dir)  # .../<ver>
        cxx_dirs = [base, triple_dir, os.path.join(base, "backward")]
        existing = os.environ.get("CPLUS_INCLUDE_PATH", "").split(os.pathsep)
        os.environ["CPLUS_INCLUDE_PATH"] = os.pathsep.join(
            [*filter(None, existing), *(d for d in cxx_dirs if d not in existing)]
        )
        break


def get_backend(name):
    """Return the array module for ``name`` ("numpy"/"cpu" or "cupy"/"gpu")."""
    name = name.lower()
    if name in ("numpy", "np", "cpu"):
        return numpy
    if name in ("cupy", "gpu"):
        _configure_rocm_env()
        import cupy

        return cupy
    raise ValueError(f"unknown backend {name!r}; expected 'numpy' or 'cupy'")
