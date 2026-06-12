"""Staggered-grid (Yee) pressure-velocity FDTD for 2D acoustics.

Array-module agnostic: pass ``xp=numpy`` (default) or ``xp=cupy`` to select
the array backend used for all simulation state.

Pressure is defined at cell centers and particle velocity components at cell
faces. Velocities are stored only for interior faces; boundary faces therefore
have zero normal velocity, which makes the domain walls rigid (perfectly
reflective).
"""

import math

import numpy

SOUND_SPEED_SEAWATER = 1500.0  # m/s
DENSITY_SEAWATER = 1000.0  # kg/m^3
# Fraction of the 2D CFL stability limit dx / (c * sqrt(2)) used for dt.
CFL_SAFETY_FACTOR = 0.95


class AcousticFDTD:
    def __init__(
        self,
        nx,
        ny,
        dx,
        c=SOUND_SPEED_SEAWATER,
        rho=DENSITY_SEAWATER,
        cfl=CFL_SAFETY_FACTOR,
        source_pos=None,
        xp=numpy,
    ):
        self.xp = xp
        self.nx, self.ny, self.dx = nx, ny, dx
        self.c, self.rho = c, rho
        self.dt = cfl * dx / (c * math.sqrt(2))
        self.source_pos = source_pos if source_pos is not None else (nx // 2, ny // 2)

        self.p = xp.zeros((nx, ny), dtype=xp.float32)
        self.vx = xp.zeros((nx - 1, ny), dtype=xp.float32)
        self.vy = xp.zeros((nx, ny - 1), dtype=xp.float32)

        self._cv = xp.float32(self.dt / (rho * dx))
        self._cp = xp.float32(rho * c * c * self.dt / dx)

    def step(self, source=0.0):
        """Advance one timestep, adding ``source`` (Pa) at the source cell."""
        p, vx, vy = self.p, self.vx, self.vy

        # rho dv/dt = -grad(p)
        vx -= self._cv * (p[1:, :] - p[:-1, :])
        vy -= self._cv * (p[:, 1:] - p[:, :-1])

        # dp/dt = -rho c^2 div(v); faces outside the grid are rigid walls (v = 0)
        div = self.xp.zeros_like(p)
        div[:-1, :] += vx
        div[1:, :] -= vx
        div[:, :-1] += vy
        div[:, 1:] -= vy
        p -= self._cp * div

        p[self.source_pos] += self.xp.float32(source)


def to_numpy(a):
    """Return ``a`` as a numpy ndarray.

    numpy arrays are returned unchanged. cupy arrays are copied from GPU
    to CPU memory. Any other type raises TypeError.
    """
    if isinstance(a, numpy.ndarray):
        return a
    # Checked by module name so cupy stays an optional dependency.
    if type(a).__module__.split(".")[0] == "cupy":
        return a.get()
    raise TypeError(f"expected a numpy or cupy ndarray, got {type(a).__name__}")
