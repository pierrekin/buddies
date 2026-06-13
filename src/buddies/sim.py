"""Staggered-grid (Yee) pressure-velocity FDTD for 2D acoustics.

Array-module agnostic: pass ``xp=numpy`` (default) or ``xp=cupy`` to select
the array backend used for all simulation state.

Pressure is defined at cell centers and particle velocity components at cell
faces. Velocities are stored only for interior faces; boundary faces therefore
have zero normal velocity, which makes the domain edges rigid (perfectly
reflective).

Materials are per-cell arrays: ``rigid`` marks cells as perfect reflectors,
``damping`` gives an amplitude attenuation rate in 1/s. ``edge_sponge``
builds the standard damping array for non-reflective domain edges.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy

SOUND_SPEED_SEAWATER = 1500.0  # m/s
DENSITY_SEAWATER = 1000.0  # kg/m^3
# Fraction of the 2D CFL stability limit dx / (c * sqrt(2)) used for dt.
CFL_SAFETY_FACTOR = 0.2375


def timestep(dx, c=SOUND_SPEED_SEAWATER, cfl=CFL_SAFETY_FACTOR):
    """The dt a simulation with these parameters will use."""
    return cfl * dx / (c * math.sqrt(2))


@dataclass(frozen=True)
class Source:
    pos: tuple[float, float]  # (x, y) in meters
    # t in seconds -> volume injection rate in m^2/s (2D, i.e. volume rate
    # per meter of line source).
    waveform: Callable[[float], float]


DEFAULT_SPONGE_CELLS = 15
# One-way amplitude attenuation through the sponge, in nepers.
# 7 nepers = exp(-7) ~ 1e-3 ~ -60 dB.
SPONGE_NEPERS = 7.0


def edge_sponge(shape, dx, cells=DEFAULT_SPONGE_CELLS, c=SOUND_SPEED_SEAWATER):
    """A damping array for ``AcousticFDTD(damping=...)`` that absorbs waves
    over the outermost ``cells`` cells, making the domain edges non-reflective.

    The rate ramps quadratically from 0 to a peak calibrated so a wave
    crossing the layer at speed ``c`` is attenuated by SPONGE_NEPERS.
    """
    nx, ny = shape
    ix, iy = numpy.arange(nx), numpy.arange(ny)
    edge_dist = numpy.minimum.outer(
        numpy.minimum(ix, nx - 1 - ix), numpy.minimum(iy, ny - 1 - iy)
    )
    depth = numpy.clip((cells - edge_dist) / cells, 0.0, 1.0)
    # One-way nepers = integral of rate over the crossing time
    # = peak * (cells * dx / c) / 3 for a quadratic ramp.
    peak = SPONGE_NEPERS * 3 * c / (cells * dx)
    return (peak * depth**2).astype(numpy.float32)


def tone(
    freq,
    amplitude=1.0,
    at=1.0,
    delay=0.0,
    ramp_periods=2.0,
    c=SOUND_SPEED_SEAWATER,
    rho=DENSITY_SEAWATER,
):
    """A sine waveform calibrated so the far-field pressure amplitude is
    ``amplitude`` Pa at ``at`` meters from the source, ramping in over
    ``ramp_periods`` periods to limit the broadband switch-on transient.

    The waveform is silent before ``delay`` seconds.

    Calibration assumes 2D free-field spreading, so it holds in open water;
    walls and reflectors add their own contributions on top.
    """
    omega = 2 * math.pi * freq
    # 2D Helmholtz Green's function far field for a source of volume-rate
    # amplitude W: |p(r)| = (rho * omega * W / 4) * sqrt(2 / (pi * k * r)).
    # Solved for W given |p(at)| = amplitude.
    w_peak = 4 * amplitude / (rho * omega) * math.sqrt(math.pi * (omega / c) * at / 2)

    def waveform(t):
        t -= delay
        if t < 0:
            return 0.0
        ramp = min(1.0, t * freq / ramp_periods)
        return w_peak * ramp * math.sin(omega * t)

    return waveform


def array(start, end, n, focus, waveform, c=SOUND_SPEED_SEAWATER):
    """``n`` sources evenly spaced on the line from ``start`` to ``end``
    (meters), with firing delays chosen so every element's wavefront arrives
    at ``focus`` simultaneously.

    ``waveform`` is called with each element's delay in seconds and must
    return a Source waveform that is silent before that delay.
    """
    positions = [
        (
            start[0] + (end[0] - start[0]) * i / (n - 1),
            start[1] + (end[1] - start[1]) * i / (n - 1),
        )
        for i in range(n)
    ]
    dists = [math.hypot(px - focus[0], py - focus[1]) for px, py in positions]
    farthest = max(dists)
    return [
        Source(pos=pos, waveform=waveform((farthest - dist) / c))
        for pos, dist in zip(positions, dists)
    ]


class AcousticFDTD:
    def __init__(
        self,
        nx,
        ny,
        dx,
        c=SOUND_SPEED_SEAWATER,
        rho=DENSITY_SEAWATER,
        cfl=CFL_SAFETY_FACTOR,
        sources=(),
        rigid=None,
        damping=None,
        xp=numpy,
    ):
        self.xp = xp
        self.nx, self.ny, self.dx = nx, ny, dx
        self.c, self.rho = c, rho
        self.dt = timestep(dx, c, cfl)

        self._open_x = self._open_y = None
        if rigid is not None:
            rigid = xp.asarray(rigid, dtype=bool)
            if rigid.shape != (nx, ny):
                raise ValueError(f"rigid mask shape {rigid.shape} != grid ({nx}, {ny})")
            # Zero the velocity on every face touching a rigid cell.
            self._open_x = xp.asarray(~(rigid[1:, :] | rigid[:-1, :]), dtype=xp.float32)
            self._open_y = xp.asarray(~(rigid[:, 1:] | rigid[:, :-1]), dtype=xp.float32)

        self._damp_p = self._damp_x = self._damp_y = None
        if damping is not None:
            damping = xp.asarray(damping, dtype=xp.float32)
            if damping.shape != (nx, ny):
                raise ValueError(f"damping shape {damping.shape} != grid ({nx}, {ny})")
            # Per-step amplitude factors; faces use the mean of adjacent cells.
            self._damp_p = xp.exp(-damping * self.dt)
            self._damp_x = xp.exp(-(damping[1:, :] + damping[:-1, :]) / 2 * self.dt)
            self._damp_y = xp.exp(-(damping[:, 1:] + damping[:, :-1]) / 2 * self.dt)

        self._sources = []
        for src in sources:
            ix, iy = round(src.pos[0] / dx), round(src.pos[1] / dx)
            if not (0 <= ix < nx and 0 <= iy < ny):
                raise ValueError(f"source at {src.pos} m is outside the {nx}x{ny} grid")
            self._sources.append(((ix, iy), src.waveform))
        # Precomputed for batched injection: one scatter-add per step instead of
        # a Python loop of N scalar writes (N kernel launches on the GPU).
        self._waveforms = [waveform for _, waveform in self._sources]
        self._src_ix = xp.asarray([cell[0] for cell, _ in self._sources], dtype=int)
        self._src_iy = xp.asarray([cell[1] for cell, _ in self._sources], dtype=int)

        self._step_count = 0
        self.p = xp.zeros((nx, ny), dtype=xp.float32)
        self.vx = xp.zeros((nx - 1, ny), dtype=xp.float32)
        self.vy = xp.zeros((nx, ny - 1), dtype=xp.float32)

        self._cv = xp.float32(self.dt / (rho * dx))
        self._cp = xp.float32(rho * c * c * self.dt / dx)
        # Continuity-equation source term: a volume rate q (m^2/s) injected
        # into one cell raises p by rho c^2 q dt / dx^2, making the radiated
        # field independent of dt and dx.
        self._cq = xp.float32(rho * c * c * self.dt / dx**2)

    def step(self):
        """Advance one timestep, injecting each source's waveform at its cell."""
        p, vx, vy = self.p, self.vx, self.vy

        # rho dv/dt = -grad(p)
        vx -= self._cv * (p[1:, :] - p[:-1, :])
        vy -= self._cv * (p[:, 1:] - p[:, :-1])
        if self._open_x is not None:
            vx *= self._open_x
            vy *= self._open_y
        if self._damp_x is not None:
            vx *= self._damp_x
            vy *= self._damp_y

        # dp/dt = -rho c^2 div(v); faces outside the grid are rigid walls (v = 0)
        div = self.xp.zeros_like(p)
        div[:-1, :] += vx
        div[1:, :] -= vx
        div[:, :-1] += vy
        div[:, 1:] -= vy
        p -= self._cp * div
        if self._damp_p is not None:
            p *= self._damp_p

        t = self._step_count * self.dt
        if self._waveforms:
            vals = self.xp.asarray(
                [waveform(t) for waveform in self._waveforms], dtype=self.xp.float32
            )
            vals *= self._cq
            self._scatter_add(p, self._src_ix, self._src_iy, vals)
        self._step_count += 1

    def _scatter_add(self, p, ix, iy, vals):
        """Accumulate ``vals`` into ``p[ix, iy]``, summing any duplicate cells.
        One scatter replaces a Python loop of per-source scalar writes (each its
        own kernel launch on the GPU). numpy and cupy spell it differently."""
        if self.xp is numpy:
            numpy.add.at(p, (ix, iy), vals)
        else:
            import cupyx

            cupyx.scatter_add(p, (ix, iy), vals)


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
