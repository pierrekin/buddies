"""Absorbing boundaries, selectable by name so a simulation can swap one for
another without changing the rest of its setup.

``make`` returns the keyword arguments to hand to ``AcousticFDTD``:

- "sponge": an amplitude-damping layer (see ``edge_sponge``), folded into the
  step. Cheap, but reflects at grazing incidence.
- "pml": a convolutional PML (``CPML``) that stays absorbing at grazing angles.
"""

import math

from buddies.sim import SOUND_SPEED_SEAWATER, edge_sponge, timestep

# CPML grading: cubic conductivity ramp targeting this reflection coefficient.
PML_ORDER = 3
PML_REFLECTION = 1e-6


def make(args, shape, dx, freq=None):
    """The ``AcousticFDTD`` keyword args for the boundary ``args.boundary`` asks
    for, sized to ``shape`` cells at spacing ``dx`` (meters). ``freq`` (the
    simulation's reference frequency, Hz) is required for the PML."""
    if args.boundary == "sponge":
        return {"damping": edge_sponge(shape, dx, cells=args.sponge_cells)}
    if args.boundary == "pml":
        if freq is None:
            raise ValueError("the PML boundary needs the simulation frequency")
        dt = timestep(dx, SOUND_SPEED_SEAWATER, args.cfl)
        cpml = CPML(shape, dx, args.sponge_cells, dt, SOUND_SPEED_SEAWATER, freq, args.xp)
        return {"boundary": cpml}
    raise ValueError(f"unknown boundary {args.boundary!r}")


class CPML:
    """Convolutional PML (Roden-Gedney CFS form) for the pressure-velocity FDTD.

    The complex coordinate stretch becomes, in the time domain, a per-step
    memory variable psi alongside each spatial derivative: psi = b*psi + a*D,
    and the field update gains a ``-coef*psi`` term. With the stretch factor
    kappa = 1, the velocity/pressure kernels already compute the un-stretched
    update, so the PML's whole job is to maintain psi (one per derivative) and
    subtract it after each kernel. The conductivity ramps in only over the outer
    ``cells``; the alpha (frequency-shift) term ramps the other way and is what
    keeps absorption working at grazing incidence, where a sponge fails.

    This first cut maintains the psi fields over the full grid. They are
    identity (b=1, a=0) in the interior, so it is correct, but it pays full-grid
    traffic the boundary slabs don't need -- restricting the updates to the
    edge slabs is the obvious optimization.
    """

    def __init__(self, shape, dx, cells, dt, c, freq, xp):
        self.xp = xp
        nx, ny = shape
        self.cells = cells

        # sigma ramps 0 -> sigma_max over the layer; alpha ramps alpha_max -> 0.
        sigma_max = (PML_ORDER + 1) * c * math.log(1.0 / PML_REFLECTION) / (2 * cells * dx)
        alpha_max = math.pi * freq

        def coeffs(length):
            """(b, a) along an axis of ``length`` samples: identity in the
            interior, CPML in the outer ``cells`` at each end."""
            i = xp.arange(length)
            edge_dist = xp.minimum(i, length - 1 - i)
            in_pml = edge_dist < cells
            depth = xp.clip((cells - edge_dist) / cells, 0.0, 1.0)  # 1 at edge -> 0 inward
            sigma = sigma_max * depth**PML_ORDER
            alpha = xp.where(in_pml, alpha_max * (1.0 - depth), 0.0)
            denom = sigma + alpha
            b = xp.exp(-(sigma + alpha) * dt)
            a = xp.where(denom > 0, sigma / denom * (b - 1.0), 0.0)
            return b.astype(xp.float32), a.astype(xp.float32)

        # Profiles for the two staggered grids: faces (velocity) and cell
        # centers (pressure divergence), per axis. [:, None] / [None, :] shape
        # them to broadcast against the 2D fields.
        self._b_fx, self._a_fx = (v[:, None] for v in coeffs(nx - 1))
        self._b_cx, self._a_cx = (v[:, None] for v in coeffs(nx))
        self._b_fy, self._a_fy = (v[None, :] for v in coeffs(ny - 1))
        self._b_cy, self._a_cy = (v[None, :] for v in coeffs(ny))

        # Memory variables, one per spatial derivative in the two updates.
        self._psi_px = xp.zeros((nx - 1, ny), dtype=xp.float32)  # d p / dx (vx update)
        self._psi_py = xp.zeros((nx, ny - 1), dtype=xp.float32)  # d p / dy (vy update)
        self._psi_vx = xp.zeros((nx, ny), dtype=xp.float32)      # d vx / dx (p update)
        self._psi_vy = xp.zeros((nx, ny), dtype=xp.float32)      # d vy / dy (p update)

    def reset(self):
        """Zero the memory accumulators so the boundary is silent again."""
        self._psi_px.fill(0)
        self._psi_py.fill(0)
        self._psi_vx.fill(0)
        self._psi_vy.fill(0)

    def apply_velocity(self, sim):
        """Correct the velocity faces after the un-stretched velocity update."""
        p, cv = sim.p, sim._cv
        self._psi_px = self._b_fx * self._psi_px + self._a_fx * (p[1:, :] - p[:-1, :])
        sim.vx -= cv * self._psi_px
        self._psi_py = self._b_fy * self._psi_py + self._a_fy * (p[:, 1:] - p[:, :-1])
        sim.vy -= cv * self._psi_py

    def apply_pressure(self, sim):
        """Correct the pressure after the un-stretched pressure update."""
        xp, vx, vy, cp = self.xp, sim.vx, sim.vy, sim._cp
        dvx = xp.zeros_like(sim.p)
        dvx[:-1, :] += vx
        dvx[1:, :] -= vx
        self._psi_vx = self._b_cx * self._psi_vx + self._a_cx * dvx
        sim.p -= cp * self._psi_vx
        dvy = xp.zeros_like(sim.p)
        dvy[:, :-1] += vy
        dvy[:, 1:] -= vy
        self._psi_vy = self._b_cy * self._psi_vy + self._a_cy * dvy
        sim.p -= cp * self._psi_vy
