"""Stateless probes that read physical quantities from live simulation
state. Positions are in meters, like sources."""


def _cell(sim, pos):
    ix, iy = round(pos[0] / sim.dx), round(pos[1] / sim.dx)
    if not (0 <= ix < sim.nx and 0 <= iy < sim.ny):
        raise ValueError(f"probe at {pos} m is outside the {sim.nx}x{sim.ny} grid")
    return ix, iy


def pressure(sim, pos):
    """Pressure at a point, in Pa."""
    return float(sim.p[_cell(sim, pos)])


def velocity(sim, pos):
    """Particle velocity (vx, vy) at a point, in m/s.

    The staggered face values around the cell are averaged to its center;
    faces on the domain edge are zero.
    """
    ix, iy = _cell(sim, pos)
    left = float(sim.vx[ix - 1, iy]) if ix > 0 else 0.0
    right = float(sim.vx[ix, iy]) if ix < sim.nx - 1 else 0.0
    down = float(sim.vy[ix, iy - 1]) if iy > 0 else 0.0
    up = float(sim.vy[ix, iy]) if iy < sim.ny - 1 else 0.0
    return ((left + right) / 2, (down + up) / 2)


def intensity(sim, pos):
    """Instantaneous acoustic intensity (ix, iy) at a point, in W/m^2.

    The product p * v: magnitude is energy flux, direction is where the
    energy is flowing.
    """
    p = pressure(sim, pos)
    vx, vy = velocity(sim, pos)
    return (p * vx, p * vy)


def energy(sim):
    """Total acoustic energy in the field, in J per meter of depth:
    p^2 / (2 rho c^2) + rho |v|^2 / 2, summed over the grid."""
    pe = float((sim.p**2).sum()) / (2 * sim.rho * sim.c**2)
    ke = sim.rho / 2 * (float((sim.vx**2).sum()) + float((sim.vy**2).sum()))
    return (pe + ke) * sim.dx**2
