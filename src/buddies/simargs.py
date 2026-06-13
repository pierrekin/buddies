"""Unified command-line knobs shared by the simulation scripts.

Every script exposes the three quality/cost dials:

--resolution     cells per wavelength (accuracy of the wave itself)
--cfl            fraction of the 2D CFL stability limit used for dt
--capture-every  record every Nth step to the capture file

Scripts state step-count constants (durations, windows) in units of
their default timestep; ``Args.steps`` rescales them so the simulated
time spans stay fixed when the knobs change. At default knob values
every quantity is identical to the constants as written.
"""

import argparse
from dataclasses import dataclass

from tqdm import tqdm

from buddies.sim import CFL_SAFETY_FACTOR, SOUND_SPEED_SEAWATER, timestep

DEFAULT_RESOLUTION = 10.0  # cells per wavelength


def progress(steps):
    """``range(steps)`` wrapped in a progress bar (percentage, rate, ETA)."""
    return tqdm(range(steps), unit="step")


@dataclass(frozen=True)
class Args:
    resolution: float
    cfl: float
    capture_every: int
    dx: float  # m, from the wavelength and resolution
    dt: float  # s, from dx and cfl
    default_dt: float  # s, dt at the script's default knobs

    def steps(self, n):
        """Rescale a step count written for the default knobs so it spans
        the same simulated time at the current ones."""
        return round(n * self.default_dt / self.dt)

    def nframes(self, steps):
        """How many frames a run of ``steps`` captures at this stride."""
        return -(-steps // self.capture_every)


def parse(
    doc,
    freq,
    c=SOUND_SPEED_SEAWATER,
    resolution=DEFAULT_RESOLUTION,
    cfl=CFL_SAFETY_FACTOR,
    capture_every=1,
):
    """Parse the shared knobs; keyword arguments set this script's defaults."""
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument(
        "--resolution", type=float, default=resolution, help="cells per wavelength"
    )
    ap.add_argument(
        "--cfl", type=float, default=cfl, help="fraction of the CFL stability limit"
    )
    ap.add_argument(
        "--capture-every", type=int, default=capture_every, help="record every Nth step"
    )
    ns = ap.parse_args()
    dx = c / freq / ns.resolution
    return Args(
        resolution=ns.resolution,
        cfl=ns.cfl,
        capture_every=ns.capture_every,
        dx=dx,
        dt=timestep(dx, c, ns.cfl),
        default_dt=timestep(c / freq / resolution, c, cfl),
    )
