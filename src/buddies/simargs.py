"""The quality/cost knobs for each pipeline stage.

Simulate (physics):
  --resolution     cells per wavelength (accuracy of the wave itself)
  --cfl            fraction of the 2D CFL stability limit used for dt
  --capture-every  record every Nth step to the master (its temporal rate)
  --gpu            run on the GPU (cupy) instead of the CPU (numpy)

Process (perception):
  --decimate       keep every Nth master frame for playback
  --percentile     |p| percentile that maps to the display's full color range

Simulations state step-count constants (durations, windows) in units of
their default timestep; ``Args.steps`` rescales them so simulated time spans
stay fixed when the knobs change. At default knobs every quantity is identical
to the constants as written.
"""

from dataclasses import dataclass

from tqdm import tqdm

from buddies.backend import get_backend
from buddies.sim import CFL_SAFETY_FACTOR, SOUND_SPEED_SEAWATER, timestep

DEFAULT_RESOLUTION = 10.0  # cells per wavelength
DEFAULT_PERCENTILE = 99.5


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
    default_dt: float  # s, dt at the simulation's default knobs
    xp: object  # the array module: numpy (CPU) or cupy (GPU)

    def steps(self, n):
        """Rescale a step count written for the default knobs so it spans the
        same simulated time at the current ones."""
        return round(n * self.default_dt / self.dt)

    def nframes(self, steps):
        """How many frames a run of ``steps`` captures at this stride."""
        return -(-steps // self.capture_every)


@dataclass(frozen=True)
class ProcessArgs:
    decimate: int
    percentile: float


def add_sim_args(ap, resolution=DEFAULT_RESOLUTION, cfl=CFL_SAFETY_FACTOR, capture_every=1):
    """Add the simulate-stage knobs, with this simulation's defaults."""
    ap.add_argument("--resolution", type=float, default=resolution, help="cells per wavelength")
    ap.add_argument("--cfl", type=float, default=cfl, help="fraction of the CFL stability limit")
    ap.add_argument("--capture-every", type=int, default=capture_every, help="record every Nth step")
    ap.add_argument("--gpu", action="store_true", help="run on the GPU via cupy")
    # Stash the baseline so ``default_dt`` (and thus step rescaling) stays tied
    # to the simulation's defaults rather than any command-line overrides.
    ap.set_defaults(_baseline_resolution=resolution, _baseline_cfl=cfl)


def sim_args(ns, freq, c=SOUND_SPEED_SEAWATER):
    """Build ``Args`` from a parsed namespace and the simulation's reference
    frequency."""
    dx = c / freq / ns.resolution
    default_dx = c / freq / ns._baseline_resolution
    return Args(
        resolution=ns.resolution,
        cfl=ns.cfl,
        capture_every=ns.capture_every,
        dx=dx,
        dt=timestep(dx, c, ns.cfl),
        default_dt=timestep(default_dx, c, ns._baseline_cfl),
        xp=get_backend("cupy" if ns.gpu else "numpy"),
    )


def add_process_args(ap, decimate=1, percentile=DEFAULT_PERCENTILE):
    """Add the process-stage knobs, with this simulation's defaults."""
    ap.add_argument("--decimate", type=int, default=decimate, help="keep every Nth frame")
    ap.add_argument("--percentile", type=float, default=percentile, help="|p| percentile mapped to full color range")


def process_args(ns):
    return ProcessArgs(decimate=ns.decimate, percentile=ns.percentile)
