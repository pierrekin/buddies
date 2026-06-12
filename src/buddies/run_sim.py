"""Run a 2D underwater acoustic FDTD simulation, recording every frame
to a capture file (see buddies.capture)."""

import argparse
import math

import numpy as np

from buddies import capture
from buddies.sim import AcousticFDTD, to_numpy

DEFAULT_SIZE = 1.0  # m
DEFAULT_DX = 0.01  # m
DEFAULT_FREQ = 15_000.0  # Hz
DEFAULT_STEPS = 2000
DEFAULT_AMPLITUDE = 1.0  # Pa
DEFAULT_OUT = "run.npz"

# Below this the wave is too coarsely sampled and dispersion dominates.
MIN_CELLS_PER_WAVELENGTH = 8
# Source periods over which the sine amplitude ramps linearly from 0 to 1,
# limiting the broadband switch-on transient.
SOURCE_RAMP_PERIODS = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=float, default=DEFAULT_SIZE, help="domain edge length (m)")
    ap.add_argument("--dx", type=float, default=DEFAULT_DX, help="cell size (m)")
    ap.add_argument("--freq", type=float, default=DEFAULT_FREQ, help="source frequency (Hz)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="number of timesteps")
    ap.add_argument(
        "--amplitude", type=float, default=DEFAULT_AMPLITUDE, help="source amplitude (Pa)"
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="output file")
    ap.add_argument(
        "--backend",
        choices=("numpy", "cupy"),
        default="numpy",
        help="array backend (cupy requires an NVIDIA CUDA GPU)",
    )
    args = ap.parse_args()

    if args.backend == "cupy":
        try:
            import cupy as xp
        except ImportError:
            ap.error("cupy is not installed (e.g. uv add cupy-cuda12x, on a CUDA machine)")
    else:
        xp = np

    n = round(args.size / args.dx)
    sim = AcousticFDTD(n, n, args.dx, xp=xp)

    wavelength = sim.c / args.freq
    cells_per_wavelength = wavelength / args.dx
    if cells_per_wavelength < MIN_CELLS_PER_WAVELENGTH:
        ap.error(
            f"{args.freq:g} Hz gives {cells_per_wavelength:.1f} cells per wavelength; "
            f"need >= {MIN_CELLS_PER_WAVELENGTH} to resolve the wave "
            f"(shrink --dx or lower --freq)"
        )
    print(
        f"grid {n}x{n} ({args.size:g} m, dx={args.dx:g} m), dt={sim.dt * 1e6:.3f} us, "
        f"lambda={wavelength * 100:.1f} cm ({cells_per_wavelength:.0f} cells), "
        f"{args.steps} steps = {args.steps * sim.dt * 1e3:.2f} ms"
    )

    frames = np.empty((args.steps, n, n), dtype=np.float32)
    for i in range(args.steps):
        t = i * sim.dt
        ramp = min(1.0, t * args.freq / SOURCE_RAMP_PERIODS)
        sim.step(args.amplitude * ramp * math.sin(2 * math.pi * args.freq * t))
        frames[i] = to_numpy(sim.p)

    capture.save(
        args.out,
        capture.Capture(
            frames=frames,
            dt=sim.dt,
            dx=args.dx,
            freq=args.freq,
            c=sim.c,
            amplitude=args.amplitude,
        ),
    )
    print(f"wrote {args.out}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")


if __name__ == "__main__":
    main()
