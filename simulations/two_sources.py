"""Two tone sources of different frequencies in a 1x1 m tank."""

import numpy as np

from buddies import capture, simargs
from buddies.sim import AcousticFDTD, Source, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz; --resolution is relative to this source's wavelength
OUT = "captures/two_sources.npz"

args = simargs.parse(__doc__, FREQ)
DX = args.dx
STEPS = args.steps(1000)

n = round(SIZE / DX)
sim = AcousticFDTD(
    n,
    n,
    DX,
    cfl=args.cfl, xp=args.xp,
    sources=[
        Source(pos=(0.3, 0.3), waveform=tone(FREQ)),
        Source(pos=(0.7, 0.7), waveform=tone(22_000.0, amplitude=0.5)),
    ],
)

frames = np.empty((args.nframes(STEPS), n, n), dtype=np.float32)
for i in simargs.progress(STEPS):
    sim.step()
    if i % args.capture_every == 0:
        frames[i // args.capture_every] = to_numpy(sim.p)

capture.save(
    OUT,
    capture.Capture(frames=frames, dt=sim.dt * args.capture_every, dx=DX, c=sim.c),
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
