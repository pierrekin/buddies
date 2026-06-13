"""A 15 kHz tone source in open water: sponged edges, no reflections."""

import numpy as np

from buddies import capture, simargs
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
OUT = "captures/open_water.npz"

args = simargs.parse(__doc__, FREQ)
DX = args.dx
STEPS = args.steps(1000)

n = round(SIZE / DX)
sim = AcousticFDTD(
    n,
    n,
    DX,
    cfl=args.cfl, xp=args.xp,
    sources=[Source(pos=(SIZE / 2, SIZE / 2), waveform=tone(FREQ))],
    damping=edge_sponge((n, n), DX),
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
