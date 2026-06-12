"""A 15 kHz tone source in open water: sponged edges, no reflections."""

import numpy as np

from buddies import capture
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
DX = 0.01  # m
FREQ = 15_000.0  # Hz
STEPS = 1000
OUT = "captures/open_water.npz"

n = round(SIZE / DX)
sim = AcousticFDTD(
    n,
    n,
    DX,
    sources=[Source(pos=(SIZE / 2, SIZE / 2), waveform=tone(FREQ))],
    damping=edge_sponge((n, n), DX),
)

frames = np.empty((STEPS, n, n), dtype=np.float32)
for i in range(STEPS):
    sim.step()
    frames[i] = to_numpy(sim.p)

capture.save(OUT, capture.Capture(frames=frames, dt=sim.dt, dx=DX, c=sim.c))
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
