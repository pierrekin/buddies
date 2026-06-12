"""Two tone sources of different frequencies in a 1x1 m tank."""

import numpy as np

from buddies import capture
from buddies.sim import AcousticFDTD, Source, to_numpy, tone

SIZE = 1.0  # m
DX = 0.01  # m
STEPS = 1000
OUT = "captures/two_sources.npz"

n = round(SIZE / DX)
sim = AcousticFDTD(
    n,
    n,
    DX,
    sources=[
        Source(pos=(0.3, 0.3), waveform=tone(15_000.0)),
        Source(pos=(0.7, 0.7), waveform=tone(22_000.0, amplitude=0.5)),
    ],
)

frames = np.empty((STEPS, n, n), dtype=np.float32)
for i in range(STEPS):
    sim.step()
    frames[i] = to_numpy(sim.p)

capture.save(OUT, capture.Capture(frames=frames, dt=sim.dt, dx=DX, c=sim.c))
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
