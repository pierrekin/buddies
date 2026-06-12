"""A 16-element line array focusing its beam on a point in open water."""

import numpy as np

from buddies import capture
from buddies.sim import AcousticFDTD, array, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
DX = 0.01  # m
FREQ = 15_000.0  # Hz
STEPS = 1000
ELEMENTS = 16
ARRAY_START = (0.2, 0.2)  # m
ARRAY_END = (0.2, 0.8)  # m
FOCUS = (0.7, 0.5)  # m
OUT = "captures/beamforming.npz"

n = round(SIZE / DX)
sim = AcousticFDTD(
    n,
    n,
    DX,
    sources=array(
        start=ARRAY_START,
        end=ARRAY_END,
        n=ELEMENTS,
        focus=FOCUS,
        waveform=lambda d: tone(FREQ, delay=d),
    ),
    damping=edge_sponge((n, n), DX),
)

frames = np.empty((STEPS, n, n), dtype=np.float32)
for i in range(STEPS):
    sim.step()
    frames[i] = to_numpy(sim.p)

capture.save(OUT, capture.Capture(frames=frames, dt=sim.dt, dx=DX, c=sim.c))
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
