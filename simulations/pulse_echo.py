"""A single pulse in open water echoing off a rigid slab."""

import math

import numpy as np

from buddies import capture
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy

SIZE = 1.0  # m
DX = 0.01  # m
FREQ = 15_000.0  # Hz
STEPS = 1000
OUT = "captures/pulse_echo.npz"


def pulse(t):
    """One cycle of a FREQ sine, then silence."""
    return math.sin(2 * math.pi * FREQ * t) if t < 1 / FREQ else 0.0


n = round(SIZE / DX)
rigid = np.zeros((n, n), dtype=bool)
# Vertical slab from (0.70, 0.20) to (0.75, 0.80).
rigid[round(0.70 / DX) : round(0.75 / DX), round(0.20 / DX) : round(0.80 / DX)] = True

sim = AcousticFDTD(
    n,
    n,
    DX,
    sources=[Source(pos=(0.25, 0.5), waveform=pulse)],
    rigid=rigid,
    damping=edge_sponge((n, n), DX),
)

frames = np.empty((STEPS, n, n), dtype=np.float32)
for i in range(STEPS):
    sim.step()
    frames[i] = to_numpy(sim.p)

capture.save(OUT, capture.Capture(frames=frames, dt=sim.dt, dx=DX, c=sim.c))
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
