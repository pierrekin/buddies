"""A single pulse in open water echoing off a rigid slab."""

import numpy as np

from buddies import capture
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
DX = 0.01  # m
FREQ = 15_000.0  # Hz
STEPS = 1000
OUT = "captures/pulse_echo.npz"

_tone = tone(FREQ, ramp_periods=1.0)


def pulse(t):
    """One cycle of a 1 Pa @ 1 m tone, then silence."""
    return _tone(t) if t < 1 / FREQ else 0.0


SLAB_COLOR = (140, 110, 70, 220)  # RGBA

n = round(SIZE / DX)
rigid = np.zeros((n, n), dtype=bool)
overlay = np.zeros((n, n, 4), dtype=np.uint8)
# Vertical slab from (0.70, 0.20) to (0.75, 0.80).
slab = (slice(round(0.70 / DX), round(0.75 / DX)), slice(round(0.20 / DX), round(0.80 / DX)))
rigid[slab] = True
overlay[slab] = SLAB_COLOR

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

capture.save(
    OUT, capture.Capture(frames=frames, dt=sim.dt, dx=DX, c=sim.c, overlay=overlay)
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
