"""Sonar in miniature: a pulse in open water, a rigid slab, and a mic that
hears the outgoing pulse and then its echo."""

import numpy as np

from buddies import capture, probe
from buddies.capture import Channel
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
DX = 0.01  # m
FREQ = 15_000.0  # Hz
STEPS = 1000
MIC = (0.4, 0.5)  # m
OUT = "captures/echo_mic.npz"

_tone = tone(FREQ, ramp_periods=1.0)


def pulse(t):
    """One cycle of a 1 Pa @ 1 m tone, then silence."""
    return _tone(t) if t < 1 / FREQ else 0.0


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

mic = Channel("mic (Pa)", kind="scalar", dt=sim.dt, pos=MIC)
frames = np.empty((STEPS, n, n), dtype=np.float32)
for i in range(STEPS):
    sim.step()
    frames[i] = to_numpy(sim.p)
    mic.append(probe.pressure(sim, MIC))

capture.save(
    OUT, capture.Capture(frames=frames, dt=sim.dt, dx=DX, c=sim.c, channels=(mic,))
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
