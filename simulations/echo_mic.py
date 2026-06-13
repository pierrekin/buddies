"""Sonar in miniature: a pulse in open water, a rigid slab, and a mic that
hears the outgoing pulse and then its echo."""

import numpy as np

from buddies import capture, probe, simargs
from buddies.capture import Channel
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
MIC = (0.4, 0.5)  # m
SLAB_COLOR = (140, 110, 70, 220)  # RGBA
OUT = "captures/echo_mic.npz"

args = simargs.parse(__doc__, FREQ)
DX = args.dx
STEPS = args.steps(1000)

_tone = tone(FREQ, ramp_periods=1.0)


def pulse(t):
    """One cycle of a 1 Pa @ 1 m tone, then silence."""
    return _tone(t) if t < 1 / FREQ else 0.0


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
    cfl=args.cfl,
    sources=[Source(pos=(0.25, 0.5), waveform=pulse)],
    rigid=rigid,
    damping=edge_sponge((n, n), DX),
)

mic = Channel("mic (Pa)", kind="scalar", dt=sim.dt, pos=MIC)
frames = np.empty((args.nframes(STEPS), n, n), dtype=np.float32)
for i in simargs.progress(STEPS):
    sim.step()
    if i % args.capture_every == 0:
        frames[i // args.capture_every] = to_numpy(sim.p)
    mic.append(probe.pressure(sim, MIC))

capture.save(
    OUT,
    capture.Capture(
        frames=frames,
        dt=sim.dt * args.capture_every,
        dx=DX,
        c=sim.c,
        channels=(mic,),
        overlay=overlay,
    ),
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
