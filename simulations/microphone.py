"""A tone source in open water observed by every channel kind: a mic
(scalar plot), an energy-flow arrow (vector), a pressure-tinted marker
(color), and total field energy (scalar, no position)."""

import numpy as np

from buddies import capture, probe, simargs
from buddies.capture import Channel
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
MIC = (0.7, 0.5)  # m
FLOW = (0.35, 0.65)  # m
TINT = (0.5, 0.75)  # m
OUT = "captures/microphone.npz"

args = simargs.parse(__doc__, FREQ)
DX = args.dx
STEPS = args.steps(1000)

n = round(SIZE / DX)
sim = AcousticFDTD(
    n,
    n,
    DX,
    cfl=args.cfl,
    sources=[Source(pos=(SIZE / 2, SIZE / 2), waveform=tone(FREQ))],
    damping=edge_sponge((n, n), DX),
)

mic = Channel("mic (Pa)", kind="scalar", dt=sim.dt, pos=MIC)
flow = Channel("flow", kind="vector", dt=sim.dt, pos=FLOW)
tint = Channel("tint", kind="color", dt=sim.dt, pos=TINT)
energy = Channel("field energy (J/m)", kind="scalar", dt=sim.dt)

frames = np.empty((args.nframes(STEPS), n, n), dtype=np.float32)
for i in simargs.progress(STEPS):
    sim.step()
    if i % args.capture_every == 0:
        frames[i // args.capture_every] = to_numpy(sim.p)
    mic.append(probe.pressure(sim, MIC))
    flow.append(probe.intensity(sim, FLOW))
    tint.append(probe.pressure(sim, TINT))
    energy.append(probe.energy(sim))

capture.save(
    OUT,
    capture.Capture(
        frames=frames,
        dt=sim.dt * args.capture_every,
        dx=DX,
        c=sim.c,
        channels=(mic, flow, tint, energy),
    ),
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
