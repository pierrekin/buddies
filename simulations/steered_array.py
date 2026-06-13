"""A 16-element line array beam-steered 25 degrees off broadside at 250 kHz,
with the domain large enough to reach the array's far field."""

import math
import time

import numpy as np

from buddies import capture, simargs
from buddies.sim import AcousticFDTD, array, edge_sponge, to_numpy, tone

C = 1500.0  # m/s
FREQ = 250_000.0  # Hz
WAVELENGTH = C / FREQ  # 6 mm
SIZE = 0.7  # m; far field starts at aperture^2 / wavelength ~ 0.34 m
ELEMENTS = 16
SPACING = WAVELENGTH / 2  # at most half a wavelength, else grating lobes
ARRAY_X = 0.05  # m
ANGLE = math.radians(25)  # steering angle off broadside (+x)
# Focus far beyond the domain: the focusing delays degenerate into the
# linear profile of traditional plane-wave steering.
FOCUS_DIST = 100.0  # m
OUT = "captures/steered_array.npz"

# Coarse playback is handled by --capture-every; no need for a fine dt.
args = simargs.parse(__doc__, FREQ, c=C, cfl=0.95, capture_every=8)
DX = args.dx
STEPS = args.steps(2200)

n = round(SIZE / DX)
aperture = (ELEMENTS - 1) * SPACING
cy = SIZE / 2
sim = AcousticFDTD(
    n,
    n,
    DX,
    c=C,
    cfl=args.cfl,
    sources=array(
        start=(ARRAY_X, cy - aperture / 2),
        end=(ARRAY_X, cy + aperture / 2),
        n=ELEMENTS,
        focus=(ARRAY_X + FOCUS_DIST * math.cos(ANGLE), cy + FOCUS_DIST * math.sin(ANGLE)),
        waveform=lambda d: tone(FREQ, delay=d),
        c=C,
    ),
    damping=edge_sponge((n, n), DX, c=C),
)

print(f"grid {n}x{n}, dt={sim.dt * 1e9:.1f} ns, {STEPS} steps = {STEPS * sim.dt * 1e6:.0f} us")
frames = np.empty((args.nframes(STEPS), n, n), dtype=np.float32)
t0 = time.perf_counter()
for i in range(STEPS):
    sim.step()
    if i % args.capture_every == 0:
        frames[i // args.capture_every] = to_numpy(sim.p)
elapsed = time.perf_counter() - t0
print(f"{elapsed:.1f} s ({STEPS / elapsed:.0f} steps/s)")

capture.save(
    OUT,
    capture.Capture(frames=frames, dt=sim.dt * args.capture_every, dx=DX, c=sim.c),
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
