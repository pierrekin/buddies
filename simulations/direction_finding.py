"""Passive direction finding: a line array of microphones hears a pulse
from an off-axis source and estimates its bearing from the times of
arrival. The estimate appears as a vector channel at the array center,
pointing toward where the array thinks the source is."""

import math

import numpy as np

from buddies import capture, probe, simargs
from buddies.capture import Channel
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
MICS = 8
ARRAY_X = 0.2  # m
ARRAY_SPAN = (0.3, 0.7)  # m, mic y positions
SOURCE = (0.65, 0.75)  # m
DETECT_THRESHOLD = 0.2  # arrival = first |p| crossing of this x own peak
OUT = "captures/direction_finding.npz"

args = simargs.parse(__doc__, FREQ)
DX = args.dx
STEPS = args.steps(1000)

_tone = tone(FREQ, ramp_periods=1.0)


def pulse(t):
    """One cycle of a 1 Pa @ 1 m tone, then silence."""
    return _tone(t) if t < 1 / FREQ else 0.0


n = round(SIZE / DX)
sim = AcousticFDTD(
    n,
    n,
    DX,
    cfl=args.cfl,
    sources=[Source(pos=SOURCE, waveform=pulse)],
    damping=edge_sponge((n, n), DX),
)

mic_pos = [(ARRAY_X, y) for y in np.linspace(*ARRAY_SPAN, MICS)]
lights = [Channel(f"m{j}", kind="color", dt=sim.dt, pos=p) for j, p in enumerate(mic_pos)]

recordings = np.empty((STEPS, MICS), dtype=np.float32)
frames = np.empty((args.nframes(STEPS), n, n), dtype=np.float32)
for i in simargs.progress(STEPS):
    sim.step()
    if i % args.capture_every == 0:
        frames[i // args.capture_every] = to_numpy(sim.p)
    for j, p in enumerate(mic_pos):
        recordings[i, j] = probe.pressure(sim, p)
        lights[j].append(recordings[i, j])

# Time of arrival per mic: first crossing of the detection threshold.
envelopes = np.abs(recordings)
arrival_steps = np.array(
    [int(np.argmax(envelopes[:, j] > DETECT_THRESHOLD * envelopes[:, j].max())) for j in range(MICS)]
)
arrivals = arrival_steps * sim.dt

# Hyperbolic localization: find the position whose predicted arrival-time
# differences best match the measured ones. A line array cannot tell front
# from back (mirror positions give identical differences), so search only
# the front half-plane.
gx, gy = np.meshgrid(
    np.linspace(ARRAY_X, SIZE, 161), np.linspace(0, SIZE, 201), indexing="ij"
)
dists = np.array([np.hypot(gx - mx, gy - my) for mx, my in mic_pos])
tdoa_pred = (dists - dists[0]) / sim.c
tdoa_meas = arrivals - arrivals[0]
err = ((tdoa_pred - tdoa_meas[:, None, None]) ** 2).sum(axis=0)
best = np.unravel_index(err.argmin(), err.shape)
estimate = (float(gx[best]), float(gy[best]))

center = (ARRAY_X, float(np.mean([p[1] for p in mic_pos])))
bearing = math.atan2(estimate[1] - center[1], estimate[0] - center[0])
true_bearing = math.atan2(SOURCE[1] - center[1], SOURCE[0] - center[0])
print(
    f"position estimate ({estimate[0]:.2f}, {estimate[1]:.2f}) m, true {SOURCE} m; "
    f"bearing {math.degrees(bearing):.1f} deg, true {math.degrees(true_bearing):.1f} deg "
    f"(arrival spread {arrival_steps.max() - arrival_steps.min()} steps)"
)

# The guess: zero until every mic has reported, then a unit vector.
ready = int(arrival_steps.max()) + 1
direction = (math.cos(bearing), math.sin(bearing))
guess = Channel("bearing", kind="vector", dt=sim.dt, pos=center)
guess.values = [(0.0, 0.0)] * ready + [direction] * (STEPS - ready)

capture.save(
    OUT,
    capture.Capture(
        frames=frames,
        dt=sim.dt * args.capture_every,
        dx=DX,
        c=sim.c,
        channels=(guess, *lights),
    ),
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
