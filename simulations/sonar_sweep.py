"""A swept-beam active sonar: the array fires one beamformed ping per
angle, listens for the first echo at its center after transmit blanking,
and converts the time of flight into a range. Each angle's result is a
true-scale vector channel, so the arrow tips should land on the rigid
obstacles in the tank."""

import math

import numpy as np

from buddies import capture, probe
from buddies.capture import Channel
from buddies.sim import (
    AcousticFDTD,
    Source,
    array,
    edge_sponge,
    timestep,
    to_numpy,
    tone,
)

SIZE = 1.0  # m
DX = 0.01  # m
FREQ = 15_000.0  # Hz
ELEMENTS = 16
ARRAY_X = 0.15  # m
APERTURE = 0.3  # m
ANGLES_DEG = range(-40, 41, 10)
PING_STEPS = 700  # round trip to the farthest obstacle is ~550 steps
# Ignore the mic until the transmit has fully passed. At ±40 deg steering
# the element delay spread stretches the transmit to ~170 steps plus ring.
BLANK_STEPS = 320
FOCUS_RANGE = 0.45  # m, sharpens the beam around the obstacle ranges
DETECT_THRESHOLD = 0.2  # of a window's transmit peak, for the emit edge
# An echo is a rise of the smoothed envelope above its running minimum:
# the transmit reverb only ever decays, so a rise means a reflection.
RISE_FACTOR = 2.5
MIN_ECHO = 0.1  # Pa, ignore rises in the quiet tail
SMOOTH_STEPS = 30  # trailing-max window, half a pulse, bridges zero crossings
CAPTURE_EVERY = 4
OUT = "captures/sonar_sweep.npz"

STEPS = PING_STEPS * len(ANGLES_DEG)
DT = timestep(DX)
CENTER = (ARRAY_X, 0.5)


def ping_waveform(ping_start):
    """A waveform factory for ``array``: one tone cycle fired at the
    element's beamforming delay, offset by this ping's start time."""

    def factory(d):
        w = tone(FREQ, delay=ping_start + d, ramp_periods=1.0)
        end = ping_start + d + 1 / FREQ
        return lambda t: w(t) if t < end else 0.0

    return factory


n = round(SIZE / DX)
sources = []
for k, deg in enumerate(ANGLES_DEG):
    a = math.radians(deg)
    sources += array(
        start=(ARRAY_X, 0.5 - APERTURE / 2),
        end=(ARRAY_X, 0.5 + APERTURE / 2),
        n=ELEMENTS,
        focus=(CENTER[0] + FOCUS_RANGE * math.cos(a), CENTER[1] + FOCUS_RANGE * math.sin(a)),
        waveform=ping_waveform(k * PING_STEPS * DT),
    )

rigid = np.zeros((n, n), dtype=bool)
rigid[45:53, 66:74] = True  # block A: ~0.35 m from the array at ~+30 deg
rigid[58:68, 34:40] = True  # block B: ~0.46 m at ~-20 deg

sim = AcousticFDTD(n, n, DX, sources=sources, rigid=rigid, damping=edge_sponge((n, n), DX))

mic = Channel("mic (Pa)", kind="scalar", dt=sim.dt, pos=CENTER)
frames = np.empty((STEPS // CAPTURE_EVERY, n, n), dtype=np.float32)
for i in range(STEPS):
    sim.step()
    if i % CAPTURE_EVERY == 0:
        frames[i // CAPTURE_EVERY] = to_numpy(sim.p)
    mic.append(probe.pressure(sim, CENTER))

# Per ping: leading edge of the transmit, then the first echo after
# blanking. Range = c * time difference / 2.
recording = np.abs(np.asarray(mic.values))
depth_channels = []
for k, deg in enumerate(ANGLES_DEG):
    window = recording[k * PING_STEPS : (k + 1) * PING_STEPS]
    emit = int(np.argmax(window > DETECT_THRESHOLD * window[:BLANK_STEPS].max()))
    env = np.array(
        [window[max(0, i - SMOOTH_STEPS) : i + 1].max() for i in range(len(window))]
    )
    listen = env[BLANK_STEPS:]
    rises = (listen > RISE_FACTOR * np.minimum.accumulate(listen)) & (listen > MIN_ECHO)
    echo_rel = int(np.argmax(rises)) if rises.any() else None

    a = math.radians(deg)
    if echo_rel is None:
        dist = None
        values = [(0.0, 0.0)] * STEPS
    else:
        dist = sim.c * (BLANK_STEPS + echo_rel - emit) * sim.dt / 2
        ready = (k + 1) * PING_STEPS
        vec = (dist * math.cos(a), dist * math.sin(a))
        values = [(0.0, 0.0)] * ready + [vec] * (STEPS - ready)
    print(f"{deg:+3d} deg: " + (f"range {dist:.3f} m" if dist else "no echo"))

    ch = Channel("", kind="vector", dt=sim.dt, pos=CENTER, scale=1.0)
    ch.values = values
    depth_channels.append(ch)

capture.save(
    OUT,
    capture.Capture(
        frames=frames,
        dt=sim.dt * CAPTURE_EVERY,
        dx=DX,
        c=sim.c,
        channels=(mic, *depth_channels),
    ),
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
