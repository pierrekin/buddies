"""A swept-beam active sonar: the array fires one beamformed ping per
angle, beamforms the same angle on receive (delay-and-sum of all element
recordings, squaring the sidelobe suppression), and converts the first
echo's time of flight into a range. Each angle's result is a true-scale
vector channel, so the arrow tip should land on the rigid block."""

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

SIZE_X = 2.0  # m
SIZE_Y = 1.0  # m
DX = 0.01  # m
FREQ = 15_000.0  # Hz
ELEMENTS = 16
ARRAY_X = 0.15  # m
APERTURE = 0.3  # m
ANGLES_DEG = range(-40, 41, 2)
PING_STEPS = 1500  # round trip to the block is ~1270 steps
# Ignore the mic until the transmit has fully passed. At ±40 deg steering
# the element delay spread stretches the transmit to ~170 steps plus ring.
BLANK_STEPS = 320
FOCUS_RANGE = 1.05  # m, sharpens the beam around the block's range
DETECT_THRESHOLD = 0.2  # of a window's transmit peak, for the emit edge
# An echo is a rise of the smoothed envelope above its running minimum:
# the transmit reverb only ever decays, so a rise means a reflection.
RISE_FACTOR = 2.5
MIN_ECHO = 0.05  # Pa, ignore rises in the quiet tail
# Color scale: loudest return is full hot, returns this many dB below the
# loudest fade to cold.
COLOR_SPAN_DB = 30.0
COLD = np.array((50, 50, 140, 230))  # RGBA
HOT = np.array((255, 180, 40, 255))
SMOOTH_STEPS = 30  # trailing-max window, half a pulse, bridges zero crossings
CAPTURE_EVERY = 4
OUT = "captures/sonar_sweep.npz"

STEPS = PING_STEPS * len(ANGLES_DEG)
DT = timestep(DX)
CENTER = (ARRAY_X, SIZE_Y / 2)


def ping_waveform(ping_start):
    """A waveform factory for ``array``: one tone cycle fired at the
    element's beamforming delay, offset by this ping's start time."""

    def factory(d):
        w = tone(FREQ, delay=ping_start + d, ramp_periods=1.0)
        end = ping_start + d + 1 / FREQ
        return lambda t: w(t) if t < end else 0.0

    return factory


nx, ny = round(SIZE_X / DX), round(SIZE_Y / DX)
sources = []
for k, deg in enumerate(ANGLES_DEG):
    a = math.radians(deg)
    sources += array(
        start=(ARRAY_X, CENTER[1] - APERTURE / 2),
        end=(ARRAY_X, CENTER[1] + APERTURE / 2),
        n=ELEMENTS,
        focus=(CENTER[0] + FOCUS_RANGE * math.cos(a), CENTER[1] + FOCUS_RANGE * math.sin(a)),
        waveform=ping_waveform(k * PING_STEPS * DT),
    )

rigid = np.zeros((nx, ny), dtype=bool)
overlay = np.zeros((nx, ny, 4), dtype=np.uint8)
# 12x12 cm block in the bottom half, ~1.06 m from the array at ~-14 deg.
block = (slice(119, 131), slice(17, 29))
rigid[block] = True
overlay[block] = (140, 110, 70, 220)  # RGBA

sim = AcousticFDTD(
    nx, ny, DX, sources=sources, rigid=rigid, damping=edge_sponge((nx, ny), DX)
)

element_y = np.linspace(CENTER[1] - APERTURE / 2, CENTER[1] + APERTURE / 2, ELEMENTS)
recordings = np.empty((STEPS, ELEMENTS), dtype=np.float32)
frames = np.empty((STEPS // CAPTURE_EVERY, nx, ny), dtype=np.float32)
for i in range(STEPS):
    sim.step()
    if i % CAPTURE_EVERY == 0:
        frames[i // CAPTURE_EVERY] = to_numpy(sim.p)
    for j, ey in enumerate(element_y):
        recordings[i, j] = probe.pressure(sim, (ARRAY_X, ey))

# Per ping: delay-and-sum the element recordings with the ping's own
# delays, find the transmit's leading edge and the first echo after
# blanking, then range = c * time difference / 2 minus the alignment
# offset (echoes aligned this way arrive (dmax - FOCUS_RANGE)/c late).
rx = np.empty(STEPS, dtype=np.float32)
results = []  # (deg, dist, loudness, ready step)
for k, deg in enumerate(ANGLES_DEG):
    a = math.radians(deg)
    focus = (CENTER[0] + FOCUS_RANGE * math.cos(a), CENTER[1] + FOCUS_RANGE * math.sin(a))
    dists = np.hypot(ARRAY_X - focus[0], element_y - focus[1])
    shifts = np.round((dists.max() - dists) / sim.c / sim.dt).astype(int)

    win = recordings[k * PING_STEPS : (k + 1) * PING_STEPS]
    beamformed = np.zeros(PING_STEPS, dtype=np.float32)
    for j, s in enumerate(shifts):
        beamformed[s:] += win[: PING_STEPS - s, j]
    beamformed /= ELEMENTS
    rx[k * PING_STEPS : (k + 1) * PING_STEPS] = beamformed

    window = np.abs(beamformed)
    emit = int(np.argmax(window > DETECT_THRESHOLD * window[:BLANK_STEPS].max()))
    env = np.array(
        [window[max(0, i - SMOOTH_STEPS) : i + 1].max() for i in range(len(window))]
    )
    listen = env[BLANK_STEPS:]
    rises = (listen > RISE_FACTOR * np.minimum.accumulate(listen)) & (listen > MIN_ECHO)
    echo_rel = int(np.argmax(rises)) if rises.any() else None

    if echo_rel is None:
        results.append((deg, None, 0.0, 0))
    else:
        dist = sim.c * (BLANK_STEPS + echo_rel - emit) * sim.dt / 2 - (
            dists.max() - FOCUS_RANGE
        )
        loudness = float(listen[echo_rel : echo_rel + 2 * SMOOTH_STEPS].max())
        results.append((deg, dist, loudness, (k + 1) * PING_STEPS))

loudest = max(loudness for _, dist, loudness, _ in results if dist is not None)
depth_channels = []
for deg, dist, loudness, ready in results:
    if dist is None:
        values = [(0.0, 0.0)] * STEPS
        color = None
        print(f"{deg:+3d} deg: no echo")
    else:
        a = math.radians(deg)
        vec = (dist * math.cos(a), dist * math.sin(a))
        values = [(0.0, 0.0)] * ready + [vec] * (STEPS - ready)
        db = 20 * math.log10(loudness / loudest)
        q = max(0.0, 1 + db / COLOR_SPAN_DB)
        color = tuple(int(v) for v in np.rint(COLD + (HOT - COLD) * q))
        print(f"{deg:+3d} deg: range {dist:.3f} m  {db:+6.1f} dB")

    ch = Channel("", kind="vector", dt=sim.dt, pos=CENTER, scale=1.0, color=color)
    ch.values = values
    depth_channels.append(ch)

mic = Channel("rx beam (Pa)", kind="scalar", dt=sim.dt, pos=CENTER)
mic.values = list(rx)

capture.save(
    OUT,
    capture.Capture(
        frames=frames,
        dt=sim.dt * CAPTURE_EVERY,
        dx=DX,
        c=sim.c,
        channels=(mic, *depth_channels),
        overlay=overlay,
    ),
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
