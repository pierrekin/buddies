"""Sonar v3: the backstop is a bumpy surface instead of a flat wall.

The bump profile is a smoothed random height field with feature height
and width of about half a wavelength, the regime where a surface stops
being a specular mirror and scatters diffusely. Compared to v2, oblique
angles should now return measurable wall echoes (no more cold edges) at
the cost of a noisier range estimate, since each beam ranges whichever
facet it happens to hit."""

import math

import numpy as np

from buddies import capture, probe, simargs
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
SIZE_Y = 1.5  # m
FREQ = 15_000.0  # Hz
ELEMENTS = 16
ARRAY_X = 0.15  # m
APERTURE = 0.3  # m
ANGLES_DEG = range(-30, 31, 2)
FOCUS_RANGE = 1.05  # m, sharpens the beam around the block's range
DETECT_THRESHOLD = 0.2  # of a window's transmit peak, for the emit edge
MIN_ECHO = 0.05  # Pa, below this a window counts as no echo
# The smoothed envelope peaks ~3/4 of a cycle behind the arrival's leading
# edge (one-cycle ramped pulse); subtracted from the time of flight.
PEAK_OFFSET_S = 0.75 / FREQ
# Color scale: loudest return is full hot, returns this many dB below the
# loudest fade to cold.
COLOR_SPAN_DB = 30.0
COLD = np.array((50, 50, 140, 230))  # RGBA
HOT = np.array((255, 180, 40, 255))
OUT = "captures/sonar_sweep3.npz"

args = simargs.parse(__doc__, FREQ, capture_every=8)
DX = args.dx
PING_STEPS = args.steps(2400)  # round trip to the backstop at +-40 deg
# Ignore the mic until the transmit has fully passed. Loudest-return
# ranging has no reverb rejection, so this must outlast the entire
# transmit tail even at ±40 deg steering (delay spread ~170 steps + ring).
BLANK_STEPS = args.steps(600)
# Trailing-max window, half a pulse, bridges zero crossings.
SMOOTH_STEPS = args.steps(30)

STEPS = PING_STEPS * len(ANGLES_DEG)
DT = timestep(DX, cfl=args.cfl)
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
def cells(lo, hi):
    return slice(round(lo / DX), round(hi / DX))


# 10x10 cm block below the array axis, ~1.0 m from the array at ~-13 deg.
block = (cells(1.11, 1.21), cells(0.47, 0.57))
rigid[block] = True
overlay[block] = (140, 110, 70, 220)  # RGBA
# 8x8 cm block above the array axis, ~0.78 m from the array at ~+11 deg.
block2 = (cells(0.91, 0.99), cells(0.86, 0.94))
rigid[block2] = True
overlay[block2] = (110, 140, 70, 220)
# Full-height bumpy backstop: deepest face at x = 1.55, bumps protruding
# up to BUMP_HEIGHT toward the array. Smoothed uniform noise gives facets
# of ~BUMP_WIDTH lateral size. Seeded so the surface is reproducible.
BUMP_HEIGHT = 0.05  # m, ~half a wavelength
BUMP_WIDTH = 0.05  # m, lateral feature size
WALL_X = 1.55  # m, face of the deepest troughs
WALL_THICKNESS = 0.05  # m

kernel_cells = round(BUMP_WIDTH / DX)
noise = np.random.default_rng(7).random(ny + kernel_cells)
profile = np.convolve(noise, np.ones(kernel_cells) / kernel_cells, mode="valid")[:ny]
profile = (profile - profile.min()) / (profile.max() - profile.min())
bump_cells = np.rint(profile * BUMP_HEIGHT / DX).astype(int)
wall_cell = round(WALL_X / DX)
back_cell = round((WALL_X + WALL_THICKNESS) / DX)
for iy in range(ny):
    rigid[wall_cell - bump_cells[iy] : back_cell, iy] = True
    overlay[wall_cell - bump_cells[iy] : back_cell, iy] = (80, 80, 90, 220)

sim = AcousticFDTD(
    nx, ny, DX, cfl=args.cfl, sources=sources, rigid=rigid, damping=edge_sponge((nx, ny), DX)
)

element_y = np.linspace(CENTER[1] - APERTURE / 2, CENTER[1] + APERTURE / 2, ELEMENTS)
recordings = np.empty((STEPS, ELEMENTS), dtype=np.float32)
frames = np.empty((args.nframes(STEPS), nx, ny), dtype=np.float32)
for i in range(STEPS):
    sim.step()
    if i % args.capture_every == 0:
        frames[i // args.capture_every] = to_numpy(sim.p)
    for j, ey in enumerate(element_y):
        recordings[i, j] = probe.pressure(sim, (ARRAY_X, ey))

# Per ping: delay-and-sum the element recordings with the ping's own
# delays, find the transmit's leading edge, then take the LOUDEST arrival
# after blanking. Range = c * time difference / 2 minus the alignment
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
    loudness = float(listen.max())
    if loudness < MIN_ECHO:
        results.append((deg, None, 0.0, 0))
    else:
        echo_rel = int(listen.argmax())
        dist = (
            sim.c * ((BLANK_STEPS + echo_rel - emit) * sim.dt - PEAK_OFFSET_S) / 2
            - (dists.max() - FOCUS_RANGE)
        )
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
        dt=sim.dt * args.capture_every,
        dx=DX,
        c=sim.c,
        channels=(mic, *depth_channels),
        overlay=overlay,
    ),
)
print(f"wrote {OUT}: frames {frames.shape}, peak |p| = {np.abs(frames).max():.3f} Pa")
