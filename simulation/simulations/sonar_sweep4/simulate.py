"""Sonar v4: rounded targets instead of axis-aligned blocks.

v1-v3 built every object out of rectangular cell slices, so a beam grazing a
target's side hit a flat facet square to the grid. Real objects are curved, and
a curved face spreads the specular return across steering angles instead of
lighting up one. This version adds two mask helpers over a physical coordinate
grid -- an ellipse and a capsule (a rectangle capped with semicircles, i.e. a
stadium / lozenge) -- and seeds the scene with one of each in front of the same
bumpy backstop as v3. The ellipse should read as a smooth arc of ranges across
the sweep; the tilted capsule shows a flat broadside flanked by two rounded
ends."""

import math

import numpy as np

from buddies import simargs
from buddies.sim import AcousticFDTD, array, edge_sponge, receiver_array, timestep, to_numpy, tone
from buddies.store import Channel

SIZE_X = 2.0  # m
SIZE_Y = 1.5  # m
FREQ = 15_000.0  # Hz
DEFAULTS = {"capture_every": 8}
ELEMENTS = 16
ARRAY_X = 0.15  # m
APERTURE = 0.3  # m
ANGLES_DEG = range(-30, 31, 2)
FOCUS_RANGE = 1.05  # m, sharpens the beam around the target range
DETECT_THRESHOLD = 0.2  # of a window's transmit peak, for the emit edge
MIN_ECHO = 0.05  # Pa, below this a window counts as no echo
# The smoothed envelope peaks ~3/4 of a cycle behind the arrival's leading edge
# (one-cycle ramped pulse); subtracted from the time of flight.
PEAK_OFFSET_S = 0.75 / FREQ
# Color scale: loudest return is full hot, returns this many dB below the
# loudest fade to cold.
COLOR_SPAN_DB = 30.0
COLD = np.array((50, 50, 140, 230))  # RGBA
HOT = np.array((255, 180, 40, 255))
CENTER = (ARRAY_X, SIZE_Y / 2)
# Bumpy backstop, same as v3.
BUMP_HEIGHT = 0.05  # m, ~half a wavelength
BUMP_WIDTH = 0.05  # m, lateral feature size
WALL_X = 1.55  # m, face of the deepest troughs
WALL_THICKNESS = 0.05  # m


def ping_waveform(ping_start):
    """A waveform factory for ``array``: one tone cycle fired at the element's
    beamforming delay, offset by this ping's start time."""

    def factory(d):
        w = tone(FREQ, delay=ping_start + d, ramp_periods=1.0)
        end = ping_start + d + 1 / FREQ
        return lambda t: w(t) if t < end else 0.0

    return factory


def run(args, out):
    DX = args.dx
    ping_steps = args.steps(2400)  # round trip to the backstop at +-40 deg
    blank_steps = args.steps(600)
    smooth_steps = args.steps(30)

    steps = ping_steps * len(ANGLES_DEG)
    dt = timestep(DX, cfl=args.cfl)

    nx, ny = round(SIZE_X / DX), round(SIZE_Y / DX)
    sources = []
    for k, deg in enumerate(ANGLES_DEG):
        a = math.radians(deg)
        sources += array(
            start=(ARRAY_X, CENTER[1] - APERTURE / 2),
            end=(ARRAY_X, CENTER[1] + APERTURE / 2),
            n=ELEMENTS,
            focus=(CENTER[0] + FOCUS_RANGE * math.cos(a), CENTER[1] + FOCUS_RANGE * math.sin(a)),
            waveform=ping_waveform(k * ping_steps * dt),
        )

    rigid = np.zeros((nx, ny), dtype=bool)
    overlay = np.zeros((nx, ny, 4), dtype=np.uint8)

    # Cell-center coordinates in metres. X varies along axis 0 (range), Y along
    # axis 1 (cross-range); broadcasting these gives a (nx, ny) field per shape.
    X = (np.arange(nx)[:, None] + 0.5) * DX
    Y = (np.arange(ny)[None, :] + 0.5) * DX

    def ellipse(cx, cy, rx, ry):
        """Filled axis-aligned ellipse, semi-axes ``rx`` (range) and ``ry``."""
        return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0

    def capsule(p0, p1, r):
        """Filled capsule (stadium): every cell within ``r`` of the segment
        ``p0``->``p1``. A rectangle of half-width ``r`` with semicircular caps,
        so it has a flat broadside and two rounded ends and can be tilted."""
        x0, y0 = p0
        x1, y1 = p1
        sx, sy = x1 - x0, y1 - y0
        length2 = sx * sx + sy * sy
        # Parameter of the nearest point on the segment, clamped to its ends.
        t = np.clip(((X - x0) * sx + (Y - y0) * sy) / length2, 0.0, 1.0)
        return np.hypot(X - (x0 + t * sx), Y - (y0 + t * sy)) <= r

    def place(mask, color):
        rigid[mask] = True
        overlay[mask] = color

    # Ellipse ~1.04 m from the array at ~-13 deg (where v3's lower block sat),
    # 12 cm across the beam by 6 cm deep -- a smoothly curved target whose face
    # slides the specular return across the beams that look at it.
    place(ellipse(1.16, 0.52, 0.03, 0.06), (140, 110, 70, 220))
    # Capsule above the axis, ~0.82 m out at ~+10 deg. Its long axis runs nearly
    # cross-range, so the flat broadside faces back toward the array and lights
    # up the mid steering angles, while the rounded end caps return to the
    # steeper +angles -- the curved-vs-flat contrast the ellipse can't show.
    place(capsule((0.935, 0.84), (0.965, 0.96), 0.025), (110, 140, 70, 220))

    # Full-height bumpy backstop, smoothed uniform noise seeded for reproducibility.
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

    mics = receiver_array(
        (ARRAY_X, CENTER[1] - APERTURE / 2), (ARRAY_X, CENTER[1] + APERTURE / 2), ELEMENTS
    )
    element_y = np.array([m.pos[1] for m in mics])
    for mx, my in (m.pos for m in mics):
        overlay[round(mx / DX), round(my / DX)] = (40, 200, 255, 255)

    sim = AcousticFDTD(
        nx, ny, DX, cfl=args.cfl, xp=args.xp, sources=sources, receivers=mics, rigid=rigid,
        damping=edge_sponge((nx, ny), DX),
    )

    recordings_dev = args.xp.empty((steps, ELEMENTS), dtype=np.float32)
    shot = out.shot("main")
    frames = shot.open((args.nframes(steps), nx, ny))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        recordings_dev[i] = sim.record()
    recordings = to_numpy(recordings_dev)

    # Per ping: delay-and-sum the element recordings with the ping's own delays,
    # find the transmit's leading edge, then take the LOUDEST arrival after
    # blanking. Range = c * time difference / 2 minus the alignment offset
    # (echoes aligned this way arrive (dmax - FOCUS_RANGE)/c late).
    rx = np.empty(steps, dtype=np.float32)
    results = []  # (deg, dist, loudness, ready step)
    for k, deg in enumerate(ANGLES_DEG):
        a = math.radians(deg)
        focus = (CENTER[0] + FOCUS_RANGE * math.cos(a), CENTER[1] + FOCUS_RANGE * math.sin(a))
        dists = np.hypot(ARRAY_X - focus[0], element_y - focus[1])
        shifts = np.round((dists.max() - dists) / sim.c / sim.dt).astype(int)

        win = recordings[k * ping_steps : (k + 1) * ping_steps]
        beamformed = np.zeros(ping_steps, dtype=np.float32)
        for j, s in enumerate(shifts):
            beamformed[s:] += win[: ping_steps - s, j]
        beamformed /= ELEMENTS
        rx[k * ping_steps : (k + 1) * ping_steps] = beamformed

        window = np.abs(beamformed)
        emit = int(np.argmax(window > DETECT_THRESHOLD * window[:blank_steps].max()))
        env = np.array(
            [window[max(0, i - smooth_steps) : i + 1].max() for i in range(len(window))]
        )
        listen = env[blank_steps:]
        loudness = float(listen.max())
        if loudness < MIN_ECHO:
            results.append((deg, None, 0.0, 0))
        else:
            echo_rel = int(listen.argmax())
            dist = (
                sim.c * ((blank_steps + echo_rel - emit) * sim.dt - PEAK_OFFSET_S) / 2
                - (dists.max() - FOCUS_RANGE)
            )
            results.append((deg, dist, loudness, (k + 1) * ping_steps))

    loudest = max((loudness for _, dist, loudness, _ in results if dist is not None), default=None)
    depth_channels = []
    detections = []
    for deg, dist, loudness, ready in results:
        if dist is None:
            values = [(0.0, 0.0)] * steps
            color = None
            print(f"{deg:+3d} deg: no echo")
            detections.append({"deg": int(deg), "range_m": None, "loudness_pa": loudness, "db": None})
        else:
            a = math.radians(deg)
            vec = (dist * math.cos(a), dist * math.sin(a))
            values = [(0.0, 0.0)] * ready + [vec] * (steps - ready)
            db = 20 * math.log10(loudness / loudest)
            q = max(0.0, 1 + db / COLOR_SPAN_DB)
            color = tuple(int(v) for v in np.rint(COLD + (HOT - COLD) * q))
            print(f"{deg:+3d} deg: range {dist:.3f} m  {db:+6.1f} dB")
            detections.append({"deg": int(deg), "range_m": float(dist), "loudness_pa": loudness, "db": float(db)})

        ch = Channel("", kind="vector", dt=sim.dt, pos=CENTER, scale=1.0, color=color)
        ch.values = values
        depth_channels.append(ch)

    mic = Channel("rx beam (Pa)", kind="scalar", dt=sim.dt, pos=CENTER)
    mic.values = list(rx)

    shot.finish(
        channels=(mic, *depth_channels), overlay=overlay,
        extras={"detections": detections},
    )
    out.finish(dt=sim.dt * args.capture_every, dx=DX, c=sim.c)
