"""Sonar v3: the backstop is a bumpy surface instead of a flat wall.

The bump profile is a smoothed random height field with feature height and
width of about half a wavelength, the regime where a surface stops being a
specular mirror and scatters diffusely. Compared to v2, oblique angles should
now return measurable wall echoes (no more cold edges) at the cost of a noisier
range estimate, since each beam ranges whichever facet it happens to hit."""

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
FOCUS_RANGE = 1.05  # m, sharpens the beam around the block's range
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
# Bumpy backstop: deepest face at WALL_X, bumps protruding up to BUMP_HEIGHT
# toward the array. Smoothed uniform noise gives facets of ~BUMP_WIDTH.
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

    # Smoothed uniform noise, seeded so the surface is reproducible.
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
