"""Sonar listen v3: multiple targets at realistic numbers, before a sea wall.

v2 formed a fan from one ping against a single target. v3 keeps the
one-ping receive-beamforming idea but moves to numbers a real high-
frequency imaging sonar would use, and populates the scene so the fan has
something to resolve:

  - 250 kHz (6 mm wavelength) and a 32-element array at half-wavelength
    spacing (3 mm), so the aperture is ~9.3 cm and there are no grating
    lobes. The far-field distance 2 D^2 / lambda is ~2.9 m, so targets sit
    beyond that, where the array's ~lambda/D ~ 3.7 deg beam is meaningful.
  - Three targets: one isolated, plus a closely spaced pair ~0.3 m apart at
    ~3.5 m (~4.9 deg, just wider than the beam) to show the fan resolving
    two returns.
  - A diffuse sea wall built from the same wavelength-scale bumpy height field
    as the sweep backstop: a rough vertical backdrop behind the targets (a
    forward-facing sonar looks at objects in front of a wall, not a floor) that
    scatters rather than mirrors.

The beamformer and ranging are unchanged from v2; only the scene and the
numbers grow up."""

import math

import numpy as np

from buddies import boundaries, simargs
from buddies.sim import (
    SOUND_SPEED_SEAWATER,
    AcousticFDTD,
    Source,
    receiver_array,
    to_numpy,
    tone,
)
from buddies.store import Channel

FREQ = 250_000.0  # Hz
WAVELENGTH = SOUND_SPEED_SEAWATER / FREQ  # 6 mm
ELEMENTS = 32
SPACING = WAVELENGTH / 2  # half-wavelength: no grating lobes
APERTURE = (ELEMENTS - 1) * SPACING  # ~9.3 cm
FAR_FIELD = 2 * APERTURE**2 / WAVELENGTH  # ~2.9 m

SIZE_X = 4.0  # m, range
SIZE_Y = 3.0  # m, cross-range
ARRAY_X = 0.1  # m
CENTER = (ARRAY_X, SIZE_Y / 2)
ANGLES_DEG = range(-25, 26, 1)  # the fan, all formed from one recording
FOCUS_RANGE = 3.4  # m, sharpens each receive beam around the target band

SRC_AMP = 150.0  # Pa at 1 m; high because one omni ping must light a 4 m scene
DETECT_THRESHOLD = 0.2  # of the pre-blank peak, for the transmit leading edge
MIN_ECHO = 0.01  # Pa, below this a beam counts as no echo
PEAK_OFFSET_S = 0.75 / FREQ  # envelope peak lags the leading edge by ~3/4 cycle

COLOR_SPAN_DB = 30.0  # returns this far below the loudest fade from hot to cold
COLD = np.array((50, 50, 140, 230))  # RGBA
HOT = np.array((255, 180, 40, 255))

# Diffuse sea wall: a rough vertical backdrop behind the targets, bumps
# ~wavelength scale so it scatters rather than mirrors.
WALL_X = 3.8  # m, mean range of the wall's near face
BUMP_HEIGHT = WAVELENGTH  # m, protrusion toward the array (in range)
BUMP_WIDTH = WAVELENGTH  # m, lateral (cross-range) feature size

DEFAULTS = {"capture_every": 16, "sponge_cells": 32}


def look(range_m, deg):
    """Point at ``range_m`` and ``deg`` off the array's broadside (meters)."""
    a = math.radians(deg)
    return (CENTER[0] + range_m * math.cos(a), CENTER[1] + range_m * math.sin(a))


def run(args, out):
    DX = args.dx
    nx, ny = round(SIZE_X / DX), round(SIZE_Y / DX)

    burst = tone(FREQ, amplitude=SRC_AMP, ramp_periods=1.0)
    burst_end = 1.0 / FREQ
    sources = [Source(pos=CENTER, waveform=lambda t: burst(t) if t < burst_end else 0.0)]

    rigid = np.zeros((nx, ny), dtype=bool)
    overlay = np.zeros((nx, ny, 4), dtype=np.uint8)

    X = (np.arange(nx)[:, None] + 0.5) * DX
    Y = (np.arange(ny)[None, :] + 0.5) * DX

    def ellipse(center, rx, ry):
        cx, cy = center
        return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0

    # One isolated target and a closely spaced pair, all in the far field.
    targets = [
        (look(3.2, -4), (140, 110, 70, 220)),  # isolated
        (look(3.5, -11), (110, 140, 70, 220)),  # pair, upper
        (look(3.5, -16), (110, 140, 70, 220)),  # pair, lower (~0.3 m away)
    ]
    for center, color in targets:
        mask = ellipse(center, 0.03, 0.03)
        rigid[mask] = True
        overlay[mask] = color

    # Diffuse sea wall: a wavelength-scale bumpy height field standing in range.
    # Inset by the boundary width on the top, bottom, and far edges so no rigid
    # cell lands in the absorbing layer -- a reflector inside a PML is unstable,
    # and the inset keeps the sponge and PML scenes identical for comparison.
    margin = args.sponge_cells
    kernel_cells = max(1, round(BUMP_WIDTH / DX))
    noise = np.random.default_rng(7).random(ny + kernel_cells)
    profile = np.convolve(noise, np.ones(kernel_cells) / kernel_cells, mode="valid")[:ny]
    profile = (profile - profile.min()) / (profile.max() - profile.min())
    bump_cells = np.rint(profile * BUMP_HEIGHT / DX).astype(int)
    wall_cell = round(WALL_X / DX)
    for iy in range(margin, ny - margin):
        rigid[wall_cell - bump_cells[iy] : nx - margin, iy] = True
        overlay[wall_cell - bump_cells[iy] : nx - margin, iy] = (90, 80, 70, 220)

    mics = receiver_array(
        (ARRAY_X, CENTER[1] - APERTURE / 2), (ARRAY_X, CENTER[1] + APERTURE / 2), ELEMENTS
    )
    element_y = np.array([m.pos[1] for m in mics])
    for mx, my in (m.pos for m in mics):
        overlay[round(mx / DX), round(my / DX)] = (40, 200, 255, 255)

    sim = AcousticFDTD(
        nx, ny, DX, cfl=args.cfl, xp=args.xp, sources=sources, receivers=mics, rigid=rigid,
        **boundaries.make(args, (nx, ny), DX, FREQ),
    )

    # Step counts from physical path lengths: cover the two-way trip across the
    # domain, blank past the direct blast, smooth over half a pulse.
    steps = args.capped(round(2.3 * SIZE_X / sim.c / sim.dt))
    blank_steps = round(1.5 / sim.c / sim.dt)
    smooth_steps = max(1, round(0.5 / FREQ / sim.dt))

    recordings_dev = args.xp.empty((steps, ELEMENTS), dtype=np.float32)
    shot = out.shot("main")
    frames = shot.open((args.nframes(steps), nx, ny))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        recordings_dev[i] = sim.record()
    recordings = to_numpy(recordings_dev)

    def beamform(deg):
        a = math.radians(deg)
        focus = look(FOCUS_RANGE, deg)
        dists = np.hypot(ARRAY_X - focus[0], element_y - focus[1])
        shifts = np.round((dists.max() - dists) / sim.c / sim.dt).astype(int)
        beam = np.zeros(steps, dtype=np.float32)
        for j, s in enumerate(shifts):
            beam[s:] += recordings[: steps - s, j]
        return beam / ELEMENTS, dists.max()

    results = []  # (deg, dist, loudness, arrival step)
    for deg in ANGLES_DEG:
        beamformed, dmax = beamform(deg)
        window = np.abs(beamformed)
        emit = int(np.argmax(window > DETECT_THRESHOLD * window[:blank_steps].max()))
        env = np.array([window[max(0, i - smooth_steps) : i + 1].max() for i in range(len(window))])
        listen = env[blank_steps:]
        loudness = float(listen.max()) if listen.size else 0.0
        if loudness < MIN_ECHO:
            results.append((deg, None, loudness, 0))
        else:
            echo_rel = int(listen.argmax())
            arrival = blank_steps + echo_rel
            dist = sim.c * ((arrival - emit) * sim.dt - PEAK_OFFSET_S) - dmax
            results.append((deg, dist, loudness, arrival))

    loudest = max((loudness for _, dist, loudness, _ in results if dist is not None), default=None)
    depth_channels = []
    detections = []
    for deg, dist, loudness, arrival in results:
        if dist is None:
            values = [(0.0, 0.0)] * steps
            color = None
            print(f"{deg:+3d} deg: no echo")
            detections.append({"deg": int(deg), "range_m": None, "loudness_pa": loudness, "db": None})
        else:
            a = math.radians(deg)
            vec = (dist * math.cos(a), dist * math.sin(a))
            values = [(0.0, 0.0)] * arrival + [vec] * (steps - arrival)
            db = 20 * math.log10(loudness / loudest)
            q = max(0.0, 1 + db / COLOR_SPAN_DB)
            color = tuple(int(v) for v in np.rint(COLD + (HOT - COLD) * q))
            print(f"{deg:+3d} deg: range {dist:.3f} m  {db:+6.1f} dB")
            detections.append({"deg": int(deg), "range_m": float(dist), "loudness_pa": loudness, "db": float(db)})

        ch = Channel("", kind="vector", dt=sim.dt, pos=CENTER, scale=1.0, color=color)
        ch.values = values
        depth_channels.append(ch)

    best = max(results, key=lambda r: r[2])
    # No pos: this is the whole array's formed output, not a point in the field.
    mic = Channel("rx beam (Pa)", kind="scalar", dt=sim.dt)
    mic.values = list(beamform(best[0])[0])

    shot.finish(
        channels=(mic, *depth_channels), overlay=overlay,
        extras={"detections": detections},
    )
    out.finish(dt=sim.dt * args.capture_every, dx=DX, c=sim.c)
