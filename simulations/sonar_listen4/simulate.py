"""Sonar listen v4: a range-azimuth energy heatmap from one omni ping.

v3 collapsed each receive beam to a single loudest return (one vector). v4
keeps the whole range profile of every beam and renders it as a radial
(range x azimuth) heatmap painted over the scene -- "how much echo energy we
estimate at each direction and range", the natural read-out of a one-ping,
beamform-everything sonar.

Two things make the map honest:
  - Spreading compensation. In this 2D (cylindrical) world a two-way echo's
    amplitude falls as 1/R, so its energy falls as 1/R^2; multiplying the
    energy by R^2 (a time-varying gain) undoes that, so a far wall and a near
    target are shown on equal footing instead of the far one fading out.
  - dB scale. Energy is mapped to colour in dB relative to the strongest cell,
    over COLOR_SPAN_DB, the way a sonar display is read.

Scene, array, transmit and boundary are identical to v3; only the read-out
changes. The ground-truth targets/wall/array are drawn back over the heatmap
so the estimate can be compared to the truth."""

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
ANGLES_DEG = range(-25, 26, 1)  # the beams, all formed from one recording
FOCUS_RANGE = 3.4  # m, sharpens each receive beam around the target band

SRC_AMP = 150.0  # Pa at 1 m; high because one omni ping must light a 4 m scene
DETECT_THRESHOLD = 0.2  # of the pre-blank peak, for the transmit leading edge
PEAK_OFFSET_S = 0.75 / FREQ  # envelope peak lags the leading edge by ~3/4 cycle

# Heatmap colour scale: energy this far below the strongest cell fades to cold.
COLOR_SPAN_DB = 30.0
COLD = np.array((50, 50, 140))  # RGB
HOT = np.array((255, 180, 40))
RANGE_BINS = 512
HEAT_ALPHA = 235  # opacity at full energy; scales to 0 as energy fades

# Diffuse sea wall: a rough vertical backdrop behind the targets.
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
    truth = np.zeros((nx, ny, 4), dtype=np.uint8)  # ground-truth overlay, drawn over the heatmap

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
        truth[mask] = color

    # Diffuse sea wall, inset by the boundary width so no rigid cell sits in the
    # absorbing layer (see v3).
    margin = args.sponge_cells
    kernel_cells = max(1, round(BUMP_WIDTH / DX))
    noise = np.random.default_rng(7).random(ny + kernel_cells)
    profile = np.convolve(noise, np.ones(kernel_cells) / kernel_cells, mode="valid")[:ny]
    profile = (profile - profile.min()) / (profile.max() - profile.min())
    bump_cells = np.rint(profile * BUMP_HEIGHT / DX).astype(int)
    wall_cell = round(WALL_X / DX)
    for iy in range(margin, ny - margin):
        rigid[wall_cell - bump_cells[iy] : nx - margin, iy] = True
        truth[wall_cell - bump_cells[iy] : nx - margin, iy] = (90, 80, 70, 220)

    mics = receiver_array(
        (ARRAY_X, CENTER[1] - APERTURE / 2), (ARRAY_X, CENTER[1] + APERTURE / 2), ELEMENTS
    )
    element_y = np.array([m.pos[1] for m in mics])
    for mx, my in (m.pos for m in mics):
        truth[round(mx / DX), round(my / DX)] = (40, 200, 255, 255)

    sim = AcousticFDTD(
        nx, ny, DX, cfl=args.cfl, xp=args.xp, sources=sources, receivers=mics, rigid=rigid,
        **boundaries.make(args, (nx, ny), DX, FREQ),
    )

    steps = args.capped(round(2.3 * SIZE_X / sim.c / sim.dt))
    blank_steps = round(1.5 / sim.c / sim.dt)
    smooth_steps = max(1, round(0.5 / FREQ / sim.dt))

    recordings_dev = args.xp.empty((steps, ELEMENTS), dtype=np.float32)
    frames = out.open((args.nframes(steps), nx, ny))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        recordings_dev[i] = sim.record()
    recordings = to_numpy(recordings_dev)

    def beamform(deg):
        focus = look(FOCUS_RANGE, deg)
        dists = np.hypot(ARRAY_X - focus[0], element_y - focus[1])
        shifts = np.round((dists.max() - dists) / sim.c / sim.dt).astype(int)
        beam = np.zeros(steps, dtype=np.float32)
        for j, s in enumerate(shifts):
            beam[s:] += recordings[: steps - s, j]
        return beam / ELEMENTS, dists.max()

    # Per beam: build the spreading-compensated energy profile over range, on a
    # shared range axis, so the beams stack into one range-azimuth map.
    r_axis = np.linspace(0.0, math.hypot(SIZE_X - ARRAY_X, SIZE_Y / 2), RANGE_BINS)
    energy = np.zeros((len(ANGLES_DEG), RANGE_BINS))
    idx = np.arange(blank_steps, steps)
    for k, deg in enumerate(ANGLES_DEG):
        beamformed, dmax = beamform(deg)
        window = np.abs(beamformed)
        emit = int(np.argmax(window > DETECT_THRESHOLD * window[:blank_steps].max()))
        # Trailing-max envelope over half a pulse, vectorized.
        env = window.copy()
        for s in range(1, smooth_steps + 1):
            env[s:] = np.maximum(env[s:], window[:-s])
        ranges = sim.c * ((idx - emit) * sim.dt - PEAK_OFFSET_S) - dmax
        # Echo energy with the 1/R^2 two-way spreading divided back out.
        e = env[blank_steps:].astype(np.float64) ** 2 * np.clip(ranges, 0.0, None) ** 2
        energy[k] = np.interp(r_axis, ranges, e, left=0.0, right=0.0)

    overlay = _heatmap(X, Y, energy, r_axis)
    np.copyto(overlay, truth, where=truth[..., 3:] > 0)

    for k, deg in enumerate(ANGLES_DEG):
        peak = int(energy[k].argmax())
        db = 10 * math.log10(energy[k, peak] / energy.max()) if energy[k, peak] > 0 else -np.inf
        print(f"{deg:+3d} deg: peak energy at {r_axis[peak]:.3f} m  {db:+6.1f} dB")

    best = int(energy.max(axis=1).argmax())
    # No pos: this is the whole array's formed output, not a point in the field.
    mic = Channel("rx beam (Pa)", kind="scalar", dt=sim.dt)
    mic.values = list(beamform(list(ANGLES_DEG)[best])[0])

    out.finish(
        dt=sim.dt * args.capture_every, dx=DX, c=sim.c, channels=(mic,), overlay=overlay,
    )


def _heatmap(X, Y, energy, r_axis):
    """Rasterize the (azimuth, range) ``energy`` map into a Cartesian RGBA fan
    over the domain: each cell takes the energy of the beam and range bin it
    falls in, coloured in dB below the strongest cell."""
    cx, cy = CENTER
    rg = np.hypot(X - cx, Y - cy)
    theta = np.degrees(np.arctan2(Y - cy, X - cx))
    amin, amax = min(ANGLES_DEG), max(ANGLES_DEG)
    beam = np.clip(np.rint(theta - amin).astype(int), 0, len(ANGLES_DEG) - 1)
    rbin = np.clip(np.rint(rg / (r_axis[1] - r_axis[0])).astype(int), 0, RANGE_BINS - 1)
    in_fan = (theta >= amin - 0.5) & (theta <= amax + 0.5) & (rg <= r_axis[-1])

    e = np.where(in_fan, energy[beam, rbin], 0.0)
    emax = energy.max()
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(np.where(e > 0, e / emax, 1e-30)) if emax > 0 else np.full(e.shape, -np.inf)
    q = np.clip(1 + db / COLOR_SPAN_DB, 0.0, 1.0)

    heat = np.zeros(X.shape[:1] + Y.shape[1:] + (4,), dtype=np.uint8)
    heat[..., :3] = np.rint(COLD + (HOT - COLD) * q[..., None]).astype(np.uint8)
    heat[..., 3] = np.rint(q * HEAT_ALPHA).astype(np.uint8)
    return heat
