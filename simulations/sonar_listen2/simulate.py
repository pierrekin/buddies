"""Sonar listen v2: a full fan formed from ONE omni ping.

v1 beamformed the single recording to one look angle. The whole point of
receive-side beamforming is that the recording holds every direction at once,
so v2 keeps the identical one-ping simulation and just loops the look angle over
the fan in post -- 31 directions out of a single pulse, where the sweep series
needed 31 separate focused pings. Each beam is delay-and-summed, ranged, and
colored by its return strength, so off-axis beams that don't face the target
fall away and the fan shows where the echo actually came from."""

import math

import numpy as np

from buddies import simargs
from buddies.sim import AcousticFDTD, Source, edge_sponge, receiver_array, to_numpy, tone
from buddies.store import Channel

SIZE_X = 2.0  # m
SIZE_Y = 1.5  # m
FREQ = 15_000.0  # Hz
DEFAULTS = {"capture_every": 8}
ELEMENTS = 16
ARRAY_X = 0.15  # m
APERTURE = 0.3  # m
ANGLES_DEG = range(-30, 31, 2)  # the fan, all formed from one recording
FOCUS_RANGE = 1.05  # m, sharpens each receive beam around the target's range
SRC_AMP = 30.0  # Pa at 1 m; high because one omni ping must light the scene
DETECT_THRESHOLD = 0.2  # of the pre-blank peak, for the transmit leading edge
MIN_ECHO = 0.02  # Pa, below this a beam counts as no echo
# The smoothed envelope peaks ~3/4 of a cycle behind the arrival's leading edge
# (one-cycle ramped pulse); subtracted from the time of flight.
PEAK_OFFSET_S = 0.75 / FREQ
# Color scale: loudest return is full hot, returns this many dB below the
# loudest fade to cold.
COLOR_SPAN_DB = 30.0
COLD = np.array((50, 50, 140, 230))  # RGBA
HOT = np.array((255, 180, 40, 255))
CENTER = (ARRAY_X, SIZE_Y / 2)


def run(args, out):
    DX = args.dx
    steps = args.capped(args.steps(2000))  # one ping: round trip plus listen
    # Ignore the mic until the direct blast has passed. No reverb rejection, so
    # this must outlast the omni pulse and its ring-down but clear a ~1 m echo.
    blank_steps = args.steps(600)
    # Trailing-max window, half a pulse, bridges zero crossings.
    smooth_steps = args.steps(30)

    nx, ny = round(SIZE_X / DX), round(SIZE_Y / DX)

    # One omnidirectional transmitter at the array center: a single point source
    # radiates a cylindrical wave that insonifies the whole scene with one ping.
    burst = tone(FREQ, amplitude=SRC_AMP, ramp_periods=1.0)
    burst_end = 1.0 / FREQ
    sources = [Source(pos=CENTER, waveform=lambda t: burst(t) if t < burst_end else 0.0)]

    rigid = np.zeros((nx, ny), dtype=bool)
    overlay = np.zeros((nx, ny, 4), dtype=np.uint8)

    # Cell-center coordinates in metres (axis 0 = range, axis 1 = cross-range).
    X = (np.arange(nx)[:, None] + 0.5) * DX
    Y = (np.arange(ny)[None, :] + 0.5) * DX

    def ellipse(cx, cy, rx, ry):
        """Filled axis-aligned ellipse, semi-axes ``rx`` (range) and ``ry``."""
        return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0

    # One rounded target ~1.04 m from the array at ~-13 deg. The fan should peak
    # on the beams that look at it and fade away from them.
    target = ellipse(1.16, 0.52, 0.05, 0.08)
    rigid[target] = True
    overlay[target] = (140, 110, 70, 220)

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
    frames = out.open((args.nframes(steps), nx, ny))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        recordings_dev[i] = sim.record()
    recordings = to_numpy(recordings_dev)

    def beamform(deg):
        """Delay-and-sum the one recording toward ``deg``. The shifts are the
        sweep's transmit focusing delays, reused on receive (reciprocity)."""
        a = math.radians(deg)
        focus = (CENTER[0] + FOCUS_RANGE * math.cos(a), CENTER[1] + FOCUS_RANGE * math.sin(a))
        dists = np.hypot(ARRAY_X - focus[0], element_y - focus[1])
        shifts = np.round((dists.max() - dists) / sim.c / sim.dt).astype(int)
        beam = np.zeros(steps, dtype=np.float32)
        for j, s in enumerate(shifts):
            beam[s:] += recordings[: steps - s, j]
        return beam / ELEMENTS, dists.max()

    # Form every direction from the SAME recording. Per beam: find the direct
    # blast's leading edge, then the loudest arrival after blanking, and range it.
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
            # One omni transmit: path at the peak is range + dists.max(), no
            # transmit-side focusing term and no halving (see v1).
            dist = sim.c * ((arrival - emit) * sim.dt - PEAK_OFFSET_S) - dmax
            results.append((deg, dist, loudness, arrival))

    loudest = max((loudness for _, dist, loudness, _ in results if dist is not None), default=None)
    depth_channels = []
    for deg, dist, loudness, arrival in results:
        if dist is None:
            values = [(0.0, 0.0)] * steps
            color = None
            print(f"{deg:+3d} deg: no echo")
        else:
            a = math.radians(deg)
            vec = (dist * math.cos(a), dist * math.sin(a))
            # Reveal each beam when its echo returns, so the fan fills in.
            values = [(0.0, 0.0)] * arrival + [vec] * (steps - arrival)
            db = 20 * math.log10(loudness / loudest)
            q = max(0.0, 1 + db / COLOR_SPAN_DB)
            color = tuple(int(v) for v in np.rint(COLD + (HOT - COLD) * q))
            print(f"{deg:+3d} deg: range {dist:.3f} m  {db:+6.1f} dB")

        ch = Channel("", kind="vector", dt=sim.dt, pos=CENTER, scale=1.0, color=color)
        ch.values = values
        depth_channels.append(ch)

    # The beamformed trace for the strongest beam, as a sanity scope.
    best = max(results, key=lambda r: r[2])
    mic = Channel("rx beam (Pa)", kind="scalar", dt=sim.dt, pos=CENTER)
    mic.values = list(beamform(best[0])[0])

    out.finish(
        dt=sim.dt * args.capture_every, dx=DX, c=sim.c,
        channels=(mic, *depth_channels), overlay=overlay,
    )
