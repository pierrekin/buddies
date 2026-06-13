"""Sonar listen v1: receive-side beamforming from a single omni ping.

The sweep series (sonar_sweep*) steers the *transmit* beam, firing one focused
ping per look angle -- 31 pings for a 31-angle fan. This series flips that:
fire ONE omnidirectional pulse, record every array element, and form the look
direction afterward by delay-and-summing the recordings. The delays are
identical to the sweep's transmit focusing (receive reciprocity), so all the
machinery carries over; the only real change is that one recording can be
beamformed to any direction we like.

v1 is the smallest thing that works: a single hardcoded look angle aimed at one
target, producing one range vector. The tradeoff to watch is SNR -- an omni
ping spreads its energy over the whole scene instead of concentrating it at the
target, so returns are weaker than the focused sweep and the source amplitude
is turned up to compensate."""

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
LOOK_DEG = -13.0  # the single direction we listen in this version
FOCUS_RANGE = 1.05  # m, sharpens the receive beam around the target's range
SRC_AMP = 30.0  # Pa at 1 m; high because one omni ping must light the scene
DETECT_THRESHOLD = 0.2  # of the pre-blank peak, for the transmit leading edge
MIN_ECHO = 0.02  # Pa, below this we call it no echo
# The smoothed envelope peaks ~3/4 of a cycle behind the arrival's leading edge
# (one-cycle ramped pulse); subtracted from the time of flight.
PEAK_OFFSET_S = 0.75 / FREQ
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

    # One rounded target ~1.04 m from the array at ~-13 deg, on the look axis.
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

    # Receive beamforming: delay each element so a wavefront focused at the look
    # point sums in phase, then add. The shifts are exactly the sweep's transmit
    # focusing delays, applied to the recordings instead of the source waveforms.
    a = math.radians(LOOK_DEG)
    focus = (CENTER[0] + FOCUS_RANGE * math.cos(a), CENTER[1] + FOCUS_RANGE * math.sin(a))
    dists = np.hypot(ARRAY_X - focus[0], element_y - focus[1])
    shifts = np.round((dists.max() - dists) / sim.c / sim.dt).astype(int)

    beamformed = np.zeros(steps, dtype=np.float32)
    for j, s in enumerate(shifts):
        beamformed[s:] += recordings[: steps - s, j]
    beamformed /= ELEMENTS

    # Find the direct blast's leading edge, then the loudest arrival after
    # blanking. With an omni transmit fired at t=0 the blast onset marks t_tx.
    window = np.abs(beamformed)
    emit = int(np.argmax(window > DETECT_THRESHOLD * window[:blank_steps].max()))
    env = np.array([window[max(0, i - smooth_steps) : i + 1].max() for i in range(len(window))])
    listen = env[blank_steps:]
    loudness = float(listen.max()) if listen.size else 0.0

    if loudness < MIN_ECHO:
        print(f"{LOOK_DEG:+.0f} deg: no echo (peak {loudness:.4f} Pa)")
        vec = (0.0, 0.0)
    else:
        echo_rel = int(listen.argmax())
        # Path measured at the peak is outgoing (center->target) plus the return
        # aligned to the farthest element, i.e. range + dists.max(). One omni
        # transmit, so no transmit-side focusing term and no halving.
        dist = sim.c * ((blank_steps + echo_rel - emit) * sim.dt - PEAK_OFFSET_S) - dists.max()
        vec = (dist * math.cos(a), dist * math.sin(a))
        print(f"{LOOK_DEG:+.0f} deg: range {dist:.3f} m  (peak {loudness:.4f} Pa)")

    ch = Channel("", kind="vector", dt=sim.dt, pos=CENTER, scale=1.0, color=(255, 180, 40, 255))
    ch.values = [vec] * steps

    mic = Channel("rx beam (Pa)", kind="scalar", dt=sim.dt, pos=CENTER)
    mic.values = list(beamformed)

    out.finish(
        dt=sim.dt * args.capture_every, dx=DX, c=sim.c,
        channels=(mic, ch), overlay=overlay,
    )
