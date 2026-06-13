"""A source that hops between positions, emitting one pulse from each, while a
microphone array re-estimates its bearing after every pulse. The final
position is nearly endfire to the array, where bearing estimates degrade.

A hopping pulsed source needs no special support: it is identical to one
static source per position, each firing on a delay.
"""

import math

import numpy as np

from buddies import probe, simargs
from buddies.sim import AcousticFDTD, Source, edge_sponge, timestep, to_numpy, tone
from buddies.store import Channel

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
MICS = 8
ARRAY_X = 0.2  # m
ARRAY_SPAN = (0.3, 0.7)  # m
POSITIONS = [(0.65, 0.75), (0.75, 0.35), (0.3, 0.9)]  # last is near endfire
DETECT_THRESHOLD = 0.2  # of the window's peak |p|


def arrival(envelope):
    """Step where the arrival that contains the window's peak rises through the
    threshold. Searching backward from the peak ignores residual energy from
    earlier pulses at the start of the window."""
    peak = int(envelope.argmax())
    below = np.nonzero(envelope[:peak] < DETECT_THRESHOLD * envelope[peak])[0]
    return int(below[-1]) + 1 if len(below) else 0


def pulse(delay):
    """One cycle of a 1 Pa @ 1 m tone starting at ``delay``, then silence."""
    w = tone(FREQ, delay=delay, ramp_periods=1.0)
    return lambda t: w(t) if t < delay + 1 / FREQ else 0.0


def run(args, out):
    DX = args.dx
    # Per position: max mic distance ~0.61 m = ~365 default steps, plus margin.
    hop_steps = args.steps(600)
    steps = hop_steps * len(POSITIONS)
    dt = timestep(DX, cfl=args.cfl)

    n = round(SIZE / DX)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[
            Source(pos=p, waveform=pulse(k * hop_steps * dt))
            for k, p in enumerate(POSITIONS)
        ],
        damping=edge_sponge((n, n), DX),
    )

    mic_pos = [(ARRAY_X, y) for y in np.linspace(*ARRAY_SPAN, MICS)]
    lights = [Channel(f"m{j}", kind="color", dt=sim.dt, pos=p) for j, p in enumerate(mic_pos)]

    recordings = np.empty((steps, MICS), dtype=np.float32)
    frames = out.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        for j, p in enumerate(mic_pos):
            recordings[i, j] = probe.pressure(sim, p)
            lights[j].append(recordings[i, j])

    # Localization grid, front half-plane only: a line array cannot tell front
    # from back (mirror positions give identical arrival-time differences).
    gx, gy = np.meshgrid(
        np.linspace(ARRAY_X, SIZE, 161), np.linspace(0, SIZE, 201), indexing="ij"
    )
    grid_dists = np.array([np.hypot(gx - mx, gy - my) for mx, my in mic_pos])
    center = (ARRAY_X, float(np.mean([p[1] for p in mic_pos])))

    # Estimate per hop window, then feed the bearing vector from the moment the
    # window's last mic has reported.
    guess_values = [(0.0, 0.0)] * steps
    for k, true_pos in enumerate(POSITIONS):
        window = np.abs(recordings[k * hop_steps : (k + 1) * hop_steps])
        arrival_steps = np.array([arrival(window[:, j]) for j in range(MICS)])
        tdoa_meas = (arrival_steps - arrival_steps[0]) * sim.dt
        tdoa_pred = (grid_dists - grid_dists[0]) / sim.c
        err = ((tdoa_pred - tdoa_meas[:, None, None]) ** 2).sum(axis=0)
        best = np.unravel_index(err.argmin(), err.shape)
        estimate = (float(gx[best]), float(gy[best]))

        bearing = math.atan2(estimate[1] - center[1], estimate[0] - center[0])
        true_bearing = math.atan2(true_pos[1] - center[1], true_pos[0] - center[0])
        print(
            f"hop {k}: bearing {math.degrees(bearing):6.1f} deg, "
            f"true {math.degrees(true_bearing):6.1f} deg, "
            f"position ({estimate[0]:.2f}, {estimate[1]:.2f}) vs true {true_pos}"
        )

        ready = k * hop_steps + int(arrival_steps.max()) + 1
        direction = (math.cos(bearing), math.sin(bearing))
        for i in range(ready, steps):
            guess_values[i] = direction

    guess = Channel("bearing", kind="vector", dt=sim.dt, pos=center)
    guess.values = guess_values

    out.finish(dt=sim.dt * args.capture_every, dx=DX, c=sim.c, channels=(guess, *lights))
