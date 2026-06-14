"""``ook_link`` with an eye diagram on the receiver.

Same OOK-on-square-wave link, but transmits a longer pseudo-random bit
pattern so the received signal can be folded against the bit period and
overlaid into an eye diagram. The eye plot shows every bit window stacked
on top of every other; a clean 'open' centre means the slicer's job is
easy, a closed eye means the link is on the edge of failing.

The mic trace lands as a normal scalar channel; the bit period and the
prop-delay-trimmed start sample go into ``extras`` so the sim's ``view.py``
can fold the trace into an eye diagram."""

import math

import numpy as np

from buddies import probe, simargs
from buddies.sim import (
    DENSITY_SEAWATER, SOUND_SPEED_SEAWATER, AcousticFDTD, Source,
    edge_sponge, to_numpy,
)
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}  # 32-bit run is long; thin frames so disk stays sane

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz, square-wave carrier
BIT_DURATION = 0.001  # s — 15 carrier cycles per bit
# 32-bit pseudo-random pattern (balanced and varied so the eye has both
# 0→1 and 1→0 transitions in many positions).
MESSAGE = (
    1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1,
    0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
)
TX = (0.2, 0.5)  # m
RX = (0.8, 0.5)  # m
AMPLITUDE = 1.0  # Pa at 1 m — same calibration convention as ``tone()``


def ook_square(freq, bits, bit_dur, amplitude, at=1.0, ramp_periods=1.0,
               c=SOUND_SPEED_SEAWATER, rho=DENSITY_SEAWATER):
    """OOK on a square carrier: sign(sin(2pi f t)) during 1-bits, 0 during 0-bits.

    ``amplitude`` is the far-field pressure (Pa) the *fundamental* component
    would reach at ``at`` meters in open water — the same calibration
    ``tone()`` uses, so ``ook_square(freq, [1], ...)`` is interchangeable
    with ``tone(freq, ...)`` for level comparisons. Each 1-bit re-zeros the
    phase and ramps in over ``ramp_periods``, so switching the wave on at a
    non-zero edge doesn't inject a click."""
    omega = 2 * math.pi * freq
    # Calibrate the fundamental to ``amplitude``. A square wave's fundamental
    # has Fourier amplitude 4/pi * peak, so scale tone()'s w_peak down by
    # pi/4 to land the fundamental at the target.
    w_peak_sine = 4 * amplitude / (rho * omega) * math.sqrt(math.pi * (omega / c) * at / 2)
    w_peak = w_peak_sine * math.pi / 4

    def waveform(t):
        if t < 0:
            return 0.0
        bit_idx = int(t / bit_dur)
        if bit_idx >= len(bits) or bits[bit_idx] == 0:
            return 0.0
        local_t = t - bit_idx * bit_dur
        ramp = min(1.0, local_t * freq / ramp_periods)
        return w_peak * ramp * (1.0 if math.sin(omega * local_t) >= 0 else -1.0)

    return waveform


def decode(mic_values, sim_dt, n_bits, bit_dur, prop_delay):
    """Per-bit RMS slicer, aligned to the propagation delay from TX to RX.

    Only the second half of each bit window is integrated, so the leading
    rise-time edge and any leftover ringing from the previous bit don't
    contaminate the energy estimate. The threshold sits midway between the
    strongest and weakest bit; this works whenever the message contains
    both 0s and 1s."""
    samples = np.asarray(mic_values, dtype=np.float32)
    spb = int(round(bit_dur / sim_dt))
    delay = int(round(prop_delay / sim_dt))
    rms = np.array([
        float(np.sqrt(np.mean(
            samples[delay + i * spb + spb // 2 : delay + (i + 1) * spb] ** 2
        )))
        for i in range(n_bits)
    ])
    threshold = (rms.min() + rms.max()) / 2
    return tuple(int(r > threshold) for r in rms), rms, threshold


def run(args, out):
    DX = args.dx
    sim_time = (len(MESSAGE) + 1) * BIT_DURATION  # one bit of tail past TX end
    steps = args.steps(round(sim_time / args.default_dt))

    n = round(SIZE / DX)
    tx_waveform = ook_square(FREQ, MESSAGE, BIT_DURATION, AMPLITUDE)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[Source(pos=TX, waveform=tx_waveform)],
        damping=edge_sponge((n, n), DX),
    )

    mic = Channel("mic (Pa)", kind="scalar", dt=sim.dt, pos=RX)

    frames = out.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        mic.append(probe.pressure(sim, RX))

    prop_delay = math.hypot(RX[0] - TX[0], RX[1] - TX[1]) / sim.c
    decoded, rms, threshold = decode(mic.values, sim.dt, len(MESSAGE), BIT_DURATION, prop_delay)
    delay_samples = int(round(prop_delay / sim.dt))

    print(f"sent:    {MESSAGE}")
    print(f"decoded: {decoded}")
    print(f"per-bit RMS (Pa): {[round(float(r), 4) for r in rms]}")
    print(f"threshold (Pa):   {threshold:.4f}")

    out.finish(
        dt=sim.dt * args.capture_every, dx=DX, c=sim.c,
        channels=(mic,),
        extras={
            "bit_duration": BIT_DURATION,
            "first_arrival_sample": delay_samples,
            "sent": list(MESSAGE),
            "decoded": list(decoded),
            "per_bit_rms": [float(r) for r in rms],
            "slicer_threshold": float(threshold),
        },
    )
