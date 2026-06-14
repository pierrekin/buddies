"""The bare-minimum acoustic link: on-off keying of a square-wave carrier.

A transmitter on the left of the tank emits a 15 kHz square wave during
each '1' bit and silence during each '0' bit. A microphone on the right
slices each per-bit window's RMS pressure against a midpoint threshold to
decide which bit it heard. Sent and decoded bits print at the end.

Note: the FDTD grid resolves the carrier (10 cells / wavelength by default)
but heavily attenuates a square wave's higher harmonics (3f, 5f, ...). At
the mic the wave is closer to a sine; that's fine here because the demod
just measures energy, but it's why this is a 'square wave' only at the
source.
"""

import math

import numpy as np

from buddies import probe, simargs
from buddies.sim import (
    DENSITY_SEAWATER, SOUND_SPEED_SEAWATER, AcousticFDTD, Source,
    edge_sponge, to_numpy,
)
from buddies.store import Channel

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz, square-wave carrier
BIT_DURATION = 0.001  # s — 15 carrier cycles per bit
MESSAGE = (1, 0, 1, 1, 0)
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


def sliding_rms(samples, window_samples):
    """Sliding-window RMS, same length as ``samples``. The window is the
    carrier period, so the result is the slow amplitude envelope of the wave."""
    sq = np.asarray(samples, dtype=np.float32) ** 2
    kernel = np.ones(window_samples, dtype=np.float32) / window_samples
    return np.sqrt(np.convolve(sq, kernel, mode="same")).astype(np.float32)


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

    tx = Channel("TX waveform (m²/s)", kind="scalar", dt=sim.dt, pos=TX)
    mic = Channel("mic (Pa)", kind="scalar", dt=sim.dt, pos=RX)

    frames = out.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        tx.append(tx_waveform(i * sim.dt))
        mic.append(probe.pressure(sim, RX))

    # Carrier-period sliding RMS — the slicer's view of the channel.
    env_samples = max(1, int(round(1 / FREQ / sim.dt)))
    envelope = Channel(
        "mic envelope (Pa, 1-period RMS)", kind="scalar", dt=sim.dt, pos=RX,
        values=sliding_rms(mic.values, env_samples).tolist(),
    )

    prop_delay = math.hypot(RX[0] - TX[0], RX[1] - TX[1]) / sim.c
    decoded, rms, threshold = decode(mic.values, sim.dt, len(MESSAGE), BIT_DURATION, prop_delay)

    print(f"sent:    {MESSAGE}")
    print(f"decoded: {decoded}")
    print(f"per-bit RMS (Pa): {[round(float(r), 4) for r in rms]}")
    print(f"threshold (Pa):   {threshold:.4f}")

    out.finish(
        dt=sim.dt * args.capture_every, dx=DX, c=sim.c,
        channels=(tx, mic, envelope),
        extras={
            "bit_duration": BIT_DURATION,
            "sent": list(MESSAGE),
            "decoded": list(decoded),
            "per_bit_rms": [float(r) for r in rms],
            "slicer_threshold": float(threshold),
        },
    )
