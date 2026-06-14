"""Channel characterisation: probe the TX→RX path with one short pulse and
look at what comes out the other side.

The transmitter fires a single Gaussian-windowed sine burst — narrow enough
in time to span a useful frequency band but with most energy at ``FREQ``.
The mic then records the channel's response. For a roughly-impulsive probe
like this, the mic trace IS the channel's impulse response ``h(t)`` (give
or take the probe's own shape). Once you have ``h(t)``, the mic output for
any other TX waveform ``x(t)`` is just ``y = x ∗ h`` — so you can audition
modulation schemes by convolving against this trace, without re-running
the FDTD.

Three channels come out: the TX waveform itself, the raw mic pressure, and
the mic envelope (sliding RMS at the carrier period) so the pulse arrival
shape is easy to read off."""

import math

import numpy as np

from buddies import probe, simargs
from buddies.sim import (
    DENSITY_SEAWATER, SOUND_SPEED_SEAWATER, AcousticFDTD, Source,
    edge_sponge, to_numpy,
)
from buddies.store import Channel

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz, probe centre frequency
PULSE_SIGMA_CYCLES = 1.0  # Gaussian envelope std-dev in carrier cycles
TX = (0.2, 0.5)  # m
RX = (0.8, 0.5)  # m
AMPLITUDE = 1.0  # Pa at 1 m — envelope-peak Pa for the carrier component


def gaussian_burst(freq, amplitude, sigma_cycles, at=1.0,
                   c=SOUND_SPEED_SEAWATER, rho=DENSITY_SEAWATER):
    """A Gaussian-windowed sine: ``A * exp(-(t-t0)²/(2σ²)) * sin(ω(t-t0))``.

    The envelope is centred 4σ after t=0 so the wave starts silent and
    finishes silent within ~8σ. ``amplitude`` is the carrier-component Pa
    at ``at`` metres, using ``tone()``'s far-field calibration."""
    omega = 2 * math.pi * freq
    sigma = sigma_cycles / freq
    t0 = 4 * sigma
    w_peak = 4 * amplitude / (rho * omega) * math.sqrt(math.pi * (omega / c) * at / 2)

    def waveform(t):
        if t < 0 or t > 2 * t0:  # 8σ wide; outside, the Gaussian is ~0 anyway
            return 0.0
        dt = t - t0
        env = math.exp(-(dt * dt) / (2 * sigma * sigma))
        return w_peak * env * math.sin(omega * dt)

    return waveform


def sliding_rms(samples, window_samples):
    """Sliding-window RMS, same length as ``samples``."""
    sq = np.asarray(samples, dtype=np.float32) ** 2
    kernel = np.ones(window_samples, dtype=np.float32) / window_samples
    return np.sqrt(np.convolve(sq, kernel, mode="same")).astype(np.float32)


def run(args, out):
    DX = args.dx

    # Span: pulse duration (8σ), straight-line propagation, plus a tail for
    # the 2D Green's-function ringdown.
    sigma = PULSE_SIGMA_CYCLES / FREQ
    distance = math.hypot(RX[0] - TX[0], RX[1] - TX[1])
    sim_time = 8 * sigma + distance / SOUND_SPEED_SEAWATER + 0.002
    steps = args.steps(round(sim_time / args.default_dt))

    n = round(SIZE / DX)
    tx_waveform = gaussian_burst(FREQ, AMPLITUDE, PULSE_SIGMA_CYCLES)
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

    env_samples = max(1, int(round(1 / FREQ / sim.dt)))
    envelope_vals = sliding_rms(mic.values, env_samples)
    envelope = Channel(
        "mic envelope (Pa, 1-period RMS)", kind="scalar", dt=sim.dt, pos=RX,
        values=envelope_vals.tolist(),
    )

    # A few one-line stats so the impulse response has numbers attached.
    mic_arr = np.asarray(mic.values, dtype=np.float32)
    peak_idx = int(np.argmax(envelope_vals))
    peak_time = peak_idx * sim.dt
    peak_pa = float(envelope_vals[peak_idx])
    expected_delay = distance / sim.c + 4 * sigma  # envelope-peak time at TX is 4σ
    # Effective bandwidth from the FFT: -3 dB half-width around the spectral peak.
    spec = np.abs(np.fft.rfft(mic_arr))
    freqs = np.fft.rfftfreq(len(mic_arr), d=sim.dt)
    spec_peak = int(np.argmax(spec))

    print(f"distance TX→RX     : {distance:.3f} m")
    print(f"expected peak time : {expected_delay*1e3:.3f} ms (geometric + 4σ probe centre)")
    print(f"measured peak time : {peak_time*1e3:.3f} ms (envelope max)")
    print(f"measured peak Pa   : {peak_pa:.4f}")
    print(f"spectral peak freq : {freqs[spec_peak]:.0f} Hz (probed centre {FREQ:.0f} Hz)")

    out.finish(
        dt=sim.dt * args.capture_every, dx=DX, c=sim.c,
        channels=(tx, mic, envelope),
        extras={
            "distance": float(distance),
            "probe_freq": float(FREQ),
            "expected_peak_time": float(expected_delay),
            "measured_peak_time": float(peak_time),
            "measured_peak_pa": peak_pa,
            "spectral_peak_freq": float(freqs[spec_peak]),
            "spectrum_freqs": freqs.astype(np.float32),
            "spectrum_mag": spec.astype(np.float32),
        },
    )
