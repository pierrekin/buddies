"""``ook_link`` reshot through a pair of transducers: voltage in, voltage out.

A Speaker (band-pass piezo) is driven with an OOK square-wave voltage
pattern. The FDTD propagates the resulting pressure wave to a Microphone
(same band-pass family) on the far side of the tank. The mic's voltage
output is what the receiver would actually see -- the same units as the
transmitter's drive, so the identity and gain baselines for the upcoming
modelling experiment finally make sense.

The transducers are deliberately resonant (Q≈4): on each bit edge the
band-pass rings down over ~Q/f0 ≈ 0.27 ms, smearing into the next bit.
The decode still works on a 1 ms / bit pattern, but the per-bit RMS
contrast is much weaker than ``ook_link``'s clean pressure trace -- the
realistic channel the modelling pass has to learn."""

import math

import numpy as np

from buddies import probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz, transducer resonance + OOK carrier
BIT_DURATION = 0.001  # s — 15 carrier cycles per bit
MESSAGE = (
    1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1,
    0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
)
TX = (0.2, 0.5)  # m
RX = (0.8, 0.5)  # m
DRIVE_V = 1.0  # peak drive voltage

# Both devices share the same band-pass curve (f0, Q) -- realistic for a
# matched piezo pair. Sensitivities pick a sensible operating point: 1 V
# at the resonance drives 1 Pa at 1 m on the way out, and 1 Pa at the mic
# reads as 1 V on the way back. The link's overall scale is then driven
# by 2D spreading + the two BPF passes, not by these calibration choices.
SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)


def ook_voltage(freq, bits, bit_dur, drive_v):
    """OOK square-wave voltage drive: sign(sin(2π f t)) during 1-bits,
    0 V during 0-bits. The Speaker's band-pass will smooth the edges; no
    explicit ramp is needed."""
    omega = 2 * math.pi * freq

    def v(t):
        if t < 0:
            return 0.0
        bit_idx = int(t / bit_dur)
        if bit_idx >= len(bits) or bits[bit_idx] == 0:
            return 0.0
        local_t = t - bit_idx * bit_dur
        return drive_v * (1.0 if math.sin(omega * local_t) >= 0 else -1.0)

    return v


def decode(rx_v, sim_dt, n_bits, bit_dur, prop_delay):
    """Mid-bit RMS slicer on the RX voltage (same shape as ook_link's,
    just operating on V instead of Pa). The threshold sits midway between
    the strongest and weakest bit."""
    samples = np.asarray(rx_v, dtype=np.float32)
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
    voltage_fn = ook_voltage(FREQ, MESSAGE, BIT_DURATION, DRIVE_V)

    # The speaker needs dt to pre-filter, but dt depends on the sim's CFL
    # choice. AcousticFDTD computes dt the same way -- call its helper.
    dt = timestep(DX, cfl=args.cfl)

    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[SPEAKER.source(pos=TX, voltage_fn=voltage_fn, steps=steps, dt=dt)],
        damping=edge_sponge((n, n), DX),
    )

    tx = Channel("TX (V)", kind="scalar", dt=sim.dt, pos=TX)
    mic_p = Channel("mic raw (Pa)", kind="scalar", dt=sim.dt, pos=RX)

    shot = out.shot("main")
    frames = shot.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        tx.append(voltage_fn(i * sim.dt))
        mic_p.append(probe.pressure(sim, RX))

    rx_v = MIC.filter(mic_p.values, sim.dt)
    rx = Channel("RX (V)", kind="scalar", dt=sim.dt, pos=RX, values=rx_v.tolist())

    prop_delay = math.hypot(RX[0] - TX[0], RX[1] - TX[1]) / sim.c
    decoded, rms, threshold = decode(rx_v, sim.dt, len(MESSAGE), BIT_DURATION, prop_delay)
    delay_samples = int(round(prop_delay / sim.dt))

    print(f"sent:    {MESSAGE}")
    print(f"decoded: {decoded}")
    print(f"per-bit RMS (V): {[round(float(r), 4) for r in rms]}")
    print(f"threshold (V):   {threshold:.4f}")
    bit_errors = sum(s != d for s, d in zip(MESSAGE, decoded))
    print(f"bit errors: {bit_errors} / {len(MESSAGE)}")

    shot.finish(
        channels=(tx, mic_p, rx),
        extras={
            "bit_duration": BIT_DURATION,
            "first_arrival_sample": delay_samples,
            "sent": list(MESSAGE),
            "decoded": list(decoded),
            "per_bit_rms_v": [float(r) for r in rms],
            "slicer_threshold_v": float(threshold),
            "bit_errors": bit_errors,
            "speaker": {"f0": FREQ, "q": SPEAKER.q, "sensitivity_pa": SPEAKER.sensitivity_pa},
            "mic": {"f0": FREQ, "q": MIC.q, "sensitivity_v_per_pa": MIC.sensitivity_v_per_pa},
        },
    )
    out.finish(dt=sim.dt * args.capture_every, dx=DX, c=sim.c)
