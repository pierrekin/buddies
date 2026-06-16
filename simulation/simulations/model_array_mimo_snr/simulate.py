"""MIMO spatial multiplexing with ambient noise: SINR per stream.

Two simultaneous TX streams transmit independent PRBSes from widely
separated positions. An 8-element RX array captures the mixture plus
ambient noise. Per noise sigma, each stream is decoded two ways:

  * ``single`` -- decode the centre element alone. The two streams sit
    on top of each other at every RX; this pipeline has no spatial
    demix at all. Bad SINR (both cross-talk and noise hit it).
  * ``beam``   -- delay-and-sum at the known TX direction, then decode.
    Coherent summing peaks the stream you want, the array's null
    suppresses the other, and uncorrelated noise across elements drops
    by the RX array gain.

The per-stream SINR (signal vs interference + noise) is the standard
metric for spatial multiplexing performance::

    SINR_A = signal_A_power / (signal_B_leakage_power + noise_power)

Beamforming buys both interference rejection and SNR -- the combined
effect shows up as the SINR gap between ``single`` and ``beam``."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.noise import AmbientNoise, noise_power_per_unit_sigma_sq
from buddies.sim import AcousticFDTD, edge_sponge, line, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 2.0
FREQ = 15_000.0
C_EST = 1500.0
LAMBDA = C_EST / FREQ
N_RX = 8
SPACING = LAMBDA / 2
APERTURE = (N_RX - 1) * SPACING

RX_X = 1.7
RX_CY = 1.0
TX_A = (0.3, 1.6)
TX_B = (0.3, 0.4)
SINGLE_RX_INDEX = N_RX // 2  # e4

SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.003
# Per model_array_mimo: the 2 m tank's prop delay is ~1 ms; the FIR
# needs room past that delay for the channel impulse response.
FIR_N_TAPS = 4096

CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

PRBS_SEED_A = 1234
PRBS_SEED_B = 5678
N_BITS = 16
BIT_DUR = 0.001

N_NOISE_SOURCES = 16
NOISE_LAYOUT_SEED = 42
NOISE_DRIVE_SEED = 11
NOISE_MARGIN = 0.2  # > default 150 mm sponge depth so noise sources land in the fluid
NOISE_SIGMAS = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5)
NOISE_SIGMA_REF = 1e-6
NOISE_CAL_WARMUP_S = 0.005  # 2 m tank: prop delay alone is ~1 ms


def linear_chirp(f_lo, f_hi, duration, amplitude=1.0):
    def v(t):
        if t < 0 or t > duration:
            return 0.0
        k = (f_hi - f_lo) / duration
        phase = 2 * math.pi * (f_lo * t + 0.5 * k * t * t)
        return amplitude * math.sin(phase)
    return v


def ook_voltage(freq, bits, bit_dur, drive_v=1.0):
    omega = 2 * math.pi * freq

    def v(t):
        if t < 0:
            return 0.0
        idx = int(t / bit_dur)
        if idx >= len(bits) or bits[idx] == 0:
            return 0.0
        local = t - idx * bit_dur
        return drive_v * (1.0 if math.sin(omega * local) >= 0 else -1.0)

    return v


def decode(rx, sim_dt, n_bits, bit_dur, prop_delay):
    samples = np.asarray(rx, dtype=np.float32)
    spb = int(round(bit_dur / sim_dt))
    delay = int(round(prop_delay / sim_dt))
    if spb < 2:
        return tuple([0] * n_bits), np.zeros(n_bits, dtype=np.float32), 0.0
    rms = np.array([
        float(np.sqrt(np.mean(
            samples[delay + i * spb + spb // 2 : delay + (i + 1) * spb] ** 2
        )))
        for i in range(n_bits)
    ])
    threshold = float((rms.min() + rms.max()) / 2) if rms.max() > rms.min() else 0.0
    return tuple(int(r > threshold) for r in rms), rms, threshold


def rx_positions():
    start = (RX_X, RX_CY - APERTURE / 2)
    end = (RX_X, RX_CY + APERTURE / 2)
    return line(start, end, N_RX)


def y_offsets_from_center(positions):
    return [p[1] - RX_CY for p in positions]


def doa_to(tx_pos):
    dy = tx_pos[1] - RX_CY
    dx = RX_X - tx_pos[0]
    return math.degrees(math.atan2(dy, dx))


def delay_and_sum(traces, dt, look_angle_deg, y_offsets, c):
    sin_look = math.sin(math.radians(look_angle_deg))
    composite = np.zeros(len(traces[0]), dtype=np.float64)
    for v, dy in zip(traces, y_offsets):
        shift = int(round(dy * sin_look / c / dt))
        x = np.asarray(v, dtype=np.float64)
        if shift > 0:
            shifted = np.concatenate([np.zeros(shift), x[:-shift]])
        elif shift < 0:
            shifted = np.concatenate([x[-shift:], np.zeros(-shift)])
        else:
            shifted = x
        composite += shifted
    return composite / len(traces)


def sinr_db(signal_power, interference_power, noise_power):
    """SINR in dB. Returns +inf when interference + noise is zero."""
    denom = interference_power + noise_power
    if denom <= 0:
        return float("inf")
    if signal_power <= 0:
        return float("-inf")
    return 10.0 * math.log10(signal_power / denom)


def _capture_rx(sim, n_steps, positions, frames, capture_every):
    mic_p = np.empty((len(positions), n_steps), dtype=np.float32)
    for i in simargs.progress(n_steps):
        sim.step()
        if i % capture_every == 0:
            frames[i // capture_every] = to_numpy(sim.p)
        for j, pos in enumerate(positions):
            mic_p[j, i] = probe.pressure(sim, pos)
    return mic_p


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    n = round(SIZE / DX)
    positions = rx_positions()
    y_offsets = y_offsets_from_center(positions)
    c = 1500.0

    look_a = doa_to(TX_A)
    look_b = doa_to(TX_B)
    prop_delay = math.hypot(RX_X - TX_A[0], RX_CY - TX_A[1]) / C_EST

    rng_a = np.random.default_rng(PRBS_SEED_A)
    rng_b = np.random.default_rng(PRBS_SEED_B)
    bits_a = tuple(int(b) for b in rng_a.integers(0, 2, size=N_BITS))
    bits_b = tuple(int(b) for b in rng_b.integers(0, 2, size=N_BITS))

    ambient = AmbientNoise(
        n_sources=N_NOISE_SOURCES,
        domain_size=SIZE,
        margin=NOISE_MARGIN,
        layout_seed=NOISE_LAYOUT_SEED,
    )

    print(f"TX_A direction: {look_a:+.1f} deg, TX_B direction: {look_b:+.1f} deg")

    # -- Phase 1: characterize each TX alone with chirp. --
    base_chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)

    def char_shot(name, tx_pos):
        steps = round((CHIRP_DURATION + prop_delay + PROP_TAIL) / dt)
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(pos=tx_pos, voltage_fn=base_chirp,
                                    steps=steps, dt=dt)],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        frames = sw.open((args.nframes(steps), n, n))
        v_tx = np.fromiter(
            (base_chirp(i * sim.dt) for i in range(steps)),
            dtype=np.float32, count=steps,
        )
        print(f"shot {name}: tx={tx_pos}, {steps} steps")
        mic_p = _capture_rx(sim, steps, positions, frames, args.capture_every)
        v_rx = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]
        firs = []
        nrmses = []
        for j in range(N_RX):
            fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
            fir.fit(v_tx, v_rx[j])
            firs.append(fir)
            pred = fir.predict(v_tx)[: len(v_rx[j])]
            nr = float(channel_model.nrmse(v_rx[j], pred))
            nrmses.append(nr)
        print(f"  {name} per-RX training NRMSE: mean = {np.mean(nrmses):.4f}")
        return sw, sim.dt, v_tx, v_rx, firs, nrmses

    char_a_writer, char_a_dt, v_tx_chirp_a, v_rx_phys_char_a, firs_a, train_a = \
        char_shot("char_a", TX_A)
    char_b_writer, char_b_dt, v_tx_chirp_b, v_rx_phys_char_b, firs_b, train_b = \
        char_shot("char_b", TX_B)
    baseline = float(np.mean(train_a + train_b))

    # -- Phase 1b: noise-only calibration. Measure at every RX element,
    # plus the beamformed composite at each look angle (a and b). All
    # three noise floors scale with sigma^2 by FDTD linearity. --
    steps_comm = round((BIT_DUR * N_BITS + prop_delay + PROP_TAIL) / dt)
    sim_cal = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=ambient.sources(
            NOISE_SIGMA_REF, steps_comm, dt, drive_seed=NOISE_DRIVE_SEED,
        ),
        damping=edge_sponge((n, n), DX),
    )
    cal_writer = out.shot("noise_calibration")
    frames_cal = cal_writer.open((args.nframes(steps_comm), n, n))
    print(f"shot noise_calibration: sigma_ref={NOISE_SIGMA_REF:.1e}, "
          f"{steps_comm} steps")
    mic_p_cal = _capture_rx(sim_cal, steps_comm, positions, frames_cal, args.capture_every)
    v_rx_noise_per_rx = [MIC.filter(mic_p_cal[j], sim_cal.dt) for j in range(N_RX)]
    warmup = int(round(NOISE_CAL_WARMUP_S / sim_cal.dt))
    noise_power_factor_single = noise_power_per_unit_sigma_sq(
        v_rx_noise_per_rx[SINGLE_RX_INDEX][warmup:], NOISE_SIGMA_REF,
    )
    noise_comp_a_ref = delay_and_sum(v_rx_noise_per_rx, sim_cal.dt, look_a, y_offsets, c)
    noise_comp_b_ref = delay_and_sum(v_rx_noise_per_rx, sim_cal.dt, look_b, y_offsets, c)
    noise_power_factor_beam_a = noise_power_per_unit_sigma_sq(
        noise_comp_a_ref[warmup:], NOISE_SIGMA_REF,
    )
    noise_power_factor_beam_b = noise_power_per_unit_sigma_sq(
        noise_comp_b_ref[warmup:], NOISE_SIGMA_REF,
    )
    print(f"  noise power at e{SINGLE_RX_INDEX}: {noise_power_factor_single:.3e}")
    print(f"  noise power at beamformed look_a: {noise_power_factor_beam_a:.3e}")
    print(f"  noise power at beamformed look_b: {noise_power_factor_beam_b:.3e}")
    cal_writer.finish(
        channels=[
            Channel(f"noise RX e{SINGLE_RX_INDEX} (V)", kind="scalar",
                    dt=sim_cal.dt, pos=positions[SINGLE_RX_INDEX],
                    values=v_rx_noise_per_rx[SINGLE_RX_INDEX].tolist()),
            Channel("noise composite look_a (V)", kind="scalar",
                    dt=sim_cal.dt,
                    values=noise_comp_a_ref.astype(np.float32).tolist()),
            Channel("noise composite look_b (V)", kind="scalar",
                    dt=sim_cal.dt,
                    values=noise_comp_b_ref.astype(np.float32).tolist()),
        ],
        extras={
            "role": "noise_calibration",
            "sigma_ref": float(NOISE_SIGMA_REF),
            "noise_power_per_sigma_sq_single": noise_power_factor_single,
            "noise_power_per_sigma_sq_beam_a": noise_power_factor_beam_a,
            "noise_power_per_sigma_sq_beam_b": noise_power_factor_beam_b,
        },
    )

    # -- Phase 2: joint TX_A + TX_B + ambient noise sweep. --
    ook_a = ook_voltage(FREQ, bits_a, BIT_DUR)
    ook_b = ook_voltage(FREQ, bits_b, BIT_DUR)

    comm_writers = {}
    comm_data = {}
    sweep_rows = []

    for sigma in NOISE_SIGMAS:
        sigma_label = ("clean" if sigma == 0.0
                       else f"{sigma:.0e}".replace("+", "p").replace("-", "m"))
        name = f"joint_{sigma_label}"
        noise_sources = ambient.sources(
            sigma, steps_comm, dt, drive_seed=NOISE_DRIVE_SEED,
        )
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[
                SPEAKER.source(pos=TX_A, voltage_fn=ook_a,
                               steps=steps_comm, dt=dt),
                SPEAKER.source(pos=TX_B, voltage_fn=ook_b,
                               steps=steps_comm, dt=dt),
                *noise_sources,
            ],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        comm_writers[name] = sw
        frames = sw.open((args.nframes(steps_comm), n, n))

        v_tx_a = np.fromiter(
            (ook_a(i * sim.dt) for i in range(steps_comm)),
            dtype=np.float32, count=steps_comm,
        )
        v_tx_b = np.fromiter(
            (ook_b(i * sim.dt) for i in range(steps_comm)),
            dtype=np.float32, count=steps_comm,
        )
        print(f"shot {name}: sigma={sigma:.1e}, {steps_comm} steps")
        mic_p = _capture_rx(sim, steps_comm, positions, frames, args.capture_every)
        v_rx_phys = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]

        # Per-RX model contributions, kept SEPARATE per stream so we can
        # measure cross-talk later (model = signal_A + signal_B at each RX).
        signal_a_per_rx = [
            firs_a[j].predict(v_tx_a)[: len(v_rx_phys[j])].astype(np.float32)
            for j in range(N_RX)
        ]
        signal_b_per_rx = [
            firs_b[j].predict(v_tx_b)[: len(v_rx_phys[j])].astype(np.float32)
            for j in range(N_RX)
        ]
        # Mixture predicted by the surrogate.
        v_rx_model = [
            (signal_a_per_rx[j].astype(np.float64) + signal_b_per_rx[j].astype(np.float64))
            .astype(np.float32) for j in range(N_RX)
        ]

        # Beamformed composites at TX_A and TX_B look angles.
        comp_phys_a = delay_and_sum(v_rx_phys, sim.dt, look_a, y_offsets, c).astype(np.float32)
        comp_phys_b = delay_and_sum(v_rx_phys, sim.dt, look_b, y_offsets, c).astype(np.float32)
        comp_signal_a_at_a = delay_and_sum(signal_a_per_rx, sim.dt, look_a, y_offsets, c).astype(np.float32)
        comp_signal_b_at_a = delay_and_sum(signal_b_per_rx, sim.dt, look_a, y_offsets, c).astype(np.float32)
        comp_signal_a_at_b = delay_and_sum(signal_a_per_rx, sim.dt, look_b, y_offsets, c).astype(np.float32)
        comp_signal_b_at_b = delay_and_sum(signal_b_per_rx, sim.dt, look_b, y_offsets, c).astype(np.float32)
        # Calibrated noise power per pipeline -- exact via FDTD linearity.
        pow_noise_a = noise_power_factor_beam_a * sigma * sigma
        pow_noise_b = noise_power_factor_beam_b * sigma * sigma
        pow_noise_single = noise_power_factor_single * sigma * sigma

        # SINR for each beamformed pipeline.
        pow_sig_a_at_a = float(np.mean(comp_signal_a_at_a.astype(np.float64) ** 2))
        pow_sig_b_at_a = float(np.mean(comp_signal_b_at_a.astype(np.float64) ** 2))
        sinr_a_beam = sinr_db(pow_sig_a_at_a, pow_sig_b_at_a, pow_noise_a)

        pow_sig_b_at_b = float(np.mean(comp_signal_b_at_b.astype(np.float64) ** 2))
        pow_sig_a_at_b = float(np.mean(comp_signal_a_at_b.astype(np.float64) ** 2))
        sinr_b_beam = sinr_db(pow_sig_b_at_b, pow_sig_a_at_b, pow_noise_b)

        # Single-element decode (no demix): centre RX.
        ce = SINGLE_RX_INDEX
        v_phys_single = v_rx_phys[ce]
        sig_a_single = signal_a_per_rx[ce]
        sig_b_single = signal_b_per_rx[ce]
        pow_sig_a_single = float(np.mean(sig_a_single.astype(np.float64) ** 2))
        pow_sig_b_single = float(np.mean(sig_b_single.astype(np.float64) ** 2))
        sinr_a_single = sinr_db(pow_sig_a_single, pow_sig_b_single, pow_noise_single)
        sinr_b_single = sinr_db(pow_sig_b_single, pow_sig_a_single, pow_noise_single)

        # Decode each pipeline.
        bits_a_beam_p, _, _ = decode(comp_phys_a, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_b_beam_p, _, _ = decode(comp_phys_b, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_a_single_p, _, _ = decode(v_phys_single, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_b_single_p, _, _ = decode(v_phys_single, sim.dt, N_BITS, BIT_DUR, prop_delay)

        # Model side: assume LTI surrogate's predicted composites.
        comp_model_a = delay_and_sum(v_rx_model, sim.dt, look_a, y_offsets, c).astype(np.float32)
        comp_model_b = delay_and_sum(v_rx_model, sim.dt, look_b, y_offsets, c).astype(np.float32)
        v_model_single = v_rx_model[ce]
        bits_a_beam_m, _, _ = decode(comp_model_a, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_b_beam_m, _, _ = decode(comp_model_b, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_a_single_m, _, _ = decode(v_model_single, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_b_single_m, _, _ = decode(v_model_single, sim.dt, N_BITS, BIT_DUR, prop_delay)

        def ber(decoded, truth):
            return sum(d != b for d, b in zip(decoded, truth)) / N_BITS

        def agree(a, b):
            return sum(x == y for x, y in zip(a, b)) / N_BITS

        ber_a_beam_p = float(ber(bits_a_beam_p, bits_a))
        ber_a_beam_m = float(ber(bits_a_beam_m, bits_a))
        ber_b_beam_p = float(ber(bits_b_beam_p, bits_b))
        ber_b_beam_m = float(ber(bits_b_beam_m, bits_b))
        ber_a_single_p = float(ber(bits_a_single_p, bits_a))
        ber_a_single_m = float(ber(bits_a_single_m, bits_a))
        ber_b_single_p = float(ber(bits_b_single_p, bits_b))
        ber_b_single_m = float(ber(bits_b_single_m, bits_b))
        agree_a_beam = float(agree(bits_a_beam_p, bits_a_beam_m))
        agree_b_beam = float(agree(bits_b_beam_p, bits_b_beam_m))
        agree_a_single = float(agree(bits_a_single_p, bits_a_single_m))
        agree_b_single = float(agree(bits_b_single_p, bits_b_single_m))

        comm_data[name] = dict(
            sigma=float(sigma), sim_dt=sim.dt,
            v_tx_a=v_tx_a, v_tx_b=v_tx_b,
            v_rx_phys=v_rx_phys, v_rx_model=v_rx_model,
            comp_phys_a=comp_phys_a, comp_phys_b=comp_phys_b,
            comp_model_a=comp_model_a, comp_model_b=comp_model_b,
            v_phys_single=v_phys_single, v_model_single=v_model_single,
            sinr_a_single_db=sinr_a_single, sinr_b_single_db=sinr_b_single,
            sinr_a_beam_db=sinr_a_beam, sinr_b_beam_db=sinr_b_beam,
            ber_a_single_phys=ber_a_single_p, ber_a_single_model=ber_a_single_m,
            ber_b_single_phys=ber_b_single_p, ber_b_single_model=ber_b_single_m,
            ber_a_beam_phys=ber_a_beam_p, ber_a_beam_model=ber_a_beam_m,
            ber_b_beam_phys=ber_b_beam_p, ber_b_beam_model=ber_b_beam_m,
            agreement_a_single=agree_a_single, agreement_b_single=agree_b_single,
            agreement_a_beam=agree_a_beam, agreement_b_beam=agree_b_beam,
            bits_a_beam_phys=list(bits_a_beam_p), bits_a_beam_model=list(bits_a_beam_m),
            bits_b_beam_phys=list(bits_b_beam_p), bits_b_beam_model=list(bits_b_beam_m),
            bits_a_single_phys=list(bits_a_single_p), bits_a_single_model=list(bits_a_single_m),
            bits_b_single_phys=list(bits_b_single_p), bits_b_single_model=list(bits_b_single_m),
        )
        sweep_rows.append({
            "sigma": float(sigma),
            "sinr_a_single_db": sinr_a_single, "sinr_b_single_db": sinr_b_single,
            "sinr_a_beam_db": sinr_a_beam, "sinr_b_beam_db": sinr_b_beam,
            "ber_a_single_phys": ber_a_single_p, "ber_b_single_phys": ber_b_single_p,
            "ber_a_beam_phys": ber_a_beam_p, "ber_b_beam_phys": ber_b_beam_p,
            "agreement_a_beam": agree_a_beam, "agreement_b_beam": agree_b_beam,
        })
        print(
            f"  {name}: A single SINR={sinr_a_single:+.1f} dB BER={ber_a_single_p:.3f}  "
            f"A beam SINR={sinr_a_beam:+.1f} dB BER={ber_a_beam_p:.3f}  "
            f"gain_A={sinr_a_beam - sinr_a_single:+.1f} dB"
        )

    # -- Array gain (SINR_beam - SINR_single, averaged over streams). --
    gains_a = [
        r["sinr_a_beam_db"] - r["sinr_a_single_db"]
        for r in sweep_rows if r["sigma"] > 0
        and math.isfinite(r["sinr_a_beam_db"]) and math.isfinite(r["sinr_a_single_db"])
    ]
    gains_b = [
        r["sinr_b_beam_db"] - r["sinr_b_single_db"]
        for r in sweep_rows if r["sigma"] > 0
        and math.isfinite(r["sinr_b_beam_db"]) and math.isfinite(r["sinr_b_single_db"])
    ]
    array_gain_db = float(np.mean(gains_a + gains_b)) if (gains_a or gains_b) else None
    print(f"\nMIMO array gain (SINR_beam - SINR_single, mean over streams):")
    print(f"  stream A mean = {float(np.mean(gains_a)):+.2f} dB  per-sigma: "
          f"{['%+.2f' % g for g in gains_a]}")
    print(f"  stream B mean = {float(np.mean(gains_b)):+.2f} dB  per-sigma: "
          f"{['%+.2f' % g for g in gains_b]}")
    print(f"  combined mean = {array_gain_db:+.2f} dB")

    def pick(key):
        return [r[key] for r in sweep_rows]

    sweep = {
        "sweep_sigmas": pick("sigma"),
        "sweep_sinr_a_single_db": pick("sinr_a_single_db"),
        "sweep_sinr_b_single_db": pick("sinr_b_single_db"),
        "sweep_sinr_a_beam_db": pick("sinr_a_beam_db"),
        "sweep_sinr_b_beam_db": pick("sinr_b_beam_db"),
        "sweep_ber_a_single_phys": pick("ber_a_single_phys"),
        "sweep_ber_b_single_phys": pick("ber_b_single_phys"),
        "sweep_ber_a_beam_phys": pick("ber_a_beam_phys"),
        "sweep_ber_b_beam_phys": pick("ber_b_beam_phys"),
        "sweep_agreement_a_beam": pick("agreement_a_beam"),
        "sweep_agreement_b_beam": pick("agreement_b_beam"),
        "array_gain_db": array_gain_db,
        "look_a_deg": look_a, "look_b_deg": look_b,
        "noise_positions": np.asarray(ambient.positions, dtype=np.float32),
        "n_noise_sources": ambient.n_sources,
        "train_nrmse_baseline": baseline,
        "bits_a_sent": list(bits_a), "bits_b_sent": list(bits_b),
    }

    # -- Phase 3: finalize. --
    rep_idx = (0, SINGLE_RX_INDEX, N_RX - 1)

    def finish_char(writer, name, tx_pos, sim_dt, v_tx, v_rx, firs, train, label):
        channels = [
            Channel("TX (V)", kind="scalar", dt=sim_dt, pos=tx_pos,
                    values=v_tx.tolist()),
        ]
        for j in rep_idx:
            channels.append(Channel(
                f"RX e{j} phys (V)", kind="scalar", dt=sim_dt,
                pos=positions[j], values=v_rx[j].tolist(),
            ))
        writer.finish(
            channels=channels,
            extras={
                "role": "characterize",
                "label": label,
                "tx_pos_x": tx_pos[0], "tx_pos_y": tx_pos[1],
                "train_nrmse_per_rx": train,
                "train_nrmse_baseline": baseline,
                **{f"fir_h_e{j}": firs[j].h for j in range(N_RX)},
                **sweep,
            },
        )

    finish_char(char_a_writer, "char_a", TX_A, char_a_dt, v_tx_chirp_a,
                v_rx_phys_char_a, firs_a, train_a, "stream A characterize")
    finish_char(char_b_writer, "char_b", TX_B, char_b_dt, v_tx_chirp_b,
                v_rx_phys_char_b, firs_b, train_b, "stream B characterize")

    for name, r in comm_data.items():
        sim_dt = r["sim_dt"]
        ce = SINGLE_RX_INDEX
        comm_writers[name].finish(
            channels=[
                Channel("TX_A (V)", kind="scalar", dt=sim_dt, pos=TX_A,
                        values=r["v_tx_a"].tolist()),
                Channel("TX_B (V)", kind="scalar", dt=sim_dt, pos=TX_B,
                        values=r["v_tx_b"].tolist()),
                Channel(f"RX e{ce} phys (mixture) (V)", kind="scalar", dt=sim_dt,
                        pos=positions[ce], values=r["v_phys_single"].tolist()),
                Channel("composite A beam phys (V)", kind="scalar", dt=sim_dt,
                        values=r["comp_phys_a"].tolist()),
                Channel("composite A beam model (V)", kind="scalar", dt=sim_dt,
                        values=r["comp_model_a"].tolist()),
                Channel("composite B beam phys (V)", kind="scalar", dt=sim_dt,
                        values=r["comp_phys_b"].tolist()),
                Channel("composite B beam model (V)", kind="scalar", dt=sim_dt,
                        values=r["comp_model_b"].tolist()),
            ],
            extras={
                "role": "comms",
                "sigma": r["sigma"],
                "sinr_a_single_db": r["sinr_a_single_db"],
                "sinr_b_single_db": r["sinr_b_single_db"],
                "sinr_a_beam_db": r["sinr_a_beam_db"],
                "sinr_b_beam_db": r["sinr_b_beam_db"],
                "ber_a_single_phys": r["ber_a_single_phys"],
                "ber_a_single_model": r["ber_a_single_model"],
                "ber_b_single_phys": r["ber_b_single_phys"],
                "ber_b_single_model": r["ber_b_single_model"],
                "ber_a_beam_phys": r["ber_a_beam_phys"],
                "ber_a_beam_model": r["ber_a_beam_model"],
                "ber_b_beam_phys": r["ber_b_beam_phys"],
                "ber_b_beam_model": r["ber_b_beam_model"],
                "agreement_a_single": r["agreement_a_single"],
                "agreement_b_single": r["agreement_b_single"],
                "agreement_a_beam": r["agreement_a_beam"],
                "agreement_b_beam": r["agreement_b_beam"],
                **{f"v_rx_phys_e{j}": np.asarray(r["v_rx_phys"][j], dtype=np.float32)
                   for j in range(N_RX)},
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=c)
