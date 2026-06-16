"""LoRa-style chirp-spread-spectrum vs OOK on the same noisy channel.

Same TX/RX geometry and ambient-noise sweep as ``model_link_noise``, so
the OOK numbers are directly comparable. Per noise sigma we run BOTH:

  * ``ook_*``  -- 1 ms OOK on a 15 kHz square carrier, RMS slicer.
  * ``lora_*`` -- binary CSS: 1 ms chirp per symbol covering the BPF
                  passband (13.1 to 16.9 kHz). 1-bit = up-chirp,
                  0-bit = down-chirp. Decoded by matched-filter
                  correlation against reference chirps.

LoRa's processing gain for binary CSS is the time-bandwidth product
``T * B`` in dB::

    10 * log10(1 ms * 3.75 kHz) = 10 * log10(3.75) ~ 5.7 dB

So at a noise level where OOK is sitting at, say, BER 0.25, LoRa should
be ~5-6 dB to the left of OOK on the BER-vs-SNR curve -- i.e. closer to
zero BER. Modest gain at this bit rate; the real LoRa-style win comes
from spreading factor (longer symbols, more bits per symbol)."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.noise import AmbientNoise, noise_power_per_unit_sigma_sq
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 1.0
FREQ = 15_000.0
TX = (0.2, 0.5)
RX = (0.8, 0.5)
SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.002
FIR_N_TAPS = 1024

# Characterization probe -- wider band than the comm signals so the FIR
# fit covers everything either decoder will see.
CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

# Same OOK params as model_link / model_link_noise so the BERs compare.
N_BITS = 32
BIT_DUR = 0.001
PRBS_SEED = 1234

# Binary CSS over the BPF's -3 dB passband (f0/Q ~ 3.75 kHz wide around 15 kHz).
CSS_F_LO = 13_125.0
CSS_F_HI = 16_875.0
CSS_BANDWIDTH = CSS_F_HI - CSS_F_LO
CSS_PROCESSING_GAIN_DB = 10.0 * math.log10(BIT_DUR * CSS_BANDWIDTH)

# Same ambient noise setup as model_link_noise.
N_NOISE_SOURCES = 16
NOISE_LAYOUT_SEED = 42
NOISE_DRIVE_SEED = 11
NOISE_MARGIN = 0.2  # > default 150 mm sponge depth so noise sources land in the fluid
# Centred around where OOK transitions from BER 0 to BER ~ 0.5, so both
# the OOK and CSS curves' transition regions are visible.
NOISE_SIGMAS = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
NOISE_SIGMA_REF = 1e-6
NOISE_CAL_WARMUP_S = 0.003


def linear_chirp(f_lo, f_hi, duration, amplitude=1.0):
    """The training chirp (wider band than the CSS symbol chirps)."""
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


def css_voltage(f_lo, f_hi, bits, bit_dur, drive_v=1.0):
    """Binary CSS: 1 = up-chirp from f_lo to f_hi over bit_dur,
    0 = down-chirp from f_hi to f_lo over bit_dur."""
    def v(t):
        if t < 0:
            return 0.0
        idx = int(t / bit_dur)
        if idx >= len(bits):
            return 0.0
        local = t - idx * bit_dur
        if bits[idx] == 1:
            f_a, f_b = f_lo, f_hi
        else:
            f_a, f_b = f_hi, f_lo
        k = (f_b - f_a) / bit_dur
        phase = 2 * math.pi * (f_a * local + 0.5 * k * local * local)
        return drive_v * math.sin(phase)

    return v


def ook_decode(rx, sim_dt, n_bits, bit_dur, prop_delay):
    """OOK RMS slicer -- same as model_link / model_link_noise."""
    samples = np.asarray(rx, dtype=np.float32)
    spb = int(round(bit_dur / sim_dt))
    delay = int(round(prop_delay / sim_dt))
    if spb < 2:
        return tuple([0] * n_bits)
    rms = np.array([
        float(np.sqrt(np.mean(
            samples[delay + i * spb + spb // 2 : delay + (i + 1) * spb] ** 2
        )))
        for i in range(n_bits)
    ])
    threshold = float((rms.min() + rms.max()) / 2) if rms.max() > rms.min() else 0.0
    return tuple(int(r > threshold) for r in rms)


def css_decode(rx, sim_dt, n_bits, bit_dur, prop_delay, f_lo, f_hi):
    """Binary CSS matched-filter decoder: correlate each symbol slot
    with stored up- and down-chirp references; pick whichever has the
    higher absolute correlation."""
    samples = np.asarray(rx, dtype=np.float64)
    spb = int(round(bit_dur / sim_dt))
    delay = int(round(prop_delay / sim_dt))
    if spb < 2:
        return tuple([0] * n_bits)
    t_ref = np.arange(spb) * sim_dt
    k_up = (f_hi - f_lo) / bit_dur
    k_down = (f_lo - f_hi) / bit_dur
    ref_up = np.sin(2 * math.pi * (f_lo * t_ref + 0.5 * k_up * t_ref ** 2))
    ref_down = np.sin(2 * math.pi * (f_hi * t_ref + 0.5 * k_down * t_ref ** 2))
    decoded = []
    for i in range(n_bits):
        start = delay + i * spb
        end = start + spb
        segment = samples[start:end]
        if len(segment) < spb:
            decoded.append(0)
            continue
        corr_up = abs(float(np.dot(segment, ref_up)))
        corr_down = abs(float(np.dot(segment, ref_down)))
        decoded.append(1 if corr_up > corr_down else 0)
    return tuple(decoded)


def signal_noise_power(v_rx_phys, v_rx_model):
    v_phys = np.asarray(v_rx_phys, dtype=np.float64)
    v_model = np.asarray(v_rx_model, dtype=np.float64)[: len(v_phys)]
    residual = v_phys - v_model
    return float(np.mean(v_model ** 2)), float(np.mean(residual ** 2))


def snr_db(sp, np_pow):
    if np_pow <= 0:
        return float("inf")
    if sp <= 0:
        return float("-inf")
    return 10.0 * math.log10(sp / np_pow)


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    n = round(SIZE / DX)
    prop_delay = math.hypot(RX[0] - TX[0], RX[1] - TX[1]) / 1500.0
    ambient = AmbientNoise(
        n_sources=N_NOISE_SOURCES,
        domain_size=SIZE,
        margin=NOISE_MARGIN,
        layout_seed=NOISE_LAYOUT_SEED,
    )

    rng = np.random.default_rng(PRBS_SEED)
    sent_bits = tuple(int(b) for b in rng.integers(0, 2, size=N_BITS))

    # -- Phase 1: characterize on a clean chirp. --
    chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)
    steps_char = round((CHIRP_DURATION + prop_delay + PROP_TAIL) / dt)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[SPEAKER.source(pos=TX, voltage_fn=chirp,
                                steps=steps_char, dt=dt)],
        damping=edge_sponge((n, n), DX),
    )
    char_writer = out.shot("char_clean")
    frames = char_writer.open((args.nframes(steps_char), n, n))
    v_tx_char = np.fromiter(
        (chirp(i * sim.dt) for i in range(steps_char)),
        dtype=np.float32, count=steps_char,
    )
    mic_p_char = np.empty(steps_char, dtype=np.float32)
    print(f"shot char_clean: {steps_char} steps")
    for i in simargs.progress(steps_char):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        mic_p_char[i] = probe.pressure(sim, RX)
    v_rx_char = MIC.filter(mic_p_char, sim.dt)
    char_dt = sim.dt

    fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
    fir.fit(v_tx_char, v_rx_char)
    char_pred = fir.predict(v_tx_char)[: len(v_rx_char)]
    train_nrmse = float(channel_model.nrmse(v_rx_char, char_pred))
    print(f"  fitted {fir.name}: training NRMSE = {train_nrmse:.4f}")
    print(f"  expected CSS processing gain over OOK: "
          f"{CSS_PROCESSING_GAIN_DB:+.2f} dB")

    # -- Phase 1b: noise-only calibration (FDTD linearity -> sigma^2 scale). --
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
    mic_p_cal = np.empty(steps_comm, dtype=np.float32)
    print(f"shot noise_calibration: sigma_ref={NOISE_SIGMA_REF:.1e}, "
          f"{steps_comm} steps")
    for i in simargs.progress(steps_comm):
        sim_cal.step()
        if i % args.capture_every == 0:
            frames_cal[i // args.capture_every] = to_numpy(sim_cal.p)
        mic_p_cal[i] = probe.pressure(sim_cal, RX)
    v_rx_noise_ref = MIC.filter(mic_p_cal, sim_cal.dt)
    warmup = int(round(NOISE_CAL_WARMUP_S / sim_cal.dt))
    noise_power_factor = noise_power_per_unit_sigma_sq(
        v_rx_noise_ref[warmup:], NOISE_SIGMA_REF,
    )
    print(f"  noise power at RX: {noise_power_factor:.3e} V^2 per sigma^2")
    cal_writer.finish(
        channels=[Channel("noise mic (V)", kind="scalar",
                          dt=sim_cal.dt, pos=RX,
                          values=v_rx_noise_ref.tolist())],
        extras={
            "role": "noise_calibration",
            "sigma_ref": float(NOISE_SIGMA_REF),
            "noise_power_per_sigma_sq": noise_power_factor,
        },
    )

    # -- Phase 2: per sigma, run OOK shot and CSS shot side-by-side. --
    ook_fn = ook_voltage(FREQ, sent_bits, BIT_DUR, drive_v=1.0)
    css_fn = css_voltage(CSS_F_LO, CSS_F_HI, sent_bits, BIT_DUR, drive_v=1.0)

    comm_writers = {}
    comm_data = {}
    sweep_rows = []

    for sigma in NOISE_SIGMAS:
        sigma_label = ("clean" if sigma == 0.0
                       else f"{sigma:.0e}".replace("+", "p").replace("-", "m"))

        for modulation, voltage_fn, decoder in [
            ("ook", ook_fn,
             lambda rx, dt_, nb, bd, pd: ook_decode(rx, dt_, nb, bd, pd)),
            ("lora", css_fn,
             lambda rx, dt_, nb, bd, pd: css_decode(
                 rx, dt_, nb, bd, pd, CSS_F_LO, CSS_F_HI,
             )),
        ]:
            name = f"{modulation}_{sigma_label}"
            noise_sources = ambient.sources(
                sigma, steps_comm, dt, drive_seed=NOISE_DRIVE_SEED,
            )
            signal_source = SPEAKER.source(
                pos=TX, voltage_fn=voltage_fn, steps=steps_comm, dt=dt,
            )
            sim = AcousticFDTD(
                n, n, DX, cfl=args.cfl, xp=args.xp,
                sources=[signal_source, *noise_sources],
                damping=edge_sponge((n, n), DX),
            )
            sw = out.shot(name)
            comm_writers[name] = sw
            frames = sw.open((args.nframes(steps_comm), n, n))
            v_tx = np.fromiter(
                (voltage_fn(i * sim.dt) for i in range(steps_comm)),
                dtype=np.float32, count=steps_comm,
            )
            mic_p = np.empty(steps_comm, dtype=np.float32)
            print(f"shot {name}: sigma={sigma:.1e}, {steps_comm} steps")
            for i in simargs.progress(steps_comm):
                sim.step()
                if i % args.capture_every == 0:
                    frames[i // args.capture_every] = to_numpy(sim.p)
                mic_p[i] = probe.pressure(sim, RX)
            v_rx_phys = MIC.filter(mic_p, sim.dt)
            v_rx_model = fir.predict(v_tx)[: len(v_rx_phys)].astype(np.float32)

            sp = float(np.mean(np.asarray(v_rx_model, dtype=np.float64) ** 2))
            np_pow = noise_power_factor * sigma * sigma
            s = snr_db(sp, np_pow)
            bits_p = decoder(v_rx_phys, sim.dt, N_BITS, BIT_DUR, prop_delay)
            bits_m = decoder(v_rx_model, sim.dt, N_BITS, BIT_DUR, prop_delay)
            ber_p = sum(d != b for d, b in zip(bits_p, sent_bits)) / N_BITS
            ber_m = sum(d != b for d, b in zip(bits_m, sent_bits)) / N_BITS
            agree = sum(p == m for p, m in zip(bits_p, bits_m)) / N_BITS
            residual = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model).astype(np.float32)

            comm_data[name] = dict(
                modulation=modulation, sigma=float(sigma),
                v_tx=v_tx, v_rx_phys=v_rx_phys, v_rx_model=v_rx_model,
                residual=residual, sim_dt=sim.dt,
                signal_power=sp, noise_power=np_pow, snr_db=s,
                ber_phys=float(ber_p), ber_model=float(ber_m),
                agreement=float(agree),
                bits_phys=list(bits_p), bits_model=list(bits_m),
            )
            sweep_rows.append({
                "modulation": modulation, "sigma": float(sigma),
                "snr_db": s, "ber_phys": float(ber_p), "ber_model": float(ber_m),
                "agreement": float(agree),
            })
            print(f"  {name}: SNR={s:+.1f} dB BER_phys={ber_p:.3f} "
                  f"BER_model={ber_m:.3f} agreement={agree:.3f}")

    def pick(modulation, key):
        return [r[key] for r in sweep_rows if r["modulation"] == modulation]

    sweep = {
        "sweep_sigmas": [float(s) for s in NOISE_SIGMAS],
        "sweep_ook_snr_db": pick("ook", "snr_db"),
        "sweep_ook_ber_phys": pick("ook", "ber_phys"),
        "sweep_ook_ber_model": pick("ook", "ber_model"),
        "sweep_ook_agreement": pick("ook", "agreement"),
        "sweep_lora_snr_db": pick("lora", "snr_db"),
        "sweep_lora_ber_phys": pick("lora", "ber_phys"),
        "sweep_lora_ber_model": pick("lora", "ber_model"),
        "sweep_lora_agreement": pick("lora", "agreement"),
        "css_processing_gain_db": float(CSS_PROCESSING_GAIN_DB),
        "noise_positions": np.asarray(ambient.positions, dtype=np.float32),
        "n_noise_sources": ambient.n_sources,
        "train_nrmse_baseline": train_nrmse,
        "sent_bits": list(sent_bits),
    }

    # -- Finalize. --
    char_writer.finish(
        channels=[
            Channel("TX (V)", kind="scalar", dt=char_dt, pos=TX,
                    values=v_tx_char.tolist()),
            Channel("RX truth (V)", kind="scalar", dt=char_dt, pos=RX,
                    values=v_rx_char.tolist()),
            Channel("RX model (V)", kind="scalar", dt=char_dt,
                    values=char_pred.tolist()),
        ],
        extras={
            "role": "characterize",
            "train_nrmse": train_nrmse,
            "fir_h": fir.h, "fir_n_taps": fir.n_taps,
            **sweep,
        },
    )

    for name, r in comm_data.items():
        sim_dt = r["sim_dt"]
        comm_writers[name].finish(
            channels=[
                Channel("TX (V)", kind="scalar", dt=sim_dt, pos=TX,
                        values=r["v_tx"].tolist()),
                Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX,
                        values=r["v_rx_phys"].tolist()),
                Channel("RX model (clean LTI) (V)", kind="scalar", dt=sim_dt,
                        values=r["v_rx_model"].tolist()),
                Channel("residual (noise est.) (V)", kind="scalar", dt=sim_dt,
                        values=r["residual"].tolist()),
            ],
            extras={
                "role": "comms",
                "modulation": r["modulation"],
                "sigma": r["sigma"],
                "signal_power": r["signal_power"],
                "noise_power": r["noise_power"],
                "snr_db": r["snr_db"],
                "ber_phys": r["ber_phys"],
                "ber_model": r["ber_model"],
                "agreement": r["agreement"],
                "decoded_phys": r["bits_phys"],
                "decoded_model": r["bits_model"],
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
