"""Proper LoRa at higher spreading factors vs OOK on the same noisy
channel.

A LoRa symbol carries ``SF`` bits via a cyclic-shifted chirp across the
BPF passband. The shift ``k`` in ``[0, 2^SF)`` is the symbol value. The
matched-filter decoder dechirps the received symbol against the basic
down-chirp and reads off the peak FFT bin.

  * Symbol duration:  ``T_sym = 2^SF / BW``
  * Bit rate:         ``SF * BW / 2^SF``
  * Processing gain:  ``10 * log10(T_sym * BW) = 10 * log10(2^SF) ~ 3*SF dB``

At ``BW = 3.75 kHz`` (the BPF passband):

  * SF=7  -> 34 ms/symbol -> 205 bps -> ~21 dB processing gain
  * SF=9  -> 137 ms/symbol -> 66 bps -> ~27 dB processing gain

The trade-off is bit rate: SF=9 sends symbols ~30x slower than OOK at
1 ms/bit. The win is enormous noise resilience. We sweep ambient noise
and compare OOK, LoRa SF=7, and LoRa SF=9 at the same sigmas.

OOK BERs come from the same code path / decoder as ``model_link_noise``
so the numbers cross-reference directly."""

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
PROP_TAIL = 0.003  # bigger tail to accommodate longer LoRa symbols
FIR_N_TAPS = 1024

CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

# OOK baseline -- same as model_lora_noise / model_link_noise.
OOK_N_BITS = 32
OOK_BIT_DUR = 0.001
OOK_PRBS_SEED = 1234

# LoRa: chirps over the BPF passband.
LORA_F_LO = 13_125.0
LORA_F_HI = 16_875.0
LORA_BW = LORA_F_HI - LORA_F_LO
# Spreading factors to test. SF=7 is the fastest (closest bit rate to OOK);
# SF=9 is much slower but ~27 dB of processing gain.
LORA_SFS = (7, 9)
# Symbols per shot. With 8 LoRa symbols at SF=7 we get 56 bits; at SF=9
# we get 72. Both give comparable BER granularity to OOK's 32-bit message.
LORA_N_SYMBOLS = 8
LORA_SYMBOL_SEED = 1234

# Ambient noise sweep -- same range as model_lora_noise, plus one
# higher level to push OOK fully into the noise floor.
N_NOISE_SOURCES = 16
NOISE_LAYOUT_SEED = 42
NOISE_DRIVE_SEED = 11
NOISE_MARGIN = 0.2  # > default 150 mm sponge depth so noise sources land in the fluid
NOISE_SIGMAS = (0.0, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5)
NOISE_SIGMA_REF = 1e-6
NOISE_CAL_WARMUP_S = 0.003


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


def ook_decode(rx, sim_dt, n_bits, bit_dur, prop_delay):
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


# ---- LoRa modulator / demodulator ----------------------------------

def lora_symbol_phase(k, sf, bw, f_lo, t_sym, t):
    """Phase at time ``t`` (in [0, t_sym]) for a LoRa symbol with shift k.

    The chirp starts at ``f_lo + k * bw / 2^SF`` and sweeps up at slope
    ``bw / t_sym``; when the frequency would exceed ``f_lo + bw`` it
    wraps to ``f_lo`` and continues sweeping. Phase is continuous across
    the wrap."""
    N = 2 ** sf
    slope = bw / t_sym
    t_wrap = t_sym * (1.0 - k / N)
    f_init = f_lo + k * bw / N
    if t <= t_wrap:
        return 2.0 * math.pi * (f_init * t + 0.5 * slope * t * t)
    phi_wrap = 2.0 * math.pi * (f_init * t_wrap + 0.5 * slope * t_wrap * t_wrap)
    dt = t - t_wrap
    return phi_wrap + 2.0 * math.pi * (f_lo * dt + 0.5 * slope * dt * dt)


def lora_voltage(symbols, sf, bw, f_lo, t_sym, drive_v=1.0):
    def v(t):
        if t < 0:
            return 0.0
        idx = int(t / t_sym)
        if idx >= len(symbols):
            return 0.0
        local = t - idx * t_sym
        k = symbols[idx]
        return drive_v * math.sin(lora_symbol_phase(k, sf, bw, f_lo, t_sym, local))
    return v


def lora_decode(rx, sim_dt, n_symbols, sf, bw, f_lo, t_sym, prop_delay):
    """Standard LoRa demod: dechirp by the base down-chirp, decimate to
    chip rate (N = 2^SF samples per symbol), N-point FFT, argmax is the
    symbol value."""
    N = 2 ** sf
    samples = np.asarray(rx, dtype=np.float64)
    sps = int(round(t_sym / sim_dt))
    delay = int(round(prop_delay / sim_dt))
    decimate_factor = max(1, sps // N)
    sps_trim = N * decimate_factor

    t_ref = np.arange(sps_trim) * sim_dt
    slope = bw / t_sym
    # The matched filter is the complex conjugate of the basic (k=0)
    # up-chirp; mixing by it leaves a constant-frequency tone at k*BW/N.
    up_chirp_phase = 2.0 * math.pi * (f_lo * t_ref + 0.5 * slope * t_ref * t_ref)
    down_chirp = np.exp(-1j * up_chirp_phase)

    decoded = []
    for i in range(n_symbols):
        start = delay + i * sps
        end = start + sps_trim
        if end > len(samples):
            decoded.append(0)
            continue
        segment = samples[start:end].astype(np.complex128)
        dechirped = segment * down_chirp
        # Decimate by averaging over each chip period.
        decimated = dechirped.reshape(N, decimate_factor).mean(axis=1)
        spectrum = np.fft.fft(decimated)
        peak_bin = int(np.argmax(np.abs(spectrum)))
        decoded.append(peak_bin)
    return tuple(decoded)


def symbols_to_bits(symbols, sf):
    bits = []
    for s in symbols:
        for b in range(sf - 1, -1, -1):
            bits.append((s >> b) & 1)
    return tuple(bits)


def oracle_sync_lora_decode(rx, sim_dt, n_symbols, sf, bw, f_lo, t_sym,
                            prop_delay, truth_bits, search_us=500.0):
    """Decode LoRa with ORACLE sync: sweep small delay offsets around
    ``prop_delay``; for each offset, decode and count bit errors against
    the known sent bits; pick the offset that minimises errors.

    This is a cheat -- a real receiver can't see the sent bits and has
    to find sync some other way (preamble-based correlation). The point
    here is to *isolate* the alignment problem from the "how to find
    alignment" problem: if oracle-sync decode hits zero error, the
    matched filter is fine and a no-sync run's failure is purely the
    receiver lacking a sync signal."""
    search_samples = int(round(search_us * 1e-6 / sim_dt))
    best_errors = float("inf")
    best_offset = 0
    best_bits = None
    for off in range(-search_samples, search_samples + 1):
        delay = prop_delay + off * sim_dt
        if delay < 0:
            continue
        syms = lora_decode(rx, sim_dt, n_symbols, sf, bw, f_lo, t_sym, delay)
        bits = symbols_to_bits(syms, sf)
        errors = sum(d != b for d, b in zip(bits, truth_bits))
        if errors < best_errors:
            best_errors = errors
            best_offset = off
            best_bits = bits
        if best_errors == 0:
            break
    return best_bits, best_offset, best_errors


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
        n_sources=N_NOISE_SOURCES, domain_size=SIZE,
        margin=NOISE_MARGIN, layout_seed=NOISE_LAYOUT_SEED,
    )

    rng_ook = np.random.default_rng(OOK_PRBS_SEED)
    ook_bits = tuple(int(b) for b in rng_ook.integers(0, 2, size=OOK_N_BITS))

    lora_symbols = {}
    lora_truth_bits = {}
    lora_t_sym = {}
    for sf in LORA_SFS:
        N = 2 ** sf
        rng_l = np.random.default_rng(LORA_SYMBOL_SEED + sf)
        syms = tuple(int(s) for s in rng_l.integers(0, N, size=LORA_N_SYMBOLS))
        lora_symbols[sf] = syms
        lora_truth_bits[sf] = symbols_to_bits(syms, sf)
        lora_t_sym[sf] = N / LORA_BW
        print(f"LoRa SF={sf}: T_sym = {lora_t_sym[sf] * 1e3:.1f} ms, "
              f"{LORA_N_SYMBOLS} symbols = {sf * LORA_N_SYMBOLS} bits, "
              f"processing gain = {10*math.log10(2**sf):+.1f} dB")

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

    # -- Phase 1b: noise-only calibration. --
    # The longest comms shot here is the SF=9 LoRa run; we calibrate over
    # the same duration so steady-state stats are meaningful for it too.
    longest_duration = max(
        OOK_BIT_DUR * OOK_N_BITS,
        max((2 ** sf) / LORA_BW * LORA_N_SYMBOLS for sf in LORA_SFS),
    )
    steps_cal = round((longest_duration + prop_delay + PROP_TAIL) / dt)
    sim_cal = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=ambient.sources(
            NOISE_SIGMA_REF, steps_cal, dt, drive_seed=NOISE_DRIVE_SEED,
        ),
        damping=edge_sponge((n, n), DX),
    )
    cal_writer = out.shot("noise_calibration")
    frames_cal = cal_writer.open((args.nframes(steps_cal), n, n))
    mic_p_cal = np.empty(steps_cal, dtype=np.float32)
    print(f"shot noise_calibration: sigma_ref={NOISE_SIGMA_REF:.1e}, "
          f"{steps_cal} steps")
    for i in simargs.progress(steps_cal):
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

    # -- Phase 2: per sigma, run OOK shot + LoRa shots. --
    ook_fn = ook_voltage(FREQ, ook_bits, OOK_BIT_DUR, drive_v=1.0)

    comm_writers = {}
    comm_data = {}
    sweep_rows = []

    for sigma in NOISE_SIGMAS:
        sigma_label = ("clean" if sigma == 0.0
                       else f"{sigma:.0e}".replace("+", "p").replace("-", "m"))

        # Each config: (mod_name, voltage_fn, duration_s, kind, sf_or_None,
        # truth_bits, n_bits). LoRa kind triggers the two-pass decode
        # (no-sync, oracle-sync) below; OOK is single-pass.
        configs = [
            ("ook", ook_fn, OOK_BIT_DUR * OOK_N_BITS, "ook", None,
             ook_bits, OOK_N_BITS),
        ]
        for sf in LORA_SFS:
            t_sym = lora_t_sym[sf]
            voltage_fn = lora_voltage(
                lora_symbols[sf], sf, LORA_BW, LORA_F_LO, t_sym, 1.0,
            )
            n_bits_lora = sf * LORA_N_SYMBOLS
            configs.append((
                f"lora_sf{sf}", voltage_fn, t_sym * LORA_N_SYMBOLS, "lora", sf,
                lora_truth_bits[sf], n_bits_lora,
            ))

        for modulation, voltage_fn, duration_s, kind, sf, truth_bits, n_bits in configs:
            name = f"{modulation}_{sigma_label}"
            steps_comm = round((duration_s + prop_delay + PROP_TAIL) / dt)
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
            print(f"shot {name}: sigma={sigma:.1e}, {steps_comm} steps, "
                  f"{duration_s*1e3:.0f} ms message")
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

            # --- decode pass 1: no sync (naive prop_delay) ---
            if kind == "ook":
                bits_p_nosync = ook_decode(
                    v_rx_phys, sim.dt, OOK_N_BITS, OOK_BIT_DUR, prop_delay,
                )
                bits_m_nosync = ook_decode(
                    v_rx_model, sim.dt, OOK_N_BITS, OOK_BIT_DUR, prop_delay,
                )
            else:
                t_sym = lora_t_sym[sf]
                bits_p_nosync = symbols_to_bits(lora_decode(
                    v_rx_phys, sim.dt, LORA_N_SYMBOLS, sf,
                    LORA_BW, LORA_F_LO, t_sym, prop_delay,
                ), sf)
                bits_m_nosync = symbols_to_bits(lora_decode(
                    v_rx_model, sim.dt, LORA_N_SYMBOLS, sf,
                    LORA_BW, LORA_F_LO, t_sym, prop_delay,
                ), sf)

            err_p_nosync = sum(d != b for d, b in zip(bits_p_nosync, truth_bits))
            err_m_nosync = sum(d != b for d, b in zip(bits_m_nosync, truth_bits))
            ber_p_nosync = err_p_nosync / n_bits
            ber_m_nosync = err_m_nosync / n_bits

            # --- decode pass 2: oracle sync (LoRa only; cheat) ---
            if kind == "lora":
                bits_p_oracle, off_p, _ = oracle_sync_lora_decode(
                    v_rx_phys, sim.dt, LORA_N_SYMBOLS, sf,
                    LORA_BW, LORA_F_LO, t_sym, prop_delay, truth_bits,
                )
                bits_m_oracle, off_m, _ = oracle_sync_lora_decode(
                    v_rx_model, sim.dt, LORA_N_SYMBOLS, sf,
                    LORA_BW, LORA_F_LO, t_sym, prop_delay, truth_bits,
                )
                oracle_offset_p_us = off_p * sim.dt * 1e6
                oracle_offset_m_us = off_m * sim.dt * 1e6
            else:
                # OOK's slicer is alignment-robust; oracle == no-sync.
                bits_p_oracle = bits_p_nosync
                bits_m_oracle = bits_m_nosync
                oracle_offset_p_us = 0.0
                oracle_offset_m_us = 0.0

            err_p_oracle = sum(d != b for d, b in zip(bits_p_oracle, truth_bits))
            err_m_oracle = sum(d != b for d, b in zip(bits_m_oracle, truth_bits))
            ber_p_oracle = err_p_oracle / n_bits
            ber_m_oracle = err_m_oracle / n_bits

            agree_nosync = sum(p == m for p, m in zip(bits_p_nosync, bits_m_nosync)) / n_bits
            agree_oracle = sum(p == m for p, m in zip(bits_p_oracle, bits_m_oracle)) / n_bits
            residual = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model).astype(np.float32)

            comm_data[name] = dict(
                modulation=modulation, kind=kind, sigma=float(sigma),
                v_tx=v_tx, v_rx_phys=v_rx_phys, v_rx_model=v_rx_model,
                residual=residual, sim_dt=sim.dt,
                signal_power=sp, noise_power=np_pow, snr_db=s,
                n_bits=n_bits,
                ber_phys_nosync=float(ber_p_nosync),
                ber_model_nosync=float(ber_m_nosync),
                ber_phys_oracle=float(ber_p_oracle),
                ber_model_oracle=float(ber_m_oracle),
                errors_phys_nosync=int(err_p_nosync),
                errors_model_nosync=int(err_m_nosync),
                errors_phys_oracle=int(err_p_oracle),
                errors_model_oracle=int(err_m_oracle),
                agreement_nosync=float(agree_nosync),
                agreement_oracle=float(agree_oracle),
                oracle_offset_phys_us=float(oracle_offset_p_us),
                oracle_offset_model_us=float(oracle_offset_m_us),
            )
            sweep_rows.append({
                "modulation": modulation, "sigma": float(sigma),
                "snr_db": s,
                "ber_phys_nosync": float(ber_p_nosync),
                "ber_model_nosync": float(ber_m_nosync),
                "ber_phys_oracle": float(ber_p_oracle),
                "ber_model_oracle": float(ber_m_oracle),
            })
            extra_info = ""
            if kind == "lora":
                extra_info = (f"  oracle offset phys={oracle_offset_p_us:+.1f} us, "
                              f"model={oracle_offset_m_us:+.1f} us")
            print(f"  {name}: SNR={s:+.1f} dB  "
                  f"BER nosync phys={ber_p_nosync:.3f} ({err_p_nosync}/{n_bits}), "
                  f"oracle phys={ber_p_oracle:.3f} ({err_p_oracle}/{n_bits})" + extra_info)

    def pick(modulation, key):
        return [r[key] for r in sweep_rows if r["modulation"] == modulation]

    sweep = {
        "sweep_sigmas": [float(s) for s in NOISE_SIGMAS],
    }
    for mod in ["ook"] + [f"lora_sf{sf}" for sf in LORA_SFS]:
        sweep[f"sweep_{mod}_snr_db"] = pick(mod, "snr_db")
        sweep[f"sweep_{mod}_ber_phys_nosync"] = pick(mod, "ber_phys_nosync")
        sweep[f"sweep_{mod}_ber_phys_oracle"] = pick(mod, "ber_phys_oracle")
    sweep["lora_sfs"] = list(LORA_SFS)
    sweep["lora_processing_gains_db"] = [10 * math.log10(2 ** sf) for sf in LORA_SFS]
    sweep["noise_positions"] = np.asarray(ambient.positions, dtype=np.float32)
    sweep["n_noise_sources"] = ambient.n_sources
    sweep["train_nrmse_baseline"] = train_nrmse

    char_writer.finish(
        channels=[
            Channel("TX (V)", kind="scalar", dt=char_dt, pos=TX, values=v_tx_char.tolist()),
            Channel("RX truth (V)", kind="scalar", dt=char_dt, pos=RX, values=v_rx_char.tolist()),
            Channel("RX model (V)", kind="scalar", dt=char_dt, values=char_pred.tolist()),
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
                Channel("TX (V)", kind="scalar", dt=sim_dt, pos=TX, values=r["v_tx"].tolist()),
                Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX, values=r["v_rx_phys"].tolist()),
                Channel("RX model (clean LTI) (V)", kind="scalar", dt=sim_dt, values=r["v_rx_model"].tolist()),
                Channel("residual (V)", kind="scalar", dt=sim_dt, values=r["residual"].tolist()),
            ],
            extras={
                "role": "comms",
                "modulation": r["modulation"], "kind": r["kind"],
                "sigma": r["sigma"],
                "signal_power": r["signal_power"], "noise_power": r["noise_power"],
                "snr_db": r["snr_db"],
                "n_bits": r["n_bits"],
                "ber_phys_nosync": r["ber_phys_nosync"],
                "ber_model_nosync": r["ber_model_nosync"],
                "ber_phys_oracle": r["ber_phys_oracle"],
                "ber_model_oracle": r["ber_model_oracle"],
                "errors_phys_nosync": r["errors_phys_nosync"],
                "errors_model_nosync": r["errors_model_nosync"],
                "errors_phys_oracle": r["errors_phys_oracle"],
                "errors_model_oracle": r["errors_model_oracle"],
                "agreement_nosync": r["agreement_nosync"],
                "agreement_oracle": r["agreement_oracle"],
                "oracle_offset_phys_us": r["oracle_offset_phys_us"],
                "oracle_offset_model_us": r["oracle_offset_model_us"],
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
