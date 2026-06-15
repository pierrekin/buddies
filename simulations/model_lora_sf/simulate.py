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
from buddies.noise import AmbientNoise
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
NOISE_MARGIN = 0.1
NOISE_SIGMAS = (0.0, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5)


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
    # LoRa's matched filter is alignment-sensitive (unlike OOK's RMS
    # slicer). The FIR's impulse-response peak sits where the channel
    # rings strongest, not necessarily where a symbol's chirp phase
    # progression best matches the dechirp reference -- BPF post-ringing
    # pulls the peak rightward. The *leading edge* (first lag where the
    # impulse response crosses half its peak) is a closer proxy for the
    # actual symbol arrival, so we use that for the LoRa demod window.
    h_abs = np.abs(fir.h.astype(np.float64))
    threshold = 0.5 * float(h_abs.max())
    above = np.where(h_abs >= threshold)[0]
    decode_delay_samples = int(above[0]) if len(above) else int(np.argmax(h_abs))
    decode_delay = decode_delay_samples * char_dt
    print(f"  fitted {fir.name}: training NRMSE = {train_nrmse:.4f}; "
          f"geometric prop_delay = {prop_delay * 1e6:.1f} us, "
          f"empirical decode_delay (FIR leading edge) "
          f"= {decode_delay * 1e6:.1f} us")

    # -- Phase 2: per sigma, run OOK shot + LoRa shots. --
    ook_fn = ook_voltage(FREQ, ook_bits, OOK_BIT_DUR, drive_v=1.0)

    comm_writers = {}
    comm_data = {}
    sweep_rows = []

    for sigma in NOISE_SIGMAS:
        sigma_label = ("clean" if sigma == 0.0
                       else f"{sigma:.0e}".replace("+", "p").replace("-", "m"))

        configs = [
            # OOK uses geometric prop_delay (slicer is alignment-robust).
            ("ook", ook_fn, OOK_BIT_DUR * OOK_N_BITS,
             lambda rx, dt_: ook_decode(rx, dt_, OOK_N_BITS, OOK_BIT_DUR, prop_delay),
             ook_bits, OOK_N_BITS),
        ]
        for sf in LORA_SFS:
            t_sym = lora_t_sym[sf]
            syms = lora_symbols[sf]
            truth = lora_truth_bits[sf]
            voltage_fn = lora_voltage(syms, sf, LORA_BW, LORA_F_LO, t_sym, 1.0)
            n_bits_lora = sf * LORA_N_SYMBOLS
            def make_dec(sf=sf, t_sym=t_sym):
                def dec(rx, dt_):
                    # LoRa uses the empirical delay (FIR peak) so the
                    # symbol windows align with the BPF-filtered signal.
                    syms_dec = lora_decode(
                        rx, dt_, LORA_N_SYMBOLS, sf, LORA_BW, LORA_F_LO, t_sym, decode_delay,
                    )
                    return symbols_to_bits(syms_dec, sf)
                return dec
            configs.append((
                f"lora_sf{sf}", voltage_fn, t_sym * LORA_N_SYMBOLS,
                make_dec(), truth, n_bits_lora,
            ))

        for modulation, voltage_fn, duration_s, decoder, truth_bits, n_bits in configs:
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

            sp, np_pow = signal_noise_power(v_rx_phys, v_rx_model)
            s = snr_db(sp, np_pow)
            bits_p = decoder(v_rx_phys, sim.dt)
            bits_m = decoder(v_rx_model, sim.dt)
            errors_p = sum(d != b for d, b in zip(bits_p, truth_bits))
            errors_m = sum(d != b for d, b in zip(bits_m, truth_bits))
            ber_p = errors_p / n_bits
            ber_m = errors_m / n_bits
            agree = sum(p == m for p, m in zip(bits_p, bits_m)) / n_bits
            residual = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model).astype(np.float32)

            comm_data[name] = dict(
                modulation=modulation, sigma=float(sigma),
                v_tx=v_tx, v_rx_phys=v_rx_phys, v_rx_model=v_rx_model,
                residual=residual, sim_dt=sim.dt,
                signal_power=sp, noise_power=np_pow, snr_db=s,
                n_bits=n_bits,
                ber_phys=float(ber_p), ber_model=float(ber_m),
                agreement=float(agree),
                errors_phys=int(errors_p), errors_model=int(errors_m),
            )
            sweep_rows.append({
                "modulation": modulation, "sigma": float(sigma),
                "snr_db": s, "ber_phys": float(ber_p), "ber_model": float(ber_m),
                "agreement": float(agree),
            })
            print(f"  {name}: SNR={s:+.1f} dB BER_phys={ber_p:.3f} "
                  f"({errors_p}/{n_bits}) BER_model={ber_m:.3f}  "
                  f"agreement={agree:.3f}")

    def pick(modulation, key):
        return [r[key] for r in sweep_rows if r["modulation"] == modulation]

    sweep = {
        "sweep_sigmas": [float(s) for s in NOISE_SIGMAS],
    }
    for mod in ["ook"] + [f"lora_sf{sf}" for sf in LORA_SFS]:
        sweep[f"sweep_{mod}_snr_db"] = pick(mod, "snr_db")
        sweep[f"sweep_{mod}_ber_phys"] = pick(mod, "ber_phys")
        sweep[f"sweep_{mod}_ber_model"] = pick(mod, "ber_model")
        sweep[f"sweep_{mod}_agreement"] = pick(mod, "agreement")
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
                "modulation": r["modulation"], "sigma": r["sigma"],
                "signal_power": r["signal_power"], "noise_power": r["noise_power"],
                "snr_db": r["snr_db"],
                "n_bits": r["n_bits"],
                "ber_phys": r["ber_phys"], "ber_model": r["ber_model"],
                "agreement": r["agreement"],
                "errors_phys": r["errors_phys"], "errors_model": r["errors_model"],
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
