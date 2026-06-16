"""LoRa with a real preamble-based sync receiver.

Same noisy channel, same TX/RX as ``model_lora_sf`` / ``model_lora_noise``,
but each LoRa frame is now ``[preamble | payload]``:

  * preamble = ``N_PREAMBLE`` known up-chirp symbols (all k=0). On the
    wire, available to any receiver, not a cheat.
  * payload  = ``N_PAYLOAD`` random symbols.

The receiver sweeps small delay offsets around the geometric prop_delay;
at each offset it decodes the preamble and counts how many symbols come
back as 0; the offset with the most preamble matches is the sync point.
Payload then decodes at ``sync_offset + N_PREAMBLE * T_sym``. This is
roughly what a real LoRa modem does (real ones correlate continuously
against the preamble template, but the idea is the same).

Per shot we decode three ways and store all three BERs on the payload
bits (preamble is overhead, doesn't count toward BER):

  * **no sync** -- decode payload at geometric prop_delay only.
  * **preamble sync** -- find the sync from the preamble, decode payload
    at sync offset.
  * **oracle sync** -- cheat and sweep against truth bits; upper bound
    on what any sync mechanism could achieve.

The red/yellow/green spread per SF shows the educational point:
preamble sync recovers most of the gap between no-sync (broken) and
oracle (best possible), at the cost of a known preamble's overhead.
At very high noise the preamble itself fails and the curve drifts back
toward no-sync."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.noise import AmbientNoise, noise_power_per_unit_sigma_sq
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

# Shots in this sim span 30 ms (OOK) to 1.6 s (LoRa SF=9), so a fixed
# capture_every gives wildly different per-shot disk costs at the default
# 16. Bumping to 1024 keeps the longest SF=9 shot at ~230 MB while still
# leaving ~30 frames of animation for the shortest OOK shot.
DEFAULTS = {"capture_every": 1024}

SIZE = 1.0
FREQ = 15_000.0
TX = (0.2, 0.5)
RX = (0.8, 0.5)
SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.003
FIR_N_TAPS = 1024

CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

OOK_N_BITS = 32
OOK_BIT_DUR = 0.001
OOK_PRBS_SEED = 1234

LORA_F_LO = 13_125.0
LORA_F_HI = 16_875.0
LORA_BW = LORA_F_HI - LORA_F_LO
LORA_SFS = (7, 9)
N_PREAMBLE = 4         # known up-chirps for sync
N_PAYLOAD = 8          # data symbols (same as model_lora_sf for compare)
LORA_SYMBOL_SEED = 1234
# How wide to sweep for sync (oracle and preamble). 500 us covers the
# BPF group-delay slop comfortably without overlapping adjacent symbols.
SYNC_SEARCH_US = 500.0

N_NOISE_SOURCES = 16
NOISE_LAYOUT_SEED = 42
NOISE_DRIVE_SEED = 11
NOISE_MARGIN = 0.2  # > default 150 mm sponge depth so noise sources land in the fluid
NOISE_SIGMAS = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5)
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


def lora_symbol_phase(k, sf, bw, f_lo, t_sym, t):
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
    N = 2 ** sf
    samples = np.asarray(rx, dtype=np.float64)
    sps = int(round(t_sym / sim_dt))
    delay = int(round(prop_delay / sim_dt))
    decimate_factor = max(1, sps // N)
    sps_trim = N * decimate_factor

    t_ref = np.arange(sps_trim) * sim_dt
    slope = bw / t_sym
    up_chirp_phase = 2.0 * math.pi * (f_lo * t_ref + 0.5 * slope * t_ref * t_ref)
    down_chirp = np.exp(-1j * up_chirp_phase)

    decoded = []
    for i in range(n_symbols):
        start = delay + i * sps
        end = start + sps_trim
        if start < 0 or end > len(samples):
            decoded.append(0)
            continue
        segment = samples[start:end].astype(np.complex128)
        dechirped = segment * down_chirp
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


def preamble_sync_lora_decode(rx, sim_dt, sf, bw, f_lo, t_sym,
                              prop_delay, n_preamble, n_payload,
                              search_us=SYNC_SEARCH_US):
    """Real preamble-based sync: sweep small offsets, decode the
    preamble at each, pick the offset with the most preamble symbols
    decoding as 0. Then decode the payload at that offset.

    Returns ``(preamble_symbols, payload_symbols, sync_offset_samples,
    preamble_matches)``."""
    search_samples = int(round(search_us * 1e-6 / sim_dt))
    best_matches = -1
    best_offset = 0
    best_preamble = None
    for off in range(-search_samples, search_samples + 1):
        delay = prop_delay + off * sim_dt
        if delay < 0:
            continue
        preamble = lora_decode(
            rx, sim_dt, n_preamble, sf, bw, f_lo, t_sym, delay,
        )
        matches = sum(1 for s in preamble if s == 0)
        if matches > best_matches:
            best_matches = matches
            best_offset = off
            best_preamble = preamble
        if matches == n_preamble:
            break
    sync_delay = prop_delay + best_offset * sim_dt
    payload_delay = sync_delay + n_preamble * t_sym
    payload = lora_decode(
        rx, sim_dt, n_payload, sf, bw, f_lo, t_sym, payload_delay,
    )
    return best_preamble, payload, best_offset, best_matches


def oracle_sync_lora_decode(rx, sim_dt, sf, bw, f_lo, t_sym,
                            prop_delay, n_preamble, n_payload, truth_payload_bits,
                            search_us=SYNC_SEARCH_US):
    """Cheat: sweep offsets, decode payload, count errors against truth,
    pick the offset with the fewest payload errors. Upper-bound reference."""
    search_samples = int(round(search_us * 1e-6 / sim_dt))
    best_errors = float("inf")
    best_offset = 0
    best_payload = None
    for off in range(-search_samples, search_samples + 1):
        delay = prop_delay + off * sim_dt
        if delay < 0:
            continue
        # Skip the preamble interval at this candidate offset.
        payload_delay = delay + n_preamble * t_sym
        payload = lora_decode(
            rx, sim_dt, n_payload, sf, bw, f_lo, t_sym, payload_delay,
        )
        bits = symbols_to_bits(payload, sf)
        errors = sum(d != b for d, b in zip(bits, truth_payload_bits))
        if errors < best_errors:
            best_errors = errors
            best_offset = off
            best_payload = payload
        if errors == 0:
            break
    return best_payload, best_offset, best_errors


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

    # LoRa frames: preamble (all zeros) + random payload.
    lora_frames = {}        # sf -> full symbol tuple (preamble + payload)
    lora_payload_syms = {}  # sf -> payload symbols (for oracle decode)
    lora_payload_bits = {}  # sf -> payload bits (truth for BER)
    lora_t_sym = {}
    for sf in LORA_SFS:
        N = 2 ** sf
        rng_l = np.random.default_rng(LORA_SYMBOL_SEED + sf)
        payload = tuple(int(s) for s in rng_l.integers(0, N, size=N_PAYLOAD))
        preamble = tuple([0] * N_PREAMBLE)
        lora_frames[sf] = preamble + payload
        lora_payload_syms[sf] = payload
        lora_payload_bits[sf] = symbols_to_bits(payload, sf)
        lora_t_sym[sf] = N / LORA_BW
        print(f"LoRa SF={sf}: T_sym = {lora_t_sym[sf] * 1e3:.1f} ms, "
              f"frame = {N_PREAMBLE} preamble + {N_PAYLOAD} payload = "
              f"{(N_PREAMBLE + N_PAYLOAD) * lora_t_sym[sf] * 1e3:.0f} ms, "
              f"{sf * N_PAYLOAD} payload bits, "
              f"processing gain = {10*math.log10(2**sf):+.1f} dB")

    # -- Phase 1: characterize. --
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
    longest_duration = max(
        OOK_BIT_DUR * OOK_N_BITS,
        max((2 ** sf) / LORA_BW * (N_PREAMBLE + N_PAYLOAD) for sf in LORA_SFS),
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

    # -- Phase 2: sigma sweep, OOK + LoRa shots with 3 decode modes. --
    ook_fn = ook_voltage(FREQ, ook_bits, OOK_BIT_DUR, drive_v=1.0)

    comm_writers = {}
    comm_data = {}
    sweep_rows = []

    for sigma in NOISE_SIGMAS:
        sigma_label = ("clean" if sigma == 0.0
                       else f"{sigma:.0e}".replace("+", "p").replace("-", "m"))

        configs = [
            ("ook", ook_fn, OOK_BIT_DUR * OOK_N_BITS, "ook", None,
             ook_bits, OOK_N_BITS),
        ]
        for sf in LORA_SFS:
            t_sym = lora_t_sym[sf]
            frame_syms = lora_frames[sf]
            voltage_fn = lora_voltage(
                frame_syms, sf, LORA_BW, LORA_F_LO, t_sym, 1.0,
            )
            n_payload_bits = sf * N_PAYLOAD
            configs.append((
                f"lora_sf{sf}", voltage_fn,
                t_sym * (N_PREAMBLE + N_PAYLOAD),
                "lora", sf, lora_payload_bits[sf], n_payload_bits,
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

            if kind == "ook":
                # OOK: one decode is enough (slicer is alignment-robust).
                bits_p = ook_decode(
                    v_rx_phys, sim.dt, OOK_N_BITS, OOK_BIT_DUR, prop_delay,
                )
                bits_m = ook_decode(
                    v_rx_model, sim.dt, OOK_N_BITS, OOK_BIT_DUR, prop_delay,
                )
                bits_nosync_p = bits_preamble_p = bits_oracle_p = bits_p
                bits_nosync_m = bits_preamble_m = bits_oracle_m = bits_m
                preamble_matches_p = preamble_matches_m = -1
                pre_off_p = pre_off_m = ora_off_p = ora_off_m = 0
            else:
                t_sym = lora_t_sym[sf]

                # -- no sync: payload starts at prop_delay + preamble interval --
                payload_delay_naive = prop_delay + N_PREAMBLE * t_sym
                bits_nosync_p = symbols_to_bits(lora_decode(
                    v_rx_phys, sim.dt, N_PAYLOAD, sf,
                    LORA_BW, LORA_F_LO, t_sym, payload_delay_naive,
                ), sf)
                bits_nosync_m = symbols_to_bits(lora_decode(
                    v_rx_model, sim.dt, N_PAYLOAD, sf,
                    LORA_BW, LORA_F_LO, t_sym, payload_delay_naive,
                ), sf)

                # -- preamble sync: receiver finds offset from preamble --
                pre_p, payload_p, pre_off_p, preamble_matches_p = \
                    preamble_sync_lora_decode(
                        v_rx_phys, sim.dt, sf, LORA_BW, LORA_F_LO, t_sym,
                        prop_delay, N_PREAMBLE, N_PAYLOAD,
                    )
                pre_m, payload_m, pre_off_m, preamble_matches_m = \
                    preamble_sync_lora_decode(
                        v_rx_model, sim.dt, sf, LORA_BW, LORA_F_LO, t_sym,
                        prop_delay, N_PREAMBLE, N_PAYLOAD,
                    )
                bits_preamble_p = symbols_to_bits(payload_p, sf)
                bits_preamble_m = symbols_to_bits(payload_m, sf)

                # -- oracle sync: cheat against truth --
                payload_p_o, ora_off_p, _ = oracle_sync_lora_decode(
                    v_rx_phys, sim.dt, sf, LORA_BW, LORA_F_LO, t_sym,
                    prop_delay, N_PREAMBLE, N_PAYLOAD, truth_bits,
                )
                payload_m_o, ora_off_m, _ = oracle_sync_lora_decode(
                    v_rx_model, sim.dt, sf, LORA_BW, LORA_F_LO, t_sym,
                    prop_delay, N_PREAMBLE, N_PAYLOAD, truth_bits,
                )
                bits_oracle_p = symbols_to_bits(payload_p_o, sf)
                bits_oracle_m = symbols_to_bits(payload_m_o, sf)

            def _ber(bits):
                return sum(d != b for d, b in zip(bits, truth_bits)) / n_bits

            ber_p_nosync = _ber(bits_nosync_p)
            ber_m_nosync = _ber(bits_nosync_m)
            ber_p_pre = _ber(bits_preamble_p)
            ber_m_pre = _ber(bits_preamble_m)
            ber_p_oracle = _ber(bits_oracle_p)
            ber_m_oracle = _ber(bits_oracle_m)
            residual = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model).astype(np.float32)

            comm_data[name] = dict(
                modulation=modulation, kind=kind, sigma=float(sigma),
                v_tx=v_tx, v_rx_phys=v_rx_phys, v_rx_model=v_rx_model,
                residual=residual, sim_dt=sim.dt,
                signal_power=sp, noise_power=np_pow, snr_db=s,
                n_bits=n_bits,
                ber_phys_nosync=float(ber_p_nosync),
                ber_model_nosync=float(ber_m_nosync),
                ber_phys_preamble=float(ber_p_pre),
                ber_model_preamble=float(ber_m_pre),
                ber_phys_oracle=float(ber_p_oracle),
                ber_model_oracle=float(ber_m_oracle),
                preamble_matches_phys=int(preamble_matches_p),
                preamble_matches_model=int(preamble_matches_m),
                preamble_offset_phys_us=float(pre_off_p * sim.dt * 1e6),
                preamble_offset_model_us=float(pre_off_m * sim.dt * 1e6),
                oracle_offset_phys_us=float(ora_off_p * sim.dt * 1e6),
                oracle_offset_model_us=float(ora_off_m * sim.dt * 1e6),
            )
            sweep_rows.append({
                "modulation": modulation, "sigma": float(sigma),
                "snr_db": s,
                "ber_phys_nosync": float(ber_p_nosync),
                "ber_phys_preamble": float(ber_p_pre),
                "ber_phys_oracle": float(ber_p_oracle),
            })
            extra_info = ""
            if kind == "lora":
                extra_info = (f"  preamble matches phys={preamble_matches_p}/{N_PREAMBLE} "
                              f"at {pre_off_p * sim.dt * 1e6:+.1f} us, "
                              f"oracle offset phys={ora_off_p * sim.dt * 1e6:+.1f} us")
            print(f"  {name}: SNR={s:+.1f} dB  "
                  f"BER nosync={ber_p_nosync:.3f}  "
                  f"preamble={ber_p_pre:.3f}  "
                  f"oracle={ber_p_oracle:.3f}" + extra_info)

    def pick(modulation, key):
        return [r[key] for r in sweep_rows if r["modulation"] == modulation]

    sweep = {
        "sweep_sigmas": [float(s) for s in NOISE_SIGMAS],
    }
    for mod in ["ook"] + [f"lora_sf{sf}" for sf in LORA_SFS]:
        sweep[f"sweep_{mod}_snr_db"] = pick(mod, "snr_db")
        sweep[f"sweep_{mod}_ber_phys_nosync"] = pick(mod, "ber_phys_nosync")
        sweep[f"sweep_{mod}_ber_phys_preamble"] = pick(mod, "ber_phys_preamble")
        sweep[f"sweep_{mod}_ber_phys_oracle"] = pick(mod, "ber_phys_oracle")
    sweep["lora_sfs"] = list(LORA_SFS)
    sweep["lora_processing_gains_db"] = [10 * math.log10(2 ** sf) for sf in LORA_SFS]
    sweep["n_preamble"] = N_PREAMBLE
    sweep["n_payload"] = N_PAYLOAD
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
                "ber_phys_preamble": r["ber_phys_preamble"],
                "ber_model_preamble": r["ber_model_preamble"],
                "ber_phys_oracle": r["ber_phys_oracle"],
                "ber_model_oracle": r["ber_model_oracle"],
                "preamble_matches_phys": r["preamble_matches_phys"],
                "preamble_matches_model": r["preamble_matches_model"],
                "preamble_offset_phys_us": r["preamble_offset_phys_us"],
                "preamble_offset_model_us": r["preamble_offset_model_us"],
                "oracle_offset_phys_us": r["oracle_offset_phys_us"],
                "oracle_offset_model_us": r["oracle_offset_model_us"],
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
