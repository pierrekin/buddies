"""TX beamforming with ambient noise: does the array buy SNR gain?

Two TX configurations send the same OOK PRBS through the same noisy
channel and we compare their BER:

  * ``single``     -- one Speaker at the array's centre point
  * ``broadside``  -- all 8 elements firing the same OOK in phase (a
                      far-field broadside beam, i.e. uniformly weighted
                      delay-and-sum focused along +x)

Both are stressed under a sweep of ambient-noise sigmas. Ambient noise
comes from sources scattered across the domain interior (not from the
array), so the noise field at RX is identical between configurations
and the only thing that changes is the signal amplitude.

Standard comms metrics on the output:

  * ``snr_db`` at the receiver, computed per shot from
    ``10 * log10(signal_power / noise_power)`` where signal power is
    ``RMS(v_rx_model)^2`` (LTI-predicted, noise-free) and noise power
    is ``RMS(v_rx_phys - v_rx_model)^2`` (the residual = noise).
  * ``BER vs SNR`` waterfall (log y, dB x) -- the canonical performance
    figure for a comm system.
  * ``array gain`` (dB) = SNR difference at a fixed BER target (default
    1%); the headline number an array-and-noise study reports.

For 8 coherent emitters the predicted array gain is ~18 dB
(20 * log10(8)): 8x amplitude at RX, noise unchanged. We measure and
quote it directly."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.noise import AmbientNoise
from buddies.sim import AcousticFDTD, edge_sponge, line, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 1.0
FREQ = 15_000.0
C_EST = 1500.0
LAMBDA = C_EST / FREQ
N_ELEMENTS = 8
SPACING = LAMBDA / 2
APERTURE = (N_ELEMENTS - 1) * SPACING
ARRAY_X = 0.1
ARRAY_CY = 0.5
RX = (0.8, 0.5)
# Single-source baseline sits at the array's geometric centre so it has
# the same average TX->RX range as the array; only the number-of-elements
# differs.
SINGLE_SOURCE_POS = (ARRAY_X, ARRAY_CY)

SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.002
FIR_N_TAPS = 1024

CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

N_BITS = 32
BIT_DUR = 0.001
PRBS_SEED = 1234

N_NOISE_SOURCES = 16
NOISE_LAYOUT_SEED = 42
NOISE_DRIVE_SEED = 11
NOISE_MARGIN = 0.1
# Wide enough to span BER 0 -> ~0.5 for both the single-source and the
# 8-element broadside curves. The broadside curve sits ~18 dB to the right
# of single, so the sweep extends well past where single fully fails.
NOISE_SIGMAS = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5)
# Target BER at which to report the array-gain headline number.
ARRAY_GAIN_TARGET_BER = 0.01


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


def array_positions():
    start = (ARRAY_X, ARRAY_CY - APERTURE / 2)
    end = (ARRAY_X, ARRAY_CY + APERTURE / 2)
    return line(start, end, N_ELEMENTS)


def signal_noise_power(v_rx_phys, v_rx_model):
    """LTI-decomposition: signal = model prediction, noise = residual.

    Returns ``(signal_power, noise_power)`` in V^2 (per the channel
    voltage scale)."""
    v_phys = np.asarray(v_rx_phys, dtype=np.float64)
    v_model = np.asarray(v_rx_model, dtype=np.float64)[: len(v_phys)]
    residual = v_phys - v_model
    return float(np.mean(v_model ** 2)), float(np.mean(residual ** 2))


def snr_db(signal_power, noise_power):
    if noise_power <= 0:
        return float("inf")
    if signal_power <= 0:
        return float("-inf")
    return 10.0 * math.log10(signal_power / noise_power)


def interpolate_snr_at_ber(snr_db_arr, ber_arr, target_ber):
    """Linear-in-(snr, log10(ber)) interpolation for the SNR at which the
    curve crosses ``target_ber``. Returns ``None`` if the curve does not
    span the target."""
    pairs = sorted(
        (s, b) for s, b in zip(snr_db_arr, ber_arr)
        if math.isfinite(s) and b > 0
    )
    if not pairs:
        return None
    # Walk from high-SNR end (low BER) toward low-SNR end (high BER).
    pairs.sort(key=lambda p: p[0], reverse=True)
    prev_s, prev_b = pairs[0]
    if prev_b > target_ber:
        return None  # Curve never gets below target
    for s, b in pairs[1:]:
        if b >= target_ber:
            # Crossed; interpolate in (snr, log10(ber)) space.
            log_prev = math.log10(prev_b)
            log_cur = math.log10(b)
            if log_cur == log_prev:
                return s
            frac = (math.log10(target_ber) - log_prev) / (log_cur - log_prev)
            return prev_s + frac * (s - prev_s)
        prev_s, prev_b = s, b
    return None  # Curve stayed below target across the whole sweep


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    n = round(SIZE / DX)
    positions = array_positions()
    prop_delay = math.hypot(RX[0] - ARRAY_X, RX[1] - ARRAY_CY) / C_EST

    rng = np.random.default_rng(PRBS_SEED)
    sent_bits = tuple(int(b) for b in rng.integers(0, 2, size=N_BITS))

    ambient = AmbientNoise(
        n_sources=N_NOISE_SOURCES,
        domain_size=SIZE,
        margin=NOISE_MARGIN,
        layout_seed=NOISE_LAYOUT_SEED,
    )

    # -- Phase 1: characterize each element (no noise) for the broadside
    # model side, plus the centre point for the single-source baseline. --
    base_chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)

    def char_shot(name, pos):
        steps = round((CHIRP_DURATION + prop_delay + PROP_TAIL) / dt)
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(pos=pos, voltage_fn=base_chirp,
                                    steps=steps, dt=dt)],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        frames = sw.open((args.nframes(steps), n, n))
        v_tx = np.fromiter(
            (base_chirp(i * sim.dt) for i in range(steps)),
            dtype=np.float32, count=steps,
        )
        mic_p = np.empty(steps, dtype=np.float32)
        print(f"shot {name}: pos={pos}, {steps} steps")
        for i in simargs.progress(steps):
            sim.step()
            if i % args.capture_every == 0:
                frames[i // args.capture_every] = to_numpy(sim.p)
            mic_p[i] = probe.pressure(sim, RX)
        v_rx = MIC.filter(mic_p, sim.dt)
        fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
        fir.fit(v_tx, v_rx)
        pred = fir.predict(v_tx)[: len(v_rx)]
        nr = float(channel_model.nrmse(v_rx, pred))
        print(f"  fitted {name}: training NRMSE = {nr:.4f}")
        return sw, sim.dt, v_tx, v_rx, fir, nr

    char_writers = {}
    char_data = {}
    firs = []
    train_nrmses = []
    for i, pos in enumerate(positions):
        sw, dt_i, v_tx, v_rx, fir, nr = char_shot(f"characterize_e{i}", pos)
        char_writers[f"characterize_e{i}"] = sw
        char_data[f"characterize_e{i}"] = (dt_i, v_tx, v_rx, fir, nr, pos)
        firs.append(fir)
        train_nrmses.append(nr)
    sw, dt_c, v_tx_c, v_rx_c, fir_center, nr_c = char_shot(
        "characterize_center", SINGLE_SOURCE_POS,
    )
    char_writers["characterize_center"] = sw
    char_data["characterize_center"] = (dt_c, v_tx_c, v_rx_c, fir_center, nr_c,
                                        SINGLE_SOURCE_POS)
    train_nrmses.append(nr_c)
    train_baseline = float(np.mean(train_nrmses))

    # -- Phase 2: comms with noise, two configurations per sigma. --
    ook_fn = ook_voltage(FREQ, sent_bits, BIT_DUR, drive_v=1.0)
    steps_comm = round((BIT_DUR * N_BITS + prop_delay + PROP_TAIL) / dt)

    comm_writers = {}
    comm_data = {}
    sweep_rows = []  # list of {config, sigma, snr_db, ber_phys, ber_model, agreement}

    for sigma in NOISE_SIGMAS:
        sigma_label = ("clean" if sigma == 0.0
                       else f"{sigma:.0e}".replace("+", "p").replace("-", "m"))
        noise_sources = ambient.sources(
            sigma, steps_comm, dt, drive_seed=NOISE_DRIVE_SEED,
        )

        # ---- single-source baseline ----
        name = f"comms_single_{sigma_label}"
        signal_source = SPEAKER.source(
            pos=SINGLE_SOURCE_POS, voltage_fn=ook_fn,
            steps=steps_comm, dt=dt,
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
            (ook_fn(i * sim.dt) for i in range(steps_comm)),
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
        v_rx_model = fir_center.predict(v_tx)[: len(v_rx_phys)].astype(np.float32)

        sp, np_pow = signal_noise_power(v_rx_phys, v_rx_model)
        s = snr_db(sp, np_pow)
        bp, _, _ = decode(v_rx_phys, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bm, _, _ = decode(v_rx_model, sim.dt, N_BITS, BIT_DUR, prop_delay)
        ber_p = sum(d != b for d, b in zip(bp, sent_bits)) / N_BITS
        ber_m = sum(d != b for d, b in zip(bm, sent_bits)) / N_BITS
        agree = sum(p == m for p, m in zip(bp, bm)) / N_BITS
        residual = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model).astype(np.float32)

        comm_data[name] = dict(
            config="single", sigma=float(sigma),
            v_tx=v_tx, v_rx_phys=v_rx_phys, v_rx_model=v_rx_model,
            residual=residual, sim_dt=sim.dt,
            signal_power=sp, noise_power=np_pow, snr_db=s,
            ber_phys=float(ber_p), ber_model=float(ber_m), agreement=float(agree),
            bits_phys=list(bp), bits_model=list(bm),
        )
        sweep_rows.append({
            "config": "single", "sigma": float(sigma),
            "snr_db": s, "signal_power": sp, "noise_power": np_pow,
            "ber_phys": float(ber_p), "ber_model": float(ber_m),
            "agreement": float(agree),
        })
        print(f"  {name}: SNR={s:+.1f} dB  BER_phys={ber_p:.3f} "
              f"BER_model={ber_m:.3f}  agreement={agree:.3f}")

        # ---- broadside (8 elements in phase) ----
        name = f"comms_broadside_{sigma_label}"
        elem_sources = [
            SPEAKER.source(pos=p, voltage_fn=ook_fn, steps=steps_comm, dt=dt)
            for p in positions
        ]
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[*elem_sources, *noise_sources],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        comm_writers[name] = sw
        frames = sw.open((args.nframes(steps_comm), n, n))
        mic_p = np.empty(steps_comm, dtype=np.float32)
        print(f"shot {name}: sigma={sigma:.1e}, {steps_comm} steps, 8 elements")
        for i in simargs.progress(steps_comm):
            sim.step()
            if i % args.capture_every == 0:
                frames[i // args.capture_every] = to_numpy(sim.p)
            mic_p[i] = probe.pressure(sim, RX)
        v_rx_phys = MIC.filter(mic_p, sim.dt)
        # Model side: per-element FIR prediction summed (LTI superposition).
        v_rx_model = np.zeros(len(v_rx_phys), dtype=np.float64)
        for j in range(N_ELEMENTS):
            pred = firs[j].predict(v_tx)[: len(v_rx_phys)]
            v_rx_model[: len(pred)] += pred
        v_rx_model = v_rx_model.astype(np.float32)

        sp, np_pow = signal_noise_power(v_rx_phys, v_rx_model)
        s = snr_db(sp, np_pow)
        bp, _, _ = decode(v_rx_phys, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bm, _, _ = decode(v_rx_model, sim.dt, N_BITS, BIT_DUR, prop_delay)
        ber_p = sum(d != b for d, b in zip(bp, sent_bits)) / N_BITS
        ber_m = sum(d != b for d, b in zip(bm, sent_bits)) / N_BITS
        agree = sum(p == m for p, m in zip(bp, bm)) / N_BITS
        residual = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model).astype(np.float32)

        comm_data[name] = dict(
            config="broadside", sigma=float(sigma),
            v_tx=v_tx, v_rx_phys=v_rx_phys, v_rx_model=v_rx_model,
            residual=residual, sim_dt=sim.dt,
            signal_power=sp, noise_power=np_pow, snr_db=s,
            ber_phys=float(ber_p), ber_model=float(ber_m), agreement=float(agree),
            bits_phys=list(bp), bits_model=list(bm),
        )
        sweep_rows.append({
            "config": "broadside", "sigma": float(sigma),
            "snr_db": s, "signal_power": sp, "noise_power": np_pow,
            "ber_phys": float(ber_p), "ber_model": float(ber_m),
            "agreement": float(agree),
        })
        print(f"  {name}: SNR={s:+.1f} dB  BER_phys={ber_p:.3f} "
              f"BER_model={ber_m:.3f}  agreement={agree:.3f}")

    # -- Array gain (dB): SNR difference at fixed sigma, broadside vs single.
    # Both configurations share the same OOK decoder so their BER-vs-SNR
    # curves overlap; the array's advantage shows up as a constant SNR
    # offset at each noise level. We average over the non-clean sigmas
    # (the residual at sigma=0 is dominated by the FIR's own fitting
    # error, not noise, so its SNR number isn't comparable).
    def per_config(rows, cfg, key):
        return [r[key] for r in rows if r["config"] == cfg]

    snr_single = per_config(sweep_rows, "single", "snr_db")
    ber_single = per_config(sweep_rows, "single", "ber_phys")
    snr_broad = per_config(sweep_rows, "broadside", "snr_db")
    ber_broad = per_config(sweep_rows, "broadside", "ber_phys")

    sigmas = list(NOISE_SIGMAS)
    snr_gains_db = [
        snr_broad[i] - snr_single[i]
        for i in range(len(sigmas))
        if sigmas[i] > 0
        and math.isfinite(snr_broad[i]) and math.isfinite(snr_single[i])
    ]
    array_gain_db = (
        float(np.mean(snr_gains_db)) if snr_gains_db else None
    )
    array_gain_theoretical_db = 20.0 * math.log10(N_ELEMENTS)
    print(f"\nArray gain (broadside - single SNR at fixed sigma):")
    for s, g in zip(sigmas[1:], snr_gains_db):
        print(f"  sigma={s:.1e}  gain = {g:+.2f} dB")
    print(f"  mean = {array_gain_db:+.2f} dB  "
          f"(theoretical 20*log10({N_ELEMENTS}) = "
          f"{array_gain_theoretical_db:+.2f} dB)")

    sweep = {
        "sweep_sigmas": [float(s) for s in NOISE_SIGMAS],
        "sweep_single_snr_db": snr_single,
        "sweep_single_ber_phys": ber_single,
        "sweep_single_ber_model": per_config(sweep_rows, "single", "ber_model"),
        "sweep_single_agreement": per_config(sweep_rows, "single", "agreement"),
        "sweep_broadside_snr_db": snr_broad,
        "sweep_broadside_ber_phys": ber_broad,
        "sweep_broadside_ber_model": per_config(sweep_rows, "broadside", "ber_model"),
        "sweep_broadside_agreement": per_config(sweep_rows, "broadside", "agreement"),
        "array_gain_db": float(array_gain_db) if array_gain_db is not None else None,
        "array_gain_theoretical_db": float(array_gain_theoretical_db),
        "noise_positions": np.asarray(ambient.positions, dtype=np.float32),
        "n_noise_sources": ambient.n_sources,
        "train_nrmse_baseline": train_baseline,
        "sent_bits": list(sent_bits),
    }

    # -- Phase 3: finalize. --
    for name, (sim_dt, v_tx, v_rx, fir, nr, pos) in char_data.items():
        pred = fir.predict(v_tx)[: len(v_rx)]
        residual = (np.asarray(v_rx, dtype=np.float32) - pred).astype(np.float32)
        char_writers[name].finish(
            channels=[
                Channel("TX (V)", kind="scalar", dt=sim_dt, pos=pos,
                        values=v_tx.tolist()),
                Channel("RX truth (V)", kind="scalar", dt=sim_dt, pos=RX,
                        values=v_rx.tolist()),
                Channel("RX model (V)", kind="scalar", dt=sim_dt,
                        values=pred.tolist()),
                Channel("residual (V)", kind="scalar", dt=sim_dt,
                        values=residual.tolist()),
            ],
            extras={
                "role": "characterize",
                "pos_x": pos[0], "pos_y": pos[1],
                "train_nrmse": nr,
                "fir_h": fir.h, "fir_n_taps": fir.n_taps,
                **sweep,
            },
        )

    for name, r in comm_data.items():
        sim_dt = r["sim_dt"]
        comm_writers[name].finish(
            channels=[
                Channel("TX (V)", kind="scalar", dt=sim_dt, pos=SINGLE_SOURCE_POS,
                        values=r["v_tx"].tolist()),
                Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX,
                        values=r["v_rx_phys"].tolist()),
                Channel("RX model (clean LTI) (V)", kind="scalar", dt=sim_dt,
                        values=r["v_rx_model"].tolist()),
                Channel("residual phys-model (noise est.) (V)", kind="scalar",
                        dt=sim_dt, values=r["residual"].tolist()),
            ],
            extras={
                "role": "comms",
                "config": r["config"],
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
