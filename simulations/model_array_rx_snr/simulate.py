"""RX beamforming with ambient noise: does coherent combining buy SNR?

Single TX fires the same OOK PRBS through a noisy channel. An 8-element
line array captures the mixture at all elements simultaneously. Two
decode pipelines share the same FDTD run per noise sigma:

  * ``single``     -- decode the centre element (RX_4) alone, treating
                      the array as just one mic.
  * ``beamformed`` -- delay-and-sum all 8 elements at the signal's
                      direction (broadside here), decode the composite.

Both phys and model traces go through both pipelines, giving a 2x2 of
BER measurements per shot.

Industry metrics:

  * ``snr_db`` at each pipeline's output, signal power from the LTI
    model prediction and noise power from phys-vs-model residual.
  * ``BER vs SNR`` waterfall per pipeline; the canonical performance
    figure for the receiver chain.
  * ``RX array gain (dB)`` = beamformed SNR - single SNR at fixed sigma.

Theoretical bound for uncorrelated noise across elements:
``10 * log10(N) = 9.03 dB``. Ambient point sources at finite range
produce noise with some spatial correlation across an aperture of order
the wavelength, so measured gain is bounded by 9 dB and typically less."""

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
N_RX = 8
SPACING = LAMBDA / 2
APERTURE = (N_RX - 1) * SPACING

RX_X = 0.9
RX_CY = 0.5
TX_POS = (0.2, 0.5)
# Centre element of the array, used as the "single mic" baseline.
SINGLE_RX_INDEX = N_RX // 2  # e4

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
NOISE_MARGIN = 0.2  # > default 150 mm sponge depth so noise sources land in the fluid
NOISE_SIGMAS = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5)

# Look angle for the beamformer. TX is broadside from the array centre.
LOOK_ANGLE_DEG = 0.0


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


def signal_noise_power(v_rx_phys, v_rx_model):
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


def _capture_rx_array(sim, n_steps, positions, frames, capture_every):
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
    prop_delay = math.hypot(RX_X - TX_POS[0], RX_CY - TX_POS[1]) / C_EST

    rng = np.random.default_rng(PRBS_SEED)
    sent_bits = tuple(int(b) for b in rng.integers(0, 2, size=N_BITS))

    ambient = AmbientNoise(
        n_sources=N_NOISE_SOURCES,
        domain_size=SIZE,
        margin=NOISE_MARGIN,
        layout_seed=NOISE_LAYOUT_SEED,
    )

    # -- Phase 1: characterize all 8 channels in one FDTD run. --
    base_chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)
    steps_char = round((CHIRP_DURATION + prop_delay + PROP_TAIL) / dt)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[SPEAKER.source(pos=TX_POS, voltage_fn=base_chirp,
                                steps=steps_char, dt=dt)],
        damping=edge_sponge((n, n), DX),
    )
    char_writer = out.shot("characterize")
    frames = char_writer.open((args.nframes(steps_char), n, n))
    v_tx_char = np.fromiter(
        (base_chirp(i * sim.dt) for i in range(steps_char)),
        dtype=np.float32, count=steps_char,
    )
    print(f"shot characterize: tx={TX_POS}, {steps_char} steps, M={N_RX} RXs")
    mic_p_char = _capture_rx_array(sim, steps_char, positions, frames, args.capture_every)
    v_rx_phys_char = [MIC.filter(mic_p_char[j], sim.dt) for j in range(N_RX)]
    char_sim_dt = sim.dt

    firs = []
    train_nrmses = []
    for j in range(N_RX):
        fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
        fir.fit(v_tx_char, v_rx_phys_char[j])
        firs.append(fir)
        pred = fir.predict(v_tx_char)[: len(v_rx_phys_char[j])]
        nr = float(channel_model.nrmse(v_rx_phys_char[j], pred))
        train_nrmses.append(nr)
        print(f"  fitted FIR_RX{j}: training NRMSE = {nr:.4f}")
    train_baseline = float(np.mean(train_nrmses))

    # -- Phase 2: comms with noise, one FDTD run per sigma. --
    ook_fn = ook_voltage(FREQ, sent_bits, BIT_DUR, drive_v=1.0)
    steps_comm = round((BIT_DUR * N_BITS + prop_delay + PROP_TAIL) / dt)

    comm_writers = {}
    comm_data = {}
    sweep_rows = []  # list of dicts per (sigma, pipeline)

    for sigma in NOISE_SIGMAS:
        sigma_label = ("clean" if sigma == 0.0
                       else f"{sigma:.0e}".replace("+", "p").replace("-", "m"))
        name = f"comms_{sigma_label}"
        noise_sources = ambient.sources(
            sigma, steps_comm, dt, drive_seed=NOISE_DRIVE_SEED,
        )
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[
                SPEAKER.source(pos=TX_POS, voltage_fn=ook_fn,
                               steps=steps_comm, dt=dt),
                *noise_sources,
            ],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        comm_writers[name] = sw
        frames = sw.open((args.nframes(steps_comm), n, n))
        v_tx = np.fromiter(
            (ook_fn(i * sim.dt) for i in range(steps_comm)),
            dtype=np.float32, count=steps_comm,
        )
        print(f"shot {name}: sigma={sigma:.1e}, {steps_comm} steps")
        mic_p = _capture_rx_array(sim, steps_comm, positions, frames, args.capture_every)
        v_rx_phys = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]
        v_rx_model = [
            firs[j].predict(v_tx)[: len(v_rx_phys[j])].astype(np.float32)
            for j in range(N_RX)
        ]

        # ---- pipeline: single element ----
        v_phys_single = v_rx_phys[SINGLE_RX_INDEX]
        v_model_single = v_rx_model[SINGLE_RX_INDEX]
        sp_s, np_s = signal_noise_power(v_phys_single, v_model_single)
        snr_single_db = snr_db(sp_s, np_s)
        bits_p_single, _, _ = decode(v_phys_single, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_m_single, _, _ = decode(v_model_single, sim.dt, N_BITS, BIT_DUR, prop_delay)
        ber_p_single = sum(d != b for d, b in zip(bits_p_single, sent_bits)) / N_BITS
        ber_m_single = sum(d != b for d, b in zip(bits_m_single, sent_bits)) / N_BITS
        agree_single = sum(p == m for p, m in zip(bits_p_single, bits_m_single)) / N_BITS

        # ---- pipeline: delay-and-sum beamformed ----
        comp_phys = delay_and_sum(
            v_rx_phys, sim.dt, LOOK_ANGLE_DEG, y_offsets, c,
        ).astype(np.float32)
        comp_model = delay_and_sum(
            v_rx_model, sim.dt, LOOK_ANGLE_DEG, y_offsets, c,
        ).astype(np.float32)
        sp_b, np_b = signal_noise_power(comp_phys, comp_model)
        snr_beam_db = snr_db(sp_b, np_b)
        bits_p_beam, _, _ = decode(comp_phys, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_m_beam, _, _ = decode(comp_model, sim.dt, N_BITS, BIT_DUR, prop_delay)
        ber_p_beam = sum(d != b for d, b in zip(bits_p_beam, sent_bits)) / N_BITS
        ber_m_beam = sum(d != b for d, b in zip(bits_m_beam, sent_bits)) / N_BITS
        agree_beam = sum(p == m for p, m in zip(bits_p_beam, bits_m_beam)) / N_BITS

        comm_data[name] = dict(
            sigma=float(sigma),
            v_tx=v_tx, v_rx_phys=v_rx_phys, v_rx_model=v_rx_model,
            sim_dt=sim.dt,
            v_phys_single=v_phys_single, v_model_single=v_model_single,
            comp_phys=comp_phys, comp_model=comp_model,
            snr_single_db=snr_single_db, snr_beam_db=snr_beam_db,
            signal_power_single=sp_s, noise_power_single=np_s,
            signal_power_beam=sp_b, noise_power_beam=np_b,
            ber_phys_single=float(ber_p_single),
            ber_model_single=float(ber_m_single),
            agreement_single=float(agree_single),
            ber_phys_beam=float(ber_p_beam),
            ber_model_beam=float(ber_m_beam),
            agreement_beam=float(agree_beam),
            bits_phys_single=list(bits_p_single),
            bits_model_single=list(bits_m_single),
            bits_phys_beam=list(bits_p_beam),
            bits_model_beam=list(bits_m_beam),
        )
        sweep_rows.append({
            "sigma": float(sigma),
            "snr_single_db": snr_single_db,
            "snr_beam_db": snr_beam_db,
            "ber_phys_single": float(ber_p_single),
            "ber_phys_beam": float(ber_p_beam),
            "agreement_single": float(agree_single),
            "agreement_beam": float(agree_beam),
        })
        print(
            f"  {name}: single SNR={snr_single_db:+.1f} dB BER={ber_p_single:.3f}  "
            f"beam SNR={snr_beam_db:+.1f} dB BER={ber_p_beam:.3f}  "
            f"gain={snr_beam_db - snr_single_db:+.2f} dB"
        )

    # -- RX array gain at fixed sigma --
    snr_gains = [
        r["snr_beam_db"] - r["snr_single_db"]
        for r in sweep_rows
        if r["sigma"] > 0
        and math.isfinite(r["snr_beam_db"]) and math.isfinite(r["snr_single_db"])
    ]
    array_gain_db = float(np.mean(snr_gains)) if snr_gains else None
    array_gain_theoretical_db = 10.0 * math.log10(N_RX)  # 10 log N for power
    print(f"\nRX array gain (beam - single SNR at fixed sigma):")
    for r, g in zip([row for row in sweep_rows if row["sigma"] > 0], snr_gains):
        print(f"  sigma={r['sigma']:.1e}  gain = {g:+.2f} dB")
    print(f"  mean = {array_gain_db:+.2f} dB  "
          f"(theoretical 10*log10({N_RX}) for uncorrelated noise = "
          f"{array_gain_theoretical_db:+.2f} dB)")

    def pick(key):
        return [r[key] for r in sweep_rows]

    sweep = {
        "sweep_sigmas": pick("sigma"),
        "sweep_single_snr_db": pick("snr_single_db"),
        "sweep_single_ber_phys": pick("ber_phys_single"),
        "sweep_single_agreement": pick("agreement_single"),
        "sweep_beam_snr_db": pick("snr_beam_db"),
        "sweep_beam_ber_phys": pick("ber_phys_beam"),
        "sweep_beam_agreement": pick("agreement_beam"),
        "array_gain_db": array_gain_db,
        "array_gain_theoretical_db": float(array_gain_theoretical_db),
        "noise_positions": np.asarray(ambient.positions, dtype=np.float32),
        "n_noise_sources": ambient.n_sources,
        "train_nrmse_baseline": train_baseline,
        "sent_bits": list(sent_bits),
    }

    # -- Phase 3: finalize. --
    # Characterize: TX, three representative RX channels (phys + model).
    rep_idx = (0, SINGLE_RX_INDEX, N_RX - 1)
    char_channels = [
        Channel("TX (V)", kind="scalar", dt=char_sim_dt, pos=TX_POS,
                values=v_tx_char.tolist()),
    ]
    for j in rep_idx:
        char_channels.append(Channel(
            f"RX e{j} phys (V)", kind="scalar", dt=char_sim_dt,
            pos=positions[j], values=v_rx_phys_char[j].tolist(),
        ))
        char_channels.append(Channel(
            f"RX e{j} model (V)", kind="scalar", dt=char_sim_dt,
            values=firs[j].predict(v_tx_char)[: len(v_rx_phys_char[j])].astype(np.float32).tolist(),
        ))
    char_writer.finish(
        channels=char_channels,
        extras={
            "role": "characterize",
            "train_nrmse_per_rx": train_nrmses,
            "train_nrmse_baseline": train_baseline,
            **{f"fir_h_e{j}": firs[j].h for j in range(N_RX)},
            **sweep,
        },
    )

    # Comms shots: TX, the centre element trace, and the beamformed
    # composite, both for phys and model. All eight RX traces and both
    # composite versions live in extras as arrays.
    for name, r in comm_data.items():
        sim_dt = r["sim_dt"]
        channels = [
            Channel("TX (V)", kind="scalar", dt=sim_dt, pos=TX_POS,
                    values=r["v_tx"].tolist()),
            Channel(f"RX e{SINGLE_RX_INDEX} phys (single) (V)",
                    kind="scalar", dt=sim_dt, pos=positions[SINGLE_RX_INDEX],
                    values=r["v_phys_single"].tolist()),
            Channel(f"RX e{SINGLE_RX_INDEX} model (single) (V)",
                    kind="scalar", dt=sim_dt,
                    values=r["v_model_single"].tolist()),
            Channel("composite phys (beamformed) (V)", kind="scalar", dt=sim_dt,
                    values=r["comp_phys"].tolist()),
            Channel("composite model (beamformed) (V)", kind="scalar", dt=sim_dt,
                    values=r["comp_model"].tolist()),
        ]
        comm_writers[name].finish(
            channels=channels,
            extras={
                "role": "comms",
                "sigma": r["sigma"],
                "snr_single_db": r["snr_single_db"],
                "snr_beam_db": r["snr_beam_db"],
                "signal_power_single": r["signal_power_single"],
                "noise_power_single": r["noise_power_single"],
                "signal_power_beam": r["signal_power_beam"],
                "noise_power_beam": r["noise_power_beam"],
                "ber_phys_single": r["ber_phys_single"],
                "ber_model_single": r["ber_model_single"],
                "agreement_single": r["agreement_single"],
                "ber_phys_beam": r["ber_phys_beam"],
                "ber_model_beam": r["ber_model_beam"],
                "agreement_beam": r["agreement_beam"],
                "bits_phys_single": r["bits_phys_single"],
                "bits_model_single": r["bits_model_single"],
                "bits_phys_beam": r["bits_phys_beam"],
                "bits_model_beam": r["bits_model_beam"],
                **{f"v_rx_phys_e{j}": np.asarray(r["v_rx_phys"][j], dtype=np.float32)
                   for j in range(N_RX)},
                **{f"v_rx_model_e{j}": np.asarray(r["v_rx_model"][j], dtype=np.float32)
                   for j in range(N_RX)},
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=c)
