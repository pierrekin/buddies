"""model_link with ambient noise in the FDTD: how does the comm system
behave as the channel gets noisier, and how does the deterministic LTI
surrogate compare?

Phase 1: fire the chirp through a noiseless channel, fit one FIR. This
is the surrogate's idealized model of the channel.

Phase 2: fire the SAME OOK PRBS at increasing ambient-noise levels. At
each level:

  * ``v_rx_phys`` -- FDTD with N point noise sources scattered across the
    domain, each driven by an iid Gaussian volume-rate noise vector.
  * ``v_rx_model`` -- ``FIR.predict(v_tx)``, deterministic, noiseless.

The surrogate predicts the *noiseless* LTI response of the channel; the
phys traces are one realization of "what would actually happen if you
ran this with noise σ." The gap between phys BER and model BER is the
noise penalty -- the bit-flip cost of noise the surrogate can't predict.

Alternative noise paths we deliberately skipped here (each is its own
future sim):

  * Userspace noise: just ``np.random.normal`` into ``v_rx`` after the
    FDTD captures. Easier, but the noise lives outside the FDTD's
    physical model, so it can't bounce off walls or interact with the
    surrogate's training data realistically.
  * Obstacles in the scene: rigid reflectors that produce real
    multipath. Stresses the FIR's truncation length, not its variance.
  * Transducer nonlinearity: would *break* the LTI assumption, exposing
    the surrogate at the model class level (FIR can't represent it),
    not just at the realization level."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.noise import AmbientNoise
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
TX = (0.2, 0.5)
RX = (0.8, 0.5)
SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.002
FIR_N_TAPS = 1024

CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

# OOK comms: match model_link so we can compare numbers directly.
N_BITS = 32
BIT_DUR = 0.001
PRBS_SEED = 1234

# Ambient noise: N point sources scattered across the domain interior,
# each iid Gaussian volume-rate. NOISE_SIGMAS sweeps the per-source RMS.
# Values calibrated empirically to span BER 0 -> 0.5 on a 1 ms OOK link;
# tune by re-running if the BER curve sits at the edges.
N_NOISE_SOURCES = 16
NOISE_LAYOUT_SEED = 42
NOISE_DRIVE_SEED = 11
NOISE_MARGIN = 0.2  # > default 150 mm sponge depth so noise sources land in the fluid
NOISE_SIGMAS = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)


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

    # -- Phase 1: characterize on a clean (noiseless) chirp. --
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

    # -- Phase 2: comms at increasing noise levels. --
    ook_fn = ook_voltage(FREQ, sent_bits, BIT_DUR, drive_v=1.0)
    steps_comm = round((BIT_DUR * N_BITS + prop_delay + PROP_TAIL) / dt)

    comm_writers = {}
    comm_data = {}
    sweep_sigmas, sweep_ber_phys, sweep_ber_model = [], [], []
    sweep_agreement, sweep_rx_phys_rms = [], []
    sweep_noise_rms_at_rx = []

    # Optional zero-signal noise calibration: one shot per nonzero sigma
    # would give us the noise's RMS at RX. Skipped here -- we just read it
    # off the actual comm shots by computing the residual.

    for sigma in NOISE_SIGMAS:
        sigma_label = f"{sigma:.0e}".replace("+", "p").replace("-", "m")
        name = ("comms_clean" if sigma == 0.0
                else f"comms_noise_{sigma_label}")
        signal_source = SPEAKER.source(
            pos=TX, voltage_fn=ook_fn, steps=steps_comm, dt=dt,
        )
        noise_sources = ambient.sources(
            sigma, steps_comm, dt, drive_seed=NOISE_DRIVE_SEED,
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
        print(f"shot {name}: sigma={sigma:.1e}, {steps_comm} steps, "
              f"{len(noise_sources)} noise sources")
        for i in simargs.progress(steps_comm):
            sim.step()
            if i % args.capture_every == 0:
                frames[i // args.capture_every] = to_numpy(sim.p)
            mic_p[i] = probe.pressure(sim, RX)
        v_rx_phys = MIC.filter(mic_p, sim.dt)
        v_rx_model = fir.predict(v_tx)[: len(v_rx_phys)]

        bits_phys, _, thr_phys = decode(
            v_rx_phys, sim.dt, N_BITS, BIT_DUR, prop_delay,
        )
        bits_model, _, thr_model = decode(
            v_rx_model, sim.dt, N_BITS, BIT_DUR, prop_delay,
        )
        ber_phys = sum(d != b for d, b in zip(bits_phys, sent_bits)) / N_BITS
        ber_model = sum(d != b for d, b in zip(bits_model, sent_bits)) / N_BITS
        agreement = sum(p == m for p, m in zip(bits_phys, bits_model)) / N_BITS
        # Noise estimate at RX: difference between phys (signal + noise)
        # and model (signal only) -- the residual is the noise plus any
        # surrogate error. With a clean FIR this is dominated by noise.
        noise_estimate = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model[: len(v_rx_phys)]).astype(np.float32)
        noise_rms_at_rx = float(np.sqrt(np.mean(noise_estimate ** 2)))
        rx_phys_rms = float(np.sqrt(np.mean(np.asarray(v_rx_phys) ** 2)))

        comm_data[name] = {
            "sigma": float(sigma),
            "v_tx": v_tx,
            "v_rx_phys": v_rx_phys,
            "v_rx_model": v_rx_model,
            "noise_estimate": noise_estimate,
            "sim_dt": sim.dt,
            "bits_phys": list(bits_phys),
            "bits_model": list(bits_model),
            "ber_phys": float(ber_phys),
            "ber_model": float(ber_model),
            "agreement": float(agreement),
            "noise_rms_at_rx": noise_rms_at_rx,
            "rx_phys_rms": rx_phys_rms,
            "threshold_phys": float(thr_phys),
            "threshold_model": float(thr_model),
        }
        sweep_sigmas.append(float(sigma))
        sweep_ber_phys.append(float(ber_phys))
        sweep_ber_model.append(float(ber_model))
        sweep_agreement.append(float(agreement))
        sweep_rx_phys_rms.append(rx_phys_rms)
        sweep_noise_rms_at_rx.append(noise_rms_at_rx)
        print(f"  {name}: BER_phys={ber_phys:.3f} BER_model={ber_model:.3f} "
              f"agreement={agreement:.3f}  "
              f"signal RMS = {rx_phys_rms:.2e} V, noise RMS = "
              f"{noise_rms_at_rx:.2e} V")

    sweep = {
        "noise_positions": np.asarray(ambient.positions, dtype=np.float32),
        "n_noise_sources": ambient.n_sources,
        "char_train_nrmse": train_nrmse,
        "sweep_sigmas": sweep_sigmas,
        "sweep_ber_phys": sweep_ber_phys,
        "sweep_ber_model": sweep_ber_model,
        "sweep_agreement": sweep_agreement,
        "sweep_rx_phys_rms": sweep_rx_phys_rms,
        "sweep_noise_rms_at_rx": sweep_noise_rms_at_rx,
        "sent_bits": list(sent_bits),
    }

    # -- Phase 3: finalize. --
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
            "fir_h": fir.h,
            "fir_n_taps": fir.n_taps,
            **sweep,
        },
    )

    for name, r in comm_data.items():
        sim_dt = r["sim_dt"]
        residual = (np.asarray(r["v_rx_phys"], dtype=np.float32)
                    - r["v_rx_model"][: len(r["v_rx_phys"])]).astype(np.float32)
        comm_writers[name].finish(
            channels=[
                Channel("TX (V)", kind="scalar", dt=sim_dt, pos=TX,
                        values=r["v_tx"].tolist()),
                Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX,
                        values=r["v_rx_phys"].tolist()),
                Channel("RX model (clean LTI) (V)", kind="scalar", dt=sim_dt,
                        values=r["v_rx_model"].tolist()),
                Channel("residual phys-model (noise est.) (V)",
                        kind="scalar", dt=sim_dt, values=residual.tolist()),
            ],
            extras={
                "role": "comms",
                "sigma": r["sigma"],
                "ber_phys": r["ber_phys"],
                "ber_model": r["ber_model"],
                "agreement": r["agreement"],
                "noise_rms_at_rx": r["noise_rms_at_rx"],
                "rx_phys_rms": r["rx_phys_rms"],
                "decoded_phys": r["bits_phys"],
                "decoded_model": r["bits_model"],
                "threshold_phys": r["threshold_phys"],
                "threshold_model": r["threshold_model"],
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
