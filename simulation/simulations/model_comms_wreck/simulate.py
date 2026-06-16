"""Comms over LOS, partial-shadow, and full NLOS in a wrecked tank.

Same geometry as ``model_link_wreck``: rigid sea floor + central wreck.
Here we drop the LTI-validation phases and run an OOK link from each of
four TX positions, sweeping ambient-noise sigma at each. The surrogate
is a *per-position* FIR characterised from a chirp probe at that
position (so the model isn't being asked to predict an off-distribution
geometry; that question was already settled by ``model_link_wreck``).

Per TX position:

  1. ``char_{pos}`` -- chirp probe -> local FIR for this geometry.
  2. ``comms_{pos}_{sigma}`` -- OOK PRBS through that channel at the
     given ambient-noise sigma. Decode both phys (FDTD with noise) and
     model (FIR.predict, deterministic noiseless prediction). Report
     BER, agreement, and the noise-floor-derived SNR.

The story the BER-vs-sigma curves should tell:

  * LOS: strong direct path -> high SNR -> BER stays at 0 well into the
    noise sweep, then drops off near where ``model_link_noise`` did
    (this is essentially the model_link_noise replication in a richer
    multipath environment).
  * partial / NLOS: weaker signal because direct path is shadowed -> SNR
    is lower at the same sigma -> BER curve shifts left, comms degrades
    earlier than LOS.
  * floor: NLOS via floor scatter -- weak and ringy. Worst case."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.noise import AmbientNoise, noise_power_per_unit_sigma_sq
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 64}

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz

SEA_FLOOR_Y = 0.10
WRECK_X_MIN, WRECK_X_MAX = 0.45, 0.55
WRECK_Y_MIN, WRECK_Y_MAX = 0.10, 0.60

RX = (0.85, 0.70)

# Four TX positions spanning LOS through floor-NLOS. Each gets its own
# FIR via a chirp characterise shot, then OOK shots at the noise sweep.
TX_POSITIONS = [
    {"name": "los",     "pos": (0.20, 0.70), "label": "LOS (above wreck)"},
    {"name": "partial", "pos": (0.20, 0.55), "label": "partial shadow (at wreck top)"},
    {"name": "nlos",    "pos": (0.20, 0.40), "label": "NLOS (wreck blocks direct path)"},
    {"name": "floor",   "pos": (0.20, 0.20), "label": "near sea floor, NLOS via scatter"},
]

SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.008
FIR_N_TAPS = 4096

CHIRP_DURATION = 0.010
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

OOK_N_BITS = 32
OOK_BIT_DUR = 0.001
OOK_PRBS_SEED = 1234

N_NOISE_SOURCES = 16
NOISE_LAYOUT_SEED = 42
NOISE_DRIVE_SEED = 11
NOISE_MARGIN = 0.2  # > default 150 mm sponge depth so noise sources land in the fluid
NOISE_SIGMAS = (0.0, 1e-7, 3e-7, 1e-6, 3e-6)
NOISE_SIGMA_REF = 1e-6
NOISE_CAL_WARMUP_S = 0.005  # extra for multipath ringup in this rich channel


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


def rigid_mask(n, dx):
    mask = np.zeros((n, n), dtype=bool)
    floor_cells = int(round(SEA_FLOOR_Y / dx))
    mask[:, :floor_cells] = True
    x_lo = int(round(WRECK_X_MIN / dx))
    x_hi = int(round(WRECK_X_MAX / dx))
    y_lo = int(round(WRECK_Y_MIN / dx))
    y_hi = int(round(WRECK_Y_MAX / dx))
    mask[x_lo:x_hi, y_lo:y_hi] = True
    return mask


def obstacle_overlay(mask):
    nx, ny = mask.shape
    overlay = np.zeros((nx, ny, 4), dtype=np.uint8)
    overlay[..., 0] = 80
    overlay[..., 1] = 80
    overlay[..., 2] = 80
    overlay[..., 3] = np.where(mask, 200, 0).astype(np.uint8)
    return overlay


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
    mask = rigid_mask(n, DX)
    overlay = obstacle_overlay(mask)
    ambient = AmbientNoise(
        n_sources=N_NOISE_SOURCES, domain_size=SIZE,
        margin=NOISE_MARGIN, layout_seed=NOISE_LAYOUT_SEED,
    )

    rng = np.random.default_rng(OOK_PRBS_SEED)
    sent_bits = tuple(int(b) for b in rng.integers(0, 2, size=OOK_N_BITS))

    print(f"grid {n}x{n}, dx={DX*1e3:.1f} mm, dt={dt*1e6:.2f} us, "
          f"FIR taps={FIR_N_TAPS} ({FIR_N_TAPS*dt*1e3:.1f} ms span)")

    # Noise-only calibration: rigid mask is the same for every TX/sigma,
    # so one shot fixes the noise power at the receiver for the entire
    # sweep. By FDTD linearity noise_power(sigma) = sigma^2 * factor.
    steps_cal = round((OOK_BIT_DUR * OOK_N_BITS + PROP_TAIL) / dt)
    sim_cal = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=ambient.sources(
            NOISE_SIGMA_REF, steps_cal, dt, drive_seed=NOISE_DRIVE_SEED,
        ),
        rigid=mask,
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
        channels=[
            Channel("noise mic (V)", kind="scalar", dt=sim_cal.dt, pos=RX,
                    values=v_rx_noise_ref.tolist()),
        ],
        overlay=overlay,
        extras={
            "role": "noise_calibration",
            "sigma_ref": float(NOISE_SIGMA_REF),
            "noise_power_per_sigma_sq": noise_power_factor,
        },
    )

    def run_shot(name, tx_pos, voltage_fn, duration_s, sigma):
        prop_delay_here = math.hypot(
            RX[0] - tx_pos[0], RX[1] - tx_pos[1],
        ) / 1500.0
        steps = round((duration_s + prop_delay_here + PROP_TAIL) / dt)
        noise_sources = (ambient.sources(sigma, steps, dt, drive_seed=NOISE_DRIVE_SEED)
                         if sigma > 0 else [])
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[
                SPEAKER.source(pos=tx_pos, voltage_fn=voltage_fn,
                               steps=steps, dt=dt),
                *noise_sources,
            ],
            rigid=mask,
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        frames = sw.open((args.nframes(steps), n, n))
        v_tx = np.fromiter(
            (voltage_fn(i * sim.dt) for i in range(steps)),
            dtype=np.float32, count=steps,
        )
        mic_p = np.empty(steps, dtype=np.float32)
        print(f"shot {name}: tx={tx_pos}, sigma={sigma:.1e}, "
              f"{steps} steps, {steps * sim.dt * 1e3:.1f} ms")
        for i in simargs.progress(steps):
            sim.step()
            if i % args.capture_every == 0:
                frames[i // args.capture_every] = to_numpy(sim.p)
            mic_p[i] = probe.pressure(sim, RX)
        v_rx = MIC.filter(mic_p, sim.dt)
        return sw, sim.dt, v_tx, v_rx, prop_delay_here

    chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)
    ook_fn = ook_voltage(FREQ, sent_bits, OOK_BIT_DUR, drive_v=1.0)

    # Per position: characterize + per-sigma comms shots.
    pos_firs = {}
    pos_baselines = {}
    char_writers = {}
    char_records = {}
    comm_writers = {}
    comm_records = {}
    sweep_rows = []

    for tx in TX_POSITIONS:
        pname, tx_pos, plabel = tx["name"], tx["pos"], tx["label"]

        # 1. Characterise at this position (chirp, no ambient noise).
        char_name = f"char_{pname}"
        sw, char_dt, v_tx_chirp, v_rx_chirp, _ = run_shot(
            char_name, tx_pos, chirp, CHIRP_DURATION, sigma=0.0,
        )
        char_writers[char_name] = sw
        fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
        fir.fit(v_tx_chirp, v_rx_chirp)
        pred = fir.predict(v_tx_chirp)[: len(v_rx_chirp)]
        train_nrmse = float(channel_model.nrmse(v_rx_chirp, pred))
        pos_firs[pname] = fir
        pos_baselines[pname] = train_nrmse
        char_records[char_name] = {
            "sim_dt": char_dt, "v_tx": v_tx_chirp, "v_rx": v_rx_chirp,
            "pred": pred, "train_nrmse": train_nrmse,
            "tx_pos": tx_pos, "label": plabel,
        }
        print(f"  {char_name}: training NRMSE = {train_nrmse:.4f}  ({plabel})")

        # 2. OOK comms shots, sweeping ambient noise sigma.
        for sigma in NOISE_SIGMAS:
            sigma_label = ("clean" if sigma == 0.0
                           else f"{sigma:.0e}".replace("+", "p").replace("-", "m"))
            cname = f"comms_{pname}_{sigma_label}"
            sw, sim_dt, v_tx_ook, v_rx_phys, prop_delay_here = run_shot(
                cname, tx_pos, ook_fn, OOK_BIT_DUR * OOK_N_BITS, sigma,
            )
            v_rx_model = fir.predict(v_tx_ook)[: len(v_rx_phys)].astype(np.float32)
            sp = float(np.mean(np.asarray(v_rx_model, dtype=np.float64) ** 2))
            np_pow = noise_power_factor * sigma * sigma
            s = snr_db(sp, np_pow)
            bits_p = ook_decode(v_rx_phys, sim_dt, OOK_N_BITS, OOK_BIT_DUR, prop_delay_here)
            bits_m = ook_decode(v_rx_model, sim_dt, OOK_N_BITS, OOK_BIT_DUR, prop_delay_here)
            ber_p = sum(d != b for d, b in zip(bits_p, sent_bits)) / OOK_N_BITS
            ber_m = sum(d != b for d, b in zip(bits_m, sent_bits)) / OOK_N_BITS
            agree = sum(p == m for p, m in zip(bits_p, bits_m)) / OOK_N_BITS

            comm_writers[cname] = sw
            comm_records[cname] = {
                "sim_dt": sim_dt,
                "v_tx": v_tx_ook, "v_rx_phys": v_rx_phys, "v_rx_model": v_rx_model,
                "tx_pos": tx_pos, "pname": pname, "plabel": plabel,
                "sigma": float(sigma), "snr_db": s,
                "signal_power": sp, "noise_power": np_pow,
                "ber_phys": float(ber_p), "ber_model": float(ber_m),
                "agreement": float(agree),
                "bits_phys": list(bits_p), "bits_model": list(bits_m),
            }
            sweep_rows.append({
                "pname": pname, "sigma": float(sigma),
                "snr_db": s,
                "ber_phys": float(ber_p), "ber_model": float(ber_m),
                "agreement": float(agree),
            })
            print(f"  {cname}: SNR={s:+.1f} dB  BER_phys={ber_p:.3f}  "
                  f"BER_model={ber_m:.3f}  agreement={agree:.3f}")

    # -- Cross-shot sweep (per-position BER curves) for the view. --
    def pick(pname, key):
        return [r[key] for r in sweep_rows if r["pname"] == pname]

    sweep = {
        "sweep_sigmas": [float(s) for s in NOISE_SIGMAS],
        "tx_pos_names": [p["name"] for p in TX_POSITIONS],
        "tx_pos_labels": [p["label"] for p in TX_POSITIONS],
        "rx_pos": list(RX),
        "noise_positions": np.asarray(ambient.positions, dtype=np.float32),
        "train_nrmse_per_pos": [pos_baselines[p["name"]] for p in TX_POSITIONS],
    }
    for p in TX_POSITIONS:
        pname = p["name"]
        sweep[f"sweep_{pname}_snr_db"] = pick(pname, "snr_db")
        sweep[f"sweep_{pname}_ber_phys"] = pick(pname, "ber_phys")
        sweep[f"sweep_{pname}_ber_model"] = pick(pname, "ber_model")
        sweep[f"sweep_{pname}_agreement"] = pick(pname, "agreement")

    # -- Finalize. --
    for name, r in char_records.items():
        residual = (np.asarray(r["v_rx"], dtype=np.float32) - r["pred"]).astype(np.float32)
        char_writers[name].finish(
            channels=[
                Channel("TX (V)", kind="scalar", dt=r["sim_dt"], pos=r["tx_pos"],
                        values=r["v_tx"].tolist()),
                Channel("RX truth (V)", kind="scalar", dt=r["sim_dt"], pos=RX,
                        values=r["v_rx"].tolist()),
                Channel("RX model (V)", kind="scalar", dt=r["sim_dt"],
                        values=r["pred"].tolist()),
                Channel("residual (V)", kind="scalar", dt=r["sim_dt"],
                        values=residual.tolist()),
            ],
            overlay=overlay,
            extras={
                "role": "characterize",
                "pname": name.removeprefix("char_"),
                "label": r["label"],
                "tx_pos_x": r["tx_pos"][0], "tx_pos_y": r["tx_pos"][1],
                "train_nrmse": r["train_nrmse"],
                **sweep,
            },
        )

    for name, r in comm_records.items():
        sim_dt = r["sim_dt"]
        residual = (np.asarray(r["v_rx_phys"], dtype=np.float32) - r["v_rx_model"]).astype(np.float32)
        comm_writers[name].finish(
            channels=[
                Channel("TX (V)", kind="scalar", dt=sim_dt, pos=r["tx_pos"],
                        values=r["v_tx"].tolist()),
                Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX,
                        values=r["v_rx_phys"].tolist()),
                Channel("RX model (clean LTI) (V)", kind="scalar", dt=sim_dt,
                        values=r["v_rx_model"].tolist()),
                Channel("residual phys-model (noise est.) (V)", kind="scalar", dt=sim_dt,
                        values=residual.tolist()),
            ],
            overlay=overlay,
            extras={
                "role": "comms",
                "pname": r["pname"], "label": r["plabel"],
                "tx_pos_x": r["tx_pos"][0], "tx_pos_y": r["tx_pos"][1],
                "sigma": r["sigma"],
                "snr_db": r["snr_db"],
                "signal_power": r["signal_power"], "noise_power": r["noise_power"],
                "ber_phys": r["ber_phys"], "ber_model": r["ber_model"],
                "agreement": r["agreement"],
                "decoded_phys": r["bits_phys"], "decoded_model": r["bits_model"],
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
