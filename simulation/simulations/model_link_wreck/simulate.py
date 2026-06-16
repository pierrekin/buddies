"""LTI surrogate in a non-trivial acoustic environment.

Does the FIR-based channel surrogate from ``model_link`` still work
when the channel impulse response is genuinely rich and when line of
sight can be fully blocked? We build a tank with two rigid features:

  * **sea floor** -- the bottom strip of the domain (y < 0.1 m) is a
    perfectly reflecting wall, so every TX-RX path acquires a strong
    floor bounce in addition to whatever multipath it already had.
  * **wreck** -- a rigid rectangle at x in [0.45, 0.55], y in [0.1,
    0.6] (10 cm wide, 50 cm tall, sitting on the floor). Placed in the
    middle of the tank between TX and RX so that some TX positions are
    line-of-sight to the RX and others are fully shadowed.

No comms in this sim -- only LTI validation. The headline metric is
per-shot NRMSE between phys (FDTD with rigid wreck + floor) and the
surrogate (FIR.predict on the in-channel chirp probe).

Phase 1 -- ``char_train`` (LOS): fire chirp at TX_TRAIN above the wreck,
capture at RX, fit FIR.

Phase 2 -- same trained TX position, different waveforms: ``val_train_ook``,
``val_train_tone``, ``val_train_prbs``. Tests that LTI predicts arbitrary
in-band waveforms once the channel is characterised.

Phase 3 -- displaced TX positions, all firing the chirp: ``val_los_high``
(still LOS, just different angle), ``val_partial`` (TX at wreck top),
``val_nlos_mid`` (wreck fully shadows the direct path), ``val_floor``
(TX hugging the sea floor; direct path partially blocked, scattered).

Expected: NRMSE rises monotonically as TX displacement and shadowing
grow. Stays low across waveforms at the trained position (LTI is real)
but breaks under geometric mismatch (the FIR encodes one channel)."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

# Longer shots in this sim (chirp + multipath ringdown is ~10 ms);
# keep frames manageable by sampling sparsely.
DEFAULTS = {"capture_every": 64}

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz

# Geometry: sea floor + central wreck.
SEA_FLOOR_Y = 0.10  # everything below this is rigid
WRECK_X_MIN, WRECK_X_MAX = 0.45, 0.55  # 10 cm wide
WRECK_Y_MIN, WRECK_Y_MAX = 0.10, 0.60  # sits on the floor, 50 cm tall

# RX is well above the wreck so LOS depends on TX y-position only.
RX = (0.85, 0.70)

# Trained TX position: above the wreck, full LOS to RX.
TX_TRAIN = (0.20, 0.70)

# Off-position TX shots, each tagged with a description for the view.
OFF_POSITIONS = [
    {"name": "val_los_high",  "tx": (0.20, 0.85), "label": "LOS, higher angle"},
    {"name": "val_partial",   "tx": (0.20, 0.55), "label": "partial shadow (TX at wreck top)"},
    {"name": "val_nlos_mid",  "tx": (0.20, 0.40), "label": "NLOS (wreck blocks direct path)"},
    {"name": "val_floor",     "tx": (0.20, 0.20), "label": "near sea floor, NLOS via scatter"},
]

SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)

# Rich multipath -> long ringdown, long FIR. 4096 taps at default
# dt ~1.18 us = ~4.8 ms; chirp duration must be much larger than that
# for the Wiener-Hopf fit to be well-conditioned.
FIR_N_TAPS = 4096
PROP_TAIL = 0.008  # s

CHIRP_DURATION = 0.010  # s, long enough to overdetermine the FIR fit
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

# Same-position waveforms for the in-distribution validation phase.
OOK_N_BITS = 16
OOK_BIT_DUR = 0.001
OOK_PRBS_SEED = 1234

TONE_DURATION = 0.0025  # ~37 cycles at 15 kHz
TONE_RAMP_PERIODS = 2.0

PRBS_N_BITS = 200
PRBS_BIT_DUR = 0.00003  # ~30 us, broad spectrum over the BPF band


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


def tone_burst(freq, duration, ramp_periods=TONE_RAMP_PERIODS, amplitude=1.0):
    """A Hann-windowed sine burst -- broadband enough to excite multipath
    without the click of a hard switch-on."""
    omega = 2 * math.pi * freq
    ramp_t = ramp_periods / freq

    def v(t):
        if t < 0 or t > duration:
            return 0.0
        # Hann ramp in/out.
        if t < ramp_t:
            env = 0.5 * (1 - math.cos(math.pi * t / ramp_t))
        elif t > duration - ramp_t:
            env = 0.5 * (1 - math.cos(math.pi * (duration - t) / ramp_t))
        else:
            env = 1.0
        return amplitude * env * math.sin(omega * t)

    return v


def prbs_voltage(bits, bit_dur, drive_v=1.0):
    """+/- drive PRBS, no carrier -- broadband across the BPF passband."""
    def v(t):
        if t < 0:
            return 0.0
        idx = int(t / bit_dur)
        if idx >= len(bits):
            return 0.0
        return drive_v * (1.0 if bits[idx] else -1.0)
    return v


def rigid_mask(n, dx):
    """Boolean mask (nx, ny). True = perfectly reflecting cell."""
    mask = np.zeros((n, n), dtype=bool)
    # mask[x_idx, y_idx]: x grows along axis 0, y along axis 1 -- matches
    # the framework's (pos -> grid) convention.
    floor_cells = int(round(SEA_FLOOR_Y / dx))
    mask[:, :floor_cells] = True
    x_lo = int(round(WRECK_X_MIN / dx))
    x_hi = int(round(WRECK_X_MAX / dx))
    y_lo = int(round(WRECK_Y_MIN / dx))
    y_hi = int(round(WRECK_Y_MAX / dx))
    mask[x_lo:x_hi, y_lo:y_hi] = True
    return mask


def obstacle_overlay(mask):
    """RGBA overlay (nx, ny, 4) uint8 -- semi-transparent dim grey on
    every rigid cell. Sits on top of the pressure field in the viewer."""
    nx, ny = mask.shape
    overlay = np.zeros((nx, ny, 4), dtype=np.uint8)
    overlay[..., 0] = 80
    overlay[..., 1] = 80
    overlay[..., 2] = 80
    overlay[..., 3] = np.where(mask, 200, 0).astype(np.uint8)
    return overlay


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    n = round(SIZE / DX)
    mask = rigid_mask(n, DX)
    overlay = obstacle_overlay(mask)
    prop_delay_train = math.hypot(
        RX[0] - TX_TRAIN[0], RX[1] - TX_TRAIN[1],
    ) / 1500.0

    rng_ook = np.random.default_rng(OOK_PRBS_SEED)
    ook_bits = tuple(int(b) for b in rng_ook.integers(0, 2, size=OOK_N_BITS))
    rng_prbs = np.random.default_rng(OOK_PRBS_SEED + 1)
    prbs_bits = tuple(int(b) for b in rng_prbs.integers(0, 2, size=PRBS_N_BITS))

    print(f"grid {n}x{n}, dx={DX*1e3:.1f} mm, dt={dt*1e6:.2f} us, "
          f"FIR taps={FIR_N_TAPS} ({FIR_N_TAPS*dt*1e3:.1f} ms span)")
    rigid_frac = float(mask.sum()) / mask.size
    print(f"rigid fraction = {rigid_frac:.3f} (sea floor + wreck)")

    # -- Helper: run one FDTD shot, capture mic + frames, return arrays. --
    def run_shot(name, tx_pos, voltage_fn, duration_s):
        prop_delay_here = math.hypot(
            RX[0] - tx_pos[0], RX[1] - tx_pos[1],
        ) / 1500.0
        steps = round((duration_s + prop_delay_here + PROP_TAIL) / dt)
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(
                pos=tx_pos, voltage_fn=voltage_fn, steps=steps, dt=dt,
            )],
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
        print(f"shot {name}: tx={tx_pos}, {steps} steps, "
              f"{steps * sim.dt * 1e3:.1f} ms")
        for i in simargs.progress(steps):
            sim.step()
            if i % args.capture_every == 0:
                frames[i // args.capture_every] = to_numpy(sim.p)
            mic_p[i] = probe.pressure(sim, RX)
        v_rx = MIC.filter(mic_p, sim.dt)
        return sw, sim.dt, v_tx, v_rx

    # -- Phase 1: characterize at TX_TRAIN with chirp. --
    chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)
    char_writer, char_dt, v_tx_train, v_rx_train = run_shot(
        "char_train", TX_TRAIN, chirp, CHIRP_DURATION,
    )

    fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
    fir.fit(v_tx_train, v_rx_train)
    char_pred = fir.predict(v_tx_train)[: len(v_rx_train)]
    train_nrmse = float(channel_model.nrmse(v_rx_train, char_pred))
    print(f"  fitted {fir.name}: training NRMSE = {train_nrmse:.4f}")

    # Pull the prop delay + impulse-response peak for view annotation.
    h_peak_lag = int(np.argmax(np.abs(fir.h.astype(np.float64))))

    # -- Phase 2: same TX position, different waveforms. --
    same_pos_specs = [
        {"name": "val_train_ook",
         "voltage_fn": ook_voltage(FREQ, ook_bits, OOK_BIT_DUR, 1.0),
         "duration": OOK_BIT_DUR * OOK_N_BITS,
         "label": "trained TX, OOK"},
        {"name": "val_train_tone",
         "voltage_fn": tone_burst(FREQ, TONE_DURATION),
         "duration": TONE_DURATION,
         "label": "trained TX, tone burst"},
        {"name": "val_train_prbs",
         "voltage_fn": prbs_voltage(prbs_bits, PRBS_BIT_DUR),
         "duration": PRBS_BIT_DUR * PRBS_N_BITS,
         "label": "trained TX, broadband PRBS"},
    ]

    same_pos_data = {}
    for spec in same_pos_specs:
        sw, sim_dt, v_tx, v_rx_phys = run_shot(
            spec["name"], TX_TRAIN, spec["voltage_fn"], spec["duration"],
        )
        v_rx_model = fir.predict(v_tx)[: len(v_rx_phys)].astype(np.float32)
        nrm = float(channel_model.nrmse(v_rx_phys, v_rx_model))
        same_pos_data[spec["name"]] = {
            "writer": sw, "sim_dt": sim_dt,
            "v_tx": v_tx, "v_rx_phys": v_rx_phys, "v_rx_model": v_rx_model,
            "nrmse": nrm, "tx_pos": TX_TRAIN,
            "label": spec["label"], "role": "validate_waveform",
        }
        print(f"  {spec['name']}: NRMSE = {nrm:.4f}")

    # -- Phase 3: displaced TX, chirp probe.
    # For each off-position we ALSO fit a fresh local FIR on its own
    # (v_tx, v_rx_phys) pair; the local fit's NRMSE should drop back to
    # baseline. That separates "the LTI surrogate is broken in complex
    # multipath" (it isn't) from "the trained FIR encodes one specific
    # geometry" (it does). --
    off_data = {}
    for spec in OFF_POSITIONS:
        sw, sim_dt, v_tx, v_rx_phys = run_shot(
            spec["name"], spec["tx"], chirp, CHIRP_DURATION,
        )
        v_rx_model_trained = fir.predict(v_tx)[: len(v_rx_phys)].astype(np.float32)
        nrm_trained = float(channel_model.nrmse(v_rx_phys, v_rx_model_trained))

        local_fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
        local_fir.fit(v_tx, v_rx_phys)
        v_rx_model_local = local_fir.predict(v_tx)[: len(v_rx_phys)].astype(np.float32)
        nrm_local = float(channel_model.nrmse(v_rx_phys, v_rx_model_local))

        off_data[spec["name"]] = {
            "writer": sw, "sim_dt": sim_dt,
            "v_tx": v_tx, "v_rx_phys": v_rx_phys,
            "v_rx_model": v_rx_model_trained,
            "v_rx_model_local": v_rx_model_local,
            "nrmse": nrm_trained,
            "nrmse_local": nrm_local,
            "local_fir_h": local_fir.h,
            "tx_pos": spec["tx"],
            "label": spec["label"], "role": "validate_position",
        }
        print(f"  {spec['name']}: NRMSE trained={nrm_trained:.4f}, "
              f"local={nrm_local:.4f}  (tx={spec['tx']}, {spec['label']})")

    # -- Aggregate sweep across all validate shots for the view.
    # ``val_nrmse_local`` is the per-geometry refit's in-sample NRMSE on
    # off-position shots, and equals ``nan`` for same-position waveform
    # shots (refitting on OOK/tone/PRBS isn't meaningful). --
    all_val = (list(same_pos_data.items()) + list(off_data.items()))
    sweep = {
        "train_nrmse": train_nrmse,
        "val_shot_names": [n for n, _ in all_val],
        "val_labels":     [r["label"] for _, r in all_val],
        "val_nrmse":      [r["nrmse"] for _, r in all_val],
        "val_nrmse_local": [
            r.get("nrmse_local", float("nan")) for _, r in all_val
        ],
        "val_kind":       [r["role"] for _, r in all_val],
        "rx_pos": list(RX),
        "tx_train_pos": list(TX_TRAIN),
        "fir_n_taps": FIR_N_TAPS,
        "fir_span_ms": float(FIR_N_TAPS * dt * 1e3),
        "h_peak_lag_us": float(h_peak_lag * dt * 1e6),
    }

    # -- Finalize shots. --
    char_writer.finish(
        channels=[
            Channel("TX (V)", kind="scalar", dt=char_dt, pos=TX_TRAIN,
                    values=v_tx_train.tolist()),
            Channel("RX truth (V)", kind="scalar", dt=char_dt, pos=RX,
                    values=v_rx_train.tolist()),
            Channel("RX model (V)", kind="scalar", dt=char_dt,
                    values=char_pred.tolist()),
        ],
        overlay=overlay,
        extras={
            "role": "characterize",
            "label": "training: chirp at TX_TRAIN (LOS)",
            "train_nrmse": train_nrmse,
            "fir_h": fir.h,
            **sweep,
        },
    )

    for name, r in all_val:
        sim_dt = r["sim_dt"]
        residual = (np.asarray(r["v_rx_phys"], dtype=np.float32)
                    - r["v_rx_model"]).astype(np.float32)
        channels = [
            Channel("TX (V)", kind="scalar", dt=sim_dt, pos=r["tx_pos"],
                    values=r["v_tx"].tolist()),
            Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX,
                    values=r["v_rx_phys"].tolist()),
            Channel("RX model (trained FIR) (V)", kind="scalar", dt=sim_dt,
                    values=r["v_rx_model"].tolist()),
            Channel("residual trained FIR (V)", kind="scalar", dt=sim_dt,
                    values=residual.tolist()),
        ]
        extras = {
            "role": r["role"],
            "label": r["label"],
            "tx_pos_x": r["tx_pos"][0], "tx_pos_y": r["tx_pos"][1],
            "waveform_nrmse": r["nrmse"],
            **sweep,
        }
        # Off-position shots also carry their local-FIR refit.
        if "v_rx_model_local" in r:
            local_residual = (np.asarray(r["v_rx_phys"], dtype=np.float32)
                              - r["v_rx_model_local"]).astype(np.float32)
            channels.append(Channel(
                "RX model (local FIR refit) (V)", kind="scalar", dt=sim_dt,
                values=r["v_rx_model_local"].tolist(),
            ))
            channels.append(Channel(
                "residual local FIR (V)", kind="scalar", dt=sim_dt,
                values=local_residual.tolist(),
            ))
            extras["waveform_nrmse_local"] = r["nrmse_local"]
            extras["local_fir_h"] = r["local_fir_h"]
        r["writer"].finish(
            channels=channels, overlay=overlay, extras=extras,
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
