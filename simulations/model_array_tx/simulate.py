"""TX-only beamforming: test the FIR model's superposition property.

Phase 1 (characterize): fire each of the N TX elements alone with a
chirp probe and record the mic. Fit one FIR per element -- 8 separate
(elem_i -> RX) channels.

Phase 2 (validate): drive ALL elements simultaneously with per-element
steering delays. Two predictions of ``v_rx``:

  * ``v_rx_phys``  : actual FDTD with the array coordinated
  * ``v_rx_model`` : ``sum_i FIR_i.predict(v_tx_i)`` -- pure post-hoc
                     composition of the per-element FIRs

If the FDTD is genuinely LTI (it is, by construction), the two should
agree to within the per-element model error baseline (~2% NRMSE from
``model_link``). A larger discrepancy at steered angles would point to
either a model artefact (FIR truncation across the spread of delays) or
a sim non-linearity. Either is informative.

Geometry: 8-element line array on the left wall at half-wavelength
spacing (50 mm at 15 kHz), 350 mm aperture, single RX broadside at
(0.8, 0.5). Steering: broadside-at-RX, plus virtual far-field targets
at +15/+30/+60° off broadside."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, line, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
C_EST = 1500.0  # m/s, for delay calculations; sim.c is the truth
LAMBDA = C_EST / FREQ  # 0.10 m
N_ELEMENTS = 8
SPACING = LAMBDA / 2  # half-wavelength avoids grating lobes
APERTURE = (N_ELEMENTS - 1) * SPACING
ARRAY_X = 0.1  # m, x of the line array
ARRAY_CY = 0.5  # m, y center of the array
RX = (0.8, 0.5)  # broadside from array center, 0.7 m away
SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.002  # s of silence after the drive ends, for BPF ringdown
FIR_N_TAPS = 1024  # matches model_link/model_compare's chosen size

CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

# Broadside (0°) plus three off-axis steering angles. Far-field virtual
# focus, not near-field, so the per-element delays are the linear
# plane-wave profile (rather than the spherical near-field correction).
STEER_ANGLES_DEG = (0.0, 15.0, 30.0, 60.0)

# -- Phase 3: comms over the steered beam. --
# Two sweeps share the broadside-at-1ms shot:
#   - steering sweep at 1 ms bits (matches the validation angles)
#   - bit-dur sweep at broadside (matches a subset of model_link's BIT_DURS)
COMMS_N_BITS = 32
COMMS_PRBS_SEED = 1234  # match model_link so we share the bit pattern
COMMS_STEER_BIT_DUR = 0.001       # bit_dur held constant across the steering sweep
COMMS_BROADSIDE_BIT_DURS = (0.001, 0.00025, 0.0001)  # bit_dur sweep at broadside
COMMS_STEER_ANGLES = (0.0, 15.0, 30.0, 60.0)


def linear_chirp(f_lo, f_hi, duration, amplitude=1.0):
    """Same chirp shape model_link/model_compare use; covers the BPF band."""
    def v(t):
        if t < 0 or t > duration:
            return 0.0
        k = (f_hi - f_lo) / duration
        phase = 2 * math.pi * (f_lo * t + 0.5 * k * t * t)
        return amplitude * math.sin(phase)
    return v


def ook_voltage(freq, bits, bit_dur, drive_v=1.0):
    """Voltage-domain OOK on a square carrier -- same as model_link."""
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
    """Per-bit RMS slicer (second-half integration) -- same as model_link."""
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


def delayed(fn, d):
    """Wrap ``fn`` to be silent before time ``d`` and equal ``fn(t-d)`` after."""
    def shifted(t):
        if t < d:
            return 0.0
        return fn(t - d)
    return shifted


def array_positions():
    """N_ELEMENTS evenly-spaced positions on the line array."""
    start = (ARRAY_X, ARRAY_CY - APERTURE / 2)
    end = (ARRAY_X, ARRAY_CY + APERTURE / 2)
    return line(start, end, N_ELEMENTS)


def steering_focus(angle_deg, far_distance=100.0):
    """Virtual far-field focus point off broadside (+x). Angle = 0 is
    broadside; positive angle steers toward +y."""
    rad = math.radians(angle_deg)
    return (
        ARRAY_X + far_distance * math.cos(rad),
        ARRAY_CY + far_distance * math.sin(rad),
    )


def element_delays(positions, focus):
    """Per-element delays so all wavefronts arrive at ``focus`` together."""
    dists = [math.hypot(p[0] - focus[0], p[1] - focus[1]) for p in positions]
    farthest = max(dists)
    return [(farthest - d) / C_EST for d in dists]


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    n = round(SIZE / DX)
    positions = array_positions()
    prop_delay = math.hypot(RX[0] - ARRAY_X, RX[1] - ARRAY_CY) / C_EST
    base_chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)

    # -- Phase 1: characterize each element alone. --
    char_data = {}  # element_idx -> (v_tx, v_rx, sim_dt)
    char_writers = {}
    for i, pos in enumerate(positions):
        name = f"characterize_e{i}"
        steps = round((CHIRP_DURATION + prop_delay + PROP_TAIL) / dt)
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(pos=pos, voltage_fn=base_chirp,
                                    steps=steps, dt=dt)],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        char_writers[i] = sw
        frames = sw.open((args.nframes(steps), n, n))

        v_tx = np.fromiter(
            (base_chirp(j * sim.dt) for j in range(steps)),
            dtype=np.float32, count=steps,
        )
        mic_p = np.empty(steps, dtype=np.float32)
        print(f"shot {name}: pos={pos}, {steps} steps")
        for j in simargs.progress(steps):
            sim.step()
            if j % args.capture_every == 0:
                frames[j // args.capture_every] = to_numpy(sim.p)
            mic_p[j] = probe.pressure(sim, RX)
        v_rx = MIC.filter(mic_p, sim.dt)
        char_data[i] = (v_tx, v_rx, sim.dt)

    # -- Fit one FIR per element. --
    firs = []
    train_nrmse = []
    for i in range(N_ELEMENTS):
        v_tx_i, v_rx_i, _ = char_data[i]
        fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
        fir.fit(v_tx_i, v_rx_i)
        firs.append(fir)
        pred = fir.predict(v_tx_i)[: len(v_rx_i)]
        nr = float(channel_model.nrmse(v_rx_i, pred))
        train_nrmse.append(nr)
        print(f"  fitted FIR_e{i}: training NRMSE = {nr:.4f}")

    # -- Phase 2: validate superposition across steering angles. --
    val_writers = {}
    val_data = {}  # name -> dict
    sweep_angles, sweep_nrmse = [], []

    for angle_deg in STEER_ANGLES_DEG:
        focus = steering_focus(angle_deg)
        delays = element_delays(positions, focus)
        max_delay = max(delays)
        duration = CHIRP_DURATION + max_delay
        steps = round((duration + prop_delay + PROP_TAIL) / dt)

        elem_fns = [delayed(base_chirp, d) for d in delays]
        sources = [
            SPEAKER.source(pos=pos, voltage_fn=fn, steps=steps, dt=dt)
            for pos, fn in zip(positions, elem_fns)
        ]
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=sources, damping=edge_sponge((n, n), DX),
        )
        if angle_deg == 0.0:
            name = "validate_broadside"
        else:
            name = f"validate_steer_{int(angle_deg):+03d}deg"

        sw = out.shot(name)
        val_writers[name] = sw
        frames = sw.open((args.nframes(steps), n, n))

        # Sample each element's drive for the model-side prediction.
        v_tx_per_elem = [
            np.fromiter(
                (fn(j * sim.dt) for j in range(steps)),
                dtype=np.float32, count=steps,
            )
            for fn in elem_fns
        ]
        mic_p = np.empty(steps, dtype=np.float32)
        print(f"shot {name}: focus={focus}, "
              f"max_delay={max_delay * 1e6:.1f}us, {steps} steps")
        for j in simargs.progress(steps):
            sim.step()
            if j % args.capture_every == 0:
                frames[j // args.capture_every] = to_numpy(sim.p)
            mic_p[j] = probe.pressure(sim, RX)
        v_rx_phys = MIC.filter(mic_p, sim.dt)

        # Model: sum of per-element FIR predictions.
        v_rx_model = np.zeros(len(v_rx_phys), dtype=np.float64)
        for i, v_tx_i in enumerate(v_tx_per_elem):
            pred = firs[i].predict(v_tx_i)[: len(v_rx_phys)]
            v_rx_model[: len(pred)] += pred
        v_rx_model = v_rx_model.astype(np.float32)

        nr = float(channel_model.nrmse(v_rx_phys, v_rx_model))
        sweep_angles.append(float(angle_deg))
        sweep_nrmse.append(nr)
        val_data[name] = {
            "angle_deg": float(angle_deg),
            "focus": focus,
            "delays": delays,
            "v_tx_per_elem": v_tx_per_elem,
            "v_rx_phys": v_rx_phys,
            "v_rx_model": v_rx_model,
            "sim_dt": sim.dt,
            "waveform_nrmse": nr,
        }
        print(f"  {name}: superposition NRMSE = {nr:.4f}")

    # -- Phase 3: comms over the steered beam. --
    # Same OOK message sent through the array. Two sweeps share the
    # broadside-at-1ms shot:
    #   - steering at COMMS_STEER_BIT_DUR (does beam direction affect BER?)
    #   - bit_dur at broadside (does the array's directivity reduce ISI?)
    rng = np.random.default_rng(COMMS_PRBS_SEED)
    sent_bits = tuple(int(b) for b in rng.integers(0, 2, size=COMMS_N_BITS))

    # Build the per-shot spec list, dedup'ed so broadside-at-1ms runs once.
    comms_specs = []
    seen = set()
    for angle_deg in COMMS_STEER_ANGLES:
        key = (angle_deg, COMMS_STEER_BIT_DUR)
        comms_specs.append({"angle_deg": angle_deg, "bit_dur": COMMS_STEER_BIT_DUR})
        seen.add(key)
    for bit_dur in COMMS_BROADSIDE_BIT_DURS:
        key = (0.0, bit_dur)
        if key in seen:
            continue
        comms_specs.append({"angle_deg": 0.0, "bit_dur": bit_dur})
        seen.add(key)

    def comms_shot_name(spec):
        bd_us = int(spec["bit_dur"] * 1e6)
        if spec["angle_deg"] == 0.0:
            return f"comms_broadside_{bd_us:04d}us"
        return f"comms_steer_{int(spec['angle_deg']):+03d}deg_{bd_us:04d}us"

    comms_writers = {}
    comms_data = {}
    for spec in comms_specs:
        name = comms_shot_name(spec)
        angle_deg = spec["angle_deg"]
        bit_dur = spec["bit_dur"]
        focus = steering_focus(angle_deg)
        delays = element_delays(positions, focus)
        max_delay = max(delays)
        duration = bit_dur * COMMS_N_BITS + max_delay
        steps = round((duration + prop_delay + PROP_TAIL) / dt)

        base_ook = ook_voltage(FREQ, sent_bits, bit_dur, drive_v=1.0)
        elem_fns = [delayed(base_ook, d) for d in delays]
        sources = [
            SPEAKER.source(pos=pos, voltage_fn=fn, steps=steps, dt=dt)
            for pos, fn in zip(positions, elem_fns)
        ]
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=sources, damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        comms_writers[name] = sw
        frames = sw.open((args.nframes(steps), n, n))

        v_tx_per_elem = [
            np.fromiter(
                (fn(j * sim.dt) for j in range(steps)),
                dtype=np.float32, count=steps,
            )
            for fn in elem_fns
        ]
        mic_p = np.empty(steps, dtype=np.float32)
        print(f"shot {name}: focus={focus}, bit_dur={bit_dur * 1e6:.0f}us, "
              f"max_delay={max_delay * 1e6:.1f}us, {steps} steps")
        for j in simargs.progress(steps):
            sim.step()
            if j % args.capture_every == 0:
                frames[j // args.capture_every] = to_numpy(sim.p)
            mic_p[j] = probe.pressure(sim, RX)
        v_rx_phys = MIC.filter(mic_p, sim.dt)

        v_rx_model = np.zeros(len(v_rx_phys), dtype=np.float64)
        for i, v_tx_i in enumerate(v_tx_per_elem):
            pred = firs[i].predict(v_tx_i)[: len(v_rx_phys)]
            v_rx_model[: len(pred)] += pred
        v_rx_model = v_rx_model.astype(np.float32)

        bits_phys, _, thr_phys = decode(
            v_rx_phys, sim.dt, COMMS_N_BITS, bit_dur, prop_delay,
        )
        bits_model, _, thr_model = decode(
            v_rx_model, sim.dt, COMMS_N_BITS, bit_dur, prop_delay,
        )
        ber_phys = sum(d != b for d, b in zip(bits_phys, sent_bits)) / COMMS_N_BITS
        ber_model = sum(d != b for d, b in zip(bits_model, sent_bits)) / COMMS_N_BITS
        agreement = sum(p == m for p, m in zip(bits_phys, bits_model)) / COMMS_N_BITS
        waveform_nrmse = float(channel_model.nrmse(v_rx_phys, v_rx_model))

        comms_data[name] = {
            "angle_deg": float(angle_deg),
            "bit_dur": float(bit_dur),
            "focus": focus,
            "delays": delays,
            "v_tx_per_elem": v_tx_per_elem,
            "v_rx_phys": v_rx_phys,
            "v_rx_model": v_rx_model,
            "sim_dt": sim.dt,
            "decoded_phys": list(bits_phys),
            "decoded_model": list(bits_model),
            "ber_phys": float(ber_phys),
            "ber_model": float(ber_model),
            "agreement": float(agreement),
            "waveform_nrmse": waveform_nrmse,
            "threshold_phys": float(thr_phys),
            "threshold_model": float(thr_model),
        }
        print(f"  {name}: BER_phys={ber_phys:.3f} BER_model={ber_model:.3f} "
              f"agreement={agreement:.3f} NRMSE={waveform_nrmse:.4f}")

    # Aggregate the two comms sweeps (steering at fixed bit_dur, bit_dur
    # at broadside) into arrays the viewer can plot directly.
    def gather(filter_fn, key):
        rows = [(spec, comms_data[comms_shot_name(spec)])
                for spec in comms_specs if filter_fn(spec)]
        rows.sort(key=key)
        return rows

    steer_rows = gather(
        lambda s: s["bit_dur"] == COMMS_STEER_BIT_DUR,
        key=lambda r: r[0]["angle_deg"],
    )
    btdur_rows = gather(
        lambda s: s["angle_deg"] == 0.0,
        key=lambda r: r[0]["bit_dur"],
    )

    def pick(rows, k):
        return [float(r[1][k]) for r in rows]

    comms_sweep = {
        "comms_steer_angles_deg": [float(r[0]["angle_deg"]) for r in steer_rows],
        "comms_steer_ber_phys": pick(steer_rows, "ber_phys"),
        "comms_steer_ber_model": pick(steer_rows, "ber_model"),
        "comms_steer_agreement": pick(steer_rows, "agreement"),
        "comms_steer_nrmse": pick(steer_rows, "waveform_nrmse"),
        "comms_bitdur_s": [float(r[0]["bit_dur"]) for r in btdur_rows],
        "comms_bitdur_ber_phys": pick(btdur_rows, "ber_phys"),
        "comms_bitdur_ber_model": pick(btdur_rows, "ber_model"),
        "comms_bitdur_agreement": pick(btdur_rows, "agreement"),
        "comms_bitdur_nrmse": pick(btdur_rows, "waveform_nrmse"),
        "comms_sent_bits": list(sent_bits),
    }

    # -- Phase 4: finalize. Sweep summaries go on every shot so the view
    # can render any curve with the current shot marked. --
    sweep = {
        "sweep_angles_deg": sweep_angles,
        "sweep_nrmse": sweep_nrmse,
        **comms_sweep,
    }
    # All eight FIR taps go on every characterize shot so the view can
    # overlay them and highlight the current element.
    fir_arrays = {f"fir_h_e{i}": firs[i].h for i in range(N_ELEMENTS)}

    for i, pos in enumerate(positions):
        v_tx_i, v_rx_i, sim_dt = char_data[i]
        pred = firs[i].predict(v_tx_i)[: len(v_rx_i)]
        residual = (np.asarray(v_rx_i, dtype=np.float32) - pred).astype(np.float32)
        char_writers[i].finish(
            channels=[
                Channel("TX (V)", kind="scalar", dt=sim_dt, pos=pos,
                        values=v_tx_i.tolist()),
                Channel("RX truth (V)", kind="scalar", dt=sim_dt, pos=RX,
                        values=v_rx_i.tolist()),
                Channel("RX model (V)", kind="scalar", dt=sim_dt,
                        values=pred.tolist()),
                Channel("residual (V)", kind="scalar", dt=sim_dt,
                        values=residual.tolist()),
            ],
            extras={
                "role": "characterize",
                "element_index": i,
                "element_pos_x": pos[0],
                "element_pos_y": pos[1],
                "fir_n_taps": FIR_N_TAPS,
                "train_nrmse": train_nrmse[i],
                "train_nrmse_per_element": train_nrmse,
                **sweep,
                **fir_arrays,
            },
        )

    for name, r in val_data.items():
        v_rx_phys = r["v_rx_phys"]
        v_rx_model = r["v_rx_model"]
        sim_dt = r["sim_dt"]
        residual = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model).astype(np.float32)
        # Show only the first and last element's drive in the viewer to
        # avoid an 11-channel scroll; all eight live in extras as arrays.
        v_tx_arrays = {
            f"v_tx_e{i}": v.astype(np.float32) for i, v in enumerate(r["v_tx_per_elem"])
        }
        val_writers[name].finish(
            channels=[
                Channel(f"TX e0 (V)", kind="scalar", dt=sim_dt,
                        pos=positions[0], values=r["v_tx_per_elem"][0].tolist()),
                Channel(f"TX e{N_ELEMENTS - 1} (V)", kind="scalar", dt=sim_dt,
                        pos=positions[-1],
                        values=r["v_tx_per_elem"][-1].tolist()),
                Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX,
                        values=v_rx_phys.tolist()),
                Channel("RX model (V)", kind="scalar", dt=sim_dt,
                        values=v_rx_model.tolist()),
                Channel("residual phys-model (V)", kind="scalar", dt=sim_dt,
                        values=residual.tolist()),
            ],
            extras={
                "role": "validate",
                "angle_deg": r["angle_deg"],
                "focus_x": float(r["focus"][0]),
                "focus_y": float(r["focus"][1]),
                "element_delays_us": [d * 1e6 for d in r["delays"]],
                "waveform_nrmse": r["waveform_nrmse"],
                **sweep,
                **v_tx_arrays,
            },
        )

    for name, r in comms_data.items():
        v_rx_phys = r["v_rx_phys"]
        v_rx_model = r["v_rx_model"]
        sim_dt = r["sim_dt"]
        residual = (np.asarray(v_rx_phys, dtype=np.float32) - v_rx_model).astype(np.float32)
        v_tx_arrays = {
            f"v_tx_e{i}": v.astype(np.float32) for i, v in enumerate(r["v_tx_per_elem"])
        }
        comms_writers[name].finish(
            channels=[
                Channel(f"TX e0 (V)", kind="scalar", dt=sim_dt,
                        pos=positions[0], values=r["v_tx_per_elem"][0].tolist()),
                Channel(f"TX e{N_ELEMENTS - 1} (V)", kind="scalar", dt=sim_dt,
                        pos=positions[-1],
                        values=r["v_tx_per_elem"][-1].tolist()),
                Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX,
                        values=v_rx_phys.tolist()),
                Channel("RX model (V)", kind="scalar", dt=sim_dt,
                        values=v_rx_model.tolist()),
                Channel("residual phys-model (V)", kind="scalar", dt=sim_dt,
                        values=residual.tolist()),
            ],
            extras={
                "role": "comms",
                "angle_deg": r["angle_deg"],
                "bit_dur": r["bit_dur"],
                "focus_x": float(r["focus"][0]),
                "focus_y": float(r["focus"][1]),
                "element_delays_us": [d * 1e6 for d in r["delays"]],
                "decoded_phys": r["decoded_phys"],
                "decoded_model": r["decoded_model"],
                "ber_phys": r["ber_phys"],
                "ber_model": r["ber_model"],
                "agreement": r["agreement"],
                "waveform_nrmse": r["waveform_nrmse"],
                "threshold_phys": r["threshold_phys"],
                "threshold_model": r["threshold_model"],
                **sweep,
                **v_tx_arrays,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
