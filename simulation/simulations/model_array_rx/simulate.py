"""RX-only beamforming: per-(TX -> RX_j) FIRs and the beam pattern.

Phase 1 (characterize): one TX at the reference position fires a chirp.
All M=8 RXs along the right wall capture simultaneously. Fit one FIR
per RX. One FDTD run, M FIRs.

Phase 2 (validate per-RX): keep TX at the reference position but fire
OOK instead of chirp. Each FIR_j predicts v_rx_j from the new v_tx;
compare to phys. LTI says it should still hit per-RX baseline NRMSE
even though the waveform is unlike training.

Phase 3 (validate off-position TX): move TX upward by 50/100/200 mm.
Same chirp, but now from a new direction. Two things to check:

  (a) Per-RX NRMSE grows with displacement -- the FIRs encode the
      trained TX position's multipath and don't generalize.
  (b) The RX-side beamformer's look-angle sweep gives a DOA estimate.
      Phys finds the new TX direction (its captures know where the
      signal came from). The model, using FIRs from the trained
      position, still thinks the TX is broadside. The angular peak
      separation = the surrogate's spatial blindness."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, line, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
C_EST = 1500.0  # m/s; sim.c is the truth, but for delay math this is enough
LAMBDA = C_EST / FREQ  # 100 mm
N_RX = 8
SPACING = LAMBDA / 2  # half-wavelength avoids grating lobes
APERTURE = (N_RX - 1) * SPACING

RX_X = 0.9  # m, x of the receive array
RX_CY = 0.5  # m, y center of the array
TX_REF = (0.2, 0.5)  # broadside-from-array-center reference TX position
# How far above center to displace the TX in the off-position shots.
TX_DISPLACEMENTS_Y_M = (0.050, 0.100, 0.200)

SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.002
FIR_N_TAPS = 1024

CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

OOK_N_BITS = 16
OOK_BIT_DUR = 0.001
OOK_PRBS_SEED = 1234

# Far-field plane-wave look angles for the beamformer, off broadside
# (broadside = the -x direction). Positive look angle = looking toward +y.
# Reaches well past the largest expected TX displacement (200 mm at 700 mm
# range ~= 16 deg).
LOOK_ANGLES_DEG = tuple(range(-25, 26))  # 51 angles at 1 deg


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
    """Per-bit RMS slicer (second-half integration). Same as model_link."""
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


def expected_doa_deg(tx_pos):
    """Angle off broadside of the line from array center to TX. Positive
    when TX is above center, since broadside points toward -x."""
    dy = tx_pos[1] - RX_CY
    dx = RX_X - tx_pos[0]  # positive when TX is to the left of the array
    return math.degrees(math.atan2(dy, dx))


def delay_and_sum(traces, dt, look_angle_deg, y_offsets, c):
    """Delay each trace by the per-element delay implied by a far-field
    plane wave from ``look_angle_deg`` off broadside, then sum. Returns
    a float64 composite array; length = trace length."""
    sin_look = math.sin(math.radians(look_angle_deg))
    n = len(traces[0])
    composite = np.zeros(n, dtype=np.float64)
    for v, dy in zip(traces, y_offsets):
        # Upper element with positive look angle = subtract this many
        # samples (the upper element saw the wave earlier; we delay it).
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


def beam_pattern(traces, dt, y_offsets, c, look_angles_deg=LOOK_ANGLES_DEG):
    """For each look angle, compute the composite's RMS. Returns
    ``(angles_array, rms_array)`` both as float64 numpy arrays."""
    angles = np.asarray(look_angles_deg, dtype=np.float64)
    rms = np.empty(len(angles), dtype=np.float64)
    for i, a in enumerate(angles):
        c_t = delay_and_sum(traces, dt, float(a), y_offsets, c)
        rms[i] = float(np.sqrt(np.mean(c_t ** 2)))
    return angles, rms


def _capture_rx_array(sim, n_steps, positions, frames, capture_every):
    """Run ``sim`` for ``n_steps``, capturing the field (subsampled) and
    the pressure at every RX position. Returns ``(n_rx, n_steps)`` array."""
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
    c = 1500.0  # sim.c after construction matches; used for beamforming math

    # -- Phase 1: characterize at TX_REF with a chirp. --
    base_chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)
    prop_delay_ref = math.hypot(RX_X - TX_REF[0], RX_CY - TX_REF[1]) / C_EST
    steps = round((CHIRP_DURATION + prop_delay_ref + PROP_TAIL) / dt)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[SPEAKER.source(pos=TX_REF, voltage_fn=base_chirp,
                                steps=steps, dt=dt)],
        damping=edge_sponge((n, n), DX),
    )
    char_writer = out.shot("characterize")
    frames = char_writer.open((args.nframes(steps), n, n))
    v_tx_char = np.fromiter(
        (base_chirp(i * sim.dt) for i in range(steps)),
        dtype=np.float32, count=steps,
    )
    print(f"shot characterize: tx={TX_REF}, {steps} steps, M={N_RX} RXs")
    mic_p = _capture_rx_array(sim, steps, positions, frames, args.capture_every)
    v_rx_phys_char = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]
    char_sim_dt = sim.dt

    # Fit one FIR per RX.
    firs = []
    train_nrmse = []
    for j in range(N_RX):
        fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
        fir.fit(v_tx_char, v_rx_phys_char[j])
        firs.append(fir)
        pred = fir.predict(v_tx_char)[: len(v_rx_phys_char[j])]
        nr = float(channel_model.nrmse(v_rx_phys_char[j], pred))
        train_nrmse.append(nr)
        print(f"  fitted FIR_RX{j}: training NRMSE = {nr:.4f}")
    train_baseline = float(np.mean(train_nrmse))

    # Characterize beam pattern: phys vs model, should match exactly.
    v_rx_model_char = [
        firs[j].predict(v_tx_char)[: len(v_rx_phys_char[j])].astype(np.float32)
        for j in range(N_RX)
    ]
    char_angles, char_phys_rms = beam_pattern(v_rx_phys_char, char_sim_dt, y_offsets, c)
    _, char_model_rms = beam_pattern(v_rx_model_char, char_sim_dt, y_offsets, c)
    char_peak_phys = float(char_angles[int(np.argmax(char_phys_rms))])
    char_peak_model = float(char_angles[int(np.argmax(char_model_rms))])
    char_expected = expected_doa_deg(TX_REF)
    print(f"  characterize beam peak: phys={char_peak_phys:+.1f} deg, "
          f"model={char_peak_model:+.1f} deg, expected={char_expected:+.1f} deg")

    # -- Phase 2: validation shots. --
    rng = np.random.default_rng(OOK_PRBS_SEED)
    sent_bits = tuple(int(b) for b in rng.integers(0, 2, size=OOK_N_BITS))
    base_ook = ook_voltage(FREQ, sent_bits, OOK_BIT_DUR)

    # All validate shots fire the same OOK PRBS so we have a comms decode
    # at every TX position, not just signal-level NRMSE.
    val_specs = [{
        "name": "validate_aligned",
        "tx_pos": TX_REF,
        "voltage_fn": base_ook,
        "duration": OOK_BIT_DUR * OOK_N_BITS,
        "tx_offset_y": 0.0,
        "kind": "same_tx_new_waveform",
    }]
    for offset in TX_DISPLACEMENTS_Y_M:
        val_specs.append({
            "name": f"validate_tx_y{int(offset * 1000):03d}mm",
            "tx_pos": (TX_REF[0], TX_REF[1] + offset),
            "voltage_fn": base_ook,
            "duration": OOK_BIT_DUR * OOK_N_BITS,
            "tx_offset_y": float(offset),
            "kind": "tx_displaced",
        })

    val_writers = {}
    val_data = {}
    for spec in val_specs:
        name = spec["name"]
        tx_pos = spec["tx_pos"]
        voltage_fn = spec["voltage_fn"]
        prop_delay_v = math.hypot(RX_X - tx_pos[0], RX_CY - tx_pos[1]) / C_EST
        steps = round((spec["duration"] + prop_delay_v + PROP_TAIL) / dt)
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(pos=tx_pos, voltage_fn=voltage_fn,
                                    steps=steps, dt=dt)],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        val_writers[name] = sw
        frames = sw.open((args.nframes(steps), n, n))

        v_tx = np.fromiter(
            (voltage_fn(i * sim.dt) for i in range(steps)),
            dtype=np.float32, count=steps,
        )
        print(f"shot {name}: tx={tx_pos}, {steps} steps")
        mic_p = _capture_rx_array(sim, steps, positions, frames, args.capture_every)
        v_rx_phys = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]
        v_rx_model = [
            firs[j].predict(v_tx)[: len(v_rx_phys[j])].astype(np.float32)
            for j in range(N_RX)
        ]
        per_rx_nrmse = [
            float(channel_model.nrmse(v_rx_phys[j], v_rx_model[j]))
            for j in range(N_RX)
        ]

        angles, beam_phys_rms = beam_pattern(v_rx_phys, sim.dt, y_offsets, c)
        _, beam_model_rms = beam_pattern(v_rx_model, sim.dt, y_offsets, c)
        peak_phys = float(angles[int(np.argmax(beam_phys_rms))])
        peak_model = float(angles[int(np.argmax(beam_model_rms))])
        expected = expected_doa_deg(tx_pos)

        # Demix + decode at two look angles: the *correct* direction
        # (toward the actual TX) and the *trained* direction (broadside,
        # where the FIRs were fit). On the model traces the trained
        # angle is the natural one; on the phys traces the correct angle
        # is the natural one. The cross-pairs tell the divergence story.
        look_correct = expected
        look_trained = 0.0
        def beamform_decode(traces, look_deg):
            composite = delay_and_sum(
                traces, sim.dt, look_deg, y_offsets, c,
            ).astype(np.float32)
            bits, _, _ = decode(
                composite, sim.dt, OOK_N_BITS, OOK_BIT_DUR, prop_delay_v,
            )
            return composite, bits

        comp_phys_correct, bits_phys_correct = beamform_decode(v_rx_phys, look_correct)
        comp_phys_trained, bits_phys_trained = beamform_decode(v_rx_phys, look_trained)
        comp_model_correct, bits_model_correct = beamform_decode(v_rx_model, look_correct)
        comp_model_trained, bits_model_trained = beamform_decode(v_rx_model, look_trained)

        def ber(decoded):
            return sum(d != b for d, b in zip(decoded, sent_bits)) / OOK_N_BITS

        def agree(a, b):
            return sum(x == y for x, y in zip(a, b)) / OOK_N_BITS

        ber_phys_correct = float(ber(bits_phys_correct))
        ber_phys_trained = float(ber(bits_phys_trained))
        ber_model_correct = float(ber(bits_model_correct))
        ber_model_trained = float(ber(bits_model_trained))
        agree_at_correct = float(agree(bits_phys_correct, bits_model_correct))
        agree_at_trained = float(agree(bits_phys_trained, bits_model_trained))
        agree_natural_pair = float(agree(bits_phys_correct, bits_model_trained))

        val_data[name] = {
            "tx_pos": tx_pos,
            "tx_offset_y": spec["tx_offset_y"],
            "v_tx": v_tx,
            "v_rx_phys": v_rx_phys,
            "v_rx_model": v_rx_model,
            "per_rx_nrmse": per_rx_nrmse,
            "beam_angles_deg": angles,
            "beam_phys_rms": beam_phys_rms,
            "beam_model_rms": beam_model_rms,
            "peak_phys_deg": peak_phys,
            "peak_model_deg": peak_model,
            "expected_deg": expected,
            "sim_dt": sim.dt,
            "kind": spec["kind"],
            "look_correct_deg": look_correct,
            "look_trained_deg": look_trained,
            "comp_phys_correct": comp_phys_correct,
            "comp_phys_trained": comp_phys_trained,
            "comp_model_correct": comp_model_correct,
            "comp_model_trained": comp_model_trained,
            "bits_phys_correct": list(bits_phys_correct),
            "bits_phys_trained": list(bits_phys_trained),
            "bits_model_correct": list(bits_model_correct),
            "bits_model_trained": list(bits_model_trained),
            "ber_phys_correct": ber_phys_correct,
            "ber_phys_trained": ber_phys_trained,
            "ber_model_correct": ber_model_correct,
            "ber_model_trained": ber_model_trained,
            "agree_at_correct": agree_at_correct,
            "agree_at_trained": agree_at_trained,
            "agree_natural_pair": agree_natural_pair,
        }
        print(f"  {name}: per-RX mean NRMSE = {np.mean(per_rx_nrmse):.4f}  "
              f"peak phys={peak_phys:+.1f} model={peak_model:+.1f} "
              f"expected={expected:+.1f} deg")
        print(f"    BER phys@correct={ber_phys_correct:.3f} "
              f"phys@trained={ber_phys_trained:.3f} "
              f"model@correct={ber_model_correct:.3f} "
              f"model@trained={ber_model_trained:.3f}")
        print(f"    natural-pair agreement (phys@correct vs model@trained) "
              f"= {agree_natural_pair:.3f}")

    # -- Build sweep summaries across the TX-displacement shots. --
    disp_shots = [s for s in val_specs if s["kind"] == "tx_displaced"]

    def pick(key):
        return [val_data[s["name"]][key] for s in disp_shots]

    sweep = {
        "char_train_nrmse": train_nrmse,
        "char_train_baseline": train_baseline,
        "char_beam_angles_deg": char_angles,
        "char_beam_phys_rms": char_phys_rms,
        "char_beam_model_rms": char_model_rms,
        "char_peak_phys_deg": char_peak_phys,
        "char_peak_model_deg": char_peak_model,
        "char_expected_deg": char_expected,
        "sweep_tx_offsets_m": [s["tx_offset_y"] for s in disp_shots],
        "sweep_per_rx_nrmse_mean": [
            float(np.mean(val_data[s["name"]]["per_rx_nrmse"])) for s in disp_shots
        ],
        "sweep_peak_phys_deg": pick("peak_phys_deg"),
        "sweep_peak_model_deg": pick("peak_model_deg"),
        "sweep_expected_deg": pick("expected_deg"),
        "sweep_ber_phys_correct": pick("ber_phys_correct"),
        "sweep_ber_phys_trained": pick("ber_phys_trained"),
        "sweep_ber_model_correct": pick("ber_model_correct"),
        "sweep_ber_model_trained": pick("ber_model_trained"),
        "sweep_agree_natural_pair": pick("agree_natural_pair"),
    }

    # -- Phase 4: finalize. --
    # Characterize: show TX, three representative RX captures, and the
    # model predictions for the same three. Eight FIR taps and all eight
    # RX traces live in extras.
    rep_idx = (0, N_RX // 2, N_RX - 1)
    char_channels = [
        Channel("TX (V)", kind="scalar", dt=char_sim_dt, pos=TX_REF,
                values=v_tx_char.tolist()),
    ]
    for j in rep_idx:
        char_channels.append(Channel(
            f"RX e{j} phys (V)", kind="scalar", dt=char_sim_dt,
            pos=positions[j], values=v_rx_phys_char[j].tolist(),
        ))
        char_channels.append(Channel(
            f"RX e{j} model (V)", kind="scalar", dt=char_sim_dt,
            values=v_rx_model_char[j].tolist(),
        ))
    char_extras = {
        "role": "characterize",
        "tx_pos_x": TX_REF[0], "tx_pos_y": TX_REF[1],
        "train_nrmse_per_rx": train_nrmse,
        "train_nrmse_baseline": train_baseline,
        "beam_angles_deg": char_angles,
        "beam_phys_rms": char_phys_rms,
        "beam_model_rms": char_model_rms,
        "peak_phys_deg": char_peak_phys,
        "peak_model_deg": char_peak_model,
        "expected_deg": char_expected,
        **{f"fir_h_e{j}": firs[j].h for j in range(N_RX)},
        **{f"v_rx_phys_e{j}": np.asarray(v_rx_phys_char[j], dtype=np.float32)
           for j in range(N_RX)},
        **sweep,
    }
    char_writer.finish(channels=char_channels, extras=char_extras)

    # Validate: TX, RX_center phys + model + residual, plus full per-RX
    # arrays in extras.
    for name, r in val_data.items():
        sim_dt = r["sim_dt"]
        v_tx = r["v_tx"]
        # Center RX is the cleanest single trace to look at.
        ce = N_RX // 2
        residual = (np.asarray(r["v_rx_phys"][ce], dtype=np.float32)
                    - r["v_rx_model"][ce]).astype(np.float32)
        channels = [
            Channel("TX (V)", kind="scalar", dt=sim_dt, pos=r["tx_pos"],
                    values=v_tx.tolist()),
            Channel(f"RX e{ce} phys (V)", kind="scalar", dt=sim_dt,
                    pos=positions[ce],
                    values=r["v_rx_phys"][ce].tolist()),
            Channel(f"RX e{ce} model (V)", kind="scalar", dt=sim_dt,
                    values=r["v_rx_model"][ce].tolist()),
            Channel(f"residual e{ce} (V)", kind="scalar", dt=sim_dt,
                    values=residual.tolist()),
            Channel(f"composite phys @ correct ({r['look_correct_deg']:+.0f} deg) (V)",
                    kind="scalar", dt=sim_dt,
                    values=r["comp_phys_correct"].tolist()),
            Channel(f"composite model @ trained (0 deg) (V)",
                    kind="scalar", dt=sim_dt,
                    values=r["comp_model_trained"].tolist()),
        ]
        extras = {
            "role": "validate",
            "kind": r["kind"],
            "tx_pos_x": r["tx_pos"][0], "tx_pos_y": r["tx_pos"][1],
            "tx_offset_y_m": r["tx_offset_y"],
            "per_rx_nrmse": r["per_rx_nrmse"],
            "per_rx_nrmse_mean": float(np.mean(r["per_rx_nrmse"])),
            "beam_angles_deg": r["beam_angles_deg"],
            "beam_phys_rms": r["beam_phys_rms"],
            "beam_model_rms": r["beam_model_rms"],
            "peak_phys_deg": r["peak_phys_deg"],
            "peak_model_deg": r["peak_model_deg"],
            "expected_deg": r["expected_deg"],
            "look_correct_deg": r["look_correct_deg"],
            "look_trained_deg": r["look_trained_deg"],
            "bits_sent": list(sent_bits),
            "bits_phys_correct": r["bits_phys_correct"],
            "bits_phys_trained": r["bits_phys_trained"],
            "bits_model_correct": r["bits_model_correct"],
            "bits_model_trained": r["bits_model_trained"],
            "ber_phys_correct": r["ber_phys_correct"],
            "ber_phys_trained": r["ber_phys_trained"],
            "ber_model_correct": r["ber_model_correct"],
            "ber_model_trained": r["ber_model_trained"],
            "agree_at_correct": r["agree_at_correct"],
            "agree_at_trained": r["agree_at_trained"],
            "agree_natural_pair": r["agree_natural_pair"],
            **{f"v_rx_phys_e{j}": np.asarray(r["v_rx_phys"][j], dtype=np.float32)
               for j in range(N_RX)},
            **{f"v_rx_model_e{j}": np.asarray(r["v_rx_model"][j], dtype=np.float32)
               for j in range(N_RX)},
            "composite_phys_correct": r["comp_phys_correct"],
            "composite_phys_trained": r["comp_phys_trained"],
            "composite_model_correct": r["comp_model_correct"],
            "composite_model_trained": r["comp_model_trained"],
            **sweep,
        }
        val_writers[name].finish(channels=channels, extras=extras)

    out.finish(dt=dt * args.capture_every, dx=DX, c=c)
