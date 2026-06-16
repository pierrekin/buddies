"""MIMO spatial multiplexing: two simultaneous streams demixed at the
RX array.

Two widely-separated TX positions each transmit an independent PRBS
at the same bit rate, same time, same carrier. The M=8 RX line array
captures the mixture. Given per-(TX, RX_j) FIR fits from per-TX
characterization, the receiver demixes by *beamforming at each TX's
known direction*: a delay-and-sum at the look-angle toward TX_A peaks
the TX_A stream and rejects the TX_B stream, and vice versa.

Phases:

  1. char_a, char_b: each TX alone fires a chirp; fit M FIRs per row of
     the 2 x M channel matrix.

  2. validate_a_alone, validate_b_alone: each TX alone fires its PRBS.
     Compute per-RX NRMSE phys vs FIR predictions, then beamform at the
     known TX direction and decode. Confirms the comm pipeline works
     for a single-stream baseline.

  3. joint: TX_A and TX_B fire their PRBSes simultaneously. Phys
     captures the mixed waveform at every RX; model predicts each RX as
     ``FIR_A_j.predict(prbs_a) + FIR_B_j.predict(prbs_b)`` (LTI
     superposition over sources). Demix both phys and model traces by
     beamforming at +/-look_angle, decode each stream, compare phys vs
     model BER and per-stream agreement.

If linearity + linear demix hold, joint phys and model agree on every
decoded bit -- even when both decoders make mistakes due to cross-talk
between the streams."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, line, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 2.0  # m, doubled to give the two streams angular room to separate
FREQ = 15_000.0  # Hz
C_EST = 1500.0  # m/s, for delay math
LAMBDA = C_EST / FREQ  # 100 mm
N_RX = 8
SPACING = LAMBDA / 2  # half-wavelength avoids grating lobes
APERTURE = (N_RX - 1) * SPACING  # 350 mm

RX_X = 1.7  # m, x of the line array
RX_CY = 1.0  # m, y center of the array
TX_A = (0.3, 1.6)  # upper-left transmitter
TX_B = (0.3, 0.4)  # lower-left transmitter

SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.003  # s, bigger tank -> longer multipath ringdown
# The 2 m tank's propagation delay is ~1 ms; at the default dt ~1.18 us
# that eats 860 of a 1024-tap FIR. Bump to 4096 (~4.8 ms) so the FIRs
# have ~3.8 ms after the prop delay for the actual impulse response.
FIR_N_TAPS = 4096

CHIRP_DURATION = 0.005
CHIRP_F_LO = 5_000.0
CHIRP_F_HI = 30_000.0

# Independent PRBSes per stream so cross-talk can't mask the demix.
PRBS_SEED_A = 1234
PRBS_SEED_B = 5678
N_BITS = 16
BIT_DUR = 0.001
DRIVE_V = 1.0

# Beam-pattern look-angle sweep covers well past +/- TX direction so the
# sidelobe structure is visible too.
LOOK_ANGLES_DEG = tuple(range(-40, 41))


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
    """Per-bit RMS slicer, same as model_link / model_array_tx."""
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


def doa_to(tx_pos):
    """Angle off broadside of the line from array center to a TX
    position. Positive when TX is above center."""
    dy = tx_pos[1] - RX_CY
    dx = RX_X - tx_pos[0]
    return math.degrees(math.atan2(dy, dx))


def delay_and_sum(traces, dt, look_angle_deg, y_offsets, c):
    """Far-field delay-and-sum on a list of M element traces. Returns a
    float64 composite array of the same length."""
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


def beam_pattern(traces, dt, y_offsets, c, look_angles_deg=LOOK_ANGLES_DEG):
    angles = np.asarray(look_angles_deg, dtype=np.float64)
    rms = np.empty(len(angles), dtype=np.float64)
    for i, a in enumerate(angles):
        composite = delay_and_sum(traces, dt, float(a), y_offsets, c)
        rms[i] = float(np.sqrt(np.mean(composite ** 2)))
    return angles, rms


def _capture_rx(sim, n_steps, positions, frames, capture_every):
    """Run the sim for n_steps; per step capture the field (subsampled)
    and the pressure at every RX position. Returns (n_rx, n_steps)."""
    mic_p = np.empty((len(positions), n_steps), dtype=np.float32)
    for i in simargs.progress(n_steps):
        sim.step()
        if i % capture_every == 0:
            frames[i // capture_every] = to_numpy(sim.p)
        for j, pos in enumerate(positions):
            mic_p[j, i] = probe.pressure(sim, pos)
    return mic_p


def _fit_row(v_tx, v_rx_list):
    """Fit one FIR per RX trace. Returns (firs, train_nrmse list)."""
    firs, nrmses = [], []
    for v_rx in v_rx_list:
        fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
        fir.fit(v_tx, v_rx)
        firs.append(fir)
        pred = fir.predict(v_tx)[: len(v_rx)]
        nrmses.append(float(channel_model.nrmse(v_rx, pred)))
    return firs, nrmses


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    n = round(SIZE / DX)
    positions = rx_positions()
    y_offsets = y_offsets_from_center(positions)
    c = 1500.0

    look_a = doa_to(TX_A)
    look_b = doa_to(TX_B)
    prop_delay_a = math.hypot(RX_X - TX_A[0], RX_CY - TX_A[1]) / C_EST
    prop_delay_b = math.hypot(RX_X - TX_B[0], RX_CY - TX_B[1]) / C_EST

    rng_a = np.random.default_rng(PRBS_SEED_A)
    rng_b = np.random.default_rng(PRBS_SEED_B)
    bits_a = tuple(int(b) for b in rng_a.integers(0, 2, size=N_BITS))
    bits_b = tuple(int(b) for b in rng_b.integers(0, 2, size=N_BITS))

    base_chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)
    ook_a = ook_voltage(FREQ, bits_a, BIT_DUR, drive_v=DRIVE_V)
    ook_b = ook_voltage(FREQ, bits_b, BIT_DUR, drive_v=DRIVE_V)

    print(f"TX_A direction from array: {look_a:+.1f} deg, "
          f"prop delay {prop_delay_a * 1e6:.0f} us")
    print(f"TX_B direction from array: {look_b:+.1f} deg, "
          f"prop delay {prop_delay_b * 1e6:.0f} us")

    # -- Phase 1: characterize each TX alone with a chirp. --
    def char_shot(name, tx_pos, prop_delay):
        steps = round((CHIRP_DURATION + prop_delay + PROP_TAIL) / dt)
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(pos=tx_pos, voltage_fn=base_chirp,
                                    steps=steps, dt=dt)],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        frames = sw.open((args.nframes(steps), n, n))
        v_tx = np.fromiter(
            (base_chirp(i * sim.dt) for i in range(steps)),
            dtype=np.float32, count=steps,
        )
        print(f"shot {name}: tx={tx_pos}, {steps} steps")
        mic_p = _capture_rx(sim, steps, positions, frames, args.capture_every)
        v_rx_phys = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]
        firs, train_nrmse = _fit_row(v_tx, v_rx_phys)
        for j, nr in enumerate(train_nrmse):
            print(f"  fitted {name} FIR_RX{j}: training NRMSE = {nr:.4f}")
        return sw, sim.dt, v_tx, v_rx_phys, firs, train_nrmse

    char_a_writer, char_a_dt, v_tx_chirp_a, v_rx_phys_char_a, firs_a, train_nrmse_a = \
        char_shot("char_a", TX_A, prop_delay_a)
    char_b_writer, char_b_dt, v_tx_chirp_b, v_rx_phys_char_b, firs_b, train_nrmse_b = \
        char_shot("char_b", TX_B, prop_delay_b)
    baseline = float(np.mean(train_nrmse_a + train_nrmse_b))

    # -- Phase 2: each TX alone with its PRBS, single-stream baseline. --
    def alone_shot(name, tx_pos, voltage_fn, ook_bits, prop_delay, firs,
                   look_angle, sent_bits, label):
        steps = round((BIT_DUR * N_BITS + prop_delay + PROP_TAIL) / dt)
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(pos=tx_pos, voltage_fn=voltage_fn,
                                    steps=steps, dt=dt)],
            damping=edge_sponge((n, n), DX),
        )
        sw = out.shot(name)
        frames = sw.open((args.nframes(steps), n, n))
        v_tx = np.fromiter(
            (voltage_fn(i * sim.dt) for i in range(steps)),
            dtype=np.float32, count=steps,
        )
        print(f"shot {name}: tx={tx_pos}, {steps} steps, "
              f"look={look_angle:+.1f} deg")
        mic_p = _capture_rx(sim, steps, positions, frames, args.capture_every)
        v_rx_phys = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]
        v_rx_model = [
            firs[j].predict(v_tx)[: len(v_rx_phys[j])].astype(np.float32)
            for j in range(N_RX)
        ]
        per_rx_nrmse = [
            float(channel_model.nrmse(v_rx_phys[j], v_rx_model[j]))
            for j in range(N_RX)
        ]
        composite_phys = delay_and_sum(
            v_rx_phys, sim.dt, look_angle, y_offsets, c,
        ).astype(np.float32)
        composite_model = delay_and_sum(
            v_rx_model, sim.dt, look_angle, y_offsets, c,
        ).astype(np.float32)
        bits_phys, _, _ = decode(composite_phys, sim.dt, N_BITS, BIT_DUR, prop_delay)
        bits_model, _, _ = decode(composite_model, sim.dt, N_BITS, BIT_DUR, prop_delay)
        ber_phys = sum(d != b for d, b in zip(bits_phys, sent_bits)) / N_BITS
        ber_model = sum(d != b for d, b in zip(bits_model, sent_bits)) / N_BITS
        agreement = sum(p == m for p, m in zip(bits_phys, bits_model)) / N_BITS
        print(f"  {name}: per-RX mean NRMSE = {np.mean(per_rx_nrmse):.4f}  "
              f"BER_phys={ber_phys:.3f} BER_model={ber_model:.3f} "
              f"agreement={agreement:.3f}")
        return {
            "writer": sw, "tx_pos": tx_pos, "v_tx": v_tx, "sim_dt": sim.dt,
            "v_rx_phys": v_rx_phys, "v_rx_model": v_rx_model,
            "per_rx_nrmse": per_rx_nrmse,
            "composite_phys": composite_phys, "composite_model": composite_model,
            "bits_phys": list(bits_phys), "bits_model": list(bits_model),
            "ber_phys": float(ber_phys), "ber_model": float(ber_model),
            "agreement": float(agreement),
            "look_angle": look_angle,
            "prop_delay": prop_delay,
            "label": label,
        }

    alone_a = alone_shot("validate_a_alone", TX_A, ook_a, bits_a,
                         prop_delay_a, firs_a, look_a, bits_a, "stream A")
    alone_b = alone_shot("validate_b_alone", TX_B, ook_b, bits_b,
                         prop_delay_b, firs_b, look_b, bits_b, "stream B")

    # -- Phase 3: joint TX_A + TX_B, two streams simultaneously. --
    duration = BIT_DUR * N_BITS
    steps = round((duration + max(prop_delay_a, prop_delay_b) + PROP_TAIL) / dt)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[
            SPEAKER.source(pos=TX_A, voltage_fn=ook_a, steps=steps, dt=dt),
            SPEAKER.source(pos=TX_B, voltage_fn=ook_b, steps=steps, dt=dt),
        ],
        damping=edge_sponge((n, n), DX),
    )
    joint_writer = out.shot("joint")
    frames = joint_writer.open((args.nframes(steps), n, n))
    v_tx_a_joint = np.fromiter(
        (ook_a(i * sim.dt) for i in range(steps)),
        dtype=np.float32, count=steps,
    )
    v_tx_b_joint = np.fromiter(
        (ook_b(i * sim.dt) for i in range(steps)),
        dtype=np.float32, count=steps,
    )
    print(f"shot joint: 2 TX firing simultaneously, {steps} steps")
    mic_p = _capture_rx(sim, steps, positions, frames, args.capture_every)
    v_rx_phys = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]

    # Model: each RX trace is the sum of two predicted contributions.
    v_rx_model = []
    for j in range(N_RX):
        contrib_a = firs_a[j].predict(v_tx_a_joint)[: len(v_rx_phys[j])]
        contrib_b = firs_b[j].predict(v_tx_b_joint)[: len(v_rx_phys[j])]
        v_rx_model.append((contrib_a + contrib_b).astype(np.float32))

    per_rx_nrmse_joint = [
        float(channel_model.nrmse(v_rx_phys[j], v_rx_model[j])) for j in range(N_RX)
    ]
    print(f"  joint per-RX mean NRMSE = {np.mean(per_rx_nrmse_joint):.4f}")

    # Demix: beamform at each TX's known direction, decode the composite.
    def demix_and_decode(traces, look_angle, prop_delay, sent_bits, label):
        composite = delay_and_sum(traces, sim.dt, look_angle, y_offsets, c)
        bits_dec, _, _ = decode(composite, sim.dt, N_BITS, BIT_DUR, prop_delay)
        ber = sum(d != b for d, b in zip(bits_dec, sent_bits)) / N_BITS
        return composite.astype(np.float32), list(bits_dec), float(ber)

    comp_a_phys, bits_a_phys, ber_a_phys = demix_and_decode(
        v_rx_phys, look_a, prop_delay_a, bits_a, "A phys",
    )
    comp_a_model, bits_a_model, ber_a_model = demix_and_decode(
        v_rx_model, look_a, prop_delay_a, bits_a, "A model",
    )
    comp_b_phys, bits_b_phys, ber_b_phys = demix_and_decode(
        v_rx_phys, look_b, prop_delay_b, bits_b, "B phys",
    )
    comp_b_model, bits_b_model, ber_b_model = demix_and_decode(
        v_rx_model, look_b, prop_delay_b, bits_b, "B model",
    )
    agreement_a = sum(p == m for p, m in zip(bits_a_phys, bits_a_model)) / N_BITS
    agreement_b = sum(p == m for p, m in zip(bits_b_phys, bits_b_model)) / N_BITS
    composite_nrmse_a = float(channel_model.nrmse(comp_a_phys, comp_a_model))
    composite_nrmse_b = float(channel_model.nrmse(comp_b_phys, comp_b_model))

    print(f"  joint stream A: BER_phys={ber_a_phys:.3f} BER_model={ber_a_model:.3f} "
          f"agreement={agreement_a:.3f}")
    print(f"  joint stream B: BER_phys={ber_b_phys:.3f} BER_model={ber_b_model:.3f} "
          f"agreement={agreement_b:.3f}")

    # Beam patterns for the joint shot phys vs model (full sweep).
    joint_angles, joint_phys_rms = beam_pattern(v_rx_phys, sim.dt, y_offsets, c)
    _, joint_model_rms = beam_pattern(v_rx_model, sim.dt, y_offsets, c)

    # -- Phase 4: finalize. --
    rep_idx = (0, N_RX // 2, N_RX - 1)

    def finish_char(writer, name, tx_pos, sim_dt, v_tx, v_rx_phys,
                    firs, train_nrmse, label):
        angles, phys_rms = beam_pattern(v_rx_phys, sim_dt, y_offsets, c)
        v_rx_model_char = [
            firs[j].predict(v_tx)[: len(v_rx_phys[j])].astype(np.float32)
            for j in range(N_RX)
        ]
        _, model_rms = beam_pattern(v_rx_model_char, sim_dt, y_offsets, c)
        expected = doa_to(tx_pos)
        channels = [
            Channel("TX (V)", kind="scalar", dt=sim_dt, pos=tx_pos,
                    values=v_tx.tolist()),
        ]
        for j in rep_idx:
            channels.append(Channel(
                f"RX e{j} phys (V)", kind="scalar", dt=sim_dt,
                pos=positions[j], values=v_rx_phys[j].tolist(),
            ))
            channels.append(Channel(
                f"RX e{j} model (V)", kind="scalar", dt=sim_dt,
                values=v_rx_model_char[j].tolist(),
            ))
        extras = {
            "role": "characterize",
            "label": label,
            "tx_pos_x": tx_pos[0], "tx_pos_y": tx_pos[1],
            "train_nrmse_per_rx": train_nrmse,
            "train_nrmse_baseline": baseline,
            "beam_angles_deg": angles,
            "beam_phys_rms": phys_rms,
            "beam_model_rms": model_rms,
            "expected_deg": expected,
            **{f"fir_h_e{j}": firs[j].h for j in range(N_RX)},
            **{f"v_rx_phys_e{j}": np.asarray(v_rx_phys[j], dtype=np.float32)
               for j in range(N_RX)},
        }
        writer.finish(channels=channels, extras=extras)

    finish_char(char_a_writer, "char_a", TX_A, char_a_dt, v_tx_chirp_a,
                v_rx_phys_char_a, firs_a, train_nrmse_a, "stream A characterize")
    finish_char(char_b_writer, "char_b", TX_B, char_b_dt, v_tx_chirp_b,
                v_rx_phys_char_b, firs_b, train_nrmse_b, "stream B characterize")

    def finish_alone(r, name, sent_bits):
        ce = N_RX // 2
        residual = (np.asarray(r["v_rx_phys"][ce], dtype=np.float32)
                    - r["v_rx_model"][ce]).astype(np.float32)
        composite_residual = (r["composite_phys"]
                              - r["composite_model"]).astype(np.float32)
        channels = [
            Channel("TX (V)", kind="scalar", dt=r["sim_dt"], pos=r["tx_pos"],
                    values=r["v_tx"].tolist()),
            Channel(f"RX e{ce} phys (V)", kind="scalar", dt=r["sim_dt"],
                    pos=positions[ce], values=r["v_rx_phys"][ce].tolist()),
            Channel(f"RX e{ce} model (V)", kind="scalar", dt=r["sim_dt"],
                    values=r["v_rx_model"][ce].tolist()),
            Channel(f"residual e{ce} (V)", kind="scalar", dt=r["sim_dt"],
                    values=residual.tolist()),
            Channel(f"composite (demixed) phys (V)", kind="scalar", dt=r["sim_dt"],
                    values=r["composite_phys"].tolist()),
            Channel(f"composite (demixed) model (V)", kind="scalar", dt=r["sim_dt"],
                    values=r["composite_model"].tolist()),
        ]
        angles, phys_rms = beam_pattern(r["v_rx_phys"], r["sim_dt"], y_offsets, c)
        _, model_rms = beam_pattern(r["v_rx_model"], r["sim_dt"], y_offsets, c)
        extras = {
            "role": "validate_alone",
            "label": r["label"],
            "tx_pos_x": r["tx_pos"][0], "tx_pos_y": r["tx_pos"][1],
            "per_rx_nrmse": r["per_rx_nrmse"],
            "per_rx_nrmse_mean": float(np.mean(r["per_rx_nrmse"])),
            "train_nrmse_baseline": baseline,
            "look_angle_deg": r["look_angle"],
            "bits_sent": list(sent_bits),
            "bits_phys": r["bits_phys"],
            "bits_model": r["bits_model"],
            "ber_phys": r["ber_phys"],
            "ber_model": r["ber_model"],
            "agreement": r["agreement"],
            "beam_angles_deg": angles,
            "beam_phys_rms": phys_rms,
            "beam_model_rms": model_rms,
            "expected_deg": doa_to(r["tx_pos"]),
            **{f"v_rx_phys_e{j}": np.asarray(r["v_rx_phys"][j], dtype=np.float32)
               for j in range(N_RX)},
            **{f"v_rx_model_e{j}": np.asarray(r["v_rx_model"][j], dtype=np.float32)
               for j in range(N_RX)},
        }
        r["writer"].finish(channels=channels, extras=extras)

    finish_alone(alone_a, "validate_a_alone", bits_a)
    finish_alone(alone_b, "validate_b_alone", bits_b)

    # Joint shot finalize.
    ce = N_RX // 2
    joint_channels = [
        Channel("TX_A (V)", kind="scalar", dt=sim.dt, pos=TX_A,
                values=v_tx_a_joint.tolist()),
        Channel("TX_B (V)", kind="scalar", dt=sim.dt, pos=TX_B,
                values=v_tx_b_joint.tolist()),
        Channel(f"RX e{ce} phys (mixture) (V)", kind="scalar", dt=sim.dt,
                pos=positions[ce], values=v_rx_phys[ce].tolist()),
        Channel(f"RX e{ce} model (mixture) (V)", kind="scalar", dt=sim.dt,
                values=v_rx_model[ce].tolist()),
        Channel("composite A demixed phys (V)", kind="scalar", dt=sim.dt,
                values=comp_a_phys.tolist()),
        Channel("composite A demixed model (V)", kind="scalar", dt=sim.dt,
                values=comp_a_model.tolist()),
        Channel("composite B demixed phys (V)", kind="scalar", dt=sim.dt,
                values=comp_b_phys.tolist()),
        Channel("composite B demixed model (V)", kind="scalar", dt=sim.dt,
                values=comp_b_model.tolist()),
    ]
    joint_writer.finish(
        channels=joint_channels,
        extras={
            "role": "joint",
            "tx_a_pos": list(TX_A), "tx_b_pos": list(TX_B),
            "look_a_deg": look_a, "look_b_deg": look_b,
            "expected_a_deg": doa_to(TX_A), "expected_b_deg": doa_to(TX_B),
            "per_rx_nrmse": per_rx_nrmse_joint,
            "per_rx_nrmse_mean": float(np.mean(per_rx_nrmse_joint)),
            "train_nrmse_baseline": baseline,
            "bits_a_sent": list(bits_a),
            "bits_a_phys": bits_a_phys,
            "bits_a_model": bits_a_model,
            "bits_b_sent": list(bits_b),
            "bits_b_phys": bits_b_phys,
            "bits_b_model": bits_b_model,
            "ber_a_phys": ber_a_phys,
            "ber_a_model": ber_a_model,
            "ber_b_phys": ber_b_phys,
            "ber_b_model": ber_b_model,
            "agreement_a": float(agreement_a),
            "agreement_b": float(agreement_b),
            "composite_nrmse_a": composite_nrmse_a,
            "composite_nrmse_b": composite_nrmse_b,
            "beam_angles_deg": joint_angles,
            "beam_phys_rms": joint_phys_rms,
            "beam_model_rms": joint_model_rms,
            **{f"v_rx_phys_e{j}": np.asarray(v_rx_phys[j], dtype=np.float32)
               for j in range(N_RX)},
            **{f"v_rx_model_e{j}": np.asarray(v_rx_model[j], dtype=np.float32)
               for j in range(N_RX)},
        },
    )

    out.finish(dt=dt * args.capture_every, dx=DX, c=c)
