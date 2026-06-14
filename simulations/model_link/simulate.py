"""Communication-through-model vs. communication-through-FDTD.

A 1024-tap FIR is fit on a chirp probe (the ``chirp_train`` shot), then
the *same* OOK message is sent through the link at a sweep of bit
durations -- one shot per duration. For each link shot we have two
receive traces:

  * ``v_rx_phys``  : the real mic voltage from the FDTD run
  * ``v_rx_model`` : ``FIR.predict(v_tx)`` -- what the model would
                     have given for that drive

The same RMS slicer decodes both, yielding ``decoded_phys`` and
``decoded_model``. The two big questions this sim answers:

  * Does the model decode agree with the physical decode? (``agreement``)
  * As we shrink the bit duration past the channel's ringdown, do they
    diverge in interesting ways? (``BER_phys`` vs ``BER_model`` curve)

The bit-duration sweep starts well above the speaker/mic BPF's ringdown
(``Q=4`` at ``f0=15 kHz`` gives roughly 400 us of tail) and walks down
through it. At long bits both pipelines should hit BER ~ 0 and full
agreement; at the bottom the ISI exceeds the slicer's per-bit window
and decisions get noisy, which is where the LTI model is most likely
to disagree with reality."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz, OOK carrier + speaker/mic resonance
TX = (0.2, 0.5)  # m
RX = (0.8, 0.5)  # m
SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.002  # s of silence after the message ends, for BPF ringdown
FIR_N_TAPS = 1024  # from model_compare: ~2.6% NRMSE at this size

# A fixed PRBS so phys and model decode the same bitstream every run.
N_BITS = 32
PRBS_SEED = 1234

# 1 ms (~6x ringdown) down to 100 us (~1.5 carrier cycles per bit). The
# top end should be trivial; the bottom is past where any sensible LTI
# decoder still works, which is the point.
BIT_DURS = (0.001, 0.0005, 0.00025, 0.00015, 0.0001)


def linear_chirp(f_lo, f_hi, duration, amplitude=1.0):
    """Linear sweep from f_lo to f_hi over ``duration``. Same probe
    ``model_compare`` uses; persistent-excitation over the BPF band so
    the FIR fit converges to the true channel impulse response."""
    def v(t):
        if t < 0 or t > duration:
            return 0.0
        k = (f_hi - f_lo) / duration
        phase = 2 * math.pi * (f_lo * t + 0.5 * k * t * t)
        return amplitude * math.sin(phase)
    return v


def ook_voltage(freq, bits, bit_dur, drive_v=1.0):
    """Voltage-domain OOK on a square carrier: +-drive_v during 1-bits,
    0 during 0-bits. Speaker turns this into pressure via its BPF."""
    omega = 2 * math.pi * freq

    def v(t):
        if t < 0:
            return 0.0
        bit_idx = int(t / bit_dur)
        if bit_idx >= len(bits) or bits[bit_idx] == 0:
            return 0.0
        local_t = t - bit_idx * bit_dur
        return drive_v * (1.0 if math.sin(omega * local_t) >= 0 else -1.0)

    return v


def decode(rx, sim_dt, n_bits, bit_dur, prop_delay):
    """Per-bit RMS slicer (same shape as ``ook_link``).

    For each bit window we integrate the *second half* only -- that skips
    the rising edge of the BPF and any leftover ringing from the previous
    bit, both of which dominate the first half and would otherwise alias
    a '0' as a '1' on the bit after a '1'. The threshold sits midway
    between the strongest and weakest per-bit RMS; works whenever the
    message contains both bit values."""
    samples = np.asarray(rx, dtype=np.float32)
    spb = int(round(bit_dur / sim_dt))
    delay = int(round(prop_delay / sim_dt))
    if spb < 2:
        # Below 2 samples/bit the slicer can't even half-window; bail with
        # all-zero decisions so the comparison still runs.
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

    rng = np.random.default_rng(PRBS_SEED)
    bits = tuple(int(b) for b in rng.integers(0, 2, size=N_BITS))
    # Geometric estimate; the exact c the sim uses is sim.c after
    # construction, but for the decode offset the estimate is enough.
    prop_delay = math.hypot(RX[0] - TX[0], RX[1] - TX[1]) / 1500.0

    shots_spec = [{
        "name": "chirp_train",
        "duration": 0.005,
        "voltage_fn": linear_chirp(5_000.0, 30_000.0, 0.005),
        "role": "train",
        "save_frames": False,
        "bit_dur": None,
    }]
    for bd in BIT_DURS:
        # Pad shot name so it sorts in the combobox by descending duration.
        shots_spec.append({
            "name": f"link_{int(bd * 1e6):04d}us",
            "duration": bd * N_BITS,
            "voltage_fn": ook_voltage(FREQ, bits, bd, drive_v=1.0),
            "role": "test",
            # Save frames only for the easiest link, as eye-candy. The
            # tighter shots are fast to re-run anyway if you want frames.
            "save_frames": bd == BIT_DURS[0],
            "bit_dur": bd,
        })

    # -- Phase 1: FDTD per shot, capture (v_tx, v_rx_phys). --
    pairs = {}  # name -> (v_tx, v_rx_phys, sim_dt)
    shot_writers = {}
    for spec in shots_spec:
        steps = round((spec["duration"] + prop_delay + PROP_TAIL) / dt)
        voltage_fn = spec["voltage_fn"]
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(pos=TX, voltage_fn=voltage_fn, steps=steps, dt=dt)],
            damping=edge_sponge((n, n), DX),
        )

        sw = out.shot(spec["name"])
        shot_writers[spec["name"]] = sw
        frames = sw.open((args.nframes(steps), n, n)) if spec["save_frames"] else None

        v_tx = np.fromiter(
            (voltage_fn(i * sim.dt) for i in range(steps)),
            dtype=np.float32, count=steps,
        )
        mic_p = np.empty(steps, dtype=np.float32)

        print(f"shot {spec['name']}: {steps} steps, {steps * sim.dt * 1e3:.2f} ms")
        for i in simargs.progress(steps):
            sim.step()
            if frames is not None and i % args.capture_every == 0:
                frames[i // args.capture_every] = to_numpy(sim.p)
            mic_p[i] = probe.pressure(sim, RX)

        v_rx = MIC.filter(mic_p, sim.dt)
        pairs[spec["name"]] = (v_tx, v_rx, sim.dt)

    # -- Phase 2: fit the FIR once on chirp_train. --
    v_tx_chirp, v_rx_chirp, _ = pairs["chirp_train"]
    fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
    fir.fit(v_tx_chirp, v_rx_chirp)
    train_pred = fir.predict(v_tx_chirp)[: len(v_rx_chirp)]
    train_nrmse = channel_model.nrmse(v_rx_chirp, train_pred)
    print(f"fitted {fir.name}: training NRMSE = {train_nrmse:.4f}")

    # -- Phase 3: predict + decode + compare on each link shot. --
    link_specs = [s for s in shots_spec if s["role"] == "test"]
    sweep_bit_durs, sweep_ber_phys, sweep_ber_model, sweep_agreement = [], [], [], []
    link_results = {}
    for spec in link_specs:
        name = spec["name"]
        v_tx, v_rx_phys, sim_dt = pairs[name]
        v_rx_model = fir.predict(v_tx)[: len(v_rx_phys)]

        bits_phys, rms_phys, thr_phys = decode(v_rx_phys, sim_dt, N_BITS, spec["bit_dur"], prop_delay)
        bits_model, rms_model, thr_model = decode(v_rx_model, sim_dt, N_BITS, spec["bit_dur"], prop_delay)

        ber_phys = sum(d != b for d, b in zip(bits_phys, bits)) / N_BITS
        ber_model = sum(d != b for d, b in zip(bits_model, bits)) / N_BITS
        agreement = sum(p == m for p, m in zip(bits_phys, bits_model)) / N_BITS

        link_results[name] = {
            "v_rx_model": v_rx_model,
            "decoded_phys": list(bits_phys),
            "decoded_model": list(bits_model),
            "ber_phys": float(ber_phys),
            "ber_model": float(ber_model),
            "agreement": float(agreement),
            "threshold_phys": float(thr_phys),
            "threshold_model": float(thr_model),
        }
        sweep_bit_durs.append(float(spec["bit_dur"]))
        sweep_ber_phys.append(float(ber_phys))
        sweep_ber_model.append(float(ber_model))
        sweep_agreement.append(float(agreement))
        print(
            f"  {name:>14}: BER_phys={ber_phys:.3f}  BER_model={ber_model:.3f}"
            f"  agreement={agreement:.3f}"
        )

    # Sweep summary lives on every shot so the viewer can highlight the
    # current shot's point on the curve without juggling a separate
    # 'summary' shot.
    sweep = {
        "sweep_bit_durs": sweep_bit_durs,
        "sweep_ber_phys": sweep_ber_phys,
        "sweep_ber_model": sweep_ber_model,
        "sweep_agreement": sweep_agreement,
        "sent_bits": list(bits),
    }

    # -- Phase 4: finalize shots. --
    v_tx_c, v_rx_c, sim_dt_c = pairs["chirp_train"]
    shot_writers["chirp_train"].finish(
        channels=[
            Channel("TX (V)", kind="scalar", dt=sim_dt_c, pos=TX, values=v_tx_c.tolist()),
            Channel("RX truth (V)", kind="scalar", dt=sim_dt_c, pos=RX, values=v_rx_c.tolist()),
            Channel(f"RX model {fir.name} (V)", kind="scalar", dt=sim_dt_c, values=train_pred.tolist()),
        ],
        extras={
            "role": "train",
            "fir_h": fir.h,  # array goes to extras.npz
            "fir_n_taps": fir.n_taps,
            "train_nrmse": float(train_nrmse),
            **sweep,
        },
    )

    for spec in link_specs:
        name = spec["name"]
        r = link_results[name]
        v_tx, v_rx_phys, sim_dt = pairs[name]
        v_rx_model = r["v_rx_model"]
        residual = (v_rx_phys[: len(v_rx_model)] - v_rx_model).astype(np.float32)
        shot_writers[name].finish(
            channels=[
                Channel("TX (V)", kind="scalar", dt=sim_dt, pos=TX, values=v_tx.tolist()),
                Channel("RX phys (V)", kind="scalar", dt=sim_dt, pos=RX, values=v_rx_phys.tolist()),
                Channel("RX model (V)", kind="scalar", dt=sim_dt, values=v_rx_model.tolist()),
                Channel("residual phys-model (V)", kind="scalar", dt=sim_dt, values=residual.tolist()),
            ],
            extras={
                "role": "test",
                "bit_dur": float(spec["bit_dur"]),
                "decoded_phys": r["decoded_phys"],
                "decoded_model": r["decoded_model"],
                "ber_phys": r["ber_phys"],
                "ber_model": r["ber_model"],
                "agreement": r["agreement"],
                "threshold_phys": r["threshold_phys"],
                "threshold_model": r["threshold_model"],
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
