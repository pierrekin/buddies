"""Phase 1 + 2: idealised 4-element TDOA direction finding, plus an
LTI substitution test that swaps phys for an FIR surrogate.

Geometry: a 60 mm square array of four point receivers in open water,
sponged edges, no reflectors. A single point source sits on a circle
of fixed range around the array centre and is swept around the azimuth
in 30 deg steps.

Per azimuth we run two shots:

  * ``char_az_X`` -- TX fires a broadband chirp from the same position.
    For each of the four receivers we fit a FIR from (v_tx, v_rx_phys),
    giving a per-azimuth surrogate of the channel. The chirp-based
    bearing is also recorded here (the Phase 1 result).

  * ``val_az_X_ook`` -- TX fires a *different* waveform (32-bit OOK at
    the same carrier) from the same position. Two bearings are
    computed: one from the FDTD-captured v_rx (``phys``), one from
    ``FIR.predict(v_tx_ook)`` (``model``). The substitution test
    is whether ``|bearing_model - bearing_phys|`` is small compared
    to the phys-vs-truth error from Phase 1.

If LTI holds (which it does in free space), substitution should pass
with sub-degree delta -- meaning a downstream pipeline (firmware,
decoder, whatever) can be exercised against model-predicted RX traces
instead of phys at the trained position without re-running FDTD."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

SIZE = 1.0  # m
FREQ = 40_000.0  # Hz
SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)

ARRAY_CENTER = (0.5, 0.5)
APERTURE = 0.060  # m
RX_POSITIONS = [
    (ARRAY_CENTER[0] - APERTURE / 2, ARRAY_CENTER[1] - APERTURE / 2),
    (ARRAY_CENTER[0] + APERTURE / 2, ARRAY_CENTER[1] - APERTURE / 2),
    (ARRAY_CENTER[0] + APERTURE / 2, ARRAY_CENTER[1] + APERTURE / 2),
    (ARRAY_CENTER[0] - APERTURE / 2, ARRAY_CENTER[1] + APERTURE / 2),
]
N_RX = len(RX_POSITIONS)

TX_RANGE = 0.40  # m
N_AZIMUTHS = 12
AZIMUTHS_DEG = [360.0 * i / N_AZIMUTHS for i in range(N_AZIMUTHS)]

# Char waveform: broadband chirp around the transducer resonance.
CHIRP_DURATION = 0.005  # s
CHIRP_F_LO = 25_000.0
CHIRP_F_HI = 55_000.0
PROP_TAIL = 0.001  # s

# Validation waveform: 32-bit OOK at the carrier frequency. Different
# spectral content from the chirp -- a meaningful LTI test rather than
# an in-sample re-fit.
OOK_N_BITS = 32
OOK_BIT_DUR = 0.0005  # s, 20 carrier cycles per bit
OOK_PRBS_SEED = 1234
OOK_PROP_TAIL = 0.001  # s

# Per-azimuth FIR. At the framework's default CFL of 0.2375 the dt is
# ~0.42 us at 40 kHz, so 2048 taps spans ~860 us -- enough to cover the
# channel's ~590 us impulse response (270 us prop delay + 320 us
# double-BPF ringdown). 1024 taps spans only ~430 us, which truncates
# the BPF tail and was the source of the bad LTI substitution.
FIR_N_TAPS = 2048


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


def matched_filter_arrival(rx, sim_dt, reference):
    """Time of arrival of ``reference`` inside ``rx`` (seconds), via
    cross-correlation peak with sub-sample parabolic interpolation."""
    rx = np.asarray(rx, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    corr = np.abs(np.correlate(rx, ref, mode="full"))
    peak = int(np.argmax(corr))
    delta = 0.0
    if 0 < peak < len(corr) - 1:
        y0, y1, y2 = corr[peak - 1], corr[peak], corr[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
    lag = (peak + delta) - (len(ref) - 1)
    return lag * sim_dt


def estimate_bearing(rx_positions, arrival_times, c):
    """Far-field plane-wave bearing from TDOAs via least squares."""
    p = np.asarray(rx_positions, dtype=np.float64)
    t = np.asarray(arrival_times, dtype=np.float64)
    dp = p[1:] - p[0]
    rhs = -c * (t[1:] - t[0])
    u, *_ = np.linalg.lstsq(dp, rhs, rcond=None)
    return math.atan2(u[1], u[0])


def wrap_pi(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def run_shot(out, name, args, n, DX, dt, tx_pos, voltage_fn, total_duration):
    """One FDTD shot at ``tx_pos`` with ``voltage_fn`` for the source.
    Returns the shot writer, sim dt, v_tx (driven voltage), and a list
    of 4 v_rx traces (one per receiver)."""
    steps = round((total_duration + PROP_TAIL) / dt)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[SPEAKER.source(pos=tx_pos, voltage_fn=voltage_fn,
                                steps=steps, dt=dt)],
        damping=edge_sponge((n, n), DX),
    )
    sw = out.shot(name)
    frames = sw.open((args.nframes(steps), n, n))
    mic_p = np.empty((N_RX, steps), dtype=np.float32)
    v_tx = np.fromiter(
        (voltage_fn(i * sim.dt) for i in range(steps)),
        dtype=np.float32, count=steps,
    )
    print(f"shot {name}: tx={tx_pos}, {steps} steps")
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        for j, p_rx in enumerate(RX_POSITIONS):
            mic_p[j, i] = probe.pressure(sim, p_rx)
    v_rx = [MIC.filter(mic_p[j], sim.dt) for j in range(N_RX)]
    return sw, sim.dt, v_tx, v_rx


def bearing_pipeline(rx_traces, reference, sim_dt, c):
    """Run the 4-channel TDOA pipeline on a set of RX traces and return
    (arrivals, bearing_rad)."""
    arrivals = np.array([
        matched_filter_arrival(rx_traces[j], sim_dt, reference)
        for j in range(N_RX)
    ])
    bearing_rad = estimate_bearing(RX_POSITIONS, arrivals, c)
    return arrivals, bearing_rad


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    n = round(SIZE / DX)
    c = 1500.0
    chirp = linear_chirp(CHIRP_F_LO, CHIRP_F_HI, CHIRP_DURATION)

    rng = np.random.default_rng(OOK_PRBS_SEED)
    ook_bits = tuple(int(b) for b in rng.integers(0, 2, size=OOK_N_BITS))
    ook_fn = ook_voltage(FREQ, ook_bits, OOK_BIT_DUR, drive_v=1.0)
    ook_duration = OOK_BIT_DUR * OOK_N_BITS

    print(f"grid {n}x{n}, dx={DX*1e3:.2f} mm, dt={dt*1e6:.3f} us")
    print(f"array: 4 corners of {APERTURE*1e3:.0f} mm square at "
          f"{ARRAY_CENTER}, TX range {TX_RANGE*1e3:.0f} mm")

    char_records = {}
    val_records = {}
    sweep_rows = []

    for az_deg in AZIMUTHS_DEG:
        az_rad = math.radians(az_deg)
        tx_pos = (
            ARRAY_CENTER[0] + TX_RANGE * math.cos(az_rad),
            ARRAY_CENTER[1] + TX_RANGE * math.sin(az_rad),
        )

        # -- Char shot: chirp at this azimuth, fit one FIR per receiver. --
        char_name = f"char_az_{int(az_deg):03d}deg"
        char_sw, char_dt, v_tx_chirp, v_rx_chirp = run_shot(
            out, char_name, args, n, DX, dt, tx_pos, chirp, CHIRP_DURATION,
        )
        firs = []
        train_nrmse = []
        for j in range(N_RX):
            fir = channel_model.FIRModel(n_taps=FIR_N_TAPS)
            fir.fit(v_tx_chirp, v_rx_chirp[j])
            firs.append(fir)
            pred = fir.predict(v_tx_chirp)[: len(v_rx_chirp[j])]
            train_nrmse.append(float(channel_model.nrmse(v_rx_chirp[j], pred)))
        ref_chirp = np.asarray(v_tx_chirp, dtype=np.float64)
        chirp_arrivals, chirp_bearing = bearing_pipeline(
            v_rx_chirp, ref_chirp, char_dt, c,
        )
        chirp_err_deg = math.degrees(wrap_pi(chirp_bearing - az_rad))
        print(f"  {char_name}: chirp bearing est={math.degrees(chirp_bearing):.2f} "
              f"truth={az_deg:.1f} err={chirp_err_deg:+.2f} deg  "
              f"(FIR train NRMSE mean = {float(np.mean(train_nrmse)):.4f})")

        char_records[char_name] = {
            "writer": char_sw, "sim_dt": char_dt,
            "tx_pos": tx_pos, "az_deg": az_deg,
            "v_rx_chirp": v_rx_chirp,
            "chirp_bearing": chirp_bearing,
            "chirp_err_deg": chirp_err_deg,
            "train_nrmse": train_nrmse,
            "firs": firs,
        }

        # -- Val shot: OOK at same azimuth, phys vs model bearing. --
        val_name = f"val_az_{int(az_deg):03d}deg_ook"
        val_sw, val_dt, v_tx_ook, v_rx_ook_phys = run_shot(
            out, val_name, args, n, DX, dt, tx_pos, ook_fn, ook_duration,
        )
        v_rx_ook_model = [
            firs[j].predict(v_tx_ook)[: len(v_rx_ook_phys[j])].astype(np.float32)
            for j in range(N_RX)
        ]
        ref_ook = np.asarray(v_tx_ook, dtype=np.float64)
        phys_arrivals, phys_bearing = bearing_pipeline(
            v_rx_ook_phys, ref_ook, val_dt, c,
        )
        model_arrivals, model_bearing = bearing_pipeline(
            v_rx_ook_model, ref_ook, val_dt, c,
        )
        phys_err_deg = math.degrees(wrap_pi(phys_bearing - az_rad))
        model_err_deg = math.degrees(wrap_pi(model_bearing - az_rad))
        model_vs_phys_delta_deg = math.degrees(wrap_pi(model_bearing - phys_bearing))
        # Per-RX NRMSE of phys vs model trace (sanity check on the LTI fit).
        ook_nrmse = [
            float(channel_model.nrmse(v_rx_ook_phys[j], v_rx_ook_model[j]))
            for j in range(N_RX)
        ]
        print(f"  {val_name}: ook phys={math.degrees(phys_bearing):.2f} "
              f"model={math.degrees(model_bearing):.2f}  "
              f"delta={model_vs_phys_delta_deg:+.2f} deg  "
              f"(per-RX NRMSE mean = {float(np.mean(ook_nrmse)):.4f})")

        val_records[val_name] = {
            "writer": val_sw, "sim_dt": val_dt,
            "tx_pos": tx_pos, "az_deg": az_deg,
            "v_tx_ook": v_tx_ook,
            "v_rx_ook_phys": v_rx_ook_phys,
            "v_rx_ook_model": v_rx_ook_model,
            "phys_arrivals": phys_arrivals,
            "model_arrivals": model_arrivals,
            "phys_bearing": phys_bearing,
            "model_bearing": model_bearing,
            "phys_err_deg": phys_err_deg,
            "model_err_deg": model_err_deg,
            "model_vs_phys_delta_deg": model_vs_phys_delta_deg,
            "ook_nrmse": ook_nrmse,
        }

        sweep_rows.append({
            "az_true_deg": float(az_deg),
            "chirp_phys_az_est_deg": float(math.degrees(chirp_bearing)),
            "chirp_phys_err_deg": float(chirp_err_deg),
            "ook_phys_az_est_deg": float(math.degrees(phys_bearing)),
            "ook_phys_err_deg": float(phys_err_deg),
            "ook_model_az_est_deg": float(math.degrees(model_bearing)),
            "ook_model_err_deg": float(model_err_deg),
            "ook_model_vs_phys_delta_deg": float(model_vs_phys_delta_deg),
        })

    # -- Aggregate sweep for the view. --
    def pick(key):
        return [r[key] for r in sweep_rows]

    sweep = {
        "rx_positions": np.asarray(RX_POSITIONS, dtype=np.float32),
        "array_center": list(ARRAY_CENTER),
        "tx_range_m": float(TX_RANGE),
        "aperture_m": float(APERTURE),
        "az_true_deg": pick("az_true_deg"),
        "chirp_phys_az_est_deg": pick("chirp_phys_az_est_deg"),
        "chirp_phys_err_deg": pick("chirp_phys_err_deg"),
        "ook_phys_az_est_deg": pick("ook_phys_az_est_deg"),
        "ook_phys_err_deg": pick("ook_phys_err_deg"),
        "ook_model_az_est_deg": pick("ook_model_az_est_deg"),
        "ook_model_err_deg": pick("ook_model_err_deg"),
        "ook_model_vs_phys_delta_deg": pick("ook_model_vs_phys_delta_deg"),
        "mean_abs_chirp_err_deg": float(np.mean(np.abs(pick("chirp_phys_err_deg")))),
        "mean_abs_ook_phys_err_deg": float(np.mean(np.abs(pick("ook_phys_err_deg")))),
        "mean_abs_model_vs_phys_delta_deg": float(np.mean(np.abs(pick("ook_model_vs_phys_delta_deg")))),
        "max_abs_model_vs_phys_delta_deg": float(np.max(np.abs(pick("ook_model_vs_phys_delta_deg")))),
    }
    print(
        f"\nSweep summary:\n"
        f"  chirp phys vs truth:   mean |err| = {sweep['mean_abs_chirp_err_deg']:.2f} deg\n"
        f"  ook phys vs truth:     mean |err| = {sweep['mean_abs_ook_phys_err_deg']:.2f} deg\n"
        f"  model vs phys (ook):   mean |delta| = {sweep['mean_abs_model_vs_phys_delta_deg']:.3f} deg, "
        f"max |delta| = {sweep['max_abs_model_vs_phys_delta_deg']:.3f} deg"
    )

    # -- Finalize char shots. --
    for name, r in char_records.items():
        sim_dt = r["sim_dt"]
        channels = []
        for j, p_rx in enumerate(RX_POSITIONS):
            channels.append(Channel(
                f"RX{j} chirp (V)", kind="scalar", dt=sim_dt, pos=p_rx,
                values=r["v_rx_chirp"][j].tolist(),
            ))
        r["writer"].finish(
            channels=channels,
            extras={
                "role": "char",
                "this_az_true_deg": r["az_deg"],
                "this_chirp_bearing_deg": math.degrees(r["chirp_bearing"]),
                "this_chirp_err_deg": r["chirp_err_deg"],
                "train_nrmse_per_rx": r["train_nrmse"],
                "tx_pos_x": r["tx_pos"][0], "tx_pos_y": r["tx_pos"][1],
                **sweep,
            },
        )

    # -- Finalize val shots. --
    for name, r in val_records.items():
        sim_dt = r["sim_dt"]
        channels = []
        for j, p_rx in enumerate(RX_POSITIONS):
            channels.append(Channel(
                f"RX{j} phys (V)", kind="scalar", dt=sim_dt, pos=p_rx,
                values=r["v_rx_ook_phys"][j].tolist(),
            ))
            channels.append(Channel(
                f"RX{j} model (V)", kind="scalar", dt=sim_dt, pos=p_rx,
                values=r["v_rx_ook_model"][j].tolist(),
            ))
        r["writer"].finish(
            channels=channels,
            extras={
                "role": "val",
                "this_az_true_deg": r["az_deg"],
                "this_phys_bearing_deg": math.degrees(r["phys_bearing"]),
                "this_model_bearing_deg": math.degrees(r["model_bearing"]),
                "this_phys_err_deg": r["phys_err_deg"],
                "this_model_err_deg": r["model_err_deg"],
                "this_model_vs_phys_delta_deg": r["model_vs_phys_delta_deg"],
                "this_phys_arrivals_s": r["phys_arrivals"].astype(np.float64),
                "this_model_arrivals_s": r["model_arrivals"].astype(np.float64),
                "this_ook_nrmse_per_rx": r["ook_nrmse"],
                "tx_pos_x": r["tx_pos"][0], "tx_pos_y": r["tx_pos"][1],
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=c)
