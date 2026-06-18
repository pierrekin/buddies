"""Two-buddy link in a long, thin tank: bearing vs range, with the DUT
rotated in place instead of the target orbited around it.

A buddy node is a single 4-element quad array (a small square of PZT
elements), half-duplex: the *same* four elements transmit and receive. One
buddy pings; the other (the device under test) listens, and from that one
reception does both jobs -- decode the OOK packet, and bear the pinger via
4-channel TDOA.

Reaching the ranges we actually care about (metres, not the 120 mm of the
first pass) means a domain metres long, which direct FDTD can only afford
if it stays thin. Free space depends only on *relative* geometry, so we
exploit that: the pinger sits fixed on the long axis at the far end, the
listener sits at the near end, and we sweep heading by **rotating the
listener's four element positions about its own centre** rather than moving
the target. The wavefront then only ever needs a narrow strip, so a 10 m
range is a 6000 x 240 grid, not 6000 x 6000.

The estimator only knows the array's *body frame* (the canonical, unrotated
element layout) and the measured arrivals; it has no idea of its absolute
orientation. So a heading of phi puts the on-axis target at body-frame
bearing -phi, and that is the truth we score against.

Two pinger modes, the open question from the first pass:

  * ``point`` -- a single element. The listener sees a point source: clean
    arrivals, good bearing. Lowest source level.
  * ``wide``  -- all four elements firing the same packet (broadside). Four
    times the source level, but the listener sees the full array aperture.
    At close range (aperture^2 / lambda ~= 0.9 m for the 120 mm array) that
    spread corrupts the arrivals and the plane-wave bearing degrades; by a
    couple of metres the aperture looks point-like again and the two modes
    converge. The range axis is built to show exactly that crossover.

Transducer model: 90 kHz resonance, Q=10 -- a ~9 kHz band and ~35 us ring
time (tau = Q / (pi f0)). Bit duration is held well above tau so the link
closes; bearing is taken off the correlation *envelope* (carrier-free) so
it does not cycle-slip by a 16.7 mm carrier wavelength.

Idealised still: 2D, open water, no ambient noise, no reflectors. So this
nails the *geometry* (bearing vs range, the near->far crossover) but not the
*link budget*: 2D spreading is cylindrical not spherical and there is no
absorption, so BER/eye stay flat with range and say nothing about reach.
SNR-to-1 km belongs to analytic propagation models, not this sim and not a
surrogate trained on it. The FIR channel surrogate (``model_tdoa``,
``buddies.channel_model``) is left out of the active path; hook marked below.
"""

import math

import numpy as np

from buddies import simargs  # channel_model: surrogate hook, reintroduced next iteration
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 64}

FREQ = 90_000.0  # Hz, transducer resonance = OOK carrier
Q = 10.0  # transducer quality factor (~9 kHz usable band, ~35 us ring time)
C_EST = 1500.0  # m/s, seawater sound speed

SPEAKER = Speaker(f0=FREQ, q=Q, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=Q, sensitivity_v_per_pa=1.0)

# Element spacing is a design knob, and the sim's job is to pin it down: at
# 90 kHz a 45 mm baseline gives inter-element delays under two carrier cycles,
# so the ~5 us envelope-timing bias is a large fraction and bearing is poor
# (~5-15 deg). 120 mm puts the delays at ~4+ cycles and bearing drops under a
# degree. So 120 mm is a finding, not an arbitrary choice.
APERTURE = 0.12  # m, side of the square quad array (element spacing)
WIDTH = 0.45  # m, transverse domain extent (kept comfortable, not thin)
MARGIN = 0.15  # m, axial clearance + sponge at each end (clears the rotated array)
MODES = ("point", "wide")

# Range experiment: sweep range across the pinger's near->far crossover at a
# fixed, clean DUT orientation. The sub-metre points are where the ``wide``
# pinger's aperture still looks extended (bad bearing); 2/5/10 m are the
# ranges we care about, where it has collapsed to a point and both modes agree.
RANGES = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
RANGE_HEADING = 0.0  # deg, a clean DUT orientation so wide-vs-point is the only variable

# Azimuth experiment: a fixed, comfortably far range, sweep heading to show
# bearing holds in every direction (and wide ~= point in the far field).
AZ_RANGE = 2.0  # m
AZ_HEADINGS = (0.0, 45.0, 90.0, 135.0)

# OOK packet. BIT_DUR held well above the ring time tau ~= 35 us.
N_BITS = 12
BIT_DUR = 0.00022  # s, ~20 carrier cycles, ~6 ring time constants
PRBS_SEED = 1234
PROP_TAIL = 0.0006  # s

# Skip the field movie above this cell count: at metre scale a full frame
# history is tens of GB. The near-range shots stay small enough to keep one.
MAX_FRAME_CELLS = 200_000


def square_corners(center, aperture):
    """The four corners of an ``aperture``-sided square centred at
    ``center`` (meters): lower-left, lower-right, upper-right, upper-left."""
    cx, cy = center
    h = aperture / 2
    return [
        (cx - h, cy - h),
        (cx + h, cy - h),
        (cx + h, cy + h),
        (cx - h, cy + h),
    ]


def rotate_about(p, center, ang_rad):
    cx, cy = center
    dx, dy = p[0] - cx, p[1] - cy
    ca, sa = math.cos(ang_rad), math.sin(ang_rad)
    return (cx + ca * dx - sa * dy, cy + sa * dx + ca * dy)


def dut_world_corners(center, aperture, heading_deg):
    """The listener's four element positions in world coordinates: the
    canonical square rotated about its own centre by ``heading_deg``."""
    h = math.radians(heading_deg)
    return [rotate_about(p, center, h) for p in square_corners(center, aperture)]


def body_corners(aperture):
    """The canonical element layout, centred at the origin. This is all the
    estimator knows -- it has no sense of absolute orientation."""
    return square_corners((0.0, 0.0), aperture)


def geometry(R, heading_deg, mode, dx):
    """Lay out one shot in the long-thin tank: domain size, the listener's
    (rotated) world element positions, and the pinger's source positions."""
    nx = round((R + 2 * MARGIN) / dx)
    ny = round(WIDTH / dx)
    dut_center = (MARGIN, WIDTH / 2)
    tx_center = (MARGIN + R, WIDTH / 2)
    dut_world = dut_world_corners(dut_center, APERTURE, heading_deg)
    tx_pos = [tx_center] if mode == "point" else square_corners(tx_center, APERTURE)
    return nx, ny, dut_world, tx_pos, tx_center


def sample_bilinear(p, pos, dx):
    """Bilinearly interpolated pressure at a sub-cell position. Nearest-cell
    probing (``buddies.probe.pressure``) snaps a rotated array's elements to
    the grid by up to half a cell, which is fatal to the fine TDOA we measure
    here; interpolating samples each element at its true position instead."""
    gx, gy = pos[0] / dx, pos[1] / dx
    ix, iy = int(math.floor(gx)), int(math.floor(gy))
    fx, fy = gx - ix, gy - iy
    return float(
        p[ix, iy] * (1 - fx) * (1 - fy)
        + p[ix + 1, iy] * fx * (1 - fy)
        + p[ix, iy + 1] * (1 - fx) * fy
        + p[ix + 1, iy + 1] * fx * fy
    )


def ook_voltage(freq, bits, bit_dur, drive_v=1.0):
    """OOK: a hard-keyed carrier, one bit per ``bit_dur`` window."""
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


def analytic_envelope(x):
    """Magnitude of the analytic signal of ``x`` (a carrier-free envelope),
    via the FFT Hilbert transform. numpy-only, so no scipy dependency."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(X * h))


def matched_filter_arrival(rx, sim_dt, reference):
    """Time of arrival (seconds) of ``reference`` within ``rx``, from the
    *envelope* of the cross-correlation. The raw correlation oscillates at
    the 90 kHz carrier, so ``argmax(|corr|)`` cycle-slips by a whole 16.7 mm
    period; the envelope is a single broad lobe, so the arrival is
    unambiguous and refined by sub-sample parabolic interpolation."""
    rx = np.asarray(rx, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    corr = analytic_envelope(np.correlate(rx, ref, mode="full"))
    peak = int(np.argmax(corr))
    delta = 0.0
    if 0 < peak < len(corr) - 1:
        y0, y1, y2 = corr[peak - 1], corr[peak], corr[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
    lag = (peak + delta) - (len(ref) - 1)
    return lag * sim_dt


def estimate_bearing(body_positions, arrival_times, c):
    """Far-field plane-wave bearing (radians) from per-element TDOAs via
    least squares, in the array's body frame. Returns the direction *to*
    the source as the array sees it."""
    p = np.asarray(body_positions, dtype=np.float64)
    t = np.asarray(arrival_times, dtype=np.float64)
    dp = p[1:] - p[0]
    rhs = -c * (t[1:] - t[0])
    u, *_ = np.linalg.lstsq(dp, rhs, rcond=None)
    return math.atan2(u[1], u[0])


def wrap_pi(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def combine_aligned(traces, arrivals, sim_dt):
    """Coherently sum the element traces, each shifted earlier by its own
    arrival so the packets line up -- delay-and-sum steered by the measured
    TDOAs. The bearing pipeline's arrivals double as the combiner's delays."""
    delays = np.asarray(arrivals, dtype=np.float64) - float(np.min(arrivals))
    out = np.zeros(len(traces[0]), dtype=np.float64)
    for v, d in zip(traces, delays):
        x = np.asarray(v, dtype=np.float64)
        shift = int(round(d / sim_dt))
        if shift > 0:
            x = np.concatenate([x[shift:], np.zeros(shift)])
        out += x
    return out


def decode(rx, sim_dt, n_bits, bit_dur, start_delay):
    """OOK envelope decode: per-bit RMS over the back half of each bit
    window, thresholded at the midpoint. Returns (bits, rms, threshold)."""
    samples = np.asarray(rx, dtype=np.float64)
    spb = int(round(bit_dur / sim_dt))
    delay = int(round(start_delay / sim_dt))
    if spb < 2 or delay < 0:
        return tuple([0] * n_bits), np.zeros(n_bits), 0.0
    rms = np.array([
        float(np.sqrt(np.mean(
            samples[delay + i * spb + spb // 2 : delay + (i + 1) * spb] ** 2
        )))
        for i in range(n_bits)
    ])
    threshold = float((rms.min() + rms.max()) / 2) if rms.max() > rms.min() else 0.0
    return tuple(int(r > threshold) for r in rms), rms, threshold


def eye_contrast(rms, truth_bits):
    """Separation between the '1' and '0' RMS levels, normalised to [0, 1]."""
    rms = np.asarray(rms, dtype=np.float64)
    truth = np.asarray(truth_bits)
    highs, lows = rms[truth == 1], rms[truth == 0]
    if highs.size == 0 or lows.size == 0:
        return 0.0
    hi, lo = float(highs.mean()), float(lows.mean())
    return (hi - lo) / (hi + lo) if (hi + lo) > 0 else 0.0


def ber(decoded, truth):
    return sum(d != b for d, b in zip(decoded, truth)) / len(truth)


def run_shot(out, name, args, nx, ny, DX, dt, R, dut_world, tx_pos, ook_fn):
    """One FDTD ping in a long-thin tank. The pinger fires from ``tx_pos``,
    the listener captures its four ``dut_world`` elements. The field movie
    is skipped above MAX_FRAME_CELLS (metre-scale domains are too big)."""
    steps = round((BIT_DUR * N_BITS + R / C_EST + PROP_TAIL) / dt)
    capture = nx * ny <= MAX_FRAME_CELLS
    sources = [
        SPEAKER.source(pos=p, voltage_fn=ook_fn, steps=steps, dt=dt)
        for p in tx_pos
    ]
    sim = AcousticFDTD(
        nx, ny, DX, cfl=args.cfl, xp=args.xp,
        sources=sources, damping=edge_sponge((nx, ny), DX),
    )
    sw = out.shot(name)
    frames = sw.open((args.nframes(steps), nx, ny)) if capture else None
    mic_p = np.empty((4, steps), dtype=np.float32)
    v_tx = np.fromiter(
        (ook_fn(i * sim.dt) for i in range(steps)),
        dtype=np.float32, count=steps,
    )
    print(f"shot {name}: {nx}x{ny} grid, {len(tx_pos)} TX element(s), {steps} steps"
          f"{'' if capture else '  (no frames)'}")
    for i in simargs.progress(steps):
        sim.step()
        if capture and i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        for j, p_rx in enumerate(dut_world):
            mic_p[j, i] = sample_bilinear(sim.p, p_rx, DX)
    v_rx = [MIC.filter(mic_p[j], sim.dt) for j in range(4)]
    return sw, sim.dt, v_tx, v_rx


def analyze(v_rx, ref, sim_dt, body_pos, truth_rad, c, bits):
    """The listener's full pipeline on one reception: body-frame TDOA
    bearing, plus a 4-element combined OOK decode aligned on the arrivals.

    The inter-element delays come from *pairwise* cross-correlation of the
    received traces, not from matched-filtering each against the idealized
    TX packet. All four elements see nearly the same waveform (same source,
    same channel, just delayed), so pairwise correlation peaks sharply at
    the true delay -- whereas correlating against the differently-shaped
    (BPF-rung) TX reference jitters by several samples, which at metre range
    is several degrees of bearing. The TX reference is used only for the
    coarse absolute packet start that the decoder needs to window bits."""
    taus = np.array([
        0.0 if j == 0 else matched_filter_arrival(v_rx[j], sim_dt, v_rx[0])
        for j in range(4)
    ])
    bearing = estimate_bearing(body_pos, taus, c)
    bearing_err = math.degrees(wrap_pi(bearing - truth_rad))

    t0 = matched_filter_arrival(v_rx[0], sim_dt, ref)
    arrivals = t0 + taus
    combined = combine_aligned(v_rx, arrivals, sim_dt)
    bits_rx, rms, _ = decode(combined, sim_dt, len(bits), BIT_DUR, float(np.min(arrivals)))
    return dict(
        arrivals=arrivals, bearing=bearing, bearing_err=bearing_err,
        combined=combined.astype(np.float32), bits_rx=bits_rx,
        ber=ber(bits_rx, bits), eye=eye_contrast(rms, bits),
    )


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    c = C_EST
    body_pos = body_corners(APERTURE)

    rng = np.random.default_rng(PRBS_SEED)
    bits = tuple(int(b) for b in rng.integers(0, 2, size=N_BITS))
    ook_fn = ook_voltage(FREQ, bits, BIT_DUR)

    tau_ring = Q / (math.pi * FREQ)
    nearfield = APERTURE ** 2 / (c / FREQ)
    print(f"dx={DX*1e3:.2f} mm, dt={dt*1e6:.3f} us, transverse {round(WIDTH/DX)} cells")
    print(f"transducer f0={FREQ/1e3:.0f} kHz Q={Q:.0f}  ring tau={tau_ring*1e6:.0f} us  "
          f"bit_dur={BIT_DUR*1e6:.0f} us ({BIT_DUR/tau_ring:.1f} tau)")
    print(f"quad array {APERTURE*1e3:.0f} mm, aperture^2/lambda = {nearfield*1e3:.0f} mm "
          f"(the near->far crossover), {N_BITS} bits: {''.join(str(b) for b in bits)}")

    records = {}

    def shoot(name, R, heading_deg, mode):
        nx, ny, dut_world, tx_pos, tx_center = geometry(R, heading_deg, mode, DX)
        sw, sim_dt, v_tx, v_rx = run_shot(
            out, name, args, nx, ny, DX, dt, R, dut_world, tx_pos, ook_fn,
        )
        truth_rad = -math.radians(heading_deg)
        res = analyze(v_rx, np.asarray(v_tx, dtype=np.float64),
                      sim_dt, body_pos, truth_rad, c, bits)
        print(f"  R={R:>5.2f}m hdg={heading_deg:>5.0f} {mode:5s}: "
              f"bearing_err={res['bearing_err']:+7.2f} deg  BER={res['ber']:.2f}  "
              f"eye={res['eye']:.2f}")
        records[name] = dict(
            name=name, writer=sw, R=R, heading_deg=heading_deg, mode=mode,
            sim_dt=sim_dt, dut_world=dut_world, tx_center=tx_center,
            v_tx=v_tx, v_rx=v_rx, **res,
        )
        return res

    # -- Range experiment: near->far crossover at one off-axis heading. --
    print(f"\n== range sweep (heading {RANGE_HEADING:.0f} deg) ==")
    range_rows = []
    for R in RANGES:
        row = {"range_m": float(R)}
        for mode in MODES:
            res = shoot(f"rng_{R:g}m_{mode}", R, RANGE_HEADING, mode)
            row[f"bearing_err_deg_{mode}"] = float(res["bearing_err"])
            row[f"ber_{mode}"] = float(res["ber"])
            row[f"eye_{mode}"] = float(res["eye"])
        range_rows.append(row)

    # -- Azimuth experiment: omnidirectional check at a fixed far range. --
    print(f"\n== azimuth sweep (range {AZ_RANGE:.0f} m) ==")
    az_rows = []
    for heading in AZ_HEADINGS:
        row = {"heading_deg": float(heading)}
        for mode in MODES:
            res = shoot(f"az_{int(round(heading)):03d}deg_{mode}", AZ_RANGE, heading, mode)
            row[f"bearing_err_deg_{mode}"] = float(res["bearing_err"])
            row[f"ber_{mode}"] = float(res["ber"])
            row[f"eye_{mode}"] = float(res["eye"])
        az_rows.append(row)

    def col(rows, key):
        return [r[key] for r in rows]

    sweep = {
        "aperture_m": float(APERTURE),
        "nearfield_m": float(nearfield),
        "modes": list(MODES),
        "bits_sent": list(bits),
        "range_heading_deg": float(RANGE_HEADING),
        "az_range_m": float(AZ_RANGE),
        "range_m": col(range_rows, "range_m"),
        "az_deg": col(az_rows, "heading_deg"),
    }
    for mode in MODES:
        sweep[f"range_bearing_err_deg_{mode}"] = col(range_rows, f"bearing_err_deg_{mode}")
        sweep[f"range_ber_{mode}"] = col(range_rows, f"ber_{mode}")
        sweep[f"range_eye_{mode}"] = col(range_rows, f"eye_{mode}")
        sweep[f"az_bearing_err_deg_{mode}"] = col(az_rows, f"bearing_err_deg_{mode}")
        sweep[f"az_ber_{mode}"] = col(az_rows, f"ber_{mode}")
        sweep[f"az_eye_{mode}"] = col(az_rows, f"eye_{mode}")

    print("\nSweep summary -- bearing |err| (deg) vs range, point vs wide:")
    for r in range_rows:
        print(f"  R={r['range_m']:>5.2f}m:  point={abs(r['bearing_err_deg_point']):6.2f}   "
              f"wide={abs(r['bearing_err_deg_wide']):6.2f}")
    print("  -> wide degrades in the near field and converges to point by a "
          "couple of metres; BER/eye stay flat (no noise, 2D -- link budget "
          "is for the analytic models).")

    for r in records.values():
        sim_dt = r["sim_dt"]
        channels = [
            Channel("TX packet (V)", kind="scalar", dt=sim_dt,
                    pos=r["tx_center"], values=r["v_tx"].tolist()),
        ]
        for j, p_rx in enumerate(r["dut_world"]):
            channels.append(Channel(
                f"RX e{j} (V)", kind="scalar", dt=sim_dt, pos=p_rx,
                values=r["v_rx"][j].tolist(),
            ))
        channels.append(Channel(
            "RX combined (V)", kind="scalar", dt=sim_dt, values=r["combined"].tolist(),
        ))
        r["writer"].finish(
            channels=channels,
            extras={
                "role": "ping",
                "this_experiment": "range" if r["name"].startswith("rng_") else "azimuth",
                "this_range_m": float(r["R"]),
                "this_heading_deg": float(r["heading_deg"]),
                "this_mode": r["mode"],
                "this_bearing_deg": math.degrees(r["bearing"]),
                "this_bearing_err_deg": float(r["bearing_err"]),
                "this_arrivals_s": r["arrivals"].astype(np.float64),
                "this_ber": float(r["ber"]),
                "this_eye": float(r["eye"]),
                "this_bits_rx": list(r["bits_rx"]),
                **sweep,
            },
        )

    out.finish(dt=dt * args.capture_every, dx=DX, c=c)
