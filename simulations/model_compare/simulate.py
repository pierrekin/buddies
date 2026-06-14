"""System identification on the transducer link, four shots in one run.

Same physical setup as ``transducer_link`` (speaker + tank + mic), but
driven by four different voltage probes back-to-back:

  * chirp       -- training shot: a linear sweep from 5 to 30 kHz
  * ook         -- test: 16-bit OOK at the carrier frequency
  * tone_burst  -- test: a Gaussian-windowed 15 kHz tone
  * prbs        -- test: short pseudo-random binary sequence

Each becomes a shot in the artifact. After all four FDTD runs, the
chirp shot's (v_tx, v_rx) pair fits each model in ``buddies.channel_model``;
the fitted models then predict v_rx for every shot (including the chirp
itself as a sanity check). Predictions and residuals land as scalar
channels alongside the truth so the viewer stacks them side-by-side, and
the per-shot NRMSE goes into ``extras``.

Only the OOK shot writes frames -- a 17 ms field history is enough to
eyeball the speaker + propagation + mic in action. The other shots
exist as channel + extras data only."""

import math

import numpy as np

from buddies import channel_model, probe, simargs
from buddies.devices import Microphone, Speaker
from buddies.sim import AcousticFDTD, edge_sponge, timestep, to_numpy
from buddies.store import Channel

DEFAULTS = {"capture_every": 16}

# Physical setup -- identical to transducer_link.
SIZE = 1.0  # m
FREQ = 15_000.0  # Hz, transducer resonance + OOK carrier
TX = (0.2, 0.5)  # m
RX = (0.8, 0.5)  # m
SPEAKER = Speaker(f0=FREQ, q=4.0, sensitivity_pa=1.0)
MIC = Microphone(f0=FREQ, q=4.0, sensitivity_v_per_pa=1.0)
PROP_TAIL = 0.002  # s of silence after the probe ends, for the channel to ring down


def linear_chirp(f_lo, f_hi, duration, amplitude=1.0):
    """A linear-frequency sweep from f_lo to f_hi over ``duration``."""
    def v(t):
        if t < 0 or t > duration:
            return 0.0
        k = (f_hi - f_lo) / duration
        phase = 2 * math.pi * (f_lo * t + 0.5 * k * t * t)
        return amplitude * math.sin(phase)
    return v


def ook_square(freq, bits, bit_dur, drive_v=1.0):
    """OOK on a square carrier: same as ``transducer_link``'s drive."""
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


def tone_burst(freq, n_cycles, amplitude=1.0):
    """A half-sine envelope on a ``freq`` carrier, ``n_cycles`` long."""
    duration = n_cycles / freq

    def v(t):
        if t < 0 or t > duration:
            return 0.0
        env = math.sin(math.pi * t / duration)
        return amplitude * env * math.sin(2 * math.pi * freq * t)

    return v


def prbs(bit_dur, n_bits, drive_v=1.0, seed=42):
    """Random +/-1 sequence at the bit rate ``1/bit_dur``."""
    rng = np.random.default_rng(seed)
    levels = (rng.integers(0, 2, size=n_bits).astype(np.float32) * 2 - 1) * drive_v

    def v(t):
        if t < 0:
            return 0.0
        i = int(t / bit_dur)
        if i >= len(levels):
            return 0.0
        return float(levels[i])

    return v


PROBES = [
    {
        "name": "chirp",
        "duration": 0.005,
        "voltage_fn": linear_chirp(5_000.0, 30_000.0, 0.005),
        "role": "train",
        "save_frames": True,
    },
    {
        "name": "ook",
        "duration": 0.016,
        # Shorter than transducer_link's 32 bits to keep the run tractable.
        "voltage_fn": ook_square(
            FREQ, (1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1), 0.001,
        ),
        "role": "test",
        "save_frames": True,
    },
    {
        "name": "tone_burst",
        "duration": 0.0025,
        "voltage_fn": tone_burst(FREQ, n_cycles=30),
        "role": "test",
        "save_frames": True,
    },
    {
        "name": "prbs",
        "duration": 0.006,
        # ~30 us bits put the PRBS main lobe (1/T_b ~ 33 kHz) over the
        # channel's f0 ~ 15 kHz, so the link actually responds.
        "voltage_fn": prbs(0.00003, n_bits=200),
        "role": "test",
        "save_frames": True,
    },
]


def _build_models():
    """The lineup. Recreated per fit so a previous run's params don't leak.

    FIR sizes are chosen to bracket the channel's actual length. At 15 kHz
    with c=1500 m/s and 0.6 m TX→RX, propagation delay alone is ~360
    samples at the default dt, plus ~150 samples of group delay through
    the two Q=4 band-pass biquads. So N=128 is too short (truncates the
    main arrival), N=512 catches most of it, N=1024 is comfortable."""
    return [
        channel_model.IdentityModel(),
        channel_model.ScaleModel(),
        channel_model.ScaleDelayModel(),
        channel_model.FIRModel(n_taps=128),
        channel_model.FIRModel(n_taps=512),
        channel_model.FIRModel(n_taps=1024),
    ]


def run(args, out):
    DX = args.dx
    dt = timestep(DX, cfl=args.cfl)
    n = round(SIZE / DX)

    # -- Phase 1: four FDTD shots, collecting (v_tx, v_rx) per probe. --
    pairs = {}  # name -> (v_tx, v_rx, role, sim_dt)
    shot_writers = {}
    for p in PROBES:
        steps = round((p["duration"] + PROP_TAIL) / dt)
        voltage_fn = p["voltage_fn"]
        sim = AcousticFDTD(
            n, n, DX, cfl=args.cfl, xp=args.xp,
            sources=[SPEAKER.source(pos=TX, voltage_fn=voltage_fn, steps=steps, dt=dt)],
            damping=edge_sponge((n, n), DX),
        )

        sw = out.shot(p["name"])
        shot_writers[p["name"]] = sw
        frames = sw.open((args.nframes(steps), n, n)) if p["save_frames"] else None

        v_tx = np.fromiter(
            (voltage_fn(i * sim.dt) for i in range(steps)),
            dtype=np.float32, count=steps,
        )
        mic_p = np.empty(steps, dtype=np.float32)

        print(f"shot {p['name']}: {steps} steps, {steps * sim.dt * 1e3:.2f} ms")
        for i in simargs.progress(steps):
            sim.step()
            if frames is not None and i % args.capture_every == 0:
                frames[i // args.capture_every] = to_numpy(sim.p)
            mic_p[i] = probe.pressure(sim, RX)

        v_rx = MIC.filter(mic_p, sim.dt)
        pairs[p["name"]] = (v_tx, v_rx, p["role"], sim.dt)

    # -- Phase 2: fit every model on the training shot's pair. --
    train_name = next(p["name"] for p in PROBES if p["role"] == "train")
    v_tx_train, v_rx_train, _, _ = pairs[train_name]
    models = _build_models()
    for m in models:
        m.fit(v_tx_train, v_rx_train)
        print(f"fitted {m.name:>18}: params summary "
              f"{ {k: (round(float(v), 4) if not hasattr(v, 'shape') else f'array({v.shape})') for k, v in m.params().items()} }")

    # -- Phase 3: predictions + NRMSE + per-shot finalization. --
    for p in PROBES:
        name = p["name"]
        v_tx, v_rx, role, sim_dt = pairs[name]
        channels = [
            Channel("TX (V)", kind="scalar", dt=sim_dt, pos=TX, values=v_tx.tolist()),
            Channel("RX truth (V)", kind="scalar", dt=sim_dt, pos=RX, values=v_rx.tolist()),
        ]
        nrmse_map = {}
        for m in models:
            pred = m.predict(v_tx)[: len(v_rx)]
            score = channel_model.nrmse(v_rx, pred)
            nrmse_map[m.name] = score
            channels.append(Channel(
                f"pred {m.name} (V)", kind="scalar", dt=sim_dt, values=pred.tolist(),
            ))
            residual = (np.asarray(v_rx, dtype=np.float32) - pred[: len(v_rx)]).astype(np.float32)
            channels.append(Channel(
                f"residual {m.name} (V)", kind="scalar", dt=sim_dt, values=residual.tolist(),
            ))
            print(f"  {name:>10}  {m.name:>18}  NRMSE = {score:.4f}")

        extras = {"role": role, "nrmse": nrmse_map}
        if role == "train":
            # Pull the fitted parameters into extras so view.py can show
            # them next to the channel coefficients learned.
            params = {}
            for m in models:
                p_map = dict(m.params())
                # FIR's h is an array; everything else is scalars. The
                # array survives via extras.npz, scalars via extras.json.
                # Mix is fine -- they go into a single dict that the
                # writer splits by type.
                for k, v in list(p_map.items()):
                    if hasattr(v, "shape"):
                        # Arrays must live at the dict's top level to land
                        # in extras.npz; nest under "{model}__{key}".
                        extras[f"{m.name}__{k}"] = v
                        del p_map[k]
                params[m.name] = p_map
            extras["fitted_params"] = params

        shot_writers[name].finish(channels=channels, extras=extras)

    out.finish(dt=dt * args.capture_every, dx=DX, c=1500.0)
