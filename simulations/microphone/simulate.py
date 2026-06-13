"""A tone source in open water observed by every channel kind: a mic
(scalar plot), an energy-flow arrow (vector), a pressure-tinted marker
(color), and total field energy (scalar, no position)."""

from buddies import probe, simargs
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone
from buddies.store import Channel

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
MIC = (0.7, 0.5)  # m
FLOW = (0.35, 0.65)  # m
TINT = (0.5, 0.75)  # m


def run(args, out):
    DX = args.dx
    steps = args.steps(1000)

    n = round(SIZE / DX)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[Source(pos=(SIZE / 2, SIZE / 2), waveform=tone(FREQ))],
        damping=edge_sponge((n, n), DX),
    )

    mic = Channel("mic (Pa)", kind="scalar", dt=sim.dt, pos=MIC)
    flow = Channel("flow", kind="vector", dt=sim.dt, pos=FLOW)
    tint = Channel("tint", kind="color", dt=sim.dt, pos=TINT)
    energy = Channel("field energy (J/m)", kind="scalar", dt=sim.dt)

    frames = out.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)
        mic.append(probe.pressure(sim, MIC))
        flow.append(probe.intensity(sim, FLOW))
        tint.append(probe.pressure(sim, TINT))
        energy.append(probe.energy(sim))

    out.finish(
        dt=sim.dt * args.capture_every, dx=DX, c=sim.c,
        channels=(mic, flow, tint, energy),
    )
