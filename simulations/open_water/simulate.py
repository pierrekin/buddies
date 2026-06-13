"""A 15 kHz tone source in open water: sponged edges, no reflections."""

from buddies import simargs
from buddies.sim import AcousticFDTD, Source, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz


def run(args, out):
    DX = args.dx
    steps = args.steps(1000)

    n = round(SIZE / DX)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=[Source(pos=(SIZE / 2, SIZE / 2), waveform=tone(FREQ))],
        damping=edge_sponge((n, n), DX),
    )

    frames = out.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)

    out.finish(dt=sim.dt * args.capture_every, dx=DX, c=sim.c)
