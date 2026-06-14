"""A 16-element line array focusing its beam on a point in open water."""

from buddies import simargs
from buddies.sim import AcousticFDTD, array, edge_sponge, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz
ELEMENTS = 16
ARRAY_START = (0.2, 0.2)  # m
ARRAY_END = (0.2, 0.8)  # m
FOCUS = (0.7, 0.5)  # m


def run(args, out):
    DX = args.dx
    steps = args.steps(1000)

    n = round(SIZE / DX)
    sim = AcousticFDTD(
        n, n, DX, cfl=args.cfl, xp=args.xp,
        sources=array(
            start=ARRAY_START, end=ARRAY_END, n=ELEMENTS, focus=FOCUS,
            waveform=lambda d: tone(FREQ, delay=d),
        ),
        damping=edge_sponge((n, n), DX),
    )

    shot = out.shot("main")
    frames = shot.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)

    shot.finish()
    out.finish(dt=sim.dt * args.capture_every, dx=DX, c=sim.c)
