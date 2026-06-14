"""A single 15 kHz tone source in the center of a 1x1 m tank."""

from buddies import simargs
from buddies.sim import AcousticFDTD, Source, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz


def run(args, out):
    steps = args.steps(1000)
    n = round(SIZE / args.dx)
    sim = AcousticFDTD(
        n, n, args.dx, cfl=args.cfl, xp=args.xp,
        sources=[Source(pos=(SIZE / 2, SIZE / 2), waveform=tone(FREQ))],
    )

    shot = out.shot("main")
    frames = shot.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)

    shot.finish()
    out.finish(dt=sim.dt * args.capture_every, dx=args.dx, c=sim.c)
