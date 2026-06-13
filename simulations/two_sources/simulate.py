"""Two tone sources of different frequencies in a 1x1 m tank."""

from buddies import simargs
from buddies.sim import AcousticFDTD, Source, to_numpy, tone

SIZE = 1.0  # m
FREQ = 15_000.0  # Hz; --resolution is relative to this source's wavelength


def run(args, out):
    steps = args.steps(1000)
    n = round(SIZE / args.dx)
    sim = AcousticFDTD(
        n, n, args.dx, cfl=args.cfl, xp=args.xp,
        sources=[
            Source(pos=(0.3, 0.3), waveform=tone(FREQ)),
            Source(pos=(0.7, 0.7), waveform=tone(22_000.0, amplitude=0.5)),
        ],
    )

    frames = out.open((args.nframes(steps), n, n))
    for i in simargs.progress(steps):
        sim.step()
        if i % args.capture_every == 0:
            frames[i // args.capture_every] = to_numpy(sim.p)

    out.finish(dt=sim.dt * args.capture_every, dx=args.dx, c=sim.c)
