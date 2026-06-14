"""The default processor: a master artifact to a display-ready one.

For each shot, it decimates frames in time and quantizes the float32
pressure field to uint8 normalized to a percentile level (so the decaying
wavefront stays visible rather than being swamped by the source peak).
Channels, overlay, and extras pass through unchanged. The level the viewer
used to recompute on every open is baked in here once -- per shot, since
different excitations can have wildly different peak levels.

A simulation can override this by shipping its own ``process.py`` exposing a
``process(master, args, out)`` function; most do not need to.
"""

import numpy as np
from tqdm import tqdm

COLORMAP = "CET-D1A"  # diverging blue-white-red
# Computing the level over every frame of a multi-GB master is slow; this
# many frames, evenly spaced, estimate it.
LEVEL_SAMPLE_FRAMES = 32
# Read and quantize this many frames at a time to bound memory.
CHUNK = 64


def compute_level(frames, percentile):
    """The display half-range: ``percentile`` of |p| over sampled frames."""
    step = max(1, len(frames) // LEVEL_SAMPLE_FRAMES)
    sample = np.abs(np.asarray(frames[::step], dtype=np.float32))
    return float(np.percentile(sample, percentile))


def process(master, args, out):
    for name, shot in master.shots.items():
        out_shot = out.shot(name)
        if shot.frames is None:
            # Channels/extras-only shot: just forward the sidecar pieces.
            out_shot.finish(channels=shot.channels, overlay=shot.overlay, extras=shot.extras)
            continue

        src = shot.frames
        indices = list(range(0, len(src), args.decimate))
        dst = out_shot.open((len(indices), *src.shape[1:]), dtype=np.uint8)

        lim = compute_level(src, args.percentile) or 1.0
        # Map [-lim, lim] Pa onto [0, 255] (128 = zero pressure = white center
        # of the diverging map); values beyond the level saturate.
        scale = 127.5 / lim
        with tqdm(total=len(indices), unit="frame", desc=name) as bar:
            for start in range(0, len(indices), CHUNK):
                block = indices[start : start + CHUNK]
                chunk = np.asarray(src[block], dtype=np.float32)
                dst[start : start + len(block)] = np.clip(
                    chunk * scale + 127.5, 0, 255
                ).astype(np.uint8)
                bar.update(len(block))

        out_shot.finish(
            channels=shot.channels, overlay=shot.overlay, extras=shot.extras,
            lim=lim,
        )

    out.finish(
        dt=master.dt * args.decimate,
        dx=master.dx,
        c=master.c,
        colormap=COLORMAP,
    )
