"""On-disk format for simulation artifacts.

Each artifact is a directory, not a single file, so the large field history
can be a plain ``.npy`` that is memory-mapped for streaming reads and writes
(a ``.npz`` zip member can be neither):

    frames.npy     (nframes, nx, ny) field history, memory-mapped
    meta.json      scalar parameters and provenance
    channels.npz   recorded channels (metadata + arrays), small enough to buffer
    overlay.npy    optional (nx, ny, 4) uint8 image drawn over the field

Masters store float32 pressure; processed artifacts store uint8 frames
normalized to a baked display level. Both use this same layout, so one reader
serves both and the differences live in ``meta``.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

FRAMES = "frames.npy"
META = "meta.json"
CHANNELS = "channels.npz"
OVERLAY = "overlay.npy"

_CHANNEL_META = ("name", "kind", "dt", "pos", "scale", "color")


@dataclass
class Channel:
    """A named time series recorded alongside the frames.

    ``kind`` tells the viewer how to render the values: "scalar", "vector",
    or "color". ``dt`` is the time between samples, which may differ from the
    frame dt. ``pos`` (meters) places the channel in the domain.
    """

    name: str
    kind: str
    dt: float  # s between samples
    pos: tuple | None = None  # (x, y) in meters
    # Multiplier from values to meters when drawn in the domain (e.g. 1.0 for
    # a vector already in meters). None = the viewer picks a scale.
    scale: float | None = None
    # RGBA 0-255 for this channel's overlay graphics. None = viewer default.
    color: tuple | None = None
    values: list = field(default_factory=list)

    def append(self, value):
        self.values.append(value)


def open_frames(path, shape, dtype=np.float32):
    """Create ``frames.npy`` under ``path`` and return it as a writable memmap.

    Frames can be written one at a time; none of the array is held in RAM.
    """
    os.makedirs(path, exist_ok=True)
    return np.lib.format.open_memmap(
        os.path.join(path, FRAMES), mode="w+", shape=shape, dtype=dtype
    )


def write_sidecar(path, meta, channels=(), overlay=None):
    """Write meta.json, channels.npz, and optionally overlay.npy into ``path``."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, META), "w") as f:
        json.dump(meta, f, indent=2)

    arrays = {}
    chmeta = []
    for i, ch in enumerate(channels):
        chmeta.append({k: getattr(ch, k) for k in _CHANNEL_META})
        arrays[f"channel_{i}"] = np.asarray(ch.values, dtype=np.float32)
    arrays["channels"] = json.dumps(chmeta)
    np.savez(os.path.join(path, CHANNELS), **arrays)

    if overlay is not None:
        np.save(os.path.join(path, OVERLAY), overlay)


class Writer:
    """Collects an artifact's pieces as they are produced: ``open`` hands out
    the memmapped frames, ``finish`` writes the sidecar."""

    def __init__(self, path, provenance=None):
        self.path = path
        self._provenance = provenance or {}
        self.frames = None

    def open(self, shape, dtype=np.float32):
        self.frames = open_frames(self.path, shape, dtype)
        return self.frames

    def finish(self, *, dt, dx, c, channels=(), overlay=None, **extra):
        if self.frames is not None:
            self.frames.flush()
        # provenance wins on key clashes
        meta = {"dt": dt, "dx": dx, "c": c, **extra, **self._provenance}
        write_sidecar(self.path, meta, channels, overlay)


@dataclass(frozen=True)
class Store:
    """A read view of an artifact directory. ``frames`` is a read-only memmap,
    so a multi-GB history is never fully resident."""

    path: str
    frames: np.ndarray
    meta: dict
    channels: tuple = ()
    overlay: np.ndarray | None = None

    @property
    def dt(self):
        return self.meta["dt"]

    @property
    def dx(self):
        return self.meta["dx"]

    @property
    def c(self):
        return self.meta["c"]


def open_store(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} is not an artifact directory")
    with open(os.path.join(path, META)) as f:
        meta = json.load(f)
    frames = np.load(os.path.join(path, FRAMES), mmap_mode="r")
    channels = _load_channels(os.path.join(path, CHANNELS))
    overlay_path = os.path.join(path, OVERLAY)
    overlay = np.load(overlay_path) if os.path.exists(overlay_path) else None
    return Store(path=path, frames=frames, meta=meta, channels=channels, overlay=overlay)


def _load_channels(path):
    if not os.path.exists(path):
        return ()
    with np.load(path) as data:
        chmeta = json.loads(data["channels"].item())
        return tuple(
            Channel(
                name=m["name"],
                kind=m["kind"],
                dt=m["dt"],
                pos=tuple(m["pos"]) if m["pos"] is not None else None,
                scale=m["scale"],
                color=tuple(m["color"]) if m["color"] is not None else None,
                values=data[f"channel_{i}"],
            )
            for i, m in enumerate(chmeta)
        )
