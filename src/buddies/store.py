"""On-disk format for simulation artifacts.

Every artifact captures one or more *shots*: separate excitations of the
same physical setup. A shot owns its own field history, channels, overlay,
and extras; the physical setup (dt, dx, c, provenance) is shared and lives
once at the artifact root.

Layout::

    <run>/
      meta.json                     shared: dt, dx, c, provenance
      shots/
        <shot>/
          frames.npy                (nframes, nx, ny) field history, memmap; optional
          channels.npz              recorded channels (metadata + arrays)
          extras.json               sim-specific JSON-able stuff (optional)
          extras.npz                sim-specific arrays (optional)
          overlay.npy               (nx, ny, 4) uint8 overlay (optional)
          meta.json                 per-shot scalars (display level, etc.)

Masters store float32 pressure; processed artifacts store uint8 frames
normalized to a baked display level. Both use this same layout.

The split between shared and per-shot is the only opinionated thing buddies
imposes. Anything sim-specific (decoded bits, fitted models, summary stats)
goes into a shot's ``extras`` dict; buddies never inspects it.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

FRAMES = "frames.npy"
META = "meta.json"
CHANNELS = "channels.npz"
OVERLAY = "overlay.npy"
EXTRAS_JSON = "extras.json"
EXTRAS_NPZ = "extras.npz"
SHOTS_DIR = "shots"

_CHANNEL_META = ("name", "kind", "dt", "pos", "scale", "color")


@dataclass
class Channel:
    """A named time series recorded alongside a shot's frames.

    ``kind`` tells the viewer how to render the values: "scalar", "vector",
    or "color". ``dt`` is the time between samples, which may differ from
    the frame dt. ``pos`` (meters) places the channel in the domain.

    Domain-specific data (model coefficients, decoded bits, eye-fold
    parameters) belongs in the shot's ``extras`` dict, not here.
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


@dataclass(frozen=True)
class Shot:
    """A read view of one shot inside an artifact. ``frames`` is a read-only
    memmap (or None if the shot didn't record a field history)."""

    name: str
    frames: np.ndarray | None
    channels: tuple = ()
    overlay: np.ndarray | None = None
    extras: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Store:
    """A read view of an artifact directory.

    ``shots`` preserves insertion order so the sim's first shot is the
    natural default the viewer opens to."""

    path: str
    meta: dict
    shots: dict  # name -> Shot

    @property
    def dt(self):
        return self.meta["dt"]

    @property
    def dx(self):
        return self.meta["dx"]

    @property
    def c(self):
        return self.meta["c"]


class ShotWriter:
    """The per-shot side of an artifact. Hands out the memmapped frames
    array, then accepts channels/overlay/extras at finish time."""

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.frames = None
        os.makedirs(path, exist_ok=True)

    def open(self, shape, dtype=np.float32):
        """Allocate ``frames.npy`` for this shot and return it as a writable
        memmap. Shots that don't record a field history simply skip this."""
        self.frames = np.lib.format.open_memmap(
            os.path.join(self.path, FRAMES), mode="w+", shape=shape, dtype=dtype
        )
        return self.frames

    def finish(self, *, channels=(), overlay=None, extras=None, **shot_meta):
        """Write the shot's sidecar files. ``shot_meta`` lands in this
        shot's own meta.json (display level, frame counts, anything
        scalar that's specific to this shot)."""
        if self.frames is not None:
            self.frames.flush()

        arrays = {}
        chmeta = []
        for i, ch in enumerate(channels):
            chmeta.append({k: getattr(ch, k) for k in _CHANNEL_META})
            arrays[f"channel_{i}"] = np.asarray(ch.values, dtype=np.float32)
        arrays["channels"] = json.dumps(chmeta)
        np.savez(os.path.join(self.path, CHANNELS), **arrays)

        if overlay is not None:
            np.save(os.path.join(self.path, OVERLAY), overlay)

        if extras:
            json_part, array_part = _split_extras(extras)
            if json_part:
                with open(os.path.join(self.path, EXTRAS_JSON), "w") as f:
                    json.dump(json_part, f, indent=2)
            if array_part:
                np.savez(os.path.join(self.path, EXTRAS_NPZ), **array_part)

        if shot_meta:
            with open(os.path.join(self.path, META), "w") as f:
                json.dump(shot_meta, f, indent=2)


class Writer:
    """The top-level artifact writer. Hands out ``ShotWriter``s and writes
    the shared meta.json once at the end.

    Idiomatic single-shot use::

        shot = out.shot("main")
        frames = shot.open((nframes, nx, ny))
        for i in ...:
            sim.step()
            ...
        shot.finish(channels=(...), extras={...})
        out.finish(dt=sim.dt, dx=dx, c=sim.c)

    Multi-shot: call ``out.shot(name)`` and ``shot.finish(...)`` once per
    shot, then ``out.finish(...)`` once at the end."""

    def __init__(self, path, provenance=None):
        self.path = path
        self._provenance = provenance or {}
        self._shot_names = []
        os.makedirs(os.path.join(path, SHOTS_DIR), exist_ok=True)

    def shot(self, name):
        """Start a new shot. ``name`` becomes its on-disk directory."""
        if name in self._shot_names:
            raise ValueError(f"shot {name!r} already started in this artifact")
        self._shot_names.append(name)
        return ShotWriter(name, os.path.join(self.path, SHOTS_DIR, name))

    def finish(self, *, dt, dx, c, **extra_meta):
        """Write the shared meta.json. ``extra_meta`` lets the
        processor record display-level scalars (colormap, etc.) that
        apply across shots; provenance always wins on key clashes."""
        meta = {"dt": dt, "dx": dx, "c": c, **extra_meta, **self._provenance}
        with open(os.path.join(self.path, META), "w") as f:
            json.dump(meta, f, indent=2)


def _split_extras(extras):
    """Partition ``extras`` into (json-able dict, numpy array dict). Numpy
    arrays go to npz; everything else must be JSON-serializable as-is."""
    json_part, array_part = {}, {}
    for k, v in extras.items():
        if isinstance(v, np.ndarray):
            array_part[k] = v
        else:
            json_part[k] = v
    return json_part, array_part


def open_store(path):
    """Open an artifact directory for reading."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} is not an artifact directory")
    with open(os.path.join(path, META)) as f:
        meta = json.load(f)
    shots_root = os.path.join(path, SHOTS_DIR)
    if not os.path.isdir(shots_root):
        raise FileNotFoundError(f"{path} has no {SHOTS_DIR}/ subdirectory")
    shots = {name: _load_shot(name, os.path.join(shots_root, name))
             for name in sorted(os.listdir(shots_root))
             if os.path.isdir(os.path.join(shots_root, name))}
    if not shots:
        raise ValueError(f"{path} has no shots")
    return Store(path=path, meta=meta, shots=shots)


def _load_shot(name, path):
    frames_path = os.path.join(path, FRAMES)
    frames = np.load(frames_path, mmap_mode="r") if os.path.exists(frames_path) else None

    channels = _load_channels(os.path.join(path, CHANNELS))

    overlay_path = os.path.join(path, OVERLAY)
    overlay = np.load(overlay_path) if os.path.exists(overlay_path) else None

    extras = {}
    json_path = os.path.join(path, EXTRAS_JSON)
    if os.path.exists(json_path):
        with open(json_path) as f:
            extras.update(json.load(f))
    npz_path = os.path.join(path, EXTRAS_NPZ)
    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            for k in data.files:
                extras[k] = data[k]

    meta_path = os.path.join(path, META)
    shot_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            shot_meta = json.load(f)

    return Shot(
        name=name, frames=frames, channels=channels,
        overlay=overlay, extras=extras, meta=shot_meta,
    )


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
