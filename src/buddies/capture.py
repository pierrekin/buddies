"""The capture file format: the full recorded pressure history of a
simulation run, optional user-recorded channels, and the parameters
needed to interpret them, as .npz."""

import json
from dataclasses import dataclass, field, fields

import numpy as np


@dataclass
class Channel:
    """A named time series recorded alongside the frames.

    ``kind`` tells the viewer how to render the values: "scalar", "vector",
    or "color". ``dt`` is the time between samples, which may differ from
    the frame dt. ``pos`` (meters) places the channel in the domain.
    """

    name: str
    kind: str
    dt: float  # s between samples
    pos: tuple | None = None  # (x, y) in meters
    # Multiplier from values to meters when drawn in the domain (e.g. 1.0
    # for a vector already in meters). None = the viewer picks a scale.
    scale: float | None = None
    values: list = field(default_factory=list)

    def append(self, value):
        self.values.append(value)


@dataclass(frozen=True)
class Capture:
    frames: np.ndarray  # (steps, nx, ny) pressure history, Pa
    dt: float  # time between frames (s)
    dx: float  # cell size (m)
    c: float  # sound speed (m/s)
    channels: tuple = ()


def save(path, cap):
    arrays = {"frames": cap.frames, "dt": cap.dt, "dx": cap.dx, "c": cap.c}
    meta = []
    for i, ch in enumerate(cap.channels):
        meta.append(
            {"name": ch.name, "kind": ch.kind, "dt": ch.dt, "pos": ch.pos, "scale": ch.scale}
        )
        arrays[f"channel_{i}"] = np.asarray(ch.values, dtype=np.float32)
    arrays["channels"] = json.dumps(meta)
    np.savez(path, **arrays)


def load(path):
    with np.load(path) as data:
        missing = {f.name for f in fields(Capture)} - set(data.files)
        if missing:
            raise ValueError(f"{path} is not a capture file: missing {sorted(missing)}")
        channels = tuple(
            Channel(
                name=m["name"],
                kind=m["kind"],
                dt=m["dt"],
                pos=tuple(m["pos"]) if m["pos"] is not None else None,
                scale=m["scale"],
                values=data[f"channel_{i}"],
            )
            for i, m in enumerate(json.loads(data["channels"].item()))
        )
        return Capture(
            frames=data["frames"],
            dt=float(data["dt"]),
            dx=float(data["dx"]),
            c=float(data["c"]),
            channels=channels,
        )
