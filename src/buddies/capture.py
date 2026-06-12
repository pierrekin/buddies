"""The capture file format: the full recorded pressure history of a
simulation run plus the parameters needed to interpret it, as .npz."""

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class Capture:
    frames: np.ndarray  # (steps, nx, ny) pressure history, Pa
    dt: float  # timestep (s)
    dx: float  # cell size (m)
    c: float  # sound speed (m/s)


def save(path, cap):
    np.savez(path, frames=cap.frames, dt=cap.dt, dx=cap.dx, c=cap.c)


def load(path):
    with np.load(path) as data:
        missing = {f.name for f in fields(Capture)} - set(data.files)
        if missing:
            raise ValueError(f"{path} is not a capture file: missing {sorted(missing)}")
        return Capture(
            frames=data["frames"],
            dt=float(data["dt"]),
            dx=float(data["dx"]),
            c=float(data["c"]),
        )
