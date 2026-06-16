"""Truncated-run profiling for the simulate stage.

A profiling run reaches a simulation's normal ``run`` and steps the same loop;
the difference is that ``progress`` (which every loop iterates) warms up, times a
fixed wall-clock window, then aborts the run by raising ``Stop`` once the budget
elapses -- so we measure steady-state throughput in a few seconds instead of
running to completion. Frame writes go to a sink, so no artifact is produced.

Throughput comes from the default mode, which adds only a boolean check to the
step loop. ``--breakdown`` additionally times step/capture/record with a device
sync around each phase; those syncs serialize the GPU, so a breakdown run's
throughput is not comparable -- read its per-phase shares, not its rate.
"""

import time
from contextlib import contextmanager

import numpy

# Read once per step by the hot loop; kept a module global so it costs a single
# attribute lookup when profiling is off.
BREAKDOWN = False

_active = None  # the Profiler driving the current run, or None


class Stop(Exception):
    """Raised from ``progress`` to unwind out of ``run`` when the budget ends."""


def active():
    return _active


def register_grid(nx, ny):
    """Called by the sim so the report can size cell-update throughput."""
    if _active is not None:
        _active.nx, _active.ny = nx, ny


@contextmanager
def section(name):
    """Time a named phase of the step loop. A no-op unless a breakdown is
    running; then it syncs the device around the phase to attribute time."""
    if not BREAKDOWN:
        yield
        return
    _active.sync()
    t = time.perf_counter()
    try:
        yield
    finally:
        _active.sync()
        slot = _active.sections.setdefault(name, [0.0, 0])
        slot[0] += time.perf_counter() - t
        slot[1] += 1


@contextmanager
def session(profiler):
    """Make ``profiler`` the active one for the duration of a run."""
    global _active, BREAKDOWN
    _active, BREAKDOWN = profiler, profiler.breakdown
    try:
        yield profiler
    finally:
        _active, BREAKDOWN = None, False


class Sink:
    """Stands in for the frames memmap: accepts writes and discards them, so a
    profiling run still pays the device->host copy but writes no file."""

    def __setitem__(self, key, value):
        pass

    def flush(self):
        pass


class ProfileWriter:
    """A ``store.Writer`` that produces no artifact."""

    def open(self, shape, dtype=numpy.float32):
        return Sink()

    def finish(self, **kwargs):
        pass


class Profiler:
    def __init__(self, xp, warmup_s=0.2, budget_s=5.0, breakdown=False):
        self.xp = xp
        self.warmup_s = warmup_s
        self.budget_s = budget_s
        self.breakdown = breakdown
        self.nx = self.ny = None
        self.measured_steps = 0
        self.elapsed = 0.0
        self.sections = {}  # name -> [total_s, calls]

    def sync(self):
        if self.xp is not numpy:
            self.xp.cuda.runtime.deviceSynchronize()

    def iterate(self, steps):
        """Yield step indices, warming up before timing a window of ``budget_s``,
        then stopping the run. Measures whatever ran if ``steps`` is too short."""
        start = time.perf_counter()
        warm_until = start + self.warmup_s
        measuring = False
        t0 = first = None
        for i in range(steps):
            if not measuring and time.perf_counter() >= warm_until:
                self.sync()
                t0, first, measuring = time.perf_counter(), i, True
            yield i
            if measuring and time.perf_counter() - t0 >= self.budget_s:
                self._stop(t0, first, i + 1)
        if measuring:
            self._stop(t0, first, steps)
        # Warmup never finished (very short or slow run): time the whole thing.
        self.sync()
        self.elapsed = time.perf_counter() - start
        self.measured_steps = steps
        raise Stop

    def _stop(self, t0, first, last):
        self.sync()
        self.elapsed = time.perf_counter() - t0
        self.measured_steps = last - first
        raise Stop


def format_report(p, args, sim):
    device = "cpu" if p.xp is numpy else "gpu"
    lines = [f"{sim}  {device}  resolution {args.resolution:g}"]
    if not p.measured_steps or not p.elapsed:
        lines.append("  run too short to measure; lower --warmup or raise the resolution")
        return "\n".join(lines)

    rate = p.measured_steps / p.elapsed
    ms = 1e3 / rate
    if p.nx:
        cells = p.nx * p.ny
        lines[0] += f"  grid {p.nx}x{p.ny} ({cells / 1e6:.2f} Mcells)"
    lines.append(f"  measured {p.measured_steps} steps in {p.elapsed:.2f} s")
    line = f"  {rate:,.0f} steps/s   {ms:.3f} ms/step"
    if p.nx:
        line += f"   {cells * rate / 1e6:,.0f} Mcell-updates/s"
    lines.append(line)

    if p.sections:
        total = sum(t for t, _ in p.sections.values())
        lines.append("  breakdown (per-phase sync; throughput above is not comparable):")
        for name, (t, calls) in sorted(p.sections.items(), key=lambda kv: -kv[1][0]):
            lines.append(f"    {name:<8} {t / total:6.1%}   {t / calls * 1e3:.3f} ms/call   {calls} calls")
    return "\n".join(lines)
