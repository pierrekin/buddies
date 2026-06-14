"""Speaker and Microphone: linear transducer models that wrap a Source and
a probe so the link reads as voltage in, voltage out.

A real piezo or audio transducer has a band-pass response: a resonance at
``f0``, a quality factor ``Q``, a peak sensitivity calibrated at that
resonance. Both devices here share that family -- a 2nd-order biquad
band-pass applied in discrete time. The acoustics in between stay in the
FDTD's native units (volume injection rate in m²/s, pressure in Pa); the
transducers convert at the boundaries.

Filtering happens outside the simulation loop:

  * Speaker.source(...) precomputes the entire filtered TX voltage to a
    volume-rate array and hands it to a normal Source via array lookup, so
    AcousticFDTD's per-step source contract is unchanged.
  * Microphone.filter(...) runs after the loop on the raw probed pressure
    trace and returns the voltage you'd read off the device.

Calibration: at the resonance frequency ``f0`` with a 1 V drive, a Speaker
with ``sensitivity_pa=1.0`` produces 1 Pa at 1 m in open water -- the same
convention ``buddies.sim.tone()`` uses. A Microphone with
``sensitivity_v_per_pa=1.0`` reports 1 V at 1 Pa at ``f0``. Off resonance
the band-pass rolls off, so the link is no longer flat: the channel the
modelling experiment learns is exactly this composite shape."""

import math

import numpy as np

from buddies.sim import DENSITY_SEAWATER, SOUND_SPEED_SEAWATER, Source


def biquad_bpf_coeffs(f0, q, dt):
    """RBJ-cookbook constant-skirt-gain band-pass biquad. Returns the
    normalised (b0, b1, b2, a1, a2); a0 has been divided through. Peak gain
    is 1 at ``f0``, falling at -6 dB/octave beyond the -3 dB band ``f0/Q``."""
    omega = 2 * math.pi * f0 * dt
    alpha = math.sin(omega) / (2 * q)
    a0 = 1 + alpha
    b0 = alpha / a0
    b1 = 0.0
    b2 = -alpha / a0
    a1 = -2 * math.cos(omega) / a0
    a2 = (1 - alpha) / a0
    return b0, b1, b2, a1, a2


def biquad_filter(x, b0, b1, b2, a1, a2):
    """Direct Form I biquad on a 1D array, returned as float32. The body is
    a Python loop because the recursion isn't trivially vectorisable; it
    only runs once per simulate, so the cost is negligible against FDTD."""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(len(x)):
        y[n] = b0 * x[n] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x[n]
        y2, y1 = y1, y[n]
    return y.astype(np.float32)


def _tone_w_peak(f0, pressure_pa, at_m=1.0,
                 c=SOUND_SPEED_SEAWATER, rho=DENSITY_SEAWATER):
    """Volume-rate amplitude (m²/s) that yields ``pressure_pa`` Pa at
    ``at_m`` metres in 2D open water at frequency ``f0`` -- the same
    formula ``tone()`` uses. Pulled out so Speaker can call it directly."""
    omega = 2 * math.pi * f0
    return (
        4 * pressure_pa / (rho * omega)
        * math.sqrt(math.pi * (omega / c) * at_m / 2)
    )


class Speaker:
    """Voltage in, FDTD volume-rate out. A 2nd-order band-pass at
    ``(f0, q)``, with a 1 V drive at ``f0`` calibrated to emit
    ``sensitivity_pa`` Pa at 1 m in open water."""

    def __init__(self, f0, q, sensitivity_pa,
                 c=SOUND_SPEED_SEAWATER, rho=DENSITY_SEAWATER):
        self.f0 = f0
        self.q = q
        self.sensitivity_pa = sensitivity_pa
        self.c = c
        self.rho = rho

    def source(self, pos, voltage_fn, steps, dt):
        """Build a ``Source`` that emits the band-pass-filtered version of
        ``voltage_fn(t)``. ``voltage_fn`` is sampled at every step, so it
        can be any plain Python callable; the filtering is offline and the
        per-step waveform is an O(1) array lookup."""
        b0, b1, b2, a1, a2 = biquad_bpf_coeffs(self.f0, self.q, dt)
        v = np.fromiter(
            (voltage_fn(i * dt) for i in range(steps)),
            dtype=np.float64, count=steps,
        )
        # BPF peak gain is 1 at f0; multiply by the V→q calibration to land
        # 1 V at f0 on the sensitivity_pa-Pa-at-1m operating point.
        q = biquad_filter(v, b0, b1, b2, a1, a2) * _tone_w_peak(
            self.f0, self.sensitivity_pa, c=self.c, rho=self.rho,
        )

        def waveform(t):
            i = int(t / dt)
            if 0 <= i < len(q):
                return float(q[i])
            return 0.0

        return Source(pos=pos, waveform=waveform)


class Microphone:
    """FDTD pressure in, voltage out. Same band-pass family as Speaker,
    plus a scalar sensitivity in V/Pa applied at the input. Stateless --
    the device only knows its own response curve; the sim owns the
    receiver position and the recorded pressure trace."""

    def __init__(self, f0, q, sensitivity_v_per_pa):
        self.f0 = f0
        self.q = q
        self.sensitivity_v_per_pa = sensitivity_v_per_pa

    def filter(self, pressure_samples, dt):
        """Run a recorded pressure trace through the device's band-pass
        and sensitivity. Returns a float32 voltage array of the same
        length, ready to drop into a scalar Channel."""
        b0, b1, b2, a1, a2 = biquad_bpf_coeffs(self.f0, self.q, dt)
        scaled = np.asarray(pressure_samples) * self.sensitivity_v_per_pa
        return biquad_filter(scaled, b0, b1, b2, a1, a2)
