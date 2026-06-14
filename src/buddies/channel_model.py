"""Linear channel models for TX->RX system identification.

Each model has the same shape: ``fit(x, y)`` learns parameters from a
training pair of voltage traces; ``predict(x)`` returns the predicted RX
voltage for any TX voltage. They are deliberately closed-form (no
optimizer) and ordered by complexity so the NRMSE curve has somewhere to
go::

    M0  identity     : y_hat = x                       (no fit)
    M1  scale        : y_hat = a x                     (one scalar)
    M2  scale+delay  : y_hat = a x(t - tau)            (gain + integer delay)
    M3  FIR(N)       : y_hat = sum_k h[k] x(t - k)     (N-tap impulse response)

The data-driven impulse response M3 is the most expressive model that
remains LTI. For long enough N it captures speaker BPF + propagation +
mic BPF + any multipath as one filter -- the channel's *learned* response."""

import numpy as np


def nrmse(y_true, y_pred):
    """RMS(error) / RMS(truth). 0 = perfect, 1 = no better than zero."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    err = y_true[:n] - y_pred[:n]
    ref = float(np.sqrt(np.mean(y_true[:n] ** 2)))
    if ref == 0:
        return float("inf") if np.any(err) else 0.0
    return float(np.sqrt(np.mean(err ** 2)) / ref)


class IdentityModel:
    """The trivial baseline: y_hat = x. No parameters; ``fit`` is a no-op."""

    name = "M0_identity"

    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.asarray(x, dtype=np.float32)

    def params(self):
        return {}


class ScaleModel:
    """Single-scalar gain: y_hat = a x. ``a`` is the LS solution."""

    name = "M1_scale"

    def __init__(self):
        self.a = 0.0

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        denom = float(np.dot(x, x))
        self.a = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        return self

    def predict(self, x):
        return (self.a * np.asarray(x, dtype=np.float64)).astype(np.float32)

    def params(self):
        return {"a": self.a}


class ScaleDelayModel:
    """Gain plus integer-sample delay: y_hat[t] = a x[t - tau].

    Delay is the lag of the cross-correlation peak; gain is the LS fit on
    the aligned pair."""

    name = "M2_scale_delay"

    def __init__(self):
        self.a = 0.0
        self.delay = 0

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # argmax of full cross-correlation = lag where y best matches a
        # shifted x. Negative lags would mean y leads x, which can't happen
        # in a causal channel; clip to 0.
        xcorr = np.correlate(y, x, mode="full")
        self.delay = max(0, int(xcorr.argmax()) - (len(x) - 1))
        # Fit gain on the aligned pair.
        if self.delay >= len(x):
            self.a = 0.0
            return self
        xa = x[: len(x) - self.delay]
        ya = y[self.delay : self.delay + len(xa)]
        n = min(len(xa), len(ya))
        xa, ya = xa[:n], ya[:n]
        denom = float(np.dot(xa, xa))
        self.a = float(np.dot(xa, ya) / denom) if denom > 0 else 0.0
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(x)
        if self.delay < len(x):
            y[self.delay :] = self.a * x[: len(x) - self.delay]
        return y.astype(np.float32)

    def params(self):
        return {"a": self.a, "delay": self.delay}


class FIRModel:
    """N-tap linear FIR: y_hat[t] = sum_{k=0..N-1} h[k] x[t - k].

    Fit by least squares on the over-determined convolution system
    ``X h = y``, where ``X`` is the (T, N) lower-triangular Toeplitz
    matrix built from x. For T >> N this is the data-driven impulse
    response of the channel; it lumps speaker BPF + propagation + mic
    BPF together, since the model has no way to disentangle them."""

    def __init__(self, n_taps):
        self.n_taps = int(n_taps)
        self.name = f"M3_FIR_{self.n_taps}"
        self.h = np.zeros(self.n_taps, dtype=np.float32)

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N = self.n_taps
        T = len(x)
        # Wiener-Hopf normal equations: R h = r, where R is the (N, N)
        # Toeplitz autocorrelation matrix of x, and r is the x↔y
        # cross-correlation. Solving R h = r is O(N³) -- still cheap for
        # the N we use here (≤ 1024) and avoids the (T, N) convolution
        # matrix that the direct lstsq form materializes.
        full_xx = np.correlate(x, x, mode="full")
        r_xx = full_xx[T - 1 : T - 1 + N]
        full_xy = np.correlate(y, x, mode="full")
        r_xy = full_xy[T - 1 : T - 1 + N]
        idx = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
        R = r_xx[idx]
        # Mild Tikhonov regularization to keep R well-conditioned when x
        # has narrow-band gaps.
        R = R + 1e-6 * float(r_xx[0]) * np.eye(N)
        self.h = np.linalg.solve(R, r_xy).astype(np.float32)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = np.convolve(x, self.h.astype(np.float64), mode="full")[: len(x)]
        return y.astype(np.float32)

    def params(self):
        return {"n_taps": self.n_taps, "h": self.h}
