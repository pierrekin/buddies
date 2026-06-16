"""Acoustic channel abstractions.

Each connected buddy has a Channel that produces RX sample blocks on
demand. The Channel hides which fidelity tier is generating samples:

  - TestSignalChannel: fixed sine wave, no propagation modelling.
    Phase-1 plumbing diagnostic.
  - ParametricChannel: geometric free-space delay + amplitude + noise.
    Real-time, tier 1.
  - FirSurrogateChannel: per-channel FIR convolution against TX history.
    Trained offline from FDTD, tier 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Channel(ABC):
    """Streaming RX-side of the acoustic environment.

    `step` advances simulated time by `n_samples / sample_rate` seconds,
    consuming one TX block (None until DAC plumbing lands) and emitting
    an RX block of shape `(n_channels, n_samples)`.
    """

    @abstractmethod
    def step(
        self, n_samples: int, tx_block: np.ndarray | None = None
    ) -> np.ndarray: ...

    @abstractmethod
    def reset(self) -> None: ...


class TestSignalChannel(Channel):
    """Emits a fixed-frequency sine on every channel, ignores TX."""

    def __init__(
        self,
        n_channels: int,
        sample_rate_hz: int,
        freq_hz: float = 40_000.0,
    ) -> None:
        self._n_channels = n_channels
        self._sample_rate = sample_rate_hz
        self._freq = freq_hz
        self._t = 0

    def step(
        self, n_samples: int, tx_block: np.ndarray | None = None
    ) -> np.ndarray:
        t = (np.arange(self._t, self._t + n_samples)).astype(np.float64)
        sig = np.sin(2.0 * np.pi * self._freq * t / self._sample_rate)
        rx = np.tile(sig.astype(np.float32), (self._n_channels, 1))
        self._t += n_samples
        return rx

    def reset(self) -> None:
        self._t = 0
