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
        self,
        n_samples: int,
        tx_block: np.ndarray | None = None,
        **kwargs,
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
        self,
        n_samples: int,
        tx_block: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        t = (np.arange(self._t, self._t + n_samples)).astype(np.float64)
        sig = np.sin(2.0 * np.pi * self._freq * t / self._sample_rate)
        rx = np.tile(sig.astype(np.float32), (self._n_channels, 1))
        self._t += n_samples
        return rx

    def reset(self) -> None:
        self._t = 0


# Chirp + sound-speed constants must match firmware/src/chirp.rs.
SOUND_SPEED_M_PER_S = 1500.0
CHIRP_F_LO_HZ = 30_000.0
CHIRP_F_HI_HZ = 50_000.0
CHIRP_LEN_SAMPLES = 500

# Receiver array geometry in body frame (metres). +y forward, +x right.
# Must match firmware/src/bearing.rs.
RX_POSITIONS_BODY = [
    (-0.03, 0.03),   # front-left
    (0.03, 0.03),    # front-right
    (0.03, -0.03),   # rear-right
    (-0.03, -0.03),  # rear-left
]


class SinglePeerChirpChannel(Channel):
    """RX = low-amplitude Gaussian noise + a copy of the chirp embedded
    at sample offset `delay_samples` (passed each step). Same chirp on
    every channel; TDOA-style per-channel delays come in phase 3.
    """

    NOISE_AMPLITUDE = 0.05

    def __init__(self, n_channels: int, sample_rate_hz: int) -> None:
        self._n_channels = n_channels
        self._sample_rate = sample_rate_hz
        self._chirp = self._make_chirp()
        self._rng = np.random.default_rng()

    def _make_chirp(self) -> np.ndarray:
        n = CHIRP_LEN_SAMPLES
        t = np.arange(n) / self._sample_rate
        duration_s = n / self._sample_rate
        k = (CHIRP_F_HI_HZ - CHIRP_F_LO_HZ) / duration_s
        phase = 2.0 * np.pi * (CHIRP_F_LO_HZ * t + 0.5 * k * t * t)
        return np.sin(phase).astype(np.float32)

    def step(
        self,
        n_samples: int,
        tx_block: np.ndarray | None = None,
        per_channel_delays: list[int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        out = (
            self._rng.standard_normal((self._n_channels, n_samples))
            * self.NOISE_AMPLITUDE
        ).astype(np.float32)
        if per_channel_delays is not None:
            for ch, d in enumerate(per_channel_delays):
                if ch >= self._n_channels:
                    break
                d = int(d)
                if d < 0 or d >= n_samples:
                    continue
                end = min(d + CHIRP_LEN_SAMPLES, n_samples)
                out[ch, d:end] += self._chirp[: end - d]
        return out

    def reset(self) -> None:
        pass
