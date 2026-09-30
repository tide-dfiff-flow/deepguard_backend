"""A_noise: additive noise attack (spec §十)."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .base import BaseAttack


class NoiseAttack(BaseAttack):
    name = "A_noise"
    params = {
        "kind": "white",     # white | pink | babble | brown
        "snr_db": 20.0,
        "seed": 0,
    }

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = audio.astype(np.float64)
        kind = self.params["kind"]
        snr = self.params["snr_db"]
        noise = _generate_noise(kind, len(x), self.rng)
        # scale noise to target SNR
        sig_p = float(np.mean(x ** 2)) + 1e-12
        noise_p = float(np.mean(noise ** 2)) + 1e-12
        scale = float(np.sqrt(sig_p / noise_p * 10 ** (-snr / 10.0)))
        return (x + scale * noise).astype(np.float32)


def _generate_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "white":
        return rng.standard_normal(n)
    if kind == "pink":
        # Voss-McCartney approximation
        white = rng.standard_normal(n)
        pink = np.zeros(n)
        rows = 16
        for r in range(rows):
            stride = 1 << r
            idx = np.arange(0, n, stride)
            vals = rng.standard_normal(len(idx))
            upsampled = np.repeat(vals, stride)[:n]
            pink += upsampled
        pink /= rows
        return pink
    if kind == "brown":
        white = rng.standard_normal(n)
        return np.cumsum(white) / np.sqrt(n)
    if kind == "babble":
        # filtered white noise mimicking multi-talker murmur
        white = rng.standard_normal(n)
        b = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        a = np.array([1.0, -0.5, 0.2])
        return lfilter(b, a, white)
    return rng.standard_normal(n)
