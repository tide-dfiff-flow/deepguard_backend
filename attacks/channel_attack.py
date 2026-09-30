"""A_channel: channel / transmission simulation attack (spec §十).

Simulates real-world channel distortions that suppress detector-relied
forensic features: reverberation, resampling, gain change, bandwidth
limitation and mild dynamic-range compression -- the kind of chain social
platforms apply.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly, fftconvolve

from .base import BaseAttack


class ChannelAttack(BaseAttack):
    name = "A_channel"
    params = {
        "rt60": 0.25,          # reverberation time (s)
        "resample_to": 8000,   # bandwidth limitation (None to skip)
        "gain_db": 1.5,        # random gain change
        "compress": True,      # mu-law-ish companding
        "seed": 0,
    }

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = audio.astype(np.float64)

        # 1. reverberation (synthetic exponential-decay noise impulse)
        x = _reverb(x, sample_rate, self.params["rt60"], self.rng)

        # 2. bandwidth limitation via round-trip resampling
        rs = self.params.get("resample_to")
        if rs and rs < sample_rate:
            from math import gcd
            g1 = gcd(int(sample_rate), int(rs))
            x = resample_poly(x, int(rs) // g1, int(sample_rate) // g1)
            g2 = gcd(int(rs), int(sample_rate))
            x = resample_poly(x, int(sample_rate) // g2, int(rs) // g2)

        # 3. random gain
        g = float(10 ** (self.params["gain_db"] / 20.0))
        if self.rng.random() < 0.5:
            g = 1.0 / g
        x = x * g

        # 4. companding (mu-law) - mild dynamic compression
        if self.params["compress"]:
            x = _mulaw_compress(x)

        return x.astype(np.float32)


def _reverb(x, sr, rt60, rng):
    if rt60 <= 0:
        return x
    # build a synthetic room impulse response: diffuse noise * exp decay
    length = int(sr * rt60)
    if length < 8:
        return x
    decay = np.exp(-3.5 * np.arange(length) / max(length, 1))
    ir = rng.standard_normal(length) * decay
    ir[0] = 1.0  # direct path
    ir /= (np.linalg.norm(ir) + 1e-9)
    y = fftconvolve(x, ir, mode="full")[:len(x)]
    # mix dry/wet to avoid over-reverberation
    return 0.7 * x + 0.3 * y


def _mulaw_compress(x, mu=255.0):
    x = np.clip(x, -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
