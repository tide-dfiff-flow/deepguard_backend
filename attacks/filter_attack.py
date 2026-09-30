"""A_filter: EQ / filtering attack (spec §十).

Randomised equaliser, band-pass / notch / shelving filters that perturb the
spectral forensic features without changing intelligibility much.  All modes
are RMS-matched to the input so the attack stays quality-preserving.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, iirnotch

from .base import BaseAttack


class FilterAttack(BaseAttack):
    name = "A_filter"
    params = {
        "kind": "random_eq",   # random_eq | notch | bandpass | shelf
        "n_stages": 2,
        "q": 2.0,
        "gain_db": 2.5,
        "seed": 0,
    }

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = audio.astype(np.float64)
        kind = self.params["kind"]
        if kind == "notch":
            y = _notch(x, sample_rate, self.params["q"], self.rng)
        elif kind == "bandpass":
            y = _bandpass(x, sample_rate, self.rng)
        elif kind == "shelf":
            y = _shelf(x, sample_rate, self.params["gain_db"], self.rng)
        else:
            y = _random_eq(x, sample_rate, self.params["n_stages"],
                           self.params["gain_db"], self.rng)
        return _match_rms(x, y).astype(np.float32)


def _match_rms(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Scale ``est`` to have the same RMS as ``ref`` (loudness preservation)."""
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    if not np.isfinite(est).all():
        est = np.nan_to_num(est, nan=0.0, posinf=0.0, neginf=0.0)
    r = float(np.sqrt(np.mean(ref ** 2)) + 1e-12)
    e = float(np.sqrt(np.mean(est ** 2)) + 1e-12)
    return est * (r / e)


def _random_eq(x, sr, n_stages, gain_db, rng):
    out = x
    for _ in range(n_stages):
        fc = float(rng.uniform(200, 5000))
        q = float(rng.uniform(1.5, 3.5))
        signed_db = gain_db if rng.random() < 0.5 else -gain_db
        b, a = _peaking(sr, fc, q, signed_db)
        out = lfilter(b, a, out)
    return out


def _peaking(sr, fc, q, gain_db):
    """RBJ peaking EQ.  ``gain_db`` may be negative (attenuation)."""
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    A = float(10 ** (gain_db / 40.0))  # always > 0
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0


def _notch(x, sr, q, rng):
    out = x
    for _ in range(2):
        fc = float(rng.uniform(800, 6000))
        b, a = iirnotch(fc, q, sr)
        out = lfilter(b, a, out)
    return out


def _bandpass(x, sr, rng):
    fc = float(rng.uniform(400, 3500))
    bw = float(rng.uniform(600, 1500))
    b, a = butter(4, [fc / (sr / 2), (fc + bw) / (sr / 2)], btype="band")
    return lfilter(b, a, x)


def _shelf(x, sr, gain_db, rng):
    out = x
    for kind in ("low", "high"):
        fc = float(rng.uniform(300, 2500)) if kind == "low" else float(rng.uniform(3500, 6500))
        b, a = _shelf_biquad(sr, fc, gain_db, kind)
        out = lfilter(b, a, out)
    return out


def _shelf_biquad(sr, fc, gain_db, kind):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    cosw = np.cos(w0)
    if kind == "low":
        b0 = A * ((A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
        b2 = A * ((A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cosw)
        a2 = (A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha
    else:
        b0 = A * ((A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
        b2 = A * ((A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cosw)
        a2 = (A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha
    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0
